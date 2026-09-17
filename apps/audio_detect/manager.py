"""音频采集管理器：多路采集 + 双模型推理的调度（Phase 1/2，spec §9.3）。

职责
----
1. 按 `Camera` 表（`source_type='onvif'` + `is_active=True`）拉起/回收 `AudioCapture`；
2. **每 N 秒重读 Camera 表**——worker 是独立进程，收不到 Django 的 post_save 信号，
   用户加/删摄像头只能靠轮询同步（spec §9.3）；
3. 每 `HEARTBEAT_SEC` 把各路的**内存状态**快照写进 `AudioRuntimeState`：
   PID 文件管"进程在不在"，DB 心跳管"进程是否健康"（spec §9.3）；
4. Phase 2 起在本进程内常驻双模型，交给 `InferenceService` 按 1s hop 出决策窗；
5. Phase 4 起由 `DescribeService` 后台轮询待描述事件，送描述模型（转写 / 结构化）
   生成描述（描述服务不可用只影响描述，不影响采集/检测/通知，spec §6.5）。

模型与采集的状态叠加（spec §10）
------------------------------
采集异常（`capture_error` / `no_audio_profile`）**优先于**模型状态——没有音频
就谈不上检测。采集正常（`ready`）时才用模型状态覆盖：两模型都挂 → `model_error`，
挂一个 → `degraded`。

为什么状态要跨进程落库
--------------------
worker 跑在独立 venv / 独立进程里，web 侧看不到它的内存。心跳 + 每路 status
是 web 侧唯一能判断"音频线是否正常"的依据。
"""

from __future__ import annotations

import logging
import threading

from django.conf import settings
from django.db import close_old_connections

from .assembler import EventAssembler
from .capture import AudioCapture, CaptureConfig, CaptureSnapshot
from .consensus import ConsensusEngine
from .describer import DescribeService
from .detector import DualModelDetector
from .inference import InferenceService
from .label_map import AudioLabelMap
from .models import AudioRuntimeState
from .paths import stop_file_path
from .worker_lock import acquire_worker_slot, release_worker_slot, touch_worker_slot

logger = logging.getLogger(__name__)

# 心跳循环里每隔多少次做一次 DB 连接回收（MySQL wait_timeout 兜底）
_CLOSE_CONN_EVERY = 15

# 每隔多少次心跳打一条 INFO 摘要（其余 DEBUG）。7×24 常驻时每路每 2s 一条
# INFO 约 13 万行/天，量级失控；30s 一条足够看健康趋势。
_HEARTBEAT_LOG_EVERY = 15

# 停止哨兵检查间隔（模型加载期间的独立监视线程用；加载完就交给心跳线程）
_STOP_WATCH_INTERVAL_SEC = 1.0


class AudioCaptureManager:
    """多路音频采集 + 推理的生命周期管理。"""

    def __init__(
        self,
        config: CaptureConfig | None = None,
        camera_ids: set[int] | None = None,
        enable_inference: bool = True,
        enable_describe: bool = True,
        stop_event: threading.Event | None = None,
    ):
        self._cfg = config or CaptureConfig.from_settings()
        self._camera_ids = set(camera_ids) if camera_ids else None
        self._captures: dict[int, AudioCapture] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        #: 命令侧的停止事件（signal handler 用）：本 manager 发现停止哨兵时一并置位，
        #: 让 audio_worker 的主循环也退出（否则它要等 Ctrl+C）
        self._external_stop = stop_event
        #: 心跳线程是否已接管哨兵检查（之前由临时的监视线程负责）
        self._heartbeat_started = threading.Event()

        self.enable_inference = bool(enable_inference)
        self.enable_describe = bool(enable_describe)
        self._inference: InferenceService | None = None
        self._describer: DescribeService | None = None
        self._model_override: str = ""       # '' / model_error / degraded
        self._model_error: str = ""
        #: 本实例的 worker 租约 epoch（start() 抢到锁后写入，stop() 释放）
        self.worker_epoch: str = ""
        #: recover 允许回收的 epoch 集合（接管前的租约 + 历史空 epoch）
        self._stale_epochs: set[str] = {""}
        self._lease_lost = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        # 抢 worker 租约：同一时刻只允许一个实例，抢不到直接抛 WorkerLockError
        # 让调用方拒绝启动（spec §9.3）。
        slot = acquire_worker_slot()
        self.worker_epoch = slot.epoch
        self._stale_epochs = slot.stale_epochs

        # 模型加载是**阻塞**的（15~40s），期间心跳线程还没起来 —— 先用一个极轻的
        # 监视线程保证这段时间也能响应网页的「关闭音频线」（否则要等加载完才处理，
        # 白白触发 grace 超时强杀，留下未释放的租约）。
        watcher = threading.Thread(
            target=self._stop_watch_loop, name="audio-stop-watch", daemon=True,
        )
        watcher.start()

        try:
            self.reload_from_db()

            if self.enable_inference:
                try:
                    self._start_inference()
                except Exception:
                    # 映射表校验失败等：必须先把已经拉起的 ffmpeg 收回来，
                    # 否则进程退出后它们会变成孤儿，继续占着 RTSP 连接。
                    logger.exception("[audio] 推理初始化失败，回收已启动的采集")
                    raise
        except Exception:
            # 启动失败必须还锁，否则用户重试会被自己挡住（心跳超时前）
            self._stop.set()               # 让 watcher 等线程退出
            self._stop_captures()
            self._release_slot()
            raise

        if self._stop.is_set():
            # 加载期间已被请求停止：不再启描述服务/心跳，直接交给命令主循环收尾
            logger.info("[audio] 启动期间收到停止请求 → 跳过后续初始化，直接收尾")
            return

        if self.enable_describe:
            self._start_describer()

        hb = threading.Thread(
            target=self._heartbeat_loop, name="audio-heartbeat", daemon=True,
        )
        poll = threading.Thread(
            target=self._poll_loop, name="audio-camera-poll", daemon=True,
        )
        self._threads = [hb, poll]
        hb.start()
        poll.start()
        self._heartbeat_started.set()       # 哨兵检查交给心跳线程，watcher 退休
        logger.info(
            "[audio] manager started: %d capture(s), heartbeat=%ss poll=%ss inference=%s",
            len(self._captures),
            settings.BABYCARE_AUDIO_HEARTBEAT_SEC,
            settings.BABYCARE_AUDIO_CAMERA_POLL_SEC,
            self.enable_inference,
        )

    def _start_inference(self) -> None:
        """加载映射表 + 双模型并启动窗口调度。

        **映射表校验失败会向上抛**（spec §4.2）：配置写错一个类名必须让 worker
        起不来，而不是悄悄跑成"永远不判阳"。
        """
        label_map = AudioLabelMap.from_settings()     # 校验失败 → raise
        detector = DualModelDetector.from_settings(label_map)
        errors = detector.load()
        if errors:
            self._model_error = "；".join(f"{k}: {v}" for k, v in errors.items())
            if len(errors) >= 2:
                self._model_override = AudioRuntimeState.STATUS_MODEL_ERROR
            else:
                self._model_override = AudioRuntimeState.STATUS_DEGRADED
            logger.error("[audio] 模型加载异常: %s", self._model_error)

        engine = ConsensusEngine.from_settings(label_map.enabled_labels())
        detector.warmup()

        # 回收上次 worker 异常退出遗留的 recording 事件行（worker 是唯一写入方）。
        # 只回收"接管前那个租约"产出的行，不碰其它实例在途的事件。
        EventAssembler.recover_stale_events(self._stale_epochs)

        svc = InferenceService(
            self._captures, detector, engine, label_map,
            enable_events=bool(settings.BABYCARE_AUDIO_EVENTS_ENABLED),
            media_root=str(settings.MEDIA_ROOT),
            worker_epoch=self.worker_epoch,
        )
        self._inference = svc
        svc.announce_models_ready()
        svc.start()

    def _start_describer(self) -> None:
        """启动描述服务（Phase 4）。

        失败不向上抛：描述是异步增强能力（spec §7.4——描述失败不影响通知），
        不能因为描述服务没配好就把整条采集/检测线拖死。

        服务地址可以是同机的 127.0.0.1（单机方案），也可以是局域网内另一台推理机
        （分体方案）——对这里没有区别，只读 `.env` 的 URL。分体部署时"服务能不能
        起来"由那边保证，本进程只管连不上就退避重试（spec §6.5）。
        """
        if not getattr(settings, "BABYCARE_AUDIO_DESCRIBE_ENABLED", True):
            logger.info("[audio] 描述服务已禁用（BABYCARE_AUDIO_DESCRIBE_ENABLED=false）")
            return
        if not getattr(settings, "BABYCARE_AUDIO_DESC_SERVER_URL", ""):
            logger.warning(
                "[audio] BABYCARE_AUDIO_DESC_SERVER_URL 未配置，描述服务不启动；"
                "事件将停留在 pending_description（录音播放不受影响）",
            )
            return
        try:
            from .audio_desc_client import AudioDescClient

            svc = DescribeService(
                client=AudioDescClient.from_settings(),
                media_root=str(settings.MEDIA_ROOT),
                worker_epoch=self.worker_epoch,
                stale_epochs=self._stale_epochs,
            )
            svc.start()
            self._describer = svc
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 描述服务启动失败（采集/检测不受影响）")

    def _stop_captures(self) -> list[AudioCapture]:
        """停掉并清空全部采集，返回被停的列表（初始化失败 / 正常收尾共用）。"""
        with self._lock:
            captures = list(self._captures.values())
            self._captures.clear()
        for cap in captures:
            try:
                cap.stop()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] stop capture cam#%s failed", cap.camera_id)
        return captures

    def stop(self) -> None:
        self._stop.set()
        if self._describer is not None:
            try:
                self._describer.stop()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] stop describer failed")
        if self._inference is not None:
            try:
                self._inference.stop()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] stop inference failed")
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []

        captures = self._stop_captures()

        # 收尾：把 stopped 状态落库，web 侧不会再看到"假活"
        for cap in captures:
            try:
                self._publish_one(cap, status_override=AudioRuntimeState.STATUS_STOPPED)
            except Exception:  # noqa: BLE001
                logger.exception("[audio] publish stopped state failed cam#%s", cap.camera_id)
        close_old_connections()
        self._release_slot()
        logger.info("[audio] manager stopped")

    def _release_slot(self) -> None:
        """释放 worker 租约（幂等）。保留 epoch 供下次启动界定 recover 范围。"""
        epoch, self.worker_epoch = self.worker_epoch, ""
        if not epoch:
            return
        try:
            release_worker_slot(epoch)
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 释放 worker 租约失败")

    # ------------------------------------------------------------------
    # 摄像头同步
    # ------------------------------------------------------------------
    def reload_from_db(self) -> None:
        """重读 Camera 表，与当前在跑的采集做 diff。"""
        from apps.streaming.models import Camera

        qs = Camera.objects.filter(
            source_type=Camera.SOURCE_ONVIF, is_active=True,
        )
        if self._camera_ids is not None:
            qs = qs.filter(id__in=self._camera_ids)
        cams = {c.id: c for c in qs}

        with self._lock:
            existing = set(self._captures)

        # 新增
        for cam_id in sorted(set(cams) - existing):
            cam = cams[cam_id]
            cap = AudioCapture(cam, config=self._cfg)
            with self._lock:
                self._captures[cam_id] = cap
            cap.start()
            logger.info("[audio] capture added: cam#%s %s", cam_id, cam.name)

        # 移除
        for cam_id in sorted(existing - set(cams)):
            with self._lock:
                cap = self._captures.pop(cam_id, None)
            if cap is None:
                continue
            try:
                cap.stop()
                self._publish_one(cap, status_override=AudioRuntimeState.STATUS_STOPPED)
            except Exception:  # noqa: BLE001
                logger.exception("[audio] remove capture cam#%s failed", cam_id)
            if self._inference is not None:
                self._inference.forget_camera(cam_id)
            logger.info("[audio] capture removed: cam#%s", cam_id)

    # ------------------------------------------------------------------
    # 心跳 / 轮询
    # ------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        interval = max(float(settings.BABYCARE_AUDIO_HEARTBEAT_SEC), 0.5)
        tick = 0
        while not self._stop.wait(interval):
            tick += 1
            if self._stop_requested_by_file():
                break               # 收到 web 的优雅停止请求 → 交给主循环收尾
            try:
                self._publish_all(verbose=(tick % _HEARTBEAT_LOG_EVERY == 1))
            except Exception:  # noqa: BLE001
                logger.exception("[audio] heartbeat publish failed")
            self._touch_lease()
            if tick % _CLOSE_CONN_EVERY == 0:
                close_old_connections()

    def _stop_watch_loop(self) -> None:
        """模型加载期间的**临时**停止哨兵监视（心跳线程起来后即退休）。

        `start()` 里 `_start_inference()` 是阻塞的（YAMNet + PANNs 要 15~40s），
        那段时间心跳线程还没启动。没有这条线程的话，用户在加载中点「关闭音频线」
        要等到加载完才被处理 → 超过优雅窗口被强杀 → 租约不释放。
        """
        while not self._stop.wait(_STOP_WATCH_INTERVAL_SEC):
            if self._heartbeat_started.is_set():
                return                      # 心跳线程已接管
            if self._stop_requested_by_file():
                return

    def _stop_requested_by_file(self) -> bool:
        """web 侧的优雅停止请求（哨兵文件）→ 自停。

        为什么用文件而不是信号：Windows 上跨进程给无 console 窗口的 python 进程发
        SIGTERM / CTRL_BREAK 不可靠，而哨兵文件跨平台可靠、且在手动调试
        （不经网页拉起）时根本不存在，不会误触发。

        检测到后置位 ``self._stop`` 并通知命令侧，让主循环走正常收尾流程
        （落 ``stopped`` 状态、释放 worker 租约、收掉 FFmpeg）。
        """
        try:
            path = stop_file_path()
            if not path.exists():
                return False
        except Exception as e:  # noqa: BLE001
            logger.warning("[audio] 检查停止哨兵失败：%s", e)
            return False

        logger.info("[audio] 检测到停止哨兵 %s → 优雅收尾", path)
        self._stop.set()
        if self._external_stop is not None:
            self._external_stop.set()
        return True

    def _touch_lease(self) -> None:
        """心跳续租。租约易主时只告警一次，避免每 2s 刷屏。"""
        try:
            if touch_worker_slot(self.worker_epoch):
                self._lease_lost = False
            elif not self._lease_lost:
                self._lease_lost = True
                logger.warning(
                    "[audio] worker 租约已被其它实例接管（epoch=%s）：本实例应尽快停止，"
                    "否则两边会重复处理同一批事件",
                    (self.worker_epoch or "-")[:8],
                )
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 续租失败")

    def _poll_loop(self) -> None:
        interval = max(float(settings.BABYCARE_AUDIO_CAMERA_POLL_SEC), 1.0)
        while not self._stop.wait(interval):
            try:
                self.reload_from_db()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] camera reload failed")

    # ------------------------------------------------------------------
    # 落库
    # ------------------------------------------------------------------
    def _publish_all(self, verbose: bool = False) -> None:
        with self._lock:
            captures = list(self._captures.values())
        for cap in captures:
            try:
                snap = self._publish_one(cap)
                # 30s 一条 INFO 汇总（其余 DEBUG）：Phase 1 验收期间可用
                # BABYCARE_AUDIO_DETECT_LOG_SEC=0 看每窗推理日志，心跳不用刷屏。
                log = logger.info if verbose else logger.debug
                log("[audio cam#%s] %s", cap.camera_id, snap.as_log())
            except Exception:  # noqa: BLE001
                logger.exception("[audio] publish state failed cam#%s", cap.camera_id)

    def _publish_one(
        self,
        cap: AudioCapture,
        status_override: str | None = None,
    ) -> CaptureSnapshot:
        snap: CaptureSnapshot = cap.snapshot()
        status = self._effective_status(status_override or snap.status)
        error = snap.error or ""
        if status in (
            AudioRuntimeState.STATUS_MODEL_ERROR,
            AudioRuntimeState.STATUS_DEGRADED,
        ) and not error:
            error = self._model_error

        yamnet_version, panns_version = self._model_versions()
        AudioRuntimeState.objects.update_or_create(
            camera_id=cap.camera_id,
            defaults={
                "status": status,
                "last_packet_at": snap.last_packet_at,
                "coverage_ok_from_ts": snap.coverage_ok_from_ts,
                "last_error": error[:2000],
                "yamnet_version": yamnet_version,
                "panns_version": panns_version,
            },
        )
        # 状态变化 → 落一条审计窗（spec §5.1 的第三类落库条件）
        if self._inference is not None:
            try:
                self._inference.notify_state_change(cap.camera_id, status, detail=error)
            except Exception:  # noqa: BLE001
                logger.exception("[audio cam#%s] 状态变化通知失败", cap.camera_id)
        return snap

    def _effective_status(self, capture_status: str) -> str:
        """采集异常优先于模型状态（spec §10）。"""
        if capture_status != AudioRuntimeState.STATUS_READY:
            return capture_status
        return self._model_override or capture_status

    def _model_versions(self) -> tuple[str, str]:
        if self._inference is None:
            return "", ""
        st = self._inference.detector.status
        return str(st["yamnet"]["version"]), str(st["panns"]["version"])

"""
VideoStreamManager：维护每路摄像头的采集线程。

- 线程模型：threading（单进程，与推理线程共享进程）
- reload_from_db()  按 Camera 表拉起/关闭线程（动态：支持新增/删除/启停）
- _StreamWorker.run() 循环 read() → FrameBus.publish() → t0.set()（首次成功）

异构源处理：未实现的 source_type 跳过 + warning，不阻塞其他摄像头。

t0 写入策略（v6）
----------------
- 文件源：在 _StreamWorker.run() 首次 publish 成功后写 cam_t0_manager.set(cam.id, ts)
  注意：**视频文件循环回 0 不重置 t0**（确认：时间一直往后走）
- ONVIF 源：同文件源；首帧 publish 成功后写一次
"""

from __future__ import annotations

import logging
import threading
import time
import traceback

from .cam_t0 import CamT0Manager
from .frame_bus import FrameBus
from .sources.base import CameraSource
from .sources.file_source import FileSource
from .sources.onvif_source import OnvifSource


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class _StreamWorker:
    """单路摄像头采集线程。"""

    def __init__(self, camera_id: int, source: CameraSource):
        self.camera_id = camera_id
        self.source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_set = False  # v6：每路 cam 首次 publish 后 set t0，不重置

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name=f"stream-{self.camera_id}", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self.source.release()
        except Exception:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._cleanup_external_state()
        self._thread = None
        self._t0_set = False

    def _cleanup_external_state(self) -> None:
        """清理 FrameBus / CamT0 中的 cam 状态（被 stop() 和熔断退出共用）。"""
        try:
            FrameBus.instance().disable_ringbuffer(self.camera_id)
        except Exception:
            pass
        try:
            CamT0Manager.instance().remove(self.camera_id)
        except Exception:
            pass

    def is_thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        bus = FrameBus.instance()
        t0 = CamT0Manager.instance()
        source = self.source

        # 简易重启熔断：连续 5 次 open/read 失败就停止线程（外部 watchdog 会重试）
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        while not self._stop.is_set():
            try:
                if not source.open():
                    logger.error("[stream %d] open failed, retry in 5s", self.camera_id)
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.error("[stream %d] too many open failures, worker exits",
                                     self.camera_id)
                        return
                    self._stop.wait(5.0)
                    continue
            except Exception:
                logger.error("[stream %d] open exception:\n%s",
                             self.camera_id, traceback.format_exc())
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    return
                self._stop.wait(5.0)
                continue

            # open 成功 → reset 熔断计数
            consecutive_errors = 0
            logger.info("[stream %d] %s source opened, fps=%.1f",
                        self.camera_id, source.info.source_type, source.info.fps)

            # 主循环：read() → publish
            try:
                while not self._stop.is_set():
                    try:
                        frame = source.read()
                    except Exception:
                        logger.error("[stream %d] read exception:\n%s",
                                     self.camera_id, traceback.format_exc())
                        # 走 release+reopen 路径
                        break
                    if frame is None:
                        # 暂时没帧（文件源末尾 / onvif 拉流阻塞）
                        time.sleep(0.1)
                        continue

                    # 写 FrameBus（ref swap）
                    bus.publish(self.camera_id, frame)

                    # v6：首次 publish 后 set t0（视频循环 / onvif 都不再重置）
                    if not self._t0_set:
                        ts_int = int(time.time())
                        t0.set(self.camera_id, ts_int)
                        self._t0_set = True
                        logger.info("[stream %d] t0 set to %d (first publish)",
                                    self.camera_id, ts_int)
            finally:
                try:
                    source.release()
                except Exception:
                    pass

            # 跳出主循环（read 失败或被 stop）→ 退避后重连
            if self._stop.is_set():
                return
            logger.warning("[stream %d] source lost, retry in 5s", self.camera_id)
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logger.error("[stream %d] too many errors, worker exits", self.camera_id)
                self._cleanup_external_state()
                return
            self._stop.wait(5.0)


# ---------------------------------------------------------------------------
class VideoStreamManager:
    """全局单例：维护 per-cam _StreamWorker。"""

    _instance: "VideoStreamManager | None" = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._workers: dict[int, _StreamWorker] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "VideoStreamManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def reload_from_db(self) -> None:
        """
        按 Camera 表当前状态拉起/关闭线程。

        - 新增 / is_active False→True / 字段变化 → 重建 worker
        - 删除 / is_active True→False → 停 worker
        - 其余保持

        锁策略：仅在"算 diff"和"提交新 worker 字典"持锁；
        实际的 stop() / _build_source() / start() 在锁外执行，
        避免 OnvifSource 网络慢操作期间阻塞整个 manager。
        """
        # 延迟导入避免 AppConfig.ready() 时还没注册模型
        from .models import Camera

        try:
            cams = list(Camera.objects.filter(is_active=True))
        except Exception as e:
            # 数据库还没 migrate 时跑 here 也不崩
            logger.warning("[streaming] reload_from_db: query failed: %s", e)
            return
        active_ids = {c.id for c in cams}

        # 阶段 1：在锁内算出"要 stop 的 id 列表"和"要 start 的 cam 列表"
        with self._lock:
            stop_ids = [cam_id for cam_id in self._workers.keys() if cam_id not in active_ids]
            # 重建列表：active 但 worker 不存在 或 worker 死了
            start_cams = []
            for cam in cams:
                existing = self._workers.get(cam.id)
                if existing is not None and existing.is_thread_alive():
                    continue  # 活着，继续用
                start_cams.append(cam)
            # 死 worker：从字典移除（cleanup 留到锁外做）
            for cam in cams:
                existing = self._workers.get(cam.id)
                if existing is not None and not existing.is_thread_alive():
                    logger.warning("[streaming] dead worker for cam=%d, recreating", cam.id)
                    self._workers.pop(cam.id)

        # 阶段 2：锁外执行 stop()（可能慢，但不再持锁）
        for cam_id in stop_ids:
            with self._lock:
                worker = self._workers.pop(cam_id, None)
            if worker is not None:
                logger.info("[streaming] stopping worker for cam=%d", cam_id)
                worker.stop()
                FrameBus.instance().remove(cam_id)

        for cam in start_cams:
            source = self._build_source(cam)
            if source is None:
                continue
            worker = _StreamWorker(camera_id=cam.id, source=source)
            with self._lock:
                # 二次检查：可能在前面 stop() 期间又 reload 了一次
                if cam.id in self._workers:
                    continue
                self._workers[cam.id] = worker
            worker.start()
            logger.info("[streaming] started worker for cam=%d (%s)",
                        cam.id, cam.source_type)

    # ------------------------------------------------------------------
    def on_cams_changed(self, cam_ids) -> None:
        """由 SamplerManager 回调：active cam 列表变了 → 同步采集线程。

        Sampler 是"主"（reload 后主动通知）；Streaming 是"从"（被动同步）。
        这样保证：新增 Camera 时采集线程和采样线程**都**会启。
        """
        try:
            self.reload_from_db()
        except Exception as e:
            logger.warning("[streaming] on_cams_changed reload failed: %s", e)

    def _build_source(self, cam) -> CameraSource | None:
        if cam.source_type == cam.SOURCE_FILE:
            path = cam.resolved_file_path()
            if not path:
                logger.warning("[streaming] cam=%d file_path is empty, skip", cam.id)
                return None
            return FileSource(path)
        if cam.source_type == cam.SOURCE_ONVIF:
            if not cam.onvif_host or not cam.onvif_username:
                logger.warning(
                    "[streaming] cam=%d ONVIF host/username missing, skip",
                    cam.id,
                )
                return None
            return OnvifSource(cam.onvif_host, cam.onvif_port, cam.onvif_username)
        logger.warning("[streaming] cam=%d unknown source_type=%s",
                       cam.id, cam.source_type)
        return None

    # ------------------------------------------------------------------
    def stop_all(self) -> None:
        with self._lock:
            for cam_id in list(self._workers.keys()):
                self._workers.pop(cam_id).stop()
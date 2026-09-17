"""音频 worker 进程管理（spec §9.3，Phase 5）。

职责
----
音频线是一个**独立进程**（`<venv-audio>/python.exe manage.py audio_worker`），
web 侧（daphne）只负责它的一生——本模块是唯一入口：

| 方法 | 作用 |
|---|---|
| :meth:`AudioWorkerManager.start` | Popen 拉起 worker；stdout/stderr 落 ``data/audio_worker.log``；写 PID 文件；**把 WorkerLockError 转成可读错误**（不静默失败） |
| :meth:`AudioWorkerManager.stop` | 优雅停 → 超时 ``taskkill /F /T /PID``（FFmpeg 是**孙进程**，只杀 worker 会留下孤儿） |
| :meth:`AudioWorkerManager.is_running` | PID 信号：worker 进程是否存活（用于**启停**判断） |
| :meth:`AudioWorkerManager.status` | 汇总页面上要展示的一切（PID + DB 心跳 + 各摄像头状态） |
| :meth:`AudioWorkerManager.maybe_autostart` | web 启动期：``ENABLED and desired == "on"`` 才拉起 |
| :meth:`AudioWorkerManager.set_desired` | 持久化用户期望状态（``AudioServiceState.desired``） |

两条独立信号，分工明确（spec §9.3「健康判定」）
----------------------------------------------
- **PID 文件**（``data/audio_worker.pid``）→ 启停判断：进程还在不在；
- **DB 心跳**（``AudioRuntimeState.updated_at``）→ 健康判断：worker 每
  ``BABYCARE_AUDIO_HEARTBEAT_SEC`` 刷新，web 侧超过
  ``BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC`` 未见更新 → **"卡死"**。

两条信号再加上**启动宽限期**（``BABYCARE_AUDIO_WORKER_STARTUP_GRACE_SEC``）组合出
控制页上的三态：``starting``（刚拉起、模型还在加载，心跳尚未出现）→ 正常 →
**``stale``（卡死）** = 过了宽限期仍无心跳（进程活着却没在干活）。
只看 PID 会把"卡死"误报成"正常运行"，只看心跳会把"启动中"误报成"卡死"。

为什么 ``set_desired`` 不碰 ``updated_at``
-----------------------------------------
``AudioServiceState.updated_at`` 是 **worker 租约心跳**（见 worker_lock.py），
web 侧刷它等于伪造心跳：轻则让一个已死的 worker 看起来还活着、重则挡住一次
合法接管。因此本模块只用 ``.update(desired=..., reason=...)`` 定点改两个字段。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .capture import terminate_process_tree
from .models import AudioRuntimeState, AudioServiceState
from .paths import log_file_path, pid_file_path, stop_file_path
from .worker_lock import _pid_alive

logger = logging.getLogger(__name__)

#: spawn 后等多久确认"没早退"；锁冲突 / 配置错误都会在这段时间内暴露。
#: 注意与 ``BABYCARE_AUDIO_WORKER_STARTUP_GRACE_SEC``（45s）区分：那个是"不判
#: 卡死的宽限期"（面向状态展示），本常量只用于启动确认。
DEFAULT_SPAWN_CONFIRM_SEC = 3.0
#: 停止时等进程退出的秒数（Windows taskkill 是阻塞调用，这里作为超时上限）
_STOP_TIMEOUT_SEC = 5.0
#: 失败时回给页面的日志尾巴长度
_LOG_TAIL_CHARS = 800
#: 每次最多从日志尾部读多少字节（避免全量读大文件）
_LOG_READ_MAX_BYTES = 64 * 1024
#: 启动确认的轮询间隔
_POLL_INTERVAL_SEC = 0.2


class AudioWorkerStartError(RuntimeError):
    """音频 worker 启动失败（总开关关闭 / 解释器缺失 / 子进程早退）。"""


def _kill_pid_tree(pid: int, timeout: float = _STOP_TIMEOUT_SEC) -> None:
    """杀进程树。

    Windows **必须 ``/T``**：FFmpeg 是 worker 的孙进程，只杀 worker 会留下多个
    FFmpeg 继续占着 RTSP 连接（spec §3.2 / §9.3）。
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=timeout,
            )
            return
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return
            time.sleep(_POLL_INTERVAL_SEC)
        logger.warning("[audio] pid=%s 未响应 SIGTERM，改用 SIGKILL", pid)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass                        # 已经退出：目标达成
    except Exception as e:          # noqa: BLE001
        logger.warning("[audio] kill pid tree %s failed: %s", pid, e)


class AudioWorkerManager:
    """音频 worker 的进程管理器（web 进程内单例）。"""

    _instance: "AudioWorkerManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._started_at: float = 0.0

    @classmethod
    def instance(cls) -> "AudioWorkerManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------
    # 配置 / 路径
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """总开关：``BABYCARE_AUDIO_ENABLED``（false → 永不启动）。"""
        return bool(getattr(settings, "BABYCARE_AUDIO_ENABLED", False))

    @property
    def python_bin(self) -> str:
        """音频线解释器（独立 venv，spec §9.2）。"""
        configured = str(getattr(settings, "BABYCARE_AUDIO_PYTHON", "") or "")
        return configured or sys.executable

    def _pid_file(self) -> Path:
        return pid_file_path()

    def _log_file(self) -> Path:
        return log_file_path()

    def _stop_file(self) -> Path:
        return stop_file_path()

    @staticmethod
    def _heartbeat_timeout_sec() -> float:
        return float(getattr(settings, "BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC", 10))

    @staticmethod
    def _startup_grace_sec() -> float:
        """worker 启动后多久内**不判"卡死"**。

        刚拉起的 worker 要先加载 YAMNet + PANNs（实测 ~15s）再等首包
        （``BABYCARE_AUDIO_START_TIMEOUT_SEC``），这段窗口里心跳本来就不会出现。
        没有宽限期的话，用户点完按钮一刷新就看到"卡死"，属误报。
        """
        return float(
            getattr(settings, "BABYCARE_AUDIO_WORKER_STARTUP_GRACE_SEC", 45),
        )

    @staticmethod
    def _stop_grace_sec() -> float:
        """等 worker 优雅收尾的上限（超时才强杀）。"""
        return float(getattr(settings, "BABYCARE_AUDIO_STOP_GRACE_SEC", 10))

    def _in_startup_grace(self) -> bool:
        """是否处于"本进程刚 spawn 的 worker"的启动宽限期内。

        ``_started_at`` 只有本 web 进程亲自 spawn 过才有值；daphne 重启后
        接管一个已在运行的旧 worker 时它是 0 → 不享宽限（该由心跳判真伪）。
        """
        if self._started_at <= 0:
            return False
        return (time.time() - self._started_at) < self._startup_grace_sec()

    # ------------------------------------------------------------------
    # PID 文件
    # ------------------------------------------------------------------
    def _read_pid(self) -> int | None:
        try:
            raw = self._pid_file().read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            return None
        return int(raw) if raw.isdigit() else None

    def _write_pid(self, pid: int) -> None:
        try:
            path = self._pid_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(pid), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("[audio] 写 PID 文件失败：%s", e)

    def _clear_pid(self) -> None:
        try:
            self._pid_file().unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("[audio] 删除 PID 文件失败：%s", e)

    def _clear_stale_pidfile(self) -> None:
        """启动前清理残留 PID 文件（写它的进程已经不在了）。"""
        pid = self._read_pid()
        if pid is not None and not _pid_alive(pid):
            logger.info("[audio] 清理残留 PID 文件（pid=%s 已不存在）", pid)
            self._clear_pid()

    def _write_stop_file(self) -> None:
        """请 worker 优雅停止（见 :func:`apps.audio_detect.paths.stop_file_path`）。"""
        try:
            path = self._stop_file()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stop", encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("[audio] 写停止哨兵失败：%s", e)

    def _clear_stop_file(self) -> None:
        """清掉停止哨兵。

        **启动前必须清**：上一个 worker 的残留哨兵会让新实例起来就自杀。
        """
        try:
            self._stop_file().unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("[audio] 删除停止哨兵失败：%s", e)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """PID 信号：worker 进程是否存活（用于**启停**判断）。"""
        pid = self._read_pid()
        if pid is None:
            return False
        if self._proc is not None and self._proc.pid == pid:
            return self._proc.poll() is None
        return _pid_alive(pid)

    def _service_state(self) -> AudioServiceState | None:
        """读 ``AudioServiceState`` 单行（读不到返回 None，不抛）。"""
        try:
            return AudioServiceState.objects.filter(
                pk=AudioServiceState.SINGLETON_PK,
            ).first()
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读 AudioServiceState 失败")
            return None

    def current_desired(self) -> str:
        """读用户期望状态。三种情况：

        - **行存在** → 以行里的值为准（用户手动开/关过的明确意愿）；
        - **行不存在** → 用户从未表过态 → 跟随总开关（返回 ``on``）：
          ``BABYCARE_AUDIO_ENABLED=true`` 就等于"要跑"，首次部署不必再手点一次；
        - **读 DB 出错** → 保守返回 ``off``（宁可不起，不可乱起）。

        注意第 2 条**不削弱**"手动关闭后重启 daphne 不再拉起"：用户点过关闭就会写入
        ``desired=off`` 的行，之后一直以行为准。
        """
        try:
            row = AudioServiceState.objects.filter(
                pk=AudioServiceState.SINGLETON_PK,
            ).first()
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读 desired 失败，按 off 处理")
            return AudioServiceState.DESIRED_OFF
        if row is None:
            return AudioServiceState.DESIRED_ON
        return row.desired

    def set_desired(self, desired: str, reason: str = "") -> None:
        """写用户期望状态（``desired`` / ``reason`` 两个字段，**不动心跳**）。

        ``updated_at`` 是 worker 租约心跳，见模块文档。
        """
        try:
            AudioServiceState.objects.update_or_create(
                pk=AudioServiceState.SINGLETON_PK,
                defaults={
                    "desired": desired,
                    "reason": reason or AudioServiceState.REASON_MANUAL,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 写 desired=%s 失败", desired)

    def _heartbeat_status(self) -> dict:
        """DB 心跳：``AudioRuntimeState.updated_at`` 是否在超时内。"""
        timeout = self._heartbeat_timeout_sec()
        try:
            latest = (
                AudioRuntimeState.objects
                .order_by("-updated_at")
                .values_list("updated_at", flat=True)
                .first()
            )
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读 AudioRuntimeState 心跳失败")
            return {"ok": False, "age": None}
        if latest is None:
            return {"ok": False, "age": None}
        age = (timezone.now() - latest).total_seconds()
        return {"ok": age <= timeout, "age": age}

    def _camera_states(self) -> list[dict]:
        """各摄像头的音频运行状态（页面表格用）。"""
        try:
            rows = (
                AudioRuntimeState.objects
                .select_related("camera")
                .order_by("camera_id")
            )
            return [
                {
                    "camera_id": r.camera_id,
                    "camera_name": getattr(r.camera, "name", "") or f"cam#{r.camera_id}",
                    "status": r.status,
                    "status_display": r.get_status_display(),
                    "last_packet_at": r.last_packet_at,
                    "last_error": r.last_error,
                    "updated_at": r.updated_at,
                }
                for r in rows
            ]
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读摄像头音频状态失败")
            return []

    def status(self) -> dict:
        """汇总状态：PID 管启停、心跳管健康（spec §9.3）。"""
        pid = self._read_pid()
        running = self.is_running()
        hb = self._heartbeat_status()
        row = self._service_state()
        # 三态：starting（刚起、还没心跳）→ ok（心跳新鲜）→ stale（卡死）
        in_grace = running and self._in_startup_grace()
        return {
            "enabled": self.enabled,
            "desired": row.desired if row else AudioServiceState.DESIRED_OFF,
            "running": running,
            # ``pid`` 来自 PID 文件：Windows 下是 venv 启动器（Popen 句柄，
            # 也是 ``taskkill /T`` 的根）；租约里的 ``lease_pid`` 才是 worker
            # 真实进程（启动器的子进程）。两者都要看，页面同时展示。
            "pid": pid if running else None,
            "lease_pid": row.pid if row else None,
            "worker_epoch": (row.worker_epoch or "") if row else "",
            # 「启动中」：刚拉起、模型还在加载，心跳尚未出现（不是故障）
            "starting": running and not hb["ok"] and in_grace,
            # 「卡死」：过了启动宽限期，进程活着但心跳停了
            "stale": running and not hb["ok"] and not in_grace,
            "heartbeat_ok": hb["ok"],
            "heartbeat_age_sec": hb["age"],
            "heartbeat_timeout_sec": self._heartbeat_timeout_sec(),
            "startup_grace_sec": self._startup_grace_sec(),
            "cameras": self._camera_states(),
            "python": self.python_bin,
            "pid_file": str(self._pid_file()),
            "log_file": str(self._log_file()),
            "uptime_sec": (
                time.time() - self._started_at
                if running and self._started_at > 0 else 0.0
            ),
            # 描述服务：地址/模型/方言都来自 .env。分体部署（模型在别的机器）时
            # 这里显示的 URL 就是那台推理机，方便在控制页确认"描述到底发去哪了"。
            "audio_desc_configured": bool(
                getattr(settings, "BABYCARE_AUDIO_DESC_SERVER_URL", ""),
            ),
            "audio_desc_url": str(
                getattr(settings, "BABYCARE_AUDIO_DESC_SERVER_URL", ""),
            ),
            "audio_desc_model": str(getattr(settings, "BABYCARE_AUDIO_DESC_MODEL", "")),
            "audio_desc_provider": str(
                getattr(settings, "BABYCARE_AUDIO_DESC_PROVIDER", ""),
            ),
            # 描述队列与熔断：分体部署要能一眼看出"攒了多少 / 卡在哪台机器上"
            "audio_desc_breaker": self._breaker_snapshot(),
            **self._queue_stats(),
        }

    @staticmethod
    def _breaker_snapshot() -> dict:
        """ASR 描述服务的熔断状态（给控制页显示）。

        **读 DB 行，不读进程内单例** —— 熔断状态住在 `audio_worker` 进程里；
        daphne 进程里 `LLMBreaker.for_service(ASR)` 永远是全新的 CLOSED，
        读它会显示"一切正常"，正好把问题藏起来（见 `LLMHealthState` 的模型说明：
        落库的理由就是**可观测性**）。
        """
        try:
            from apps.core.models import LLMHealthState

            row = LLMHealthState.objects.filter(
                service=LLMHealthState.SERVICE_ASR,
            ).first()
            return {
                "enabled": bool(
                    getattr(settings, "BABYCARE_LLM_BREAKER_ENABLED", True)
                ),
                "state": row.state if row else LLMHealthState.STATE_CLOSED,
                "consecutive_failures": row.consecutive_failures if row else 0,
                "last_failure_at": row.last_failure_at if row else None,
                "last_probe_at": row.last_probe_at if row else None,
                "last_success_at": row.last_success_at if row else None,
                "last_error": (row.last_error or "") if row else "",
            }
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读熔断状态失败")
            return {}

    def _queue_stats(self) -> dict:
        """描述队列深度 + 最老一条的年龄。

        `queue_oldest_age_sec` 是"积压多久了"最直观的指标：配合
        `QUEUE_MAX_AGE_SEC` 能看出"是不是快被收割了"，也用来判断控制页上
        "待描述"数字不归零到底是**还在攒**还是**卡死**。
        """
        try:
            from django.utils import timezone

            from .models import AudioEvent

            qs = AudioEvent.objects.filter(
                status__in=[
                    AudioEvent.STATUS_PENDING_DESCRIPTION,
                    AudioEvent.STATUS_DESCRIBING,
                ],
            )
            oldest = (
                qs.order_by("started_at_ts")
                .values_list("started_at_ts", flat=True)
                .first()
            )
            age = None
            if oldest:
                age = max(0, int(timezone.now().timestamp()) - int(oldest))
            return {
                "queue_pending": qs.filter(
                    status=AudioEvent.STATUS_PENDING_DESCRIPTION
                ).count(),
                "queue_describing": qs.filter(
                    status=AudioEvent.STATUS_DESCRIBING
                ).count(),
                "queue_oldest_age_sec": age,
            }
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 读描述队列统计失败")
            return {
                "queue_pending": 0,
                "queue_describing": 0,
                "queue_oldest_age_sec": None,
            }

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _log_size(self) -> int:
        try:
            return self._log_file().stat().st_size
        except Exception:  # noqa: BLE001
            return 0

    def tail_log(self, since: int = 0, max_chars: int = _LOG_TAIL_CHARS) -> str:
        """读日志尾部；``since`` > 0 时只读该字节偏移之后的内容（本次启动新增）。

        只读尾部一段（见 ``_LOG_READ_MAX_BYTES``）：worker 是 7×24 常驻，日志
        可能已到几百 MB，``read_bytes()`` 全量读会把 web 进程拖垮。
        """
        try:
            path = self._log_file()
            if not path.exists():
                return "（日志文件不存在）"
            with open(path, "rb") as fp:
                fp.seek(0, os.SEEK_END)
                size = fp.tell()
                if since > 0:
                    if since >= size:
                        return "（本次启动没有产生日志）"
                    start = since
                    length = min(size - since, _LOG_READ_MAX_BYTES)
                else:
                    length = min(size, _LOG_READ_MAX_BYTES)
                    start = size - length
                fp.seek(start)
                data = fp.read(length)
            text = data.decode("utf-8", errors="replace").strip()
            return text[-max_chars:] if text else "（日志为空）"
        except Exception as e:  # noqa: BLE001
            return f"（读取日志失败：{e}）"

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        py = self.python_bin
        if os.sep in py and not Path(py).exists():
            raise AudioWorkerStartError(
                f"找不到音频解释器：{py}（请设 BABYCARE_AUDIO_PYTHON 或建好 .venv-audio）"
            )
        log_path = self._log_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 音频 venv 里没有 daphne / channels（spec §9.2），必须显式用
        # ``config.settings_audio``；若继承 web 进程的 ``config.settings``，
        # 子进程会在 Django setup 阶段 ModuleNotFoundError: daphne 秒退。
        env = os.environ.copy()
        env["DJANGO_SETTINGS_MODULE"] = str(
            getattr(settings, "BABYCARE_AUDIO_SETTINGS_MODULE", "")
            or "config.settings_audio",
        )
        # Windows: CREATE_NEW_PROCESS_GROUP 让子进程能收 CTRL_BREAK
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt" else 0
        )
        cmd = [py, "manage.py", "audio_worker"]
        logger.info("[audio] spawning: %s（log=%s）", " ".join(cmd), log_path)
        fp = open(log_path, "ab", buffering=0)
        try:
            return subprocess.Popen(
                cmd,
                cwd=str(Path(getattr(settings, "BASE_DIR", Path.cwd()))),
                stdout=fp,
                stderr=fp,
                env=env,
                creationflags=creationflags,
            )
        except Exception as e:  # noqa: BLE001
            raise AudioWorkerStartError(f"Popen 失败：{e}") from e
        finally:
            # 子进程已继承自己的句柄；父进程不关会每次泄漏一个 fd
            fp.close()

    def _wait_ready(
        self, proc: subprocess.Popen, wait_sec: float, offset: int = 0,
    ) -> str:
        """等 ``wait_sec``；子进程早退 → 返回带日志尾巴的错误消息（空串 = 正常）。

        早退的典型原因：抢不到 worker 租约（``CommandError``）、类名映射表写错
        （``LabelMapError``）。此时必须把日志读出来给页面，否则用户只看到
        "点了没反应"（spec §9.3 要求 ``WorkerLockError`` 不能静默失败）。

        ``offset`` 是 spawn **之前**的日志长度，由调用方传入——不能在这里现取：
        子进程很可能在 ``_log_size()`` 之前就把错误写进日志了，现取会把新内容
        当成旧内容，tail 出来是空的。
        """
        if wait_sec <= 0:
            return ""
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = self.tail_log(since=offset)
                return (
                    f"audio_worker 启动后立即退出（rc={proc.returncode}）：{tail}"
                )
            time.sleep(_POLL_INTERVAL_SEC)
        return ""

    def start(
        self,
        reason: str = AudioServiceState.REASON_MANUAL,
        wait_sec: float = DEFAULT_SPAWN_CONFIRM_SEC,
    ) -> dict:
        """拉起 worker 进程。

        Returns:
            ``{"started": bool, "pid": int|None, "note": str}``；已在运行时不重复拉起。

        Raises:
            AudioWorkerStartError: 总开关关闭 / 解释器缺失 / 子进程早退。
        """
        if not self.enabled:
            raise AudioWorkerStartError(
                "BABYCARE_AUDIO_ENABLED=false，音频线总开关未开启。"
                "请先在 .env 打开并重启 daphne。"
            )
        with self._lock:
            if self.is_running():
                pid = self._read_pid()
                logger.info("[audio] start 跳过：worker 已在运行 pid=%s", pid)
                return {"started": False, "pid": pid, "note": "already_running"}
            self._clear_stale_pidfile()
            # 残留的停止哨兵会让刚拉起的 worker 立刻自杀 → 启动前必须清掉
            self._clear_stop_file()
            # offset 必须在 spawn 之前取（见 _wait_ready 的说明）
            offset = self._log_size()
            proc = self._spawn()
            self._proc = proc
            self._started_at = time.time()
            self._write_pid(proc.pid)

        err = self._wait_ready(proc, wait_sec, offset)
        if err:
            self._reap(proc)
            raise AudioWorkerStartError(err)

        self.set_desired(AudioServiceState.DESIRED_ON, reason=reason)
        logger.info("[audio] worker 已拉起：pid=%s reason=%s", proc.pid, reason)
        return {"started": True, "pid": proc.pid, "note": "started"}

    def _reap(self, proc: subprocess.Popen) -> None:
        """启动失败后的收尾：杀掉半死的子进程（含它已拉起的 FFmpeg）并清 PID 文件。"""
        with self._lock:
            if self._proc is proc:
                self._proc = None
                self._started_at = 0.0
        try:
            terminate_process_tree(proc, timeout=_STOP_TIMEOUT_SEC)
        except Exception:  # noqa: BLE001
            logger.exception("[audio] 回收启动失败进程异常")
        self._clear_pid()

    def stop(self, reason: str = AudioServiceState.REASON_MANUAL) -> bool:
        """停止 worker：**先请它自己优雅收尾**，超时才强杀。

        spec §9.3 要求"先优雅停（SIGTERM / CTRL_BREAK），超时 → ``taskkill /F /T``"。
        Windows 上跨进程给无窗口进程发信号不可靠，所以优雅路径走**哨兵文件**：

        ```
        web 写 data/audio_worker.stop
          → worker 心跳（≤2s）发现 → 跑完整 manager.stop()
               （落 AudioRuntimeState=stopped、释放 worker 租约、收掉 FFmpeg）
          → 进程自然退出
        ```

        只有它不响应（卡死）时才 ``taskkill /F /T`` 兜底。直接 /F 强杀会跳过上面
        全部收尾，留下"租约指向死进程 + 运行状态停在 ready"的脏数据。

        Returns:
            是否确实终结了一个存活进程（本来就是死的 → False）。
        """
        with self._lock:
            proc, self._proc = self._proc, None
            self._started_at = 0.0
            pid = self._read_pid()

        # 用户期望状态先落库：无论走哪条路径，都不该再被自动拉起
        self.set_desired(AudioServiceState.DESIRED_OFF, reason=reason)

        if pid is None or not _pid_alive(pid):
            self._clear_pid()           # 残留 PID 文件的进程早没了
            self._clear_stop_file()
            logger.info("[audio] stop 完成：进程本来就不在，已清残留")
            return False

        # --- 1) 优雅路径：写哨兵，等它自己收尾 ---
        grace = self._stop_grace_sec()
        self._write_stop_file()
        logger.info(
            "[audio] 已请求优雅停止（哨兵=%s），最多等 %.0fs", self._stop_file(), grace,
        )
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                break
            time.sleep(_POLL_INTERVAL_SEC)

        if not _pid_alive(pid):
            self._clear_pid()
            self._clear_stop_file()
            logger.info("[audio] 优雅停止完成：pid=%s 已自行退出", pid)
            return True

        # --- 2) 兜底：卡死/无响应 → 强杀整棵进程树 ---
        logger.warning(
            "[audio] 优雅停止超时（%.0fs），改用 taskkill /F /T pid=%s", grace, pid,
        )
        if proc is not None and proc.pid == pid:
            terminate_process_tree(proc, timeout=_STOP_TIMEOUT_SEC)
        else:
            _kill_pid_tree(pid, _STOP_TIMEOUT_SEC)

        if _pid_alive(pid):
            # 杀不掉（权限等）：**保留 PID 文件**，页面继续如实显示"运行中"
            logger.error("[audio] pid=%s 仍然存活，保留 PID 文件", pid)
            self._clear_stop_file()
            return False

        self._clear_pid()
        self._clear_stop_file()
        logger.info("[audio] 强杀完成：pid=%s", pid)
        return True

    # ------------------------------------------------------------------
    # 启动期自动拉起
    # ------------------------------------------------------------------
    def maybe_autostart(self) -> bool:
        """web 启动期逻辑：``ENABLED and desired == "on"`` 才拉起（spec §9.3）。

        用户手动关闭过（``desired=off``）→ 重启 daphne 也**不会**被自动拉起。
        """
        if not self.enabled:
            logger.info("[audio] 启动期不拉起：BABYCARE_AUDIO_ENABLED=false")
            return False
        desired = self.current_desired()
        if desired != AudioServiceState.DESIRED_ON:
            logger.info("[audio] 启动期不拉起：desired=%s（用户未开启/已手动关闭）", desired)
            return False
        if self.is_running():
            logger.info("[audio] 启动期跳过：worker 已在运行 pid=%s", self._read_pid())
            return False
        try:
            info = self.start(reason=AudioServiceState.REASON_AUTOSTART)
        except AudioWorkerStartError as e:
            logger.error("[audio] 启动期自动拉起失败：%s", e)
            return False
        logger.info("[audio] 启动期自动拉起成功：pid=%s", info.get("pid"))
        return True


__all__ = [
    "AudioWorkerManager",
    "AudioWorkerStartError",
    "DEFAULT_SPAWN_CONFIRM_SEC",
]

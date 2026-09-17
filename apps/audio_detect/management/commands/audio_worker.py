"""音频线独立进程入口（Phase 1，spec §9.3）。

用法
----
前台调试（推荐，日志直接打屏）：

    cd 09_web_vlm_manage
    $env:DJANGO_SETTINGS_MODULE = "config.settings_audio"
    .\\.venv-audio\\Scripts\\python.exe manage.py audio_worker

    # 只跑 60 秒便于验证
    .\\.venv-audio\\Scripts\\python.exe manage.py audio_worker --duration 60

    # 只跑指定摄像头
    .\\.venv-audio\\Scripts\\python.exe manage.py audio_worker --cam-id 3055

前置条件
--------
1. `.env` 里 `BABYCARE_AUDIO_ENABLED=true`（总开关；否则本命令拒绝启动，可用 `--force` 绕过）；
2. **`"audio_worker"` 必须在 `apps/streaming` / `apps/yolo_detect` / `apps/vlm` 三处
   `_MGMT_SKIP` 里**。否则本进程的 `apps.ready()` 会把整条视频线也拉起来 →
   第二个 llama-server（8082 端口冲突）、每路摄像头第二条 RTSP、VLM 重复调度；
3. **同一时刻只允许一个实例**：启动时通过 `AudioServiceState` 单行表抢租约，
   已有活跃实例（心跳未超时）→ 直接 `CommandError` 拒绝启动（见 worker_lock.py）。

为什么不在 `ready()` 里启动
--------------------------
`ready()` 是所有进程共用的；音频线只在"用户/网页明确要求"时启动（spec §9.3），
且必须跑在独立 venv 里。见 apps.py 的说明。
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.audio_detect.manager import AudioCaptureManager
from apps.audio_detect.worker_lock import WorkerLockError

logger = logging.getLogger(__name__)


def _install_signal_handlers(stop: threading.Event) -> None:
    """Ctrl+C / SIGTERM / Ctrl+Break → 置 stop 事件，走正常收尾流程。"""

    def _handler(signum, _frame):
        logger.info("[audio] signal %s received, shutting down ...", signum)
        stop.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as e:
            logger.debug("[audio] register signal %s failed: %s", name, e)


class Command(BaseCommand):
    help = "音频线独立进程：每路 ONVIF 摄像头 FFmpeg 采集 16kHz PCM + 健康状态心跳"

    def add_arguments(self, parser):
        parser.add_argument(
            "--duration", type=float, default=0.0,
            help="跑多少秒后自动退出（0 = 一直跑，直到 Ctrl+C）。",
        )
        parser.add_argument(
            "--cam-id", type=int, action="append", default=None,
            help="只跑指定摄像头 id（可重复）；缺省 = 全部 active ONVIF 摄像头。",
        )
        parser.add_argument(
            "--force", action="store_true",
            help="忽略 BABYCARE_AUDIO_ENABLED=false，强制启动（调试用）。",
        )
        parser.add_argument(
            "--no-inference", action="store_true",
            help="只做采集、不加载双模型（Phase 1 行为；省 15s 模型加载时间）。",
        )
        parser.add_argument(
            "--no-describe", action="store_true",
            help="不启动描述服务（Phase 4；事件停留在 pending_description）。",
        )

    def handle(self, *args, **opts):
        if not getattr(settings, "BABYCARE_AUDIO_ENABLED", False) and not opts["force"]:
            self.stderr.write(
                "BABYCARE_AUDIO_ENABLED=false，音频线未开启。"
                "如确需调试请加 --force。"
            )
            return

        cam_ids = set(opts["cam_id"]) if opts["cam_id"] else None
        duration = float(opts["duration"] or 0.0)
        enable_inference = not opts["no_inference"]
        enable_describe = not opts["no_describe"]

        stop = threading.Event()
        _install_signal_handlers(stop)

        manager = AudioCaptureManager(
            camera_ids=cam_ids,
            enable_inference=enable_inference,
            enable_describe=enable_describe,
            # 网页「关闭音频线」时 web 写停止哨兵；manager 心跳发现后置位这个事件，
            # 主循环随即走正常收尾（否则只能等 Ctrl+C）
            stop_event=stop,
        )
        # 映射表类名写错 → LabelMapError 直接冒泡到这里，让命令失败退出（spec §4.2）；
        # 已有活跃实例 → WorkerLockError，明确提示而不是打一堆 traceback
        try:
            manager.start()
        except WorkerLockError as e:
            raise CommandError(f"拒绝启动：{e}")

        desc_url = getattr(settings, "BABYCARE_AUDIO_DESC_SERVER_URL", "")
        desc_on = enable_describe and bool(desc_url)
        desc_provider = getattr(settings, "BABYCARE_AUDIO_DESC_PROVIDER", "")
        desc_where = f"（{desc_provider}@{desc_url}）" if desc_on else ""
        self.stdout.write(
            f"audio_worker 已启动（ffmpeg={settings.BABYCARE_FFMPEG_BIN}，"
            f"sample_rate={settings.BABYCARE_AUDIO_SAMPLE_RATE}，"
            f"cameras={'全部 ONVIF' if cam_ids is None else sorted(cam_ids)}，"
            f"inference={enable_inference}，"
            f"consensus={settings.BABYCARE_AUDIO_CONSENSUS}，"
            f"describe={desc_on}{desc_where}，"
            f"duration={'∞' if duration <= 0 else f'{duration:.0f}s'}）"
        )

        deadline = (time.monotonic() + duration) if duration > 0 else None
        try:
            while not stop.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                stop.wait(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stdout.write("audio_worker 正在停止 ...")
            manager.stop()
            self.stdout.write(self.style.SUCCESS("audio_worker 已停止"))

"""
YoloDetect app config。

启动期行为
----------
- 启动 SamplerManager（拉起所有 active cam 的 1Hz 采样线程）
- 启动 YoloLoop（从所有 cam FrameQueue 取未推理帧 → detect）
- 设置 GpuManager.mode（从 settings.BABYCARE_GPU_MODE 读）

信号（动态 reload）
------------------
- post_save / post_delete on Camera → SamplerManager.reload_from_db()
- 节流：1s 内多次触发合并为 1 次
"""

import logging
import os
import threading
import time

from django.apps import AppConfig
from django.db.models.signals import m2m_changed, post_save, post_delete

from apps.core.startup import should_autostart


logger = logging.getLogger(__name__)


# 不需要自动启动采样的 Django 管理命令
# 注：必须包含 "audio_worker"——音频线是独立进程，它的 ready() 不能把视频线也拉起来
#     （否则 YOLO 会被重复拉起；spec §9.3）。
_MGMT_SKIP = frozenset({
    "migrate", "makemigrations", "shell", "dbshell", "check",
    "createsuperuser", "changepassword", "showmigrations",
    "sqlmigrate", "loaddata", "dumpdata", "test", "collectstatic",
    "audio_worker",
})


def _should_autostart() -> bool:
    if os.environ.get("YOLO_AUTOSTART") == "0":
        return False
    # cleanup_* 只删数据/文件，没必要拉起采样与 YOLO；
    # runserver 的 reloader 父进程也不启（见 apps/core/startup.py）。
    return should_autostart(_MGMT_SKIP)


class YoloDetectConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.yolo_detect"
    verbose_name = "YOLO 推理（Step 3：baby 检测）"

    def ready(self):
        # 1) 注册信号（无论是否 autostart 都要注册；后台 reload 不依赖启动期）
        self._register_camera_signals()

        if not _should_autostart():
            return

        # 2) 设 GpuManager.mode
        from django.conf import settings
        from .gpu_manager import GpuManager
        GpuManager.instance().set_mode(settings.BABYCARE_GPU_MODE)

        # 3) 注册跨模块回调 + 启动 YoloLoop + 触发首次 reload
        try:
            from .sampler import SamplerManager
            from .yolo_loop import YoloLoopManager
            from apps.streaming.manager import VideoStreamManager

            sm = SamplerManager.instance()
            sm.register_callback(
                "yolo_loop",
                YoloLoopManager.instance().on_cams_changed,
            )
            sm.register_callback(
                "video_stream",
                VideoStreamManager.instance().on_cams_changed,
            )

            # Sampler 首次 reload；末尾会触发两个回调
            # - VideoStreamManager.on_cams_changed → 拉起采集线程
            # - YoloLoop.on_cams_changed → 同步 _cam_order
            sm.reload_from_db()

            # YoloLoop 启动线程（在 on_cams_changed 之后，确保 _cam_order 不空）
            YoloLoopManager.instance().start()

            logger.info("[yolo_detect] ready(): callbacks registered, YoloLoop started")
        except Exception as e:
            logger.error("[yolo_detect] ready() failed: %s", e)

    # ------------------------------------------------------------------
    def _register_camera_signals(self):
        """
        动态 reload：Camera save / delete → SamplerManager.reload_from_db()

        - bulk 操作不触发信号（queryset.update / .delete）—— 这些情况需手动 reload
        - 节流：1s 内的多次触发合并为 1 次 reload（用 _pending_reload + _reload_lock）
        """
        from .sampler import SamplerManager
        from apps.streaming.models import Camera

        # 模块级状态（同一进程内只一份）
        if getattr(self, "_signals_registered", False):
            return
        self._signals_registered = True
        self._reload_lock = threading.Lock()
        self._reload_pending = False
        self._reload_thread: threading.Thread | None = None

        def _schedule_reload(*args, **kwargs):
            with self._reload_lock:
                if self._reload_pending:
                    return
                self._reload_pending = True
                if self._reload_thread is not None and self._reload_thread.is_alive():
                    return
                t = threading.Thread(target=_run_debounced, name="sampler-reload", daemon=True)
                self._reload_thread = t
                t.start()

        def _run_debounced():
            while True:
                time.sleep(1.0)  # 节流 1s
                with self._reload_lock:
                    if not self._reload_pending:
                        return
                    self._reload_pending = False
                try:
                    SamplerManager.instance().reload_from_db()
                    logger.info("[yolo_detect] signal-triggered SamplerManager reload")
                except Exception as e:
                    logger.warning("[yolo_detect] signal reload failed: %s", e)

        post_save.connect(_schedule_reload, sender=Camera)
        post_delete.connect(_schedule_reload, sender=Camera)

        # Camera.yolo_models M2M 改了 → 刷新 YoloLoop 的 cam_models 缓存
        # m2m_changed 在 add/remove/clear 三种 action 后触发
        try:
            through = Camera.yolo_models.through

            def _on_cam_yolo_models_m2m(sender, instance, action, **kwargs):
                if action not in ("post_add", "post_remove", "post_clear"):
                    return
                try:
                    from .yolo_loop import YoloLoopManager
                    YoloLoopManager.instance().on_cam_models_changed(instance.id)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[yolo_detect] on_cam_models_changed failed cam=%d: %s",
                        instance.id, e,
                    )

            m2m_changed.connect(_on_cam_yolo_models_m2m, sender=through)
            logger.info("[yolo_detect] Camera.yolo_models m2m_changed signal registered")
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[yolo_detect] Camera.yolo_models m2m_changed signal register failed: %s",
                e,
            )
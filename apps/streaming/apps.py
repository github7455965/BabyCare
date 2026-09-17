"""
Streaming app config。

启动条件
--------
- 不在 manage.py <管路命令> 中（migrate/shell/check/...）
- 不在 STREAMING_AUTOSTART=0 时

runserver 自动满足；daphne / gunicorn 等生产入口也自动满足。
"""

import logging
import os

from django.apps import AppConfig

from apps.core.startup import manage_py_command, should_autostart

logger = logging.getLogger(__name__)


# 不需要自动启动采集线程的 Django 管理命令
# 注：必须包含 "audio_worker"——音频线是独立进程，它的 ready() 不能把视频线也拉起来
#     （否则每路摄像头多开一条 RTSP；spec §9.3）。
_MGMT_SKIP = frozenset({
    "migrate", "makemigrations", "shell", "dbshell", "check",
    "createsuperuser", "changepassword", "showmigrations",
    "sqlmigrate", "loaddata", "dumpdata", "test", "collectstatic",
    "audio_worker",
})


class StreamingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.streaming"
    verbose_name = "视频流"

    def ready(self):
        if not self._should_autostart():
            return
        # 不直接 reload_from_db——交给 SamplerManager 统一调度
        # （SamplerManager.reload_from_db() 末尾会回调本 manager.on_cams_changed）
        logger.info("[streaming] ready(): VideoStreamManager waiting for SamplerManager")

    @staticmethod
    def _should_autostart() -> bool:
        if os.environ.get("STREAMING_AUTOSTART") == "0":
            logger.info("[streaming] autostart disabled via STREAMING_AUTOSTART=0")
            return False
        # cleanup_* 只删数据/文件，拉起采集线程既浪费又会开多余 RTSP；
        # runserver 的 reloader 父进程也不启（否则双进程各拉一份同一路 RTSP）。
        if not should_autostart(_MGMT_SKIP):
            logger.debug(
                "[streaming] skip autostart (cmd=%s RUN_MAIN=%s)",
                manage_py_command(), os.environ.get("RUN_MAIN"),
            )
            return False
        return True
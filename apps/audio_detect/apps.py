"""AudioDetect app config（Step 15：音频线）。

启动期行为
----------
**本 app 自己不启采集/模型**（那是 worker 独立进程的事，见下），只做一件事：

- 若 ``BABYCARE_AUDIO_ENABLED=true`` **且** DB 里 ``AudioServiceState.desired == "on"``
  → 调 :meth:`AudioWorkerManager.maybe_autostart` 把 worker 子进程拉起来（spec §9.3）。

为什么不在这里启动采集
----------------------
1. daphne 进程里**不该**有 ffmpeg / YAMNet / PANNs —— 这正是"独立进程 + 独立 venv"
   要避免的耦合（spec §9.2）；
2. 音频线的启停由用户通过网页控制（spec §9.3），启动时机不是"daphne 起来"；
3. 若在这里直接起线程，`manage.py migrate` 之类的命令也会被牵连。

注意 `desired` 的语义：用户手动关闭后写 ``desired=off``，**重启 daphne 也不会
再自动拉起**——这正是「用户期望状态」要解决的问题。
"""

import logging
import os
import threading

from django.apps import AppConfig

from apps.core.startup import should_autostart


logger = logging.getLogger(__name__)


# 不需要"启动期自动拉起音频 worker"的 Django 管理命令
# （含自身 audio_worker：子进程不该再拉起子进程）
_MGMT_SKIP = frozenset({
    "migrate", "makemigrations", "shell", "dbshell", "check",
    "createsuperuser", "changepassword", "showmigrations",
    "sqlmigrate", "loaddata", "dumpdata", "test", "collectstatic",
    "audio_worker",
})

# runserver 自动重载 / 多 worker 下 ready() 可能被调多次，只试一次
_autostart_lock = threading.Lock()
_autostart_attempted = False


def _should_autostart() -> bool:
    """管理命令期不拉起（含 runserver 的 reloader 父进程，见 apps/core/startup.py）。"""
    if os.environ.get("AUDIO_AUTOSTART") == "0":
        return False
    return should_autostart(_MGMT_SKIP)


class AudioDetectConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.audio_detect"
    verbose_name = "声音检测（Step 15：音频线）"

    def ready(self):
        if not _should_autostart():
            return

        global _autostart_attempted
        with _autostart_lock:
            if _autostart_attempted:
                return
            _autostart_attempted = True

        # 延迟 import：ready() 期 app registry 尚未完全就绪，模型必须在函数内引入
        try:
            from .worker_manager import AudioWorkerManager

            manager = AudioWorkerManager.instance()
            if not manager.enabled:
                logger.info(
                    "[audio] 启动期不拉起：BABYCARE_AUDIO_ENABLED=false",
                )
                return
            fired = manager.maybe_autostart()
            logger.info("[audio] ready(): maybe_autostart -> %s", fired)
        except Exception:  # noqa: BLE001
            # 自动拉起失败绝不能影响 daphne 启动：用户在控制页手动开即可
            logger.exception("[audio] ready() 自动拉起 worker 失败（可在控制页手动开启）")

"""音频 worker 专用 Django settings。

为什么需要它
------------
音频 worker（`manage.py audio_worker`）跑在**独立 venv**（`.venv-audio/`）里，
它不是一个 web 进程，用不到 daphne / channels。

但主 `config/settings.py` 的 `INSTALLED_APPS` 以 `"daphne"` 开头，若音频侧直接用主
settings，就必须在音频 venv 里也装上 daphne + twisted + autobahn 一整套 web 栈——
这正是"独立 venv"想要避免的耦合。

这个模块复用主 settings 的全部配置，只把 web/ASGI 相关 app 摘掉。

用法
----
    $env:DJANGO_SETTINGS_MODULE = "config.settings_audio"
    .venv-audio\\Scripts\\python.exe manage.py audio_worker

设计说明见 docs/superpowers/specs/2026-09-07-audio-detection-design.md §9.2 / §9.3。
"""

from .settings import *  # noqa: F401,F403

# 摘掉 web / ASGI 栈：音频 worker 不提供 HTTP，也不需要 Channels
INSTALLED_APPS = [  # noqa: F405
    app for app in INSTALLED_APPS  # noqa: F405
    if app not in ("daphne", "channels")
]

# 音频 worker 也不需要 ASGI application
ASGI_APPLICATION = None  # noqa: F811

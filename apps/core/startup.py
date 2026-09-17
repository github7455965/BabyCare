"""启动期 autostart 的公共判据（4 个 app 的 ``ready()`` 共用）。

为什么需要这个模块
------------------
`manage.py runserver` 默认开 autoreload，会跑**两个进程**：

| 进程 | 职责 | ``RUN_MAIN`` |
|---|---|---|
| 父进程 | 监视文件变化，负责重启子进程 | 未设置 |
| 子进程 | 真正跑 daphne 服务 | ``"true"`` |

而**两个进程都会执行 `apps.ready()`**。各 app 在 `ready()` 里做的都是"进程级"的事
（起 `LlamaManager`、起采集线程、拉起 llama-server / audio_worker），但它们的单例
作用域只在**进程内** —— 于是双进程各跑一份，互相破坏：

- **`LlamaManager`**：两个实例共享同一个 ``data/llama_server.pid``，各自调用
  ``_cleanup_stale_pidfile()`` 时会把对方刚拉起的 llama-server 判成"Ctrl+C 残留的孤儿"
  并 kill 掉 → **启动期互相屠杀**（现场实测循环 5 轮，`killed stale llama-server pid=…`
  紧跟在对方的 `spawning` 之后）；
- **`VideoStreamManager`**：两个进程各拉一份同一路 RTSP，再叠加音频 worker，
  设备并发受限时整条视频线 `too many open failures` 退出。

所以 runserver 下只让**子进程**（``RUN_MAIN=true``）启动。daphne / gunicorn 等生产
入口不走 runserver 分支，行为完全不变。

本模块只依赖标准库，**不 import django / 不碰 ORM**，因此在 `ready()` 的任何阶段
import 都安全（`apps.core` 在 `INSTALLED_APPS` 中排在 streaming/yolo/vlm 之后，
那些 app 的 `ready()` 里 import 本模块时 app registry 尚未就绪）。
"""

from __future__ import annotations

import os
import sys

#: Django autoreload 用它标记"我是子进程"（`django.utils.autoreload.DJANGO_AUTORELOAD_ENV`）
RUN_MAIN_ENV = "RUN_MAIN"

#: 没有额外参数时，视为"一次性命令"、不该拉起后台服务的前缀
_CLEANUP_PREFIX = "cleanup_"


def manage_py_command() -> str:
    """当前 ``manage.py <cmd>`` 的子命令名；不是 manage.py 调用时返回空串。"""
    argv = list(getattr(sys, "argv", []) or [])
    if argv and argv[0].endswith("manage.py"):
        return argv[1] if len(argv) > 1 else ""
    return ""


def is_runserver_reloader() -> bool:
    """是否是 ``runserver`` 的 autoreload **父进程**（不该启动任何后台服务）。

    ``--noreload`` 时只有一个进程、且不会设置 ``RUN_MAIN``，那种情况下父进程
    就是服务进程 → 返回 False（必须正常启动）。
    """
    if "--noreload" in list(getattr(sys, "argv", []) or []):
        return False
    return os.environ.get(RUN_MAIN_ENV) != "true"


def should_autostart(skip_commands) -> bool:
    """本进程要不要跑后台服务（各 app 的 ``_should_autostart()`` 都转发到这里）。

    Args:
        skip_commands: 该 app 自己的跳过集合（如 ``_MGMT_SKIP``），加上通用的
            ``cleanup_*`` 前缀规则与 runserver reloader 判定。
    """
    cmd = manage_py_command()
    if not cmd:
        # daphne / gunicorn / 自定义 ASGI 入口：照常启动
        return True
    if cmd in skip_commands or cmd.startswith(_CLEANUP_PREFIX):
        return False
    if cmd == "runserver" and is_runserver_reloader():
        return False
    return True


__all__ = [
    "RUN_MAIN_ENV",
    "is_runserver_reloader",
    "manage_py_command",
    "should_autostart",
]

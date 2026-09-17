"""音频线的文件路径（web 侧与 worker 侧共用）。

单独成模块的原因：`worker_manager`（web 侧）只需要拼几个路径，而 `manager`
（worker 侧）会连带 import 推理 / 描述服务；把路径逻辑放这里，两边都能用而不
互相拖依赖，也保证"同一个配置项在两侧解析出同一个路径"。
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings


def data_dir() -> Path:
    return Path(getattr(settings, "BASE_DIR", Path.cwd())) / "data"


def _resolve(configured: str | None, default_name: str) -> Path:
    raw = str(configured or "").strip()
    return Path(raw) if raw else data_dir() / default_name


def pid_file_path() -> Path:
    """worker PID 文件（web 侧写自己的 Popen pid；判断启停用）。"""
    return _resolve(getattr(settings, "BABYCARE_AUDIO_PID_FILE", ""), "audio_worker.pid")


def log_file_path() -> Path:
    """worker stdout/stderr 落盘位置。"""
    return _resolve(getattr(settings, "BABYCARE_AUDIO_LOG_FILE", ""), "audio_worker.log")


def stop_file_path() -> Path:
    """**优雅停止哨兵**：web 侧创建 → worker 侧轮询到即收尾退出。

    为什么用文件而不是信号：Windows 上跨进程给"无 console 窗口的 python 进程"
    发 SIGTERM / CTRL_BREAK 不可靠（``GenerateConsoleCtrlEvent`` 要求目标与调用者
    在同一个 console，且可能误伤整个进程组）。哨兵文件跨平台可靠，也不会在手动
    调试（``manage.py audio_worker``）时被误触发——那时文件根本不存在。
    """
    return _resolve(getattr(settings, "BABYCARE_AUDIO_STOP_FILE", ""), "audio_worker.stop")


__all__ = [
    "data_dir",
    "log_file_path",
    "pid_file_path",
    "stop_file_path",
]

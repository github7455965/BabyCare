"""
CamT0Manager：每路摄像头维护一个 t0（int 秒）。

v6 设计要点
-----------
- t0 用于 PromptCursor 初值（Step 3 才用）
- 文件源：FileSource 第一次成功 read 后由 manager.set(...)；**视频循环时 t0 不重置**
  （你确认："时间一直往后走"）
- ONVIF 源：首帧通过 FrameBus publish 时由 manager.set(...)（v6 设计原则同源）

为什么不放在 source 内
----------------------
- 单源无法独立判定"第一次 publish"（因为 publish 是 manager 调 FrameBus 的事）
- 集中式 manager 维护所有 cam 的 t0，便于 Step 3 调度查询
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class CamT0Manager:
    """全局单例：per-cam 维护 t0（int 秒）。"""

    _instance: Optional["CamT0Manager"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._t0: Dict[int, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "CamT0Manager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def set(self, camera_id: int, ts_int: int | None = None) -> int:
        """
        设置（或覆盖）cam 的 t0。
        ts_int=None → 用 int(time.time())
        返回实际写入的 t0。
        """
        v = int(ts_int) if ts_int is not None else int(time.time())
        with self._lock:
            self._t0[camera_id] = v
        return v

    def get(self, camera_id: int) -> int:
        """读取 cam 的 t0；未设置返回 0。"""
        with self._lock:
            return self._t0.get(camera_id, 0)

    def has(self, camera_id: int) -> bool:
        """是否已 set 过。"""
        with self._lock:
            return camera_id in self._t0

    def remove(self, camera_id: int) -> None:
        with self._lock:
            self._t0.pop(camera_id, None)
"""
帧总线（精简版）：每路摄像头一份最新帧 + 序号 + 环形缓冲。

09 与 08 的差异（精简点）：
- 不持久化 / 不写录像（步骤 10 才涉及）
- 不支持 seq 历史回放录像切片（Step 2 用不上）
- 不算 fps（消费方按需现算）
- 默认启用 25 帧环形缓冲（用于 Step 3 的 1Hz 采样拿最近帧）

线程模型
--------
- 采集线程：publish (写 path)   —— 只换 ref + append 环形缓冲
- 推理 / 采样 consumer  N 个：get_snapshot / get_ringbuffer (读 path)

数据布局
--------
每路摄像头一个 Slot：
    _frame_ref      : ndarray 引用（采集线程不断替换为新帧）
    _frame_seq      : 单调递增帧序号（消费者可用来判断"是否有新帧"）
    _last_ts        : 最近一次 publish 的 wall-clock 时间戳（float 秒）
    _ring_buffer    : 环形缓冲 [(seq, ts, frame), ...]  —— 默认关闭（容量 0）
    _seq_index      : seq → ndarray 查表（O(1) 拿指定帧）
    _lock           : slot 级锁（只在 publish 内部短暂持有）

接口：
- publish(cam_id, frame) -> seq
- get_snapshot(cam_id) -> (seq, frame, ts)
- get_frame_by_seq(cam_id, seq) -> (frame, ts, hit)
- get_ringbuffer(cam_id, n) -> List[(seq, ts, frame)]
- enable_ringbuffer / disable_ringbuffer
- remove(cam_id)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# 默认环形缓冲容量。
#
# 2026-09-15 改为 0（**默认关闭**）：09 里这个环缓没有任何消费者 ——
# ``get_ringbuffer`` / ``get_frame_by_seq`` 全项目零调用，而 25 帧的容量意味着
# 每路白存 25 个全尺寸帧：实测摄像头 2304×1296 = 8.5MB/帧 → 单路 212MB、
# 两路 425MB（daphne 内存的最大单项之一）。
# 旧注释（保留背景）：25 ≈ 1s @25fps，原为 08 的"实时画框/回放对齐"准备；
# 若以后真要做历史帧对齐，显式调 ``enable_ringbuffer(cam_id, capacity)`` 开启。
DEFAULT_RING_CAPACITY = 0

# 显式 enable 时（不传 capacity）用的容量：≈1s @25fps，与历史默认值一致
_ENABLE_DEFAULT_CAPACITY = 25


@dataclass
class _Slot:
    camera_id: int
    _frame_ref: object | None = None          # ndarray | None
    _frame_seq: int = 0                        # 单调递增
    _last_ts: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # 帧环形缓冲：[(seq, ts, frame), ...]
    _ring_buffer: deque = field(default_factory=deque)
    _ring_capacity: int = DEFAULT_RING_CAPACITY  # 默认 0 = 关闭
    # _seq_index: seq → ndarray 的查表（O(1) 拿指定帧，避免 deque 线性扫描）
    _seq_index: dict = field(default_factory=dict)


class FrameBus:
    """全局单例（进程级）。"""

    _instance: Optional["FrameBus"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._slots: dict[int, _Slot] = {}
        self._slots_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "FrameBus":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def _slot(self, camera_id: int) -> _Slot:
        with self._slots_lock:
            slot = self._slots.get(camera_id)
            if slot is None:
                slot = _Slot(camera_id=camera_id)
                self._slots[camera_id] = slot
            return slot

    # ------------------------------------------------------------------
    # 写路径：采集线程调用
    # ------------------------------------------------------------------
    def publish(self, camera_id: int, frame) -> int:
        """
        写入最新帧。返回写入后的帧序号（>= 1）。

        持锁时间仅"seq 自增 + ref 替换 + 环形 append"，与 ndarray 大小无关 → 微秒级。
        """
        slot = self._slot(camera_id)
        now = time.time()
        with slot._lock:
            slot._frame_seq += 1
            slot._frame_ref = frame
            slot._last_ts = now
            if slot._ring_capacity > 0:
                seq = slot._frame_seq
                slot._ring_buffer.append((seq, now, frame))
                slot._seq_index[seq] = frame
                # deque(maxlen=capacity) 自动截断，但 maxlen 命中时不会通知我们丢了哪个 seq
                # 所以 popleft + 同步 _seq_index，让 seq → frame 查表保持一致
                while len(slot._ring_buffer) > slot._ring_capacity:
                    old_seq, _, _ = slot._ring_buffer.popleft()
                    slot._seq_index.pop(old_seq, None)
            return slot._frame_seq

    # ------------------------------------------------------------------
    # 环形缓冲（默认关闭；09 无消费者，开启只为未来的历史帧对齐需求）
    # ------------------------------------------------------------------
    def enable_ringbuffer(self, camera_id: int, capacity: int = _ENABLE_DEFAULT_CAPACITY) -> None:
        """为指定摄像头启用/重置环形缓冲。capacity <= 0 走 disable 分支。

        注意：_ring_buffer 用无 maxlen 的 deque，截断由 publish() 手动 while popleft 完成。
        原因：deque(maxlen=N) 自动截断时不会通知丢了哪个 seq，导致 _seq_index 同步不上
        —— 历史 bug，每帧 ndarray 永久驻留 dict，内存线性增长。
        """
        if capacity <= 0:
            self.disable_ringbuffer(camera_id)
            return
        slot = self._slot(camera_id)
        with slot._lock:
            slot._ring_capacity = capacity
            slot._ring_buffer = deque()       # 不带 maxlen，手动截断
            slot._seq_index.clear()

    def disable_ringbuffer(self, camera_id: int) -> None:
        """关闭环形缓冲（注意：消费者拿历史帧将 fallback 到最新帧）。"""
        slot = self._slot(camera_id)
        with slot._lock:
            slot._ring_capacity = 0
            slot._ring_buffer.clear()
            slot._seq_index.clear()

    def get_frame_by_seq(self, camera_id: int, seq: int) -> Tuple[object | None, float | None, bool]:
        """
        按 frame_seq 拿指定历史帧。

        返回 (frame, ts, hit)：
        - hit=True：命中 ring_buffer（seq 在容量范围内），frame/ts 有效
        - hit=False：seq 已超出容量（太老或还没到），frame=None，ts=None
                     调用方应 fallback 到 get_snapshot() 的最新帧 + 日志 warn
        """
        slot = self._slot(camera_id)
        with slot._lock:
            frame = slot._seq_index.get(seq)
            if frame is not None:
                for s, t, _ in slot._ring_buffer:
                    if s == seq:
                        return frame, t, True
                return frame, None, True  # 索引有但 deque 没，理论上不应发生
            return None, None, False

    def get_ringbuffer(self, camera_id: int, n: int | None = None) -> List[Tuple[int, float, object]]:
        """
        拿环形缓冲最近 n 帧（默认全部）。返回 [(seq, ts, frame), ...]，按时间正序。
        返回引用，调用方不应长期持有（防止 ndarray 内存膨胀）。
        """
        slot = self._slot(camera_id)
        with slot._lock:
            if slot._ring_capacity == 0 or not slot._ring_buffer:
                return []
            items = list(slot._ring_buffer)
        if n is not None and n > 0 and len(items) > n:
            items = items[-n:]
        return items

    # ------------------------------------------------------------------
    # 读路径：消费者调用
    # ------------------------------------------------------------------
    def get_snapshot(self, camera_id: int) -> Tuple[int, object | None, float]:
        """
        一次原子快照：(seq, frame, ts)。
        - seq == 0 表示还没帧
        - frame 可能为 None（依然）
        持锁时间仅读 3 个字段的瞬时快照。
        """
        slot = self._slot(camera_id)
        with slot._lock:
            return slot._frame_seq, slot._frame_ref, slot._last_ts

    def current_seq(self, camera_id: int) -> int:
        """当前帧序号。consumer 比较这个值判断是否有新帧（用于跳帧）。"""
        slot = self._slot(camera_id)
        with slot._lock:
            return slot._frame_seq

    def is_alive(self, camera_id: int, stale_sec: float = 5.0) -> bool:
        slot = self._slot(camera_id)
        with slot._lock:
            if slot._last_ts == 0.0:
                return False
            return (time.time() - slot._last_ts) < stale_sec

    def remove(self, camera_id: int) -> None:
        with self._slots_lock:
            self._slots.pop(camera_id, None)

    def list_cameras(self) -> list[int]:
        """已注册过的摄像头 id（包括已经不活跃的）。"""
        with self._slots_lock:
            return list(self._slots.keys())
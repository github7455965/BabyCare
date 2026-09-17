"""
FrameQueue（per-cam FIFO，v6 设计）

设计要点
--------
- 数据结构：`deque[(ts_int, ndarray, has_baby)]`，**只存 ndarray**，不存 path
- 容量：`MAX_WINDOW_FRAMES=90`（默认；约 30s × 3 窗口）
- 满则 pop 最旧（自然 FIFO 滚出）
- **每个 has_<class> 三态**：None=未推理；True/False=YOLO 已推理
- **``detected``**：该帧"本 cam 的模型都跑过了"的标记，``peek_oldest`` 只看它 ——
  v1 遗留实现用 ``has_baby is None`` 判定，cam 没配 baby 模型时会让 YoloLoop
  反复 detect 队头同一帧（其余算力空转），2026-09-14 修
- **线程安全**：所有操作持锁（push / mark_detected / get_in_range / peek_oldest）
- **入队即限长边**（``MAX_STORED_LONG_SIDE``，2026-09-15）：摄像头原始帧是
  2304×1296（**8.5MB/帧**），但下游 VLM 只要长边 1024、YOLO 反正 letterbox 到
  640 —— 存全尺寸纯浪费。入队时缩一次（1Hz，CPU 可忽略），队列内存
  1.53GB → ~0.32GB（2 路 90 帧）。代价：落盘证据图也是 1024 长边

为什么不用 `deque(maxlen=90)`
----------------------------
- `_ts_index` 是 dict，deque(maxlen) 自动剔旧时不通知，导致 _ts_index 残留
- 所以用 `append()` + 手动 length check + pop oldest + _ts_index.pop
  （与 frame_bus 同思路）
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from apps.core.imaging import fit_long_side


# 单 cam 容量上限（≈30s × 3 窗口 / 1Hz = 90 帧）
MAX_WINDOW_FRAMES = 90

# 入队帧的长边上限（2026-09-15 起）。
# 取值依据：这是**下游最宽的消费方**所需 —— VLM 送模型前缩到 1024
# （``apps/vlm/frame_storage._VLM_MAX_LONG_SIDE``），YOLO 内部 letterbox 到 640。
# 存 1024 让 VLM 那条路径的 ``fit_long_side`` 直接变成 no-op（省掉每窗 3 帧的现缩），
# 同时把队列内存压到 1/5。再小会丢 VLM 细节，再大是纯浪费。
MAX_STORED_LONG_SIDE = 1024


@dataclass
class FrameItem:
    ts: int                  # int 秒
    ndarray: object          # BGR ndarray（shape=(H,W,3)）—— 用 object 避免 numpy import 蔓延
    # v2: 多类别字段（每个值 None=未推理；True=有；False=确认无）
    # - 未配该 class 的 model 的 cam：永远 None（YoloLoop 端不会更新此 attr）
    # - 已配的 cam：YoloLoop 跑完对应 model 后置 True/False
    has_baby: Optional[bool] = None
    has_person: Optional[bool] = None
    has_cat: Optional[bool] = None
    #: YoloLoop 是否已对该帧跑完"本 cam 当前全部模型"。
    #: 与具体类别无关：cam 只配了 person/cat 时 ``has_baby`` 会永远是 None，
    #: 只有用这个标记才能正确判断"这帧不用再跑了"。
    detected: bool = False


class FrameQueue:
    """per-cam FIFO 队列，存最近 MAX_WINDOW_FRAMES 帧。"""

    def __init__(self, cam_id: int, maxlen: int = MAX_WINDOW_FRAMES):
        self.cam_id = cam_id
        self._maxlen = maxlen
        self._items: deque = deque()  # 不带 maxlen，手动截断
        self._ts_index: Dict[int, FrameItem] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def push(self, ts: int, ndarray, **attrs) -> None:
        """SamplerWorker 调：push 新帧（所有 has_<name> 字段默认 None）。

        v1 兼容：push(ts, ndarray, has_baby=...) 仍可用
        v2 用法：push(ts, ndarray, has_baby=..., has_person=..., has_cat=...)

        帧的 ``ndarray`` 会先按 ``MAX_STORED_LONG_SIDE`` 缩一次（只缩不放）——
        缩图在**锁外**做（耗时不持锁），``ndarray=None``（测试/占位帧）原样跳过。
        """
        # 提取已知字段（v2 硬编码 baby/person/cat）；未知 key 静默丢弃（与 mark_detected 一致）
        has_baby = attrs.get("has_baby", None)
        has_person = attrs.get("has_person", None)
        has_cat = attrs.get("has_cat", None)
        if ndarray is not None:
            ndarray = fit_long_side(ndarray, MAX_STORED_LONG_SIDE)
        with self._lock:
            if len(self._items) >= self._maxlen:
                # 满了 → 剔最旧
                old = self._items.popleft()
                self._ts_index.pop(old.ts, None)
            item = FrameItem(
                ts=ts, ndarray=ndarray,
                has_baby=has_baby, has_person=has_person, has_cat=has_cat,
            )
            self._items.append(item)
            self._ts_index[ts] = item

    def mark_detected(self, ts: int, **detections) -> None:
        """YoloLoop 调：标记已推理。

        v1 兼容：mark_detected(ts, has_baby=...) 仍可用。
        v2 多类别来源：YoloRegistry.detect(frame, models) 返回
        Dict[class_name, bool]，key 不带 has_ 前缀（如 {'baby': True, ...}）；
        本方法自动加 has_ 前缀（baby -> has_baby）。

        无论写入哪几个类别，都置 ``detected=True`` —— 语义是"这一帧该跑的模型
        都跑完了"，而不是"某个特定类别有值"。
        """
        with self._lock:
            item = self._ts_index.get(ts)
            if item is not None:
                for k, v in detections.items():
                    attr = k if k.startswith("has_") else f"has_{k}"
                    if hasattr(item, attr):
                        setattr(item, attr, v)
                item.detected = True

    # ------------------------------------------------------------------
    def get_in_range(self, ts_set: Set[int]) -> List[FrameItem]:
        """VlmWorker 调：取 ts 在 ts_set 内的帧（按 ts 升序）。"""
        with self._lock:
            items = [it for it in self._items if it.ts in ts_set]
        items.sort(key=lambda f: f.ts)
        return items

    def snapshot_sorted(self) -> List[FrameItem]:
        """PromptCursor 调：返回所有帧的快照（按 ts 升序）。"""
        with self._lock:
            items = list(self._items)
        items.sort(key=lambda f: f.ts)
        return items

    def peek_oldest(self) -> Optional[FrameItem]:
        """YoloLoop 调：取最早**未推理**的帧。

        "未推理"看 ``detected``，**不再**看 ``has_baby``：cam 没配 baby 模型时
        ``has_baby`` 永远是 None，用它判定会一直返回队列里最早的同一帧 →
        YoloLoop 反复 detect 同一帧，每秒实际只推进 1 帧（其余算力全空转）。
        """
        with self._lock:
            for it in self._items:
                if not it.detected:
                    return it
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> dict:
        """监控用：长度 / 待推理数。"""
        with self._lock:
            pending = sum(1 for it in self._items if not it.detected)
            return {"len": len(self._items), "pending": pending, "maxlen": self._maxlen}


# ---------------------------------------------------------------------------
class FrameQueueManager:
    """全局单例：管理所有 cam 的 FrameQueue。"""

    _instance: Optional["FrameQueueManager"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._queues: Dict[int, FrameQueue] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "FrameQueueManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_or_create(self, cam_id: int) -> FrameQueue:
        with self._lock:
            q = self._queues.get(cam_id)
            if q is None:
                q = FrameQueue(cam_id)
                self._queues[cam_id] = q
            return q

    def get(self, cam_id: int) -> Optional[FrameQueue]:
        with self._lock:
            return self._queues.get(cam_id)

    def remove(self, cam_id: int) -> None:
        with self._lock:
            self._queues.pop(cam_id, None)

    def all_cam_ids(self) -> List[int]:
        with self._lock:
            return list(self._queues.keys())
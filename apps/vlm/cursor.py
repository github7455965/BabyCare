"""
PromptCursor（v6 进度指针）

设计要点
--------
- 每个 (cam_id, prompt_id) 一份 cursor，in-memory，不持久化
- `max_read_ts` = 已处理窗口的最大 ts（下次从 max_read_ts+1 起扫）
- `last_invoke_ts` = 已入队的窗口起点（防 worker 慢时 prompt_runner 重复入队）
- 初始化时 `max_read_ts = -1`（行为等价"从 ts=0 开始扫"——方案 A：启动瞬间消化历史堆积帧）

`try_advance` 语义
------------------
- 从 `max_read_ts + 1` 起扫 FrameQueue
- 找"首个 window_sec 个 ts 连续 + target_classes 对应 attr 全部 ≠ None 的窗口"
- ts 本身必须连续（不允许中间丢帧）；每个 target 对应 has_<name> 必须 ≠ None（YOLO 已推理）
- 找到 → 返回 WindowSpec(ts_list, start_ts)；否则返回 None
- 不修改 cursor（mark_done 由 worker 处理完一窗后调）

v2 多类别
---------
- target_classes: VLMPromptConfig.target_classes 解析列表（如 ["baby"] / ["person"]）
- 窗口判定改为：每帧所有 target_classes 对应 attr 都 ≠ None
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from apps.yolo_detect.frame_queue import FrameQueue


# ---------------------------------------------------------------------------
@dataclass
class WindowSpec:
    """一窗已凑齐的窗口描述。"""
    ts_list: List[int]   # window_sec 个连续的 ts（按时间升序）
    start_ts: int        # ts_list[0]


def _all_targets_known(item, target_classes: List[str]) -> bool:
    """该帧所有 target_classes 对应 has_<name> attr 都 ≠ None。

    - 未定义的 attr（如 FrameItem 没有 has_dog）：按 False 处理 → 视为未推理
    - 全部有值 → True（无论 True/False）
    """
    for c in target_classes:
        if getattr(item, f"has_{c}", None) is None:
            return False
    return True


# ---------------------------------------------------------------------------
@dataclass
class PromptCursor:
    """单个 (cam, prompt) 的进度指针。

    字段含义（Step 8）
    ------------------
    - max_read_ts: 已处理窗口的最大 ts
    - last_invoke_ts: 已入队的窗口起点（防重复入队）
    - fail_count: 连续失败次数（>=2 时 Runner 跳过当前 window）
    - is_cold_start: True=本 prompt 的第一次请求；第一次请求一发出就置 False，
      后续都走 normal timeout。Runner 据此决定本次调 llama 用 first / normal timeout。
    """
    cam_id: int
    prompt_id: int
    window_sec: int
    max_read_ts: int = -1          # 已处理窗口的最大 ts（-1 = 还没处理过任何 ts）
    last_invoke_ts: int = 0        # 已入队的窗口起点（防重复入队）
    fail_count: int = 0            # 连续失败次数（>=2 时 Runner 跳过当前 window）
    is_cold_start: bool = True     # 冷启动标志：第一次请求后置 False

    def try_advance(
        self,
        fq: FrameQueue,
        target_classes: Optional[List[str]] = None,
    ) -> Optional[WindowSpec]:
        """从 max_read_ts+1 起扫 FrameQueue，找首个"连续 window_sec 个 ts + target attrs ≠ None"的窗口。

        Args:
            fq: 该 cam 的 FrameQueue
            target_classes: 目标类别列表（默认 ["baby"] 兼容 v1）

        Returns:
            WindowSpec 或 None
        """
        target_classes = target_classes or ["baby"]
        if fq is None:
            return None
        # 拿 deque 的快照（按 ts 升序）
        items = fq.snapshot_sorted()

        if not items:
            return None

        start_search = self.max_read_ts + 1
        # 退化分支：cursor 落后于 queue（items[0].ts > start_search）时，
        # 把 cursor 推到 queue 头部，避免 lag_sec 无限增长
        if items[0].ts > start_search:
            self.max_read_ts = items[0].ts - 1
            start_search = self.max_read_ts + 1
        # 二分找第一个 ts >= start_search 的位置
        lo, hi = 0, len(items)
        while lo < hi:
            mid = (lo + hi) // 2
            if items[mid].ts < start_search:
                lo = mid + 1
            else:
                hi = mid
        idx = lo

        # 滑动窗口找连续 window_sec 个 target attrs ≠ None 且 ts 连续
        n = len(items)
        w = self.window_sec
        for i in range(idx, n - w + 1):
            # ts 连续检查
            ok_ts = True
            for k in range(w):
                if items[i + k].ts != items[i].ts + k:
                    ok_ts = False
                    break
            if not ok_ts:
                continue
            # target attrs 全部 ≠ None 检查
            ok_tgt = True
            for k in range(w):
                if not _all_targets_known(items[i + k], target_classes):
                    ok_tgt = False
                    break
            if not ok_tgt:
                continue
            # 命中
            ts_list = [items[i + k].ts for k in range(w)]
            return WindowSpec(ts_list=ts_list, start_ts=ts_list[0])
        return None

    def mark_done(self, window_ts_max: int) -> None:
        """worker 处理完一窗后调；推进 max_read_ts。"""
        if window_ts_max > self.max_read_ts:
            self.max_read_ts = window_ts_max

    def inc_fail_count(self) -> None:
        """VLM 调用失败时调；连续失败 +1。"""
        self.fail_count += 1

    def reset_fail_count(self) -> None:
        """VLM 调用成功时调；fail_count 归零。"""
        self.fail_count = 0

    def consume_cold_start(self) -> bool:
        """取走冷启动标志并返回它（一次性）。

        Runner 在每次准备发 VLM 请求前调一次：
        - 返回 True → 用 first timeout（长）
        - 返回 False → 用 normal timeout（短）
        调用后无论返回 True/False 都置 is_cold_start=False，
        这样本 prompt 的"第一次请求"语义只生效一次（A1 方案）。
        """
        cold = self.is_cold_start
        self.is_cold_start = False
        return cold


# ---------------------------------------------------------------------------
class PromptCursorManager:
    """per (cam, prompt) 一份 cursor（Step 8 实例化为全局单例）。"""

    def __init__(self):
        self._cursors: Dict[tuple, PromptCursor] = {}
        self._lock = threading.Lock()

    def get(
        self, cam_id: int, prompt_id: int, window_sec: int
    ) -> PromptCursor:
        """取或创建 cursor（方案 A：max_read_ts=-1，启动瞬间消化历史堆积帧）。"""
        key = (cam_id, prompt_id)
        with self._lock:
            cur = self._cursors.get(key)
            if cur is None:
                cur = PromptCursor(
                    cam_id=cam_id,
                    prompt_id=prompt_id,
                    window_sec=window_sec,
                )
                self._cursors[key] = cur
            return cur

    def remove(self, cam_id: int, prompt_id: int) -> None:
        with self._lock:
            self._cursors.pop((cam_id, prompt_id), None)

    def remove_cam(self, cam_id: int) -> None:
        with self._lock:
            for key in list(self._cursors.keys()):
                if key[0] == cam_id:
                    self._cursors.pop(key, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "cursor_count": len(self._cursors),
                "keys": list(self._cursors.keys()),
            }

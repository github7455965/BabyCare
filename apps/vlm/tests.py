"""
Step 6 单元测试：PromptCursor + FrameSelector

范围
----
- 不需要 DB、不需要 Django 启动（纯算法 + FrameQueue in-memory）
- 用 unittest 直接跑（不依赖 Django TestCase 的 DB 初始化）
- FrameItem 用最小 stub：ts / ndarray / has_baby

跑法
----
.venv\\Scripts\\python.exe -m unittest apps.vlm.tests -v
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import uuid


def _test_name(base: str) -> str:
    """测试 cam/prompt name 加 uuid 后缀，避免撞用户 DB unique 约束。"""
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


# Step 8: PromptRunner 测试需要 mock apps.vlm.models.VLMCheckState.objects.create
# 这会触发 Django settings 检查。在测试入口配 settings（不连 DB，只走 ORM mock）。
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

import responses  # noqa: E402

from apps.yolo_detect.frame_queue import FrameItem, FrameQueue  # noqa: E402
from apps.vlm.cursor import PromptCursor, PromptCursorManager, WindowSpec  # noqa: E402
from apps.vlm.frame_selector import select_three_frames  # noqa: E402
from apps.streaming.models import Camera  # noqa: E402
from apps.vlm.llama_client import (  # noqa: E402
    LlamaClient,
    LlamaNetworkError,
    LlamaParseError,
    LlamaTimeoutError,
)
from apps.vlm.runner import PromptRunner, PromptRunnerManager  # noqa: E402


# ---------------------------------------------------------------------------
# helper
def _make_frames(
    has_baby_list: List[Optional[bool]],
    start_ts: int = 0,
) -> List[FrameItem]:
    """构造连续的 FrameItem 列表（ts = start_ts, start_ts+1, ...）。"""
    return [
        FrameItem(ts=start_ts + i, ndarray=None, has_baby=b)
        for i, b in enumerate(has_baby_list)
    ]


def _feed_fq(fq: FrameQueue, has_baby_list: List[Optional[bool]], start_ts: int = 0) -> None:
    """直接 push 到 FrameQueue（绕过 SamplerWorker）。"""
    for i, b in enumerate(has_baby_list):
        fq.push(start_ts + i, ndarray=None, has_baby=b)


# ===========================================================================
# PromptCursor tests
# ===========================================================================
class PromptCursorTest(unittest.TestCase):
    """PromptCursor.try_advance / mark_done"""

    def test_empty_fq_returns_none(self):
        fq = FrameQueue(cam_id=1)
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq))

    def test_fewer_than_window_returns_none(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True, True, True])  # 3 帧 < 5
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq))

    def test_full_window_returns_spec(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True, True, False, True, False])
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        spec = cur.try_advance(fq)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.start_ts, 0)
        self.assertEqual(spec.ts_list, [0, 1, 2, 3, 4])

    def test_has_baby_none_breaks_window(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True, True, None, True, True])  # 第三个未推理
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq))

    def test_gap_in_ts_breaks_window(self):
        """ts 不连续时（中间缺帧）→ try_advance 找不到窗口。"""
        fq2 = FrameQueue(cam_id=2)
        # push ts=0, 1, 3, 4, 5（缺 2）→ deque 里 ts=[0,1,3,4,5] 但 ts 不连续
        for ts in [0, 1, 3, 4, 5]:
            fq2.push(ts, ndarray=None, has_baby=True)
        cur = PromptCursor(cam_id=2, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq2))

    def test_max_read_ts_skips_already_done(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True, True, True, True, True, False, True, True])
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)  # 默认 max_read_ts=-1
        spec1 = cur.try_advance(fq)
        self.assertEqual(spec1.ts_list, [0, 1, 2, 3, 4])
        cur.mark_done(4)
        # 缺 8, 9 → None
        spec2 = cur.try_advance(fq)
        self.assertIsNone(spec2)
        # push 8, 9
        fq.push(8, ndarray=None, has_baby=False)
        fq.push(9, ndarray=None, has_baby=False)
        spec3 = cur.try_advance(fq)
        self.assertEqual(spec3.ts_list, [5, 6, 7, 8, 9])

    def test_window_sec_10(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 10)
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=10)
        spec = cur.try_advance(fq)
        self.assertEqual(len(spec.ts_list), 10)
        self.assertEqual(spec.ts_list[0], 0)
        self.assertEqual(spec.ts_list[-1], 9)

    def test_window_sec_30(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 30)
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=30)
        spec = cur.try_advance(fq)
        self.assertEqual(len(spec.ts_list), 30)
        self.assertEqual(spec.ts_list[0], 0)
        self.assertEqual(spec.ts_list[-1], 29)

    def test_max_read_ts_in_middle_of_fq(self):
        """max_read_ts 落在 fq 中间 → 二分正确跳过已读 ts。"""
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 20)  # ts=0..19
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5, max_read_ts=9)
        # 应该跳过 0..9，从 10 起找 → [10..14]
        spec = cur.try_advance(fq)
        self.assertEqual(spec.ts_list, [10, 11, 12, 13, 14])

    def test_fq_starts_after_max_read_ts(self):
        """fq 里的 ts 全部 > max_read_ts → 正常返回。"""
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 5, start_ts=100)  # ts=100..104
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5, max_read_ts=0)
        spec = cur.try_advance(fq)
        self.assertEqual(spec.ts_list, [100, 101, 102, 103, 104])

    def test_max_read_ts_past_fq_end_returns_none(self):
        """max_read_ts > fq 最后一个 ts → 全部跳过 → None。"""
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 5)  # ts=0..4
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5, max_read_ts=100)
        self.assertIsNone(cur.try_advance(fq))

    def test_try_advance_fq_none_returns_none(self):
        """fq=None 显式覆盖。"""
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(None))

    def test_mark_done_no_op_on_smaller_ts(self):
        """mark_done(ts_max <= current) → 不变（保护倒退）。"""
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5, max_read_ts=10)
        cur.mark_done(5)   # 比当前小
        self.assertEqual(cur.max_read_ts, 10)
        cur.mark_done(10)  # 等于当前
        self.assertEqual(cur.max_read_ts, 10)

    def test_mark_done_then_try_advance_consistent(self):
        """mark_done 后立刻 try_advance → 不会再返回同一窗口。"""
        fq = FrameQueue(cam_id=1)
        _feed_fq(fq, [True] * 5)
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        spec = cur.try_advance(fq)
        cur.mark_done(spec.ts_list[-1])
        # 同 fq 状态再 try_advance → None（已处理完）
        self.assertIsNone(cur.try_advance(fq))


# ===========================================================================
# PromptCursorManager tests
# ===========================================================================
class PromptCursorManagerTest(unittest.TestCase):
    def test_get_creates_lazily(self):
        mgr = PromptCursorManager()
        cur1 = mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        # 方案 A：max_read_ts=-1（行为等价"从 ts=0 开始扫"）
        self.assertEqual(cur1.max_read_ts, -1)
        self.assertEqual(cur1.last_invoke_ts, 0)
        # 同 key → 同对象
        cur2 = mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIs(cur1, cur2)

    def test_remove_creates_new(self):
        mgr = PromptCursorManager()
        mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        mgr.remove(cam_id=1, prompt_id=10)
        cur = mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        self.assertEqual(cur.max_read_ts, -1)

    def test_remove_cam_clears_all_prompts(self):
        mgr = PromptCursorManager()
        mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        mgr.get(cam_id=1, prompt_id=11, window_sec=10)
        mgr.get(cam_id=2, prompt_id=10, window_sec=5)
        mgr.remove_cam(cam_id=1)
        # cam=1 全部清掉，cam=2 保留
        self.assertEqual(mgr.stats()["cursor_count"], 1)

    def test_stats_returns_keys(self):
        mgr = PromptCursorManager()
        mgr.get(cam_id=1, prompt_id=10, window_sec=5)
        mgr.get(cam_id=2, prompt_id=20, window_sec=10)
        stats = mgr.stats()
        self.assertEqual(stats["cursor_count"], 2)
        self.assertEqual(set(stats["keys"]), {(1, 10), (2, 20)})


# ===========================================================================
# FrameSelector tests
# ===========================================================================
class FrameSelectorTest(unittest.TestCase):
    """select_three_frames（V2 算法：最大化 True + 保证 ts 升序）"""

    # ---- True >= 3：严格三段 ----
    def test_all_true_returns_default_strict_three(self):
        """5 帧全 True → 严格三段 [ts[0], ts[2], ts[4]]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([True] * 5, start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertIsNotNone(result)
        self.assertEqual([f.ts for f in result], [10, 12, 14])
        self.assertTrue(all(f.has_baby for f in result))
        # ts 升序
        self.assertEqual(result, sorted(result, key=lambda f: f.ts))

    def test_four_true_one_false_strict_three(self):
        """n_true=4（边界：≥3 分支）→ 严格三段从 true_ts 取"""
        ts_list = [10, 11, 12, 13, 14]
        # True=[10, 11, 13, 14]（中间隔一个 False），False=[12]
        frames = _make_frames([True, True, False, True, True], start_ts=10)
        result = select_three_frames(frames, ts_list)
        # true_ts=[10, 11, 13, 14]，len=4，mid=true_ts[2]=13 → [10, 13, 14]
        self.assertEqual([f.ts for f in result], [10, 13, 14])

    def test_five_true_window_10(self):
        """10 帧 5 True [T,T,T,T,T,F,F,F,F,F] → True 数=5 ≥ 3 → 严格三段"""
        ts_list = list(range(10))
        frames = _make_frames([True]*5 + [False]*5, start_ts=0)
        result = select_three_frames(frames, ts_list)
        # true_ts=[0,1,2,3,4] → [0, 4//2=2, 4] = [0,2,4]
        self.assertEqual([f.ts for f in result], [0, 2, 4])

    def test_three_true_in_long_window(self):
        """30 帧只有 3 True [True@0, False*28, True@29, True 不在末尾] → True 数=3"""
        # True=[0, 28, 29] → 严格三段 [0, 28, 29]
        # True=[0, 15, 29] → 严格三段 [0, 15, 29]
        has_baby = [True] + [False]*27 + [True, True]
        ts_list = list(range(30))
        frames = _make_frames(has_baby, start_ts=0)
        result = select_three_frames(frames, ts_list)
        # true_ts=[0, 28, 29] → [0, 28, 29]（mid = 28）
        self.assertEqual([f.ts for f in result], [0, 28, 29])

    # ---- True == 2：2 True + 1 False ----
    def test_two_true_one_false(self):
        """窗口 [F, T, F, T, F]；True=[11,13]，False=[10,12,14]
        → 选 [true[0]=11, true[-1]=13, false_mid=12] → 排序 [11,12,13]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([False, True, False, True, False], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertIsNotNone(result)
        self.assertEqual([f.ts for f in result], [11, 12, 13])
        # ts 升序
        ts_result = [f.ts for f in result]
        self.assertEqual(ts_result, sorted(ts_result))

    def test_two_true_at_boundaries(self):
        """窗口 [T, F, F, F, T]；True=[10,14]，False=[11,12,13] → false_mid=12 → [10,12,14]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([True, False, False, False, True], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [10, 12, 14])

    def test_two_true_adjacent(self):
        """窗口 [T, T, F, F, F]；True=[10,11] 相邻 → true[0]=10, true[-1]=11
        False=[12,13,14] → false_mid=false_ts[1]=13 → 排序 [10, 11, 13]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([True, True, False, False, False], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [10, 11, 13])

    # ---- True == 1：1 True + 2 False ----
    def test_one_true_two_false(self):
        """窗口 [F, F, T, F, F]；True=[12]，False=[10,11,13,14]
        → 2 张"位置最靠中"的 False = mid_right=13, mid_left=11 → 排序 [11,12,13]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([False, False, True, False, False], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [11, 12, 13])
        # 校验：has_baby=[F,T,F]，中间是 True
        self.assertEqual([f.has_baby for f in result], [False, True, False])

    def test_one_true_at_index_1(self):
        """窗口 [F, T, F, F, F]；True=[11]，False=[10,12,13,14]
        → 选 [10, false_mid=13, 11] → 排序 [10, 11, 13]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([False, True, False, False, False], start_ts=10)
        result = select_three_frames(frames, ts_list)
        # false_ts=[10,12,13,14]，len=4，mid_right=14//2=13, mid_left=3//2=1→12
        # sorted([11, 13, 12]) = [11, 12, 13]
        self.assertEqual([f.ts for f in result], [11, 12, 13])

    def test_one_true_at_start(self):
        """窗口 [T, F, F, F, F]；True=[10]，False=[11,12,13,14]
        → mid_right=14//2=13 (false_ts[2]=13), mid_left=12 (false_ts[1]=12) → 排序 [10,12,13]"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([True, False, False, False, False], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [10, 12, 13])

    def test_one_true_at_end(self):
        """窗口 [F, F, F, F, T]；True=[14]，False=[10,11,12,13]（长度 4）
        → mid_right=false_ts[2]=12, mid_left=false_ts[1]=11 → 排序 [11,12,14]
        注：窗口中心=12，[11,12,14] 比 [12,13,14] 更居中"""
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([False, False, False, False, True], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [11, 12, 14])

    # ---- True == 0：返回 None ----
    def test_all_false_returns_none(self):
        ts_list = [10, 11, 12, 13, 14]
        frames = _make_frames([False] * 5, start_ts=10)
        self.assertIsNone(select_three_frames(frames, ts_list))

    # ---- 边界 ----
    def test_window_3_all_true(self):
        """窗口长度恰好 3 全 True → [ts[0], ts[1], ts[2]]"""
        ts_list = [10, 11, 12]
        frames = _make_frames([True, True, True], start_ts=10)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [10, 11, 12])

    def test_window_30_all_true(self):
        """30 帧全 True → true_ts=[100..129] → mid=true_ts[15]=115 → [100, 115, 129]"""
        ts_list = list(range(100, 130))
        frames = _make_frames([True] * 30, start_ts=100)
        result = select_three_frames(frames, ts_list)
        self.assertEqual([f.ts for f in result], [100, 115, 129])

    def test_empty_ts_list_returns_none(self):
        self.assertIsNone(select_three_frames([1, 2, 3], []))

    def test_empty_frames_returns_none(self):
        self.assertIsNone(select_three_frames([], [10, 11, 12]))

    def test_ts_not_in_frames_returns_none(self):
        """ts_list=[10,11,12]，frames 里只有 ts=20,21 → 默认候选 ts=10 查不到 → None"""
        ts_list = [10, 11, 12]
        frames = _make_frames([True, True], start_ts=20)
        self.assertIsNone(select_three_frames(frames, ts_list))

    # ---- ts 升序不变量：所有用例都不破 ----
    def test_picked_ts_always_sorted(self):
        """构造多种 has_baby 组合，校验 3 张图始终 ts 升序。"""
        for combo in itertools.product([True, False], repeat=5):
            ts_list = [10, 11, 12, 13, 14]
            frames = _make_frames(list(combo), start_ts=10)
            result = select_three_frames(frames, ts_list)
            if result is not None:
                ts = [f.ts for f in result]
                self.assertEqual(ts, sorted(ts), f"not sorted: combo={combo}")
                # 且 ts 都在 ts_list 里
                for t in ts:
                    self.assertIn(t, ts_list)


# ===========================================================================
# PromptCursor fail_count / cold_start tests
# ===========================================================================
class PromptCursorFailCountTest(unittest.TestCase):
    """PromptCursor.fail_count + inc/reset + is_cold_start/consume_cold_start"""

    def test_initial_fail_count_zero(self):
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertEqual(cur.fail_count, 0)
        self.assertTrue(cur.is_cold_start)

    def test_inc_fail_count(self):
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        cur.inc_fail_count()
        self.assertEqual(cur.fail_count, 1)
        cur.inc_fail_count()
        self.assertEqual(cur.fail_count, 2)

    def test_reset_fail_count(self):
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        cur.inc_fail_count()
        cur.inc_fail_count()
        cur.inc_fail_count()
        cur.reset_fail_count()
        self.assertEqual(cur.fail_count, 0)

    def test_consume_cold_start_first_call_returns_true(self):
        """首次调用 consume_cold_start → True（提示 Runner 用 first timeout）。"""
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertTrue(cur.consume_cold_start())
        # 调用后立即置 False
        self.assertFalse(cur.is_cold_start)

    def test_consume_cold_start_subsequent_returns_false(self):
        """后续调用 → False（提示 Runner 用 normal timeout）。"""
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        cur.consume_cold_start()  # 第一次
        self.assertFalse(cur.consume_cold_start())  # 第二次
        self.assertFalse(cur.consume_cold_start())  # 第三次

    def test_consume_cold_start_independent_of_fail_count(self):
        """consume_cold_start 与 fail_count 互不影响（Step 8 解耦）。"""
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        cur.inc_fail_count()
        cur.inc_fail_count()
        # fail_count=2 仍返回 True（首次请求）
        self.assertTrue(cur.consume_cold_start())
        # 后续 false
        self.assertFalse(cur.consume_cold_start())


# ===========================================================================
# LlamaClient tests
# ===========================================================================
def _make_jpeg_bytes(n: int) -> bytes:
    """构造 n 字节的 jpg 字节串（模拟；实际无所谓，base64 编码不校验内容）。"""
    return b"\xff\xd8\xff\xe0" + b"\x00" * (n - 4)


def _ok_response(content: str) -> dict:
    """构造 OpenAI 兼容的成功响应 dict。"""
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


class LlamaClientTest(unittest.TestCase):
    """LlamaClient HTTP + 解析；用 `responses` 库 mock requests。"""

    def setUp(self):
        self.client = LlamaClient(
            base_url="http://test-llama:8082",
            timeout_sec_first=2,
            timeout_sec_normal=1,
            max_retries=1,
            retry_sleep_sec=0.01,
        )
        self.images = [_make_jpeg_bytes(100) for _ in range(3)]

    # ---- helpers ----
    ERASE_URL_RE = re.compile(r"^http://test-llama:8082/slots/\d+\?action=erase$")

    @staticmethod
    def _add_erase_ok(times: int = 1) -> None:
        """注册 `_erase_slot` 的 mock 响应（每次 chat 结束会调 1 次）。

        `_erase_slot` 本身容忍失败，所以不注册也能过；但那样 `responses.calls`
        里会混进一条 ConnectionError 脏调用，也把"清 KV 有没有真发出去"掩盖掉。
        """
        for _ in range(times):
            responses.add(responses.POST, LlamaClientTest.ERASE_URL_RE, status=200)

    @staticmethod
    def _chat_calls():
        """只统计 /v1/chat/completions 调用（排除 `_erase_slot`）。"""
        return [c for c in responses.calls if "/v1/chat/completions" in c.request.url]

    # ---- base_url 拼接 ----
    def test_base_url_trailing_slash_stripped(self):
        c = LlamaClient(base_url="http://x:8082/", timeout_sec_first=1)
        self.assertEqual(c.base_url, "http://x:8082")

    def test_base_url_default_from_env(self):
        import os
        os.environ["BABYCARE_LLAMA_SERVER_URL"] = "http://from-env:9999"
        try:
            c = LlamaClient()
            self.assertEqual(c.base_url, "http://from-env:9999")
        finally:
            del os.environ["BABYCARE_LLAMA_SERVER_URL"]

    # ---- chat 成功 ----
    @responses.activate
    def test_chat_success_returns_content(self):
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json=_ok_response("是"),
            status=200,
        )
        self._add_erase_ok()
        result = self.client.chat(self.images, "prompt", max_tokens=64)
        self.assertEqual(result, "是")

    @responses.activate
    def test_chat_sends_correct_payload(self):
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json=_ok_response("否"),
            status=200,
        )
        self._add_erase_ok()
        self.client.chat(self.images, "prompt-test", max_tokens=128)
        # 校验请求体（第 1 条是 chat；_erase_slot 的调用排在后面）
        call = self._chat_calls()[0]
        body = json.loads(call.request.body)
        self.assertEqual(body["model"], "qwen3vl")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 128)
        # messages
        msgs = body["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        content = msgs[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "prompt-test")
        # 3 张图（base64 data URL）
        image_parts = [c for c in content if c["type"] == "image_url"]
        self.assertEqual(len(image_parts), 3)
        for img in image_parts:
            url = img["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    # ---- 重试 ----
    @responses.activate
    def test_chat_timeout_retries_once(self):
        """首次 timeout → 重试 1 次成功。"""
        import requests
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            body=requests.exceptions.Timeout("simulated"),
        )
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json=_ok_response("是"),
            status=200,
        )
        self._add_erase_ok()
        result = self.client.chat(self.images, "p", max_tokens=64)
        self.assertEqual(result, "是")
        self.assertEqual(len(self._chat_calls()), 2)

    @responses.activate
    def test_chat_timeout_raises_after_retry_exhausted(self):
        """两次都 timeout → LlamaTimeoutError。"""
        import requests
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            body=requests.exceptions.Timeout("simulated 1"),
        )
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            body=requests.exceptions.Timeout("simulated 2"),
        )
        self._add_erase_ok()
        with self.assertRaises(LlamaTimeoutError):
            self.client.chat(self.images, "p", max_tokens=64)

    @responses.activate
    def test_chat_5xx_raises_network_error(self):
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json={"error": "internal"},
            status=500,
        )
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json={"error": "internal"},
            status=502,
        )
        self._add_erase_ok()
        with self.assertRaises(LlamaNetworkError):
            self.client.chat(self.images, "p", max_tokens=64)

    @responses.activate
    def test_chat_4xx_raises_parse_error_no_retry(self):
        """4xx → parse_error，不重试（参数错重试无意义）。"""
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json={"error": "bad request"},
            status=400,
        )
        self._add_erase_ok()
        with self.assertRaises(LlamaParseError):
            self.client.chat(self.images, "p", max_tokens=64)
        # 不重试：chat 只调 1 次（_erase_slot 的调用不计）
        self.assertEqual(len(self._chat_calls()), 1)

    @responses.activate
    def test_chat_bad_json_raises_parse_error(self):
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            body="not a json",
            status=200,
        )
        self._add_erase_ok()
        with self.assertRaises(LlamaParseError):
            self.client.chat(self.images, "p", max_tokens=64)

    @responses.activate
    def test_chat_survives_erase_slot_failure(self):
        """`_erase_slot` 失败不得影响 chat 结果。

        回归：`_erase_slot` 的 except 分支曾引用未定义的 `logger`。
        真实场景里清 KV 很容易失败（llama-server 重启 / 路由不存在），
        那时会抛 NameError 把一次成功的 chat 一起带崩。
        """
        responses.add(
            responses.POST,
            "http://test-llama:8082/v1/chat/completions",
            json=_ok_response("是"),
            status=200,
        )
        # 故意不注册 /slots/0 → _erase_slot 内部拿到 ConnectionError
        result = self.client.chat(self.images, "p", max_tokens=64)
        self.assertEqual(result, "是")

    # ---- 解析 ----
    def test_parse_plain_hit(self):
        hit, status = self.client.parse("是，宝宝口鼻遮蔽", positive_keyword="是")
        self.assertTrue(hit)
        self.assertEqual(status, "宝宝口鼻遮蔽")

    def test_parse_plain_miss(self):
        hit, status = self.client.parse("否，宝宝正常", positive_keyword="是")
        self.assertFalse(hit)
        self.assertEqual(status, "否，宝宝正常")

    def test_parse_plain_strips_separator(self):
        hit, status = self.client.parse("是：脸部遮盖", positive_keyword="是")
        self.assertTrue(hit)
        self.assertEqual(status, "脸部遮盖")

    def test_parse_plain_strips_whitespace(self):
        hit, status = self.client.parse("  是   ", positive_keyword="是")
        self.assertTrue(hit)
        # status = ""（关键词后只有空白）
        self.assertEqual(status, "")

    def test_parse_plain_avoid_substring(self):
        """避免"不是"被误判为"是"（首字符匹配仅命中开头的"是"）。"""
        hit, status = self.client.parse("不是，宝宝正常", positive_keyword="是")
        self.assertFalse(hit)
        self.assertEqual(status, "不是，宝宝正常")

    def test_parse_plain_empty_response(self):
        hit, status = self.client.parse("", positive_keyword="是")
        self.assertFalse(hit)
        self.assertEqual(status, "")

    def test_parse_json_hit(self):
        hit, status = self.client.parse(
            '{"verdict": "是", "description": "口鼻遮蔽"}',
            positive_keyword="是",
            result_format="json",
        )
        self.assertTrue(hit)
        self.assertEqual(status, "口鼻遮蔽")

    def test_parse_json_miss(self):
        hit, status = self.client.parse(
            '{"verdict": "否", "description": "正常"}',
            positive_keyword="是",
            result_format="json",
        )
        self.assertFalse(hit)
        self.assertEqual(status, "正常")

    def test_parse_json_fallback_to_plain_on_invalid(self):
        """JSON 格式但内容不是 JSON → fallback 到 plain 解析。"""
        hit, status = self.client.parse(
            "是，状态正常",
            positive_keyword="是",
            result_format="json",
        )
        self.assertTrue(hit)
        self.assertEqual(status, "状态正常")

    def test_parse_json_non_dict_fallback(self):
        """JSON 格式但顶层不是 dict → fallback 到 plain。"""
        hit, status = self.client.parse(
            '"just a string"',
            positive_keyword="是",
            result_format="json",
        )
        # fallback 到 plain：startswith("是")? "just a string" 不以"是"开头
        self.assertFalse(hit)
        self.assertEqual(status, '"just a string"')

    # ---- _extract_content 边界 ----
    def test_extract_content_choices_empty(self):
        with self.assertRaises(LlamaParseError):
            LlamaClient._extract_content({"choices": []})

    def test_extract_content_missing_message(self):
        with self.assertRaises(LlamaParseError):
            LlamaClient._extract_content({"choices": [{"finish_reason": "stop"}]})

    def test_extract_content_non_string(self):
        with self.assertRaises(LlamaParseError):
            # content 是 list（多模态响应但本次不应出现）
            LlamaClient._extract_content({
                "choices": [{"message": {"content": [{"type": "text", "text": "x"}]}}]
            })

    # ---- describe 描述型 ----
    def test_parse_describe_keeps_full_text(self):
        """描述型：status = 全文；hit=False。"""
        hit, status = self.client.parse(
            "宝宝在床左侧趴睡，呼吸平稳，眼睛半闭",
            positive_keyword="是",
            kind="describe",
        )
        self.assertFalse(hit)
        self.assertEqual(status, "宝宝在床左侧趴睡，呼吸平稳，眼睛半闭")

    def test_parse_describe_strips_whitespace(self):
        """描述型也做 strip。"""
        hit, status = self.client.parse(
            "  宝宝醒了  \n",
            positive_keyword="是",
            kind="describe",
        )
        self.assertFalse(hit)
        self.assertEqual(status, "宝宝醒了")

    def test_parse_describe_empty(self):
        hit, status = self.client.parse("", positive_keyword="是", kind="describe")
        self.assertFalse(hit)
        self.assertEqual(status, "")

    def test_parse_describe_ignores_positive_keyword(self):
        """描述型：即使文本以"是"开头也不命中（hit 永远 False）。"""
        hit, status = self.client.parse(
            "是，宝宝正在睡觉",
            positive_keyword="是",
            kind="describe",
        )
        self.assertFalse(hit)
        self.assertEqual(status, "是，宝宝正在睡觉")

    def test_parse_describe_ignores_result_format(self):
        """描述型忽略 result_format（无论 plain 还是 json 都返回全文）。"""
        # json 但内容不是 JSON → 描述型也不该 fallback 到首字符匹配
        hit, status = self.client.parse(
            '{"verdict": "是", "description": "宝宝清醒"}',
            positive_keyword="是",
            result_format="json",
            kind="describe",
        )
        self.assertFalse(hit)
        self.assertEqual(status, '{"verdict": "是", "description": "宝宝清醒"}')

    def test_parse_judge_default_kind(self):
        """默认 kind="judge"：保持原有首字符匹配行为。"""
        hit, status = self.client.parse("是，宝宝口鼻遮蔽", positive_keyword="是")
        self.assertTrue(hit)
        self.assertEqual(status, "宝宝口鼻遮蔽")

    def test_parse_describe_long_text(self):
        """描述型：长文本（> 64 字符）保留完整。"""
        long_text = "宝宝" + "在床左侧趴睡" * 30  # 约 180 字
        hit, status = self.client.parse(long_text, positive_keyword="是", kind="describe")
        self.assertFalse(hit)
        self.assertEqual(status, long_text)
        # 长度校验：>64 字符（旧 CharField 限制）
        self.assertGreater(len(status), 64)


# ===========================================================================
# PromptRunner / PromptRunnerManager tests（Step 8）
# ===========================================================================
class _FakePrompt:
    """跑 Runner 用的最小 prompt stub（不依赖 DB）。"""

    def __init__(
        self,
        pid: int = 10,
        text: str = "宝宝是否在床？",
        positive_keyword: str = "是",
        result_format: str = "plain",
        kind: str = "judge",
        window_sec: int = 5,
        max_tokens: int = 128,
        timeout_first: int = 60,
        timeout_normal: int = 15,
        silence_count_after_dismiss: int = 0,
        notify_on_hit: bool = False,
        target_classes: str = "baby",
    ):
        self.id = pid
        self.prompt = text
        self.positive_keyword = positive_keyword
        self.result_format = result_format
        self.kind = kind
        self.window_sec = window_sec
        self.max_tokens = max_tokens
        self.timeout_sec_first = timeout_first
        self.timeout_sec_normal = timeout_normal
        self.silence_count_after_dismiss = silence_count_after_dismiss
        self.notify_on_hit = notify_on_hit
        self.target_classes = target_classes


def _make_runner_with_mocks(
    cam_id: int = 1,
    prompt: Optional[_FakePrompt] = None,
    chat_return: str = "是",
    chat_side_effect=None,
):
    """构造一个带 mock llama_client + FrameQueue 的 Runner。

    默认 patch _acquire_vlm / _release_vlm 为 no-op（避免测试每轮真启 llama-server ~5s
    health probe 拖慢）；需要真 GPU 调度的测试用 prompt_run_real_acquire=True 关闭。
    """
    prompt = prompt or _FakePrompt()
    fq = FrameQueue(cam_id=cam_id)
    cursor = PromptCursor(cam_id=cam_id, prompt_id=prompt.id, window_sec=prompt.window_sec)

    llama = MagicMock()
    if chat_side_effect is not None:
        llama.chat.side_effect = chat_side_effect
    else:
        llama.chat.return_value = chat_return
    llama.parse.return_value = (True, "在床") if chat_return == "是" else (False, chat_return)

    executor = ThreadPoolExecutor(max_workers=2)
    runner = PromptRunner(
        cam_id=cam_id,
        prompt_id=prompt.id,
        prompt_text=prompt.prompt,
        positive_keyword=prompt.positive_keyword,
        result_format=prompt.result_format,
        kind=prompt.kind,
        window_sec=prompt.window_sec,
        max_tokens=prompt.max_tokens,
        timeout_sec_first=prompt.timeout_sec_first,
        timeout_sec_normal=prompt.timeout_sec_normal,
        cursor=cursor,
        frame_queue=fq,
        llama_client=llama,
        executor=executor,
        prompt_obj=prompt,
        poll_interval=0.05,  # 测试加速
    )
    # 默认 no-op GPU acquire/release。需要真启 llama 的测试自己 patch 回来。
    runner._acquire_vlm = lambda: None  # type: ignore[assignment]
    runner._release_vlm = lambda: None  # type: ignore[assignment]
    return runner, llama, fq, cursor, executor


def _drive_runner(runner, run_sec: float = 0.3):
    """在测试自己的 loop 上跑 runner → 等 run_sec → stop。

    PromptRunner.start() 现在是 async（要在 Manager loop 上调）；
    测试里直接 asyncio.run 启动一个临时 loop 调它即可。

    apps.ready() 在测试启动阶段就跑了 RunnerManager.start()，会起一个真实 Runner
    在后台跑，可能污染 VLMCheckState.objects.create patch。测试入口统一 stop 一下。
    """
    from apps.vlm.runner import PromptRunnerManager
    try:
        PromptRunnerManager.instance().stop()
    except Exception:
        pass

    # 熔断器是**进程内单例**（`LLMBreaker.for_service`），状态会跨用例残留：
    # 某个用例连撞 3 次连接类失败后，后面的用例会看到 `is_blocking()=True`
    # → 窗口全被跳过、chat 一次都不调。每个用例进场前清一次。
    from apps.core.llm_breaker import LLMBreaker
    LLMBreaker.reset_instances()

    async def _go():
        await runner.start()
        await asyncio.sleep(run_sec)
        await runner.stop()
    asyncio.run(_go())


class PromptRunnerTest(unittest.TestCase):
    """PromptRunner 主循环行为（mock llama_client）。"""

    def test_empty_fq_no_chat_call(self):
        """fq 空 → 不调 llama。"""
        runner, llama, fq, cursor, executor = _make_runner_with_mocks()
        try:
            _drive_runner(runner, run_sec=0.3)
        finally:
            executor.shutdown(wait=False)
        self.assertEqual(llama.chat.call_count, 0)
        self.assertEqual(runner.stats.invoke_count, 0)

    def test_window_ready_calls_chat_and_marks_done(self):
        """fq 有完整窗口 → 调 llama 1 次 → cursor.mark_done。"""
        from apps.yolo_detect.frame_queue import FrameItem
        runner, llama, fq, cursor, executor = _make_runner_with_mocks()
        try:
            # 喂 5 帧全 True
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            # mock VLMCheckState.objects.create（不真写 DB）
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            self.assertGreaterEqual(llama.chat.call_count, 1)
            # cursor 应该推进（mark_done）
            self.assertGreaterEqual(cursor.max_read_ts, 4)
            # 至少写了一条库（成功）
            self.assertGreaterEqual(m_create.call_count, 1)
            # 调用时 cold_start 第一次 → first timeout
            call_kwargs = llama.chat.call_args
            self.assertEqual(call_kwargs.kwargs.get("timeout_sec"), 60)
        finally:
            executor.shutdown(wait=False)

    def test_cold_start_consumed_after_first_call(self):
        """第一次调 llama 后 is_cold_start → False。"""
        from apps.yolo_detect.frame_queue import FrameItem
        runner, llama, fq, cursor, executor = _make_runner_with_mocks()
        try:
            for i in range(10):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"):
                _drive_runner(runner, run_sec=0.3)
            self.assertFalse(cursor.is_cold_start)
        finally:
            executor.shutdown(wait=False)

    def test_no_baby_window_does_not_call_llama(self):
        """窗口全 False → FrameSelector 返回 None → 不调 llama，写 no_baby 记录。"""
        runner, llama, fq, cursor, executor = _make_runner_with_mocks()
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=False)
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            self.assertEqual(llama.chat.call_count, 0)
            self.assertGreaterEqual(m_create.call_count, 1)
            # 查最近一次 create 的 failure_reason
            kwargs = m_create.call_args.kwargs
            self.assertEqual(kwargs.get("failure_reason"), "no_baby")
        finally:
            executor.shutdown(wait=False)

    def test_fail_count_increments_on_timeout(self):
        """LlamaTimeoutError → cursor.inc_fail_count。"""
        from apps.vlm.llama_client import LlamaTimeoutError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaTimeoutError("timeout"),
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"):
                _drive_runner(runner, run_sec=0.3)
            self.assertGreaterEqual(cursor.fail_count, 1)
            self.assertGreaterEqual(runner.stats.fail_count_total, 1)
        finally:
            executor.shutdown(wait=False)

    def test_fail_count_2_skips_window(self):
        """fail_count >= 2 → 跳过 window + mark_done 推进 cursor + fail_count -= 1 衰减。

        修复：之前 skip 时 fail_count 不变 + 不 mark_done → 永久锁死在同一 window；
        现在每次 skip 消费 1 次配额 + mark_done 让 cursor 推进到下一 window。
        """
        from apps.vlm.llama_client import LlamaTimeoutError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaTimeoutError("timeout"),
        )
        try:
            cursor.inc_fail_count()
            cursor.inc_fail_count()  # =2
            for i in range(10):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"):
                _drive_runner(runner, run_sec=0.05)
            # 修复：skip 时 mark_done 推进 cursor（避免反复同一 window）
            self.assertGreaterEqual(cursor.max_read_ts, 0)
            # fail_count 衰减（之前永久 lock 在 2，现在应 < 2）
            self.assertLess(cursor.fail_count, 2)
            # skip_window 统计 +1
            self.assertGreaterEqual(runner.stats.skip_window_count, 1)
        finally:
            executor.shutdown(wait=False)

    def test_skip_then_lama_recovery_invokes_chat(self):
        """skip 衰减后 llama 恢复 → Runner 调 chat 成功 → fail_count 永久清零。

        模拟：fail_count=2 → 1 次 skip（mark_done 推进 cursor，衰减到 1）→ 下一 window
        chat 仍 side_effect=timeout → inc_fail_count=2 → 又 skip（衰减到 1）→ ...
        改 chat side_effect 为成功（lambda 计数第 2 次起返回 "是"），验证 fail_count 清零。

        注意：必须 patch _acquire_vlm/_release_vlm 为 no-op，避免每轮真启 llama-server (~5s)
        拖慢 + 测试环境副作用。
        """
        from apps.vlm.llama_client import LlamaTimeoutError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaTimeoutError("timeout"),
        )
        try:
            cursor.inc_fail_count()
            cursor.inc_fail_count()  # =2
            for i in range(200):
                fq.push(i, ndarray=None, has_baby=True)

            # 第 1 次 timeout，第 2 次起返回成功（模拟 llama 恢复）
            call_count_box = {"n": 0}

            def chat_side(*a, **kw):
                call_count_box["n"] += 1
                if call_count_box["n"] >= 2:
                    return "是"
                raise LlamaTimeoutError("first timeout")

            with patch("apps.vlm.models.VLMCheckState.objects.create"), \
                 patch.object(llama, "chat", side_effect=chat_side), \
                 patch.object(runner, "_acquire_vlm", lambda: None), \
                 patch.object(runner, "_release_vlm", lambda: None):
                _drive_runner(runner, run_sec=2.0)
            # 至少调过一次 chat
            self.assertGreaterEqual(call_count_box["n"], 2)
            # 成功后 fail_count 永久清零
            self.assertEqual(cursor.fail_count, 0)
            # success_count 至少 +1
            self.assertGreaterEqual(runner.stats.success_count, 1)
        finally:
            executor.shutdown(wait=False)

    def test_timeout_writes_failure_record(self):
        """LlamaTimeoutError → 写一条 failure_reason='llama_timeout' + cursor 推进。"""
        from apps.vlm.llama_client import LlamaTimeoutError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaTimeoutError("timeout"),
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            self.assertGreaterEqual(m_create.call_count, 1)
            kwargs = m_create.call_args.kwargs
            self.assertEqual(kwargs.get("failure_reason"), "llama_timeout")
            # mark_done 后 cursor 推进（≠-1），避免下一 window 重复触发
            self.assertGreaterEqual(cursor.max_read_ts, 0)
        finally:
            executor.shutdown(wait=False)

    def test_startup_failed_enqueues_instead_of_dropping(self):
        """LlamaStartError（acquire_vlm 阶段抛）→ **入队等回放**，不再丢弃。

        行为变更（Phase 3，checklist §2 #8）：旧行为写 failure_reason='startup_failed'
        后 mark_done，窗口就此蒸发 —— 分体部署下"推理机没开"期间**所有**窗口都会丢。
        现在改成写 VLMQueuedTask，等 Drainer 在服务恢复后回放。

        仍保持：不计 fail_count（这不是 VLM 调用失败，是可用性问题），
        下次触发会继续重试 acquire_vlm。
        """
        from apps.vlm.llama_manager import LlamaStartError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=None,  # 不让 chat 抛
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            initial_fail = cursor.fail_count
            with patch.object(runner, "_acquire_vlm",
                       side_effect=LlamaStartError("llama-server 启失败（重试 1 次）")), \
                 patch("apps.vlm.models.VLMQueuedTask.objects.create") as m_enq, \
                 patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            # 窗口进了队列，而不是被丢掉
            self.assertGreaterEqual(m_enq.call_count, 1)
            frs = [c.kwargs.get("failure_reason") for c in m_create.call_args_list]
            self.assertNotIn("startup_failed", frs, f"应入队而非丢弃: {frs}")
            # mark_done 推进 cursor（避免反复同一 window）
            self.assertGreaterEqual(cursor.max_read_ts, 0)
            # 不计 fail_count（让下次重试 acquire_vlm）
            self.assertEqual(cursor.fail_count, initial_fail)
        finally:
            executor.shutdown(wait=False)

    def test_network_error_enqueues_instead_of_dropping(self):
        """LlamaNetworkError → **入队等回放**（不再是 llama_unreachable 丢弃）。

        行为变更（Phase 3，checklist §8.2）：旧行为写 failure_reason='llama_unreachable'
        后 mark_done —— 分体部署下"推理机没开"期间的窗口全丢。现在改成写
        VLMQueuedTask，等 Drainer 在服务恢复后回放。

        同时这个失败会记进熔断器：连撞 `FAIL_THRESHOLD` 次后，后续窗口在 4a 就被
        拦进队列，不再一个个白等一个超时。
        """
        from apps.vlm.llama_client import LlamaNetworkError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaNetworkError("conn refused"),
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMQueuedTask.objects.create") as m_enq, \
                 patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            self.assertGreaterEqual(m_enq.call_count, 1)
            frs = [c.kwargs.get("failure_reason") for c in m_create.call_args_list]
            self.assertNotIn("llama_unreachable", frs, f"应入队而非丢弃: {frs}")
            self.assertGreaterEqual(cursor.max_read_ts, 0)
        finally:
            executor.shutdown(wait=False)

    def test_loading_error_does_not_write_or_advance(self):
        """LlamaLoadingError（llama 正在加载模型）→ 不写库、cursor 推进、不计 fail_count。"""
        from apps.vlm.llama_client import LlamaLoadingError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaLoadingError("Loading model"),
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            initial_fail = cursor.fail_count
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.05)
            # 不写库（loading 是临时状态）
            self.assertEqual(m_create.call_count, 0)
            # cursor 已 mark_done（避免同 window 死循环重试）
            self.assertEqual(cursor.max_read_ts, 4)
            # 不计 fail_count
            self.assertEqual(cursor.fail_count, initial_fail)
        finally:
            executor.shutdown(wait=False)

    def test_success_resets_fail_count(self):
        """成功后 fail_count 归零。"""
        runner, llama, fq, cursor, executor = _make_runner_with_mocks()
        try:
            # 先把 fail_count 拉到 1（< 2，Runner 仍会调 llama）
            cursor.inc_fail_count()
            for i in range(10):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"):
                _drive_runner(runner, run_sec=0.3)
            # 成功调用后 cursor.reset_fail_count → fail_count=0
            self.assertEqual(cursor.fail_count, 0)
            self.assertGreaterEqual(runner.stats.success_count, 1)
        finally:
            executor.shutdown(wait=False)

    def test_parse_error_writes_failure_record(self):
        """LlamaParseError → 写 failure_reason='parse_error'，不调重试。"""
        from apps.vlm.llama_client import LlamaParseError

        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            chat_side_effect=LlamaParseError("bad json"),
        )
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create:
                _drive_runner(runner, run_sec=0.3)
            # ParseError 不重试，只调一次
            self.assertEqual(llama.chat.call_count, 1)
            kwargs = m_create.call_args.kwargs
            self.assertEqual(kwargs.get("failure_reason"), "parse_error")
            # ParseError 不影响 cursor.fail_count
            self.assertEqual(cursor.fail_count, 0)
        finally:
            executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Step 9 集成测试：notify_hit 触发逻辑（mock apps.vlm.notify）
    # ------------------------------------------------------------------
    def test_hit_with_notify_calls_notify_hit(self):
        """hit=True + notify_on_hit=True → notify_hit 被调。"""
        prompt = _FakePrompt(notify_on_hit=True)
        runner, llama, fq, cursor, executor = _make_runner_with_mocks(prompt=prompt)
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"), \
                 patch("apps.vlm.notify.notify_hit") as m_notify:
                _drive_runner(runner, run_sec=0.3)
            self.assertGreaterEqual(llama.chat.call_count, 1)
            self.assertGreaterEqual(m_notify.call_count, 1)
        finally:
            executor.shutdown(wait=False)

    def test_hit_with_notify_off_does_not_call_notify_hit(self):
        """hit=True + notify_on_hit=False → notify_hit 不被调。"""
        prompt = _FakePrompt(notify_on_hit=False)
        runner, llama, fq, cursor, executor = _make_runner_with_mocks(prompt=prompt)
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"), \
                 patch("apps.vlm.notify.notify_hit") as m_notify:
                _drive_runner(runner, run_sec=0.3)
            self.assertEqual(m_notify.call_count, 0)
        finally:
            executor.shutdown(wait=False)

    def test_describe_hit_does_not_call_notify_hit(self):
        """describe 类型 hit=False → notify_hit 不被调。"""
        prompt = _FakePrompt(kind="describe", notify_on_hit=True, positive_keyword="是")
        runner, llama, fq, cursor, executor = _make_runner_with_mocks(
            prompt=prompt, chat_return="侧躺，无异常",
        )
        try:
            # describe parse 返回 (False, 全文) → hit=False
            llama.parse.return_value = (False, "侧躺，无异常")
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create"), \
                 patch("apps.vlm.notify.notify_hit") as m_notify:
                _drive_runner(runner, run_sec=0.3)
            self.assertEqual(m_notify.call_count, 0)
        finally:
            executor.shutdown(wait=False)

    def test_hit_but_auto_silenced_does_not_call_notify_hit(self):
        """hit=True 但 auto_silenced=True → notify_hit 不被调（VLMCheckState 仍写）。"""
        prompt = _FakePrompt(notify_on_hit=True, silence_count_after_dismiss=3)
        runner, llama, fq, cursor, executor = _make_runner_with_mocks(prompt=prompt)
        try:
            for i in range(5):
                fq.push(i, ndarray=None, has_baby=True)
            with patch("apps.vlm.models.VLMCheckState.objects.create") as m_create, \
                 patch("apps.vlm.notify.compute_auto_silenced", return_value=True), \
                 patch("apps.vlm.notify.notify_hit") as m_notify:
                _drive_runner(runner, run_sec=0.3)
            # 命中 + auto_silenced=True → 通知不应被调
            self.assertEqual(m_notify.call_count, 0)
            # 但 VLMCheckState 仍要写库（auto_silenced=True 记录）
            self.assertGreaterEqual(m_create.call_count, 1)
        finally:
            executor.shutdown(wait=False)


# ===========================================================================
# Step 12 单元测试：frame_storage 落盘 + 去重
# ===========================================================================
class FrameStorageTest(unittest.TestCase):
    """apps/vlm/frame_storage.py + PromptRunner._resolve_img_paths 真落盘。

    用 @override_settings(MEDIA_ROOT=tmp_path) 把 MEDIA_ROOT 重定向到临时目录。
    """

    def setUp(self):
        import shutil
        import tempfile

        from django.test import override_settings

        self.tmpdir = Path(tempfile.mkdtemp(prefix="frame_storage_test_"))
        self._override = override_settings(MEDIA_ROOT=str(self.tmpdir))
        self._override.enable()
        self.addCleanup(self._override.disable)
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def _fake_frame(self, ts: int, has_baby=True):
        """构造真实 ndarray（10x10 BGR 渐变）。"""
        import numpy as np
        arr = np.full((10, 10, 3), fill_value=ts % 255, dtype="uint8")
        return FrameItem(ts=ts, ndarray=arr, has_baby=has_baby)

    def test_save_frame_writes_jpg_and_dedups(self):
        """真落盘 + 同 cam_id+ts 第二次调不重复写。"""
        from apps.vlm.frame_storage import save_frame

        arr = self._fake_frame(1700000000).ndarray
        p1 = save_frame(arr, cam_id=7, ts=1700000000, media_root=str(self.tmpdir))
        self.assertIsNotNone(p1)
        self.assertTrue(p1.exists())
        # path 绝对
        self.assertTrue(str(p1).startswith(str(self.tmpdir)))
        # 文件非空
        self.assertGreater(p1.stat().st_size, 0)
        # 第二次同 cam+ts → 仍返回 p1（去重），不重写
        mtime_before = p1.stat().st_mtime
        p2 = save_frame(arr, cam_id=7, ts=1700000000, media_root=str(self.tmpdir))
        self.assertEqual(p1, p2)
        # mtime 不变 → 真没重写（容忍 ±1s 的文件系统精度）
        self.assertEqual(int(p2.stat().st_mtime), int(mtime_before))

    def test_save_frame_none_ndarray_returns_none(self):
        """ndarray=None → 返回 None（不抛、不写盘）。"""
        from apps.vlm.frame_storage import save_frame

        result = save_frame(None, cam_id=1, ts=1700000000, media_root=str(self.tmpdir))
        self.assertIsNone(result)
        # 也没文件
        date_dir = self.tmpdir / "frames"
        if date_dir.exists():
            jpg_count = sum(1 for _ in date_dir.rglob("*.jpg"))
            self.assertEqual(jpg_count, 0)

    def test_runner_resolve_img_paths_writes_three_jpgs(self):
        """PromptRunner._resolve_img_paths(frames, cam_id) 真落盘 3 张 jpg。

        用 _make_runner_with_mocks 构造 Runner，但只调 _resolve_img_paths（不启动 loop）。
        4 元组返回是 (runner, llama, fq, cursor, executor)；executor 要 shutdown。
        """
        from apps.vlm.frame_storage import frame_path

        result = _make_runner_with_mocks(cam_id=9)
        runner = result[0]
        executor = result[4]
        try:
            frames = [self._fake_frame(1700000100 + i) for i in range(3)]
            paths = runner._resolve_img_paths(frames, cam_id=9)  # noqa: SLF001
        finally:
            executor.shutdown(wait=False)

        self.assertEqual(len(paths), 3)
        for path_str, ts in paths:
            self.assertTrue(path_str != "", f"empty path for ts={ts}")
            self.assertTrue(Path(path_str).exists(), f"file not exist: {path_str}")
            # 路径 = frame_path(media_root, cam_id=9, ts)
            expected = frame_path(str(self.tmpdir), cam_id=9, ts=ts)
            self.assertEqual(Path(path_str), expected)
            # path 绝对
            self.assertTrue(os.path.isabs(path_str))


# ===========================================================================
# Step 9 单元测试：notify_hit + compute_auto_silenced（mock 一切，不依赖 DB）
# ===========================================================================
class _FakeStateQS:
    """mock VLMCheckState.objects.filter() 链。"""

    def __init__(self, first_item=None, items=None):
        self._first_item = first_item
        self._items = items or []

    def filter(self, **kwargs):  # noqa: A003
        return self

    def order_by(self, *args):
        return self

    def first(self):
        return self._first_item

    def __iter__(self):
        return iter(self._items)


class NotifyTest(unittest.TestCase):
    """Step 9 notify_hit / compute_auto_silenced 单元测试（mock 一切）。

    B5 改造：notify_hit 现在遍历 prompt.notify_targets 调 _send_mobile_app /
    _send_speaker（直发，不再走 .env 全局 targets）。
    """

    @staticmethod
    def _make_state(hit=True, status="在床", ts_list=None, notify_on_hit=True):
        """构造 mock VLMCheckState 实例（pure mock，不碰 DB）。"""
        from datetime import datetime, timezone
        state = MagicMock()
        state.id = 1
        state.hit = hit
        state.status = status
        state.camera_id = 7
        state.prompt_config_id = 10
        state.dismissed_as_false = False
        state.auto_silenced = False
        # 2025-01-01 00:00:00 UTC（转 +8 = 2025-01-01 08:00:00）
        state.ts_list = ts_list or [int(
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        )]
        state.camera = MagicMock()
        state.camera.name = "客厅"
        state.prompt_config = MagicMock()
        state.prompt_config.pk = 10
        state.prompt_config.name = "口鼻遮蔽"
        state.prompt_config.notify_on_hit = notify_on_hit
        return state

    @staticmethod
    def _target(name, kind="mobile_app", target_id=None):
        """构造 mock NotifyTarget（带 id/kind/target_id，供 through 路由用）。"""
        t = MagicMock()
        t.id = 1000 + (abs(hash(name)) % 9000)
        t.name = name
        t.kind = kind
        t.target_id = target_id or f"notify.{name}"
        t.enabled = True
        return t

    @staticmethod
    def _audio_outcome(result):
        """构造音频三态结果（不碰 DB / 真实音频）。"""
        from apps.vlm.audio_rule import AudioRuleOutcome

        return AudioRuleOutcome(
            result=result, condition="cry", rule_enabled=True, window_sec=60,
        )

    def _run_notify(self, state, links, result, outcome=None):
        """统一跑 notify_hit：patch through 行 / 音频三态 / 投递审计 / HA 发送。

        Returns:
            (返回值, _send_mobile_app mock, _send_speaker mock, _record_delivery mock)
        """
        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._load_target_links", return_value=links), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=outcome or self._audio_outcome(result)), \
             patch("apps.vlm.notify._record_delivery") as m_rec, \
             patch("apps.vlm.notify._send_mobile_app") as m_mobile, \
             patch("apps.vlm.notify._send_speaker") as m_speaker:
            ok = notify_hit(state)
        return ok, m_mobile, m_speaker, m_rec

    @staticmethod
    def _recorded(m_rec, target_name):
        """从 `_record_delivery` 调用参数里取某 target 的审计行参数。"""
        for call in m_rec.call_args_list:
            args = call.args
            if getattr(args[1], "name", None) != target_name:
                continue
            delivered = call.kwargs.get("delivered")
            if delivered is None and len(args) > 4:
                delivered = args[4]
            error = call.kwargs.get("error", "")
            return {
                "condition": args[2],
                "audio_state": args[3].audio_state,
                "delivered": bool(delivered),
                "error": error,
            }
        return None

    # ---- notify_hit ----
    def test_hit_with_mobile_app_target_calls_send(self):
        """hit=True + 1 个 mobile_app target（always）→ 调 _send_mobile_app 一次。"""
        target = self._target("dad_phone", target_id="notify.mobile_app_pgt_an10_donganzhuo")
        state = self._make_state(hit=True, status="在床")

        from apps.vlm.models import PromptNotifyTarget

        with patch("apps.vlm.notify._load_target_links",
                   return_value=[(target, PromptNotifyTarget.CONDITION_ALWAYS)]), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=self._audio_outcome("UNKNOWN")), \
             patch("apps.vlm.notify._record_delivery"), \
             patch("apps.vlm.notify._send_mobile_app") as m_send:
            from apps.vlm.notify import notify_hit

            result = notify_hit(state)
            self.assertTrue(result)
            self.assertEqual(m_send.call_count, 1)
            self.assertEqual(state.notified, True)
            state.save.assert_called_once_with(update_fields=["notified"])

    def test_miss_does_not_notify(self):
        """hit=False → 不发任何 target。"""
        state = self._make_state(hit=False)
        ok, m_mobile, m_speaker, m_rec = self._run_notify(
            state, [(self._target("x"), "always")], "UNKNOWN",
        )
        self.assertFalse(ok)
        self.assertEqual(m_mobile.call_count, 0)
        self.assertEqual(m_speaker.call_count, 0)
        self.assertEqual(m_rec.call_count, 0)
        state.save.assert_not_called()

    def test_notify_on_hit_false_skips(self):
        """prompt.notify_on_hit=False → 不发任何 target。"""
        state = self._make_state(hit=True, notify_on_hit=False)
        ok, m_mobile, m_speaker, m_rec = self._run_notify(
            state, [(self._target("x"), "always")], "UNKNOWN",
        )
        self.assertFalse(ok)
        self.assertEqual(m_mobile.call_count, 0)
        self.assertEqual(m_rec.call_count, 0)
        state.save.assert_not_called()

    def test_send_value_error_warns_and_returns_false(self):
        """_send_mobile_app 抛 ValueError（HA 未配置）→ 不抛，结果 False，仍写审计行。"""
        target = self._target("dad_phone")
        state = self._make_state(hit=True)

        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._load_target_links", return_value=[(target, "always")]), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=self._audio_outcome("UNKNOWN")), \
             patch("apps.vlm.notify._record_delivery") as m_rec, \
             patch("apps.vlm.notify._send_mobile_app",
                   side_effect=ValueError("缺少配置 HA_URL")):
            result = notify_hit(state)
            self.assertFalse(result)
            state.save.assert_not_called()
            rec = self._recorded(m_rec, target.name)
            self.assertIsNotNone(rec)
            self.assertFalse(rec["delivered"])
            self.assertIn("HA_URL", rec["error"])

    def test_send_ha_error_warns_and_returns_false(self):
        """_send_mobile_app 抛 HANoticeError → 不抛，结果 False。"""
        from apps.vlm.ha_notice import HANoticeError
        from apps.vlm.notify import notify_hit

        target = self._target("dad_phone")
        state = self._make_state(hit=True)
        with patch("apps.vlm.notify._load_target_links", return_value=[(target, "always")]), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=self._audio_outcome("UNKNOWN")), \
             patch("apps.vlm.notify._record_delivery"), \
             patch("apps.vlm.notify._send_mobile_app",
                   side_effect=HANoticeError("连接失败")):
            result = notify_hit(state)
            self.assertFalse(result)
            state.save.assert_not_called()

    def test_send_unexpected_exception_warns(self):
        """_send_mobile_app 抛 RuntimeError → 不抛，结果 False。"""
        from apps.vlm.notify import notify_hit

        target = self._target("dad_phone")
        state = self._make_state(hit=True)
        with patch("apps.vlm.notify._load_target_links", return_value=[(target, "always")]), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=self._audio_outcome("UNKNOWN")), \
             patch("apps.vlm.notify._record_delivery"), \
             patch("apps.vlm.notify._send_mobile_app",
                   side_effect=RuntimeError("boom")):
            result = notify_hit(state)
            self.assertFalse(result)
            state.save.assert_not_called()

    # ---- B5 多目标 ----
    def test_notify_hit_no_targets_skips(self):
        """through 行为空 → 不发任何 target，notified 不置 True。"""
        state = self._make_state(hit=True)
        ok, m_mobile, m_speaker, m_rec = self._run_notify(state, [], "UNKNOWN")
        self.assertFalse(ok)
        self.assertEqual(m_mobile.call_count, 0)
        self.assertEqual(m_speaker.call_count, 0)
        self.assertEqual(m_rec.call_count, 0)
        state.save.assert_not_called()

    def test_notify_hit_mobile_app_succeeds(self):
        """1 个 mobile_app target → _send_mobile_app 调 1 次，notified=True。"""
        target = self._target("dad_phone", target_id="notify.mobile_app_dong_wei_de_iphone")
        state = self._make_state(hit=True, status="在床")
        ok, m_mobile, _m_speaker, m_rec = self._run_notify(
            state, [(target, "always")], "UNKNOWN",
        )
        self.assertTrue(ok)
        self.assertEqual(m_mobile.call_count, 1)
        # 传给 _send_mobile_app 的参数：(notify_service, title, message)
        args, _ = m_mobile.call_args
        self.assertEqual(args[0], "notify.mobile_app_dong_wei_de_iphone")
        self.assertIn("【宝宝告警·口鼻遮蔽】", args[1])
        self.assertIn("客厅", args[2])
        self.assertEqual(state.notified, True)
        rec = self._recorded(m_rec, "dad_phone")
        self.assertTrue(rec["delivered"])
        self.assertEqual(rec["condition"], "always")

    def test_notify_hit_speaker_succeeds(self):
        """1 个 speaker target → _send_speaker 调 1 次，notified=True。"""
        target = self._target("speaker_xiaomi", kind="speaker",
                              target_id="text.xiaomi_lx06_c4d3")
        state = self._make_state(hit=True, status="在床")
        ok, _m_mobile, m_speaker, _m_rec = self._run_notify(
            state, [(target, "always")], "UNKNOWN",
        )
        self.assertTrue(ok)
        self.assertEqual(m_speaker.call_count, 1)
        args, _ = m_speaker.call_args
        self.assertEqual(args[0], "text.xiaomi_lx06_c4d3")
        self.assertIn("客厅", args[1])
        self.assertEqual(state.notified, True)

    def test_notify_hit_disabled_target_skipped(self):
        """enabled=False 的目标由 ORM 过滤掉（through 加载阶段）→ 无 target 不发。"""
        state = self._make_state(hit=True)
        ok, m_mobile, _m, _r = self._run_notify(state, [], "UNKNOWN")
        self.assertFalse(ok)
        self.assertEqual(m_mobile.call_count, 0)
        state.save.assert_not_called()

    def test_notify_hit_partial_failure_still_marks_notified(self):
        """1 个 target 成功 + 1 个失败 → notified=True（部分成功算成功）。"""
        from apps.vlm.ha_notice import HANoticeError
        from apps.vlm.notify import notify_hit

        target_ok = self._target("phone", kind="mobile_app", target_id="notify.mobile_app_x")
        target_fail = self._target("speaker", kind="speaker", target_id="text.xiaomi_x")
        state = self._make_state(hit=True)
        with patch("apps.vlm.notify._load_target_links",
                   return_value=[(target_ok, "always"), (target_fail, "always")]), \
             patch("apps.vlm.audio_rule.evaluate_audio_rule",
                   return_value=self._audio_outcome("UNKNOWN")), \
             patch("apps.vlm.notify._record_delivery"), \
             patch("apps.vlm.notify._send_mobile_app") as m_mobile, \
             patch("apps.vlm.notify._send_speaker",
                   side_effect=HANoticeError("boom")) as m_speaker:
            result = notify_hit(state)
            self.assertTrue(result)
            self.assertEqual(m_mobile.call_count, 1)
            self.assertEqual(m_speaker.call_count, 1)
            self.assertEqual(state.notified, True)
            state.save.assert_called_once_with(update_fields=["notified"])

    # ---- Phase 6：condition × 音频三态（6 组合，spec §7.2）----
    def test_always_target_sends_on_all_three_states(self):
        """always × SATISFIED / NOT_SATISFIED / UNKNOWN → 都发。"""
        for result in ("SATISFIED", "NOT_SATISFIED", "UNKNOWN"):
            with self.subTest(result=result):
                target = self._target("phone")
                state = self._make_state(hit=True)
                ok, m_mobile, _m, m_rec = self._run_notify(
                    state, [(target, "always")], result,
                )
                self.assertTrue(ok)
                self.assertEqual(m_mobile.call_count, 1)
                rec = self._recorded(m_rec, "phone")
                self.assertTrue(rec["delivered"])

    def test_audio_rule_target_sends_on_satisfied_and_unknown(self):
        """audio_rule × SATISFIED → 发；audio_rule × UNKNOWN → **也发**（不筛选）。"""
        for result in ("SATISFIED", "UNKNOWN"):
            with self.subTest(result=result):
                target = self._target("speaker", kind="speaker")
                state = self._make_state(hit=True)
                ok, _m, m_speaker, m_rec = self._run_notify(
                    state, [(target, "audio_rule")], result,
                )
                self.assertTrue(ok)
                self.assertEqual(m_speaker.call_count, 1)
                rec = self._recorded(m_rec, "speaker")
                self.assertTrue(rec["delivered"])
                self.assertEqual(
                    rec["audio_state"],
                    "satisfied" if result == "SATISFIED" else "unknown",
                )

    def test_audio_rule_target_skipped_on_not_satisfied(self):
        """audio_rule × NOT_SATISFIED → 不发，但写一条 delivered=False 审计行。"""
        target = self._target("speaker", kind="speaker")
        state = self._make_state(hit=True)
        ok, m_mobile, m_speaker, m_rec = self._run_notify(
            state, [(target, "audio_rule")], "NOT_SATISFIED",
        )
        self.assertFalse(ok)
        self.assertEqual(m_speaker.call_count, 0)
        self.assertEqual(m_mobile.call_count, 0)
        state.save.assert_not_called()
        rec = self._recorded(m_rec, "speaker")
        self.assertIsNotNone(rec)
        self.assertFalse(rec["delivered"])
        self.assertEqual(rec["audio_state"], "not_satisfied")
        self.assertEqual(rec["condition"], "audio_rule")

    def test_mixed_conditions_route_independently(self):
        """手机（always）+ 音箱（audio_rule）× NOT_SATISFIED → 手机发、音箱不发。"""
        phone = self._target("phone", kind="mobile_app")
        speaker = self._target("speaker", kind="speaker")
        state = self._make_state(hit=True)
        ok, m_mobile, m_speaker, m_rec = self._run_notify(
            state, [(phone, "always"), (speaker, "audio_rule")], "NOT_SATISFIED",
        )
        self.assertTrue(ok)
        self.assertEqual(m_mobile.call_count, 1)
        self.assertEqual(m_speaker.call_count, 0)
        self.assertTrue(self._recorded(m_rec, "phone")["delivered"])
        self.assertFalse(self._recorded(m_rec, "speaker")["delivered"])
        self.assertEqual(state.notified, True)

    def test_unknown_kind_counts_as_failure(self):
        """未知 target.kind → 记失败 + 审计行，不影响其它目标。"""
        weird = self._target("weird", kind="email")
        state = self._make_state(hit=True)
        ok, _m, _s, m_rec = self._run_notify(state, [(weird, "always")], "UNKNOWN")
        self.assertFalse(ok)
        rec = self._recorded(m_rec, "weird")
        self.assertFalse(rec["delivered"])
        self.assertIn("unknown target kind", rec["error"])

    # ---- compute_auto_silenced ----
    def test_silence_count_zero_returns_false(self):
        """silence_count=0 → 不静默（开关关闭）。"""
        from apps.vlm.notify import compute_auto_silenced

        result = compute_auto_silenced(
            cam_id=1, prompt_id=10, status="在床", silence_count=0,
        )
        self.assertFalse(result)

    def test_no_dismiss_returns_false(self):
        """无 dismiss 记录 → 不静默。"""
        from apps.vlm.notify import compute_auto_silenced

        no_dismiss_qs = _FakeStateQS(first_item=None)
        with patch("apps.vlm.models.VLMCheckState.objects") as m_objs:
            m_objs.filter.return_value = no_dismiss_qs
            result = compute_auto_silenced(
                cam_id=1, prompt_id=10, status="在床", silence_count=3,
            )
            self.assertFalse(result)

    def test_within_silence_window_returns_true(self):
        """dismiss 后连续 auto_silenced < N → 静默。"""
        from apps.vlm.notify import compute_auto_silenced

        dismiss_qs = _FakeStateQS(first_item=MagicMock(id=100))
        hits_qs = _FakeStateQS(items=[MagicMock(id=101, auto_silenced=True)])

        with patch("apps.vlm.models.VLMCheckState.objects") as m_objs:
            m_objs.filter.side_effect = [dismiss_qs, hits_qs]
            result = compute_auto_silenced(
                cam_id=1, prompt_id=10, status="在床", silence_count=3,
            )
            self.assertTrue(result)

    def test_status_changed_breaks_silence(self):
        """dismiss 后中间有非 auto_silenced 的 hit=True → 静默失效。"""
        from apps.vlm.notify import compute_auto_silenced

        # 中间有条 status 不同的 hit=True 且 auto_silenced=False
        # 但 compute_auto_silenced 只看 status 一致的 hit=True，所以 status 不一致的记录不影响本查询
        # 实际打破条件：中间有 hit=True 且 !auto_silenced
        dismiss_qs = _FakeStateQS(first_item=MagicMock(id=100))
        hits_qs = _FakeStateQS(items=[
            MagicMock(id=101, auto_silenced=True),
            MagicMock(id=102, auto_silenced=False),  # 又报了
        ])

        with patch("apps.vlm.models.VLMCheckState.objects") as m_objs:
            m_objs.filter.side_effect = [dismiss_qs, hits_qs]
            result = compute_auto_silenced(
                cam_id=1, prompt_id=10, status="在床", silence_count=3,
            )
            self.assertFalse(result)

    def test_consecutive_silenced_at_limit_breaks(self):
        """连续 auto_silenced=True 计数已达 N → 静默失效。"""
        from apps.vlm.notify import compute_auto_silenced

        dismiss_qs = _FakeStateQS(first_item=MagicMock(id=100))
        # N=3，已经有 3 条 auto_silenced=True → 达上限
        hits_qs = _FakeStateQS(items=[
            MagicMock(id=101, auto_silenced=True),
            MagicMock(id=102, auto_silenced=True),
            MagicMock(id=103, auto_silenced=True),
        ])

        with patch("apps.vlm.models.VLMCheckState.objects") as m_objs:
            m_objs.filter.side_effect = [dismiss_qs, hits_qs]
            result = compute_auto_silenced(
                cam_id=1, prompt_id=10, status="在床", silence_count=3,
            )
            self.assertFalse(result)


class PromptRunnerManagerTest(unittest.TestCase):
    """PromptRunnerManager 增删 Runner + 独立 loop 线程管理。"""

    def _make_mgr_with_loop(self):
        """构造一个带真 loop 线程（不连 DB）的 manager。

        不调 start()（会拉 LlamaClient + ORM）；手动构造 _loop / _executor / _llama_client。
        Loop 线程跑 run_forever；测试结束 stop loop。
        """
        mgr = PromptRunnerManager()
        mgr._executor = ThreadPoolExecutor(max_workers=1)  # noqa: SLF001
        mgr._llama_client = MagicMock()  # noqa: SLF001
        mgr._loop = asyncio.new_event_loop()  # noqa: SLF001

        def _loop_runner():
            asyncio.set_event_loop(mgr._loop)
            mgr._loop.run_forever()

        mgr._loop_thread = threading.Thread(  # noqa: SLF001
            target=_loop_runner, daemon=True,
        )
        mgr._loop_thread.start()
        # 等 loop 真起来
        for _ in range(50):
            if not mgr._loop.is_closed() and mgr._loop.is_running():
                break
            time.sleep(0.02)
        mgr._started = True  # noqa: SLF001
        return mgr

    def test_add_runner_creates_entry(self):
        mgr = self._make_mgr_with_loop()
        try:
            prompt = _FakePrompt(pid=10)
            mgr.add_runner(1, prompt)
            mgr.add_runner(2, prompt)
            self.assertEqual(len(mgr._runners), 2)  # noqa: SLF001
            self.assertIn((1, 10), mgr._runners)  # noqa: SLF001
            self.assertIn((2, 10), mgr._runners)  # noqa: SLF001
        finally:
            # 先停所有 Runner（避免 task 在 loop stop 后被 destroy 警告）
            runners = list(mgr._runners.values())  # noqa: SLF001
            if mgr._loop is not None and runners:  # noqa: SLF001
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        mgr._astop_all(runners), mgr._loop,  # noqa: SLF001
                    )
                    future.result(timeout=2.0)
                except Exception:
                    pass
            mgr._loop.call_soon_threadsafe(mgr._loop.stop)  # noqa: SLF001
            mgr._executor.shutdown(wait=False)  # noqa: SLF001

    def test_remove_camera_removes_runners(self):
        mgr = self._make_mgr_with_loop()
        try:
            prompt = _FakePrompt(pid=10)
            mgr.add_runner(1, prompt)
            mgr.add_runner(2, prompt)
            mgr.remove_camera(1)
            self.assertEqual(len(mgr._runners), 1)  # noqa: SLF001
            self.assertIn((2, 10), mgr._runners)  # noqa: SLF001
        finally:
            # 先停所有 Runner（避免 task 在 loop stop 后被 destroy 警告）
            runners = list(mgr._runners.values())  # noqa: SLF001
            if mgr._loop is not None and runners:  # noqa: SLF001
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        mgr._astop_all(runners), mgr._loop,  # noqa: SLF001
                    )
                    future.result(timeout=2.0)
                except Exception:
                    pass
            mgr._loop.call_soon_threadsafe(mgr._loop.stop)  # noqa: SLF001
            mgr._executor.shutdown(wait=False)  # noqa: SLF001

    def test_stats_returns_summary(self):
        mgr = self._make_mgr_with_loop()
        try:
            prompt = _FakePrompt(pid=10)
            mgr.add_runner(1, prompt)
            stats = mgr.stats()
            self.assertTrue(stats["started"])
            self.assertEqual(stats["runner_count"], 1)
            self.assertEqual(len(stats["runners"]), 1)
            self.assertEqual(stats["runners"][0]["cam_id"], 1)
            self.assertEqual(stats["runners"][0]["prompt_id"], 10)
        finally:
            # 先停所有 Runner（避免 task 在 loop stop 后被 destroy 警告）
            runners = list(mgr._runners.values())  # noqa: SLF001
            if mgr._loop is not None and runners:  # noqa: SLF001
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        mgr._astop_all(runners), mgr._loop,  # noqa: SLF001
                    )
                    future.result(timeout=2.0)
                except Exception:
                    pass
            mgr._loop.call_soon_threadsafe(mgr._loop.stop)  # noqa: SLF001
            mgr._executor.shutdown(wait=False)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# B6 单元测试：compute_recent_hit_dedup（DB 依赖；同 dashboard 清理模式）
# ===========================================================================
class NotifyDedupTest(unittest.TestCase):
    """B6 `compute_recent_hit_dedup` 单元测试（DB-backed；沿用 dashboard 模式）。

    关注：短时同 (cam, prompt, status) 命中是否跳过通知。
    依赖：真实 MySQL（apps.vlm.models.VLMCheckState），所以在 setUp 中清理
    Camera / VLMPromptConfig / VLMCheckState。
    """

    def setUp(self):
        from apps.streaming.models import Camera
        from apps.vlm.models import VLMCheckState, VLMPromptConfig

        self.cam = Camera.objects.create(name=_test_name("dedup_cam"), source_type=Camera.SOURCE_FILE, is_active=False)
        self.p1 = VLMPromptConfig.objects.create(
            name=_test_name("dedup_p1"), prompt="p", positive_keyword="是",
        )

    def _make_hit_state(self, status="是", notified=True):
        """构造一条 hit=True 历史状态（默认 notified=True；不传 ts 保留当前）。"""
        from apps.vlm.models import VLMCheckState
        return VLMCheckState.objects.create(
            camera=self.cam,
            prompt_config=self.p1,
            ts_list=[1, 2, 3],
            window_sec=10,
            hit=True,
            status=status,
            notified=notified,
            has_target_in_window=True,
        )

    def test_dedup_skips_within_window(self):
        """同 (cam, prompt, status) 在 30s 内连续 hit → dedup 命中。"""
        from apps.vlm.notify import compute_recent_hit_dedup

        self._make_hit_state(status="是", notified=True)
        self.assertTrue(compute_recent_hit_dedup(self.cam.id, self.p1.id, "是", 30))

    def test_dedup_window_expires(self):
        """第 1 条 hit 距今 > window → 不 dedup。"""
        from datetime import timedelta

        from django.utils import timezone

        from apps.vlm.models import VLMCheckState
        from apps.vlm.notify import compute_recent_hit_dedup

        s = self._make_hit_state(status="是", notified=True)
        VLMCheckState.objects.filter(pk=s.pk).update(
            created_at=timezone.now() - timedelta(seconds=120),
        )
        self.assertFalse(compute_recent_hit_dedup(self.cam.id, self.p1.id, "是", 30))

    def test_dedup_different_status_no_dedup(self):
        """不同 status → 不 dedup。"""
        from apps.vlm.notify import compute_recent_hit_dedup

        self._make_hit_state(status="是", notified=True)
        self.assertFalse(compute_recent_hit_dedup(self.cam.id, self.p1.id, "否", 30))

    def test_dedup_window_zero_disables(self):
        """window_sec=0 → 关闭去重，永远 False。"""
        from apps.vlm.notify import compute_recent_hit_dedup

        self._make_hit_state(status="是", notified=True)
        self.assertFalse(compute_recent_hit_dedup(self.cam.id, self.p1.id, "是", 0))

    def test_dedup_unnotified_does_not_count(self):
        """历史 hit=True 但 notified=False（被 dedup 跳过）→ 不算 dedup 命中。"""
        from apps.vlm.notify import compute_recent_hit_dedup

        self._make_hit_state(status="是", notified=False)
        self.assertFalse(compute_recent_hit_dedup(self.cam.id, self.p1.id, "是", 30))


# ===========================================================================
# Step 14：常驻模式相关测试（LlamaManager force_*, GpuManager resident,
#          Runner 队列化, DrainerThread）
# ===========================================================================
class ResidentModeTest(unittest.TestCase):
    """GpuManager / LlamaManager 常驻模式行为测试。"""

    def setUp(self):
        from apps.vlm.llama_manager import LlamaManager
        from apps.yolo_detect.gpu_manager import GpuManager
        # 重置单例
        LlamaManager._instance = None
        GpuManager._instance = None
        self.gpu = GpuManager.instance()
        self.llama = LlamaManager.instance()
        self.gpu.attach_llama_manager(self.llama)

    def test_gpu_resident_short_circuits_release(self):
        """set_resident(True) 后 release_yolo / release_vlm 不动 state 不调 unload。"""
        from apps.yolo_detect.gpu_manager import GpuState
        from apps.yolo_detect.detector import BabyDetector

        with patch.object(BabyDetector.instance(), "ensure_loaded") as m_load, \
             patch.object(BabyDetector.instance(), "unload") as m_unload, \
             patch.object(self.llama, "ensure_running"), \
             patch.object(self.llama, "unload") as m_llama_unload:
            self.gpu.set_resident(True)
            self.gpu.acquire_yolo()
            self.assertTrue(self.gpu._yolo_request_pending)
            m_load.assert_not_called()

            self.gpu.release_yolo_if_idle()
            self.assertFalse(self.gpu._yolo_request_pending)
            m_unload.assert_not_called()
            self.assertEqual(self.gpu._state, GpuState.IDLE)

            self.gpu.acquire_vlm()
            self.gpu.release_vlm_if_idle()
            m_llama_unload.assert_not_called()

    def test_llama_force_on_off_round_trip(self):
        """force_off 后 is_forced_off=True + ensure_running raise；force_on 清旗标。"""
        from apps.vlm.llama_manager import LlamaStartError

        self.assertFalse(self.llama.is_forced_off())

        with patch.object(self.llama, "_stop_blocking"):
            self.llama.force_off()
        self.assertTrue(self.llama.is_forced_off())

        with self.assertRaises(LlamaStartError):
            self.llama.ensure_running()

        with patch.object(self.llama, "ensure_running") as m_ensure:
            self.llama.force_on()
        self.assertFalse(self.llama.is_forced_off())
        m_ensure.assert_called_once()

    def test_auto_restart_scheduler_cycles(self):
        """scheduler wait(0) → force_off+force_on 各调一次。

        _auto_restart_loop 用 self._auto_restart_stop.wait(hours*3600)：
        第一次 wait 返回 False（继续）+ force_off + force_on；第二次 wait 返回 True（退出）。
        """
        self.llama._auto_restart_hours = 0
        stop = self.llama._auto_restart_stop
        stop.clear()
        # wait 序列：第一次 0s→False；第二次 0s→True（退出）
        with patch.object(self.llama, "force_off") as m_off, \
             patch.object(self.llama, "force_on") as m_on, \
             patch("time.sleep"), \
             patch.object(stop, "wait", side_effect=[False, True]):
            self.llama._auto_restart_loop()
        m_off.assert_called_once()
        m_on.assert_called_once()


class RunnerEnqueueTest(unittest.TestCase):
    """Runner._llama_is_off / _enqueue 行为测试。"""

    def setUp(self):
        from apps.vlm.llama_manager import LlamaManager
        # 熔断器是进程内单例：不清会让上个用例的失败计数泄漏进来
        from apps.core.llm_breaker import LLMBreaker
        LLMBreaker.reset_instances()
        LlamaManager._instance = None

    def test_llama_is_off_uses_forced_off_flag(self):
        """is_forced_off=True → _llama_is_off() True；False → False。"""
        from apps.vlm.llama_manager import LlamaManager
        from apps.vlm.runner import PromptRunner

        llama = LlamaManager.instance()
        with patch.object(PromptRunner, "__init__", lambda self: None):
            runner = PromptRunner()
        runner.cam_id = 1
        runner.prompt_id = 1

        with patch.object(llama, "is_forced_off", return_value=True):
            self.assertTrue(runner._llama_is_off())
        with patch.object(llama, "is_forced_off", return_value=False):
            self.assertFalse(runner._llama_is_off())

    def test_llama_is_off_when_breaker_blocks(self):
        """熔断打开 → 也要入队（否则每次都要白等一个超时才失败）。"""
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState
        from apps.vlm.runner import PromptRunner

        with patch.object(PromptRunner, "__init__", lambda self: None):
            runner = PromptRunner()
        breaker = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)
        for _ in range(5):
            breaker.record_failure("conn refused")
        self.assertTrue(breaker.is_blocking())
        self.assertTrue(runner._llama_is_off())

    def test_llama_is_off_when_external_remote_down(self):
        """外部模式下远端 health 不通 → 也要入队。

        这一条**不能靠熔断器代劳**：熔断要连撞 3 次才打开，而分体下"推理机没开"
        是启动就存在的状态 —— 不提前拦，前两个窗口会被白丢。
        """
        from apps.vlm.llama_manager import LlamaManager
        from apps.vlm.runner import PromptRunner

        llama = LlamaManager.instance()
        with patch.object(PromptRunner, "__init__", lambda self: None):
            runner = PromptRunner()
        with patch.object(llama, "is_forced_off", return_value=False), \
             patch.object(llama, "is_external", return_value=True), \
             patch.object(llama, "is_running", return_value=False):
            self.assertTrue(runner._llama_is_off())
        # 远端健康 + 熔断关闭 → 照常发请求
        with patch.object(llama, "is_forced_off", return_value=False), \
             patch.object(llama, "is_external", return_value=True), \
             patch.object(llama, "is_running", return_value=True):
            self.assertFalse(runner._llama_is_off())


class DrainerThreadTest(unittest.TestCase):
    """DrainerThread._drain_one / _process_task / _mark_retry_or_fail 测试。"""

    def setUp(self):
        from apps.vlm.drainer import DrainerThread
        from apps.vlm.llama_manager import LlamaManager
        from apps.yolo_detect.gpu_manager import GpuManager
        # 熔断器是进程内单例：不清会让上个用例的失败计数泄漏进来（is_blocking → 全跳过）
        from apps.core.llm_breaker import LLMBreaker
        LLMBreaker.reset_instances()
        DrainerThread._instance = None
        LlamaManager._instance = None
        GpuManager._instance = None
        self.drainer = DrainerThread.instance()
        self.llama = LlamaManager.instance()
        self.gpu = GpuManager.instance()
        self.gpu.attach_llama_manager(self.llama)

    def test_drainer_skips_when_llama_forced_off(self):
        """force_off → _drain_one 返回 IDLE（不 query DB；主循环按整间隔慢轮询）。"""
        from apps.vlm.drainer import DRAIN_IDLE

        with patch.object(self.llama, "is_forced_off", return_value=True), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs:
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)
            m_qs.filter.assert_not_called()

    def test_drainer_skips_when_llama_not_running(self):
        """is_forced_off=False 但 is_running=False → _drain_one 返回 IDLE。"""
        from apps.vlm.drainer import DRAIN_IDLE

        with patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=False), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs:
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)
            m_qs.filter.assert_not_called()

    def test_drainer_skips_when_breaker_open(self):
        """熔断打开 → _drain_one 返回 IDLE（连队列都不查）。

        没有这道闸，回放会把每条任务的 `MAX_RETRIES` 在毫秒内烧光
        —— llama 明摆着不可用，却一条条硬撞，最后把整个队列判死。
        """
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState
        from apps.vlm.drainer import DRAIN_IDLE

        breaker = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)
        for _ in range(5):                       # 远超阈值 → 打开
            breaker.record_failure("conn refused")
        self.assertTrue(breaker.is_blocking())

        with patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=True), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs:
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)
            m_qs.filter.assert_not_called()

    def test_drainer_skips_when_gpu_busy(self):
        """vlm_request_pending=True → 返回 BUSY（主循环只用短睡重探，不白等一个整间隔）。"""
        from apps.vlm.drainer import DRAIN_BUSY

        self.gpu._vlm_request_pending = True
        try:
            with patch.object(self.llama, "is_forced_off", return_value=False), \
                 patch.object(self.llama, "is_running", return_value=True), \
                 patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs:
                self.assertEqual(self.drainer._drain_one(), DRAIN_BUSY)
                m_qs.filter.assert_not_called()
        finally:
            self.gpu._vlm_request_pending = False

    def test_drainer_returns_idle_when_queue_empty(self):
        """无 pending → IDLE（llama 正常、GPU 不忙，只是没活干）。"""
        from apps.vlm.drainer import DRAIN_IDLE

        with patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=True), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs:
            m_qs.select_related.return_value.filter.return_value \
                .order_by.return_value.first.return_value = None
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)

    def test_drainer_returns_retry_when_frame_missing(self):
        """取到了但帧读不到 → RETRY（那条仍 pending，主循环要等一个整间隔再试）。"""
        from apps.vlm.drainer import DRAIN_RETRY

        task = MagicMock()
        task.id = 1
        task.retry_count = 0
        task.status = "pending"

        with patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=True), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs, \
             patch("apps.vlm.frame_storage.load_frame_bytes",
                   side_effect=OSError("no such frame")):
            m_qs.select_related.return_value.filter.return_value \
                .order_by.return_value.first.return_value = task
            self.assertEqual(self.drainer._drain_one(), DRAIN_RETRY)

        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.status, "pending")

    def test_drainer_processes_pending_task(self):
        """1 条 pending task → drainer 调 llama + 写 VLMCheckState + task.status='done'。"""
        from apps.vlm.llama_client import LlamaClient
        from apps.vlm.models import VLMCheckState

        # 建 mock task
        task = MagicMock()
        task.id = 1
        task.camera_id = 7
        task.prompt_config_id = 11
        task.ts_list = [1000, 1001, 1002]
        task.window_sec = 10
        task.img1 = "/tmp/img1.jpg"
        task.img2 = "/tmp/img2.jpg"
        task.img3 = "/tmp/img3.jpg"
        task.img1_ts = 1000
        task.img2_ts = 1001
        task.img3_ts = 1002
        task.has_target_in_window = True
        prompt = MagicMock()
        prompt.prompt = "test"
        prompt.max_tokens = 64
        prompt.timeout_sec_normal = 15
        prompt.positive_keyword = "是"
        prompt.result_format = "plain"
        prompt.kind = "judge"
        prompt.silence_count_after_dismiss = 0
        prompt.notify_on_hit = False
        task.prompt_config = prompt

        llama_client = MagicMock()
        llama_client.chat.return_value = "是"
        llama_client.parse.return_value = (True, "是")

        with patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=True), \
             patch("apps.vlm.models.VLMQueuedTask.objects") as m_qs, \
             patch("apps.vlm.frame_storage.load_frame_bytes", return_value=b"\xff\xd8\xff\xe0"), \
             patch.object(LlamaClient, "instance", return_value=llama_client), \
             patch("apps.vlm.models.VLMCheckState.objects.create") as m_create, \
             patch("apps.vlm.notify.compute_auto_silenced", return_value=False):
            m_qs.select_related.return_value.filter.return_value.order_by.return_value.first.return_value = task
            result = self.drainer._drain_one()

        llama_client.chat.assert_called_once()
        m_create.assert_called_once()
        self.assertEqual(task.status, "done")
        task.save.assert_called()
        # 成功 → DONE：主循环据此立刻抢下一条（不 sleep）
        from apps.vlm.drainer import DRAIN_DONE

        self.assertEqual(result, DRAIN_DONE)

    def test_drainer_marks_failed_after_max_retries(self):
        """load_frame 连续失败 → retry_count++; 超过 max → status='failed'。"""
        task = MagicMock()
        task.retry_count = 0
        task.status = "pending"

        with patch("django.conf.settings") as m_settings:
            m_settings.BABYCARE_LLM_QUEUE_MAX_RETRIES = 3
            self.drainer._mark_retry_or_fail(task, "boom")
            self.assertEqual(task.retry_count, 1)
            self.assertNotEqual(task.status, "failed")
            self.drainer._mark_retry_or_fail(task, "boom")
            self.assertEqual(task.retry_count, 2)
            self.drainer._mark_retry_or_fail(task, "boom")
            self.assertEqual(task.retry_count, 3)
            self.assertEqual(task.status, "failed")


    def test_expired_reaping_runs_even_when_service_down(self):
        """超期收割**必须先于**"服务不可用"闸门 —— 那正是它最该跑的场景。

        回归：原先 `_reap_expired()` 排在三个 `return`（手动关 / 远端不通 /
        熔断打开）**之后**，于是"推理机长期不在"时队列只增不减、`MEDIA_ROOT`
        一直涨，而收割一次也不跑 —— 与 checklist §8.5 / §8.8 的意图正好相反。
        """
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState
        from apps.vlm.drainer import DRAIN_IDLE

        # 1) 远端不通 → 仍是 IDLE，但收割必须已经发生
        with patch.object(self.drainer, "_reap_expired") as m_reap, \
             patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=False):
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)
            m_reap.assert_called_once()

        # 2) 熔断打开同理
        breaker = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)
        for _ in range(5):
            breaker.record_failure("down")
        self.assertTrue(breaker.is_blocking())
        with patch.object(self.drainer, "_reap_expired") as m_reap, \
             patch.object(self.llama, "is_forced_off", return_value=False), \
             patch.object(self.llama, "is_running", return_value=True):
            self.assertEqual(self.drainer._drain_one(), DRAIN_IDLE)
            m_reap.assert_called_once()


class DrainerLoopPacingTest(unittest.TestCase):
    """DrainerThread._loop 的等待策略（成功不空转 / 让位短探 / burst 让位）。

    用假 stop 事件替代 ``threading.Event``：``wait()`` 只记录时长并计数，跑满预期
    次数后 ``is_set()`` 置真让循环退出——测试不会真的 sleep。
    """

    def setUp(self):
        from apps.vlm.drainer import DrainerThread
        from apps.vlm.llama_manager import LlamaManager
        from apps.yolo_detect.gpu_manager import GpuManager
        DrainerThread._instance = None
        LlamaManager._instance = None
        GpuManager._instance = None
        self.drainer = DrainerThread.instance()

    def _run_loop(self, results, interval=7.0, max_waits=1):
        """跑一遍 _loop，返回每次 ``wait`` 的时长列表。"""
        waits = []
        seq = iter(results)

        class _Stop:
            def __init__(self, limit):
                self.limit = limit
                self.n = 0

            def is_set(self):
                return self.n >= self.limit

            def wait(self, timeout=None):
                self.n += 1
                waits.append(timeout)
                return False

        self.drainer._stop = _Stop(max_waits)
        with patch.object(self.drainer, "_drain_one", side_effect=lambda: next(seq)), \
             patch("apps.vlm.drainer.settings") as m_settings:
            m_settings.BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC = interval
            self.drainer._loop()
        return waits

    def test_loop_chains_after_success_and_paces_by_result(self):
        """DONE 不等待（连着做）；BUSY 短睡重探；RETRY / IDLE 等一个完整间隔。"""
        from apps.vlm.drainer import (
            DRAIN_BUSY, DRAIN_DONE, DRAIN_IDLE, DRAIN_RETRY, _BUSY_RETRY_SEC,
        )

        waits = self._run_loop(
            [DRAIN_DONE, DRAIN_DONE, DRAIN_BUSY, DRAIN_RETRY, DRAIN_IDLE],
            interval=7.0,
            max_waits=3,
        )
        self.assertEqual(waits, [_BUSY_RETRY_SEC, 7.0, 7.0])

    def test_loop_yields_once_every_burst(self):
        """连抢 _MAX_BURST 条只让一次短睡（burst 内一次 wait 都没有）。"""
        from apps.vlm.drainer import DRAIN_DONE, _BUSY_RETRY_SEC, _MAX_BURST

        waits = self._run_loop(
            [DRAIN_DONE] * _MAX_BURST, interval=7.0, max_waits=1,
        )
        self.assertEqual(waits, [_BUSY_RETRY_SEC])


class FrameStorageLoadTest(unittest.TestCase):
    """frame_storage.load_frame_bytes 行为测试。"""

    def test_load_relative_path(self):
        from apps.vlm.frame_storage import load_frame_bytes
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.jpg"
            p.write_bytes(b"\xff\xd8\xff")
            data = load_frame_bytes(str(p), tmp)  # 绝对路径传入也走 is_absolute 分支
            self.assertEqual(data, b"\xff\xd8\xff")   # 解不开 → 退回原始 bytes

    def test_load_missing_raises(self):
        from apps.vlm.frame_storage import load_frame_bytes
        with self.assertRaises(FileNotFoundError):
            load_frame_bytes("/nonexistent/x.jpg", "/tmp")

    def test_load_empty_raises(self):
        from apps.vlm.frame_storage import load_frame_bytes
        with self.assertRaises(FileNotFoundError):
            load_frame_bytes("", "/tmp")

    def test_load_resizes_full_res_frame_to_1024(self):
        """落盘是全分辨率 → 回放取字节必须缩到长边 1024（否则视觉 token 差 5 倍）。"""
        import cv2
        import numpy as np
        import tempfile
        from apps.vlm.frame_storage import load_frame_bytes, _VLM_MAX_LONG_SIDE

        rng = np.random.default_rng(0)
        big = rng.integers(0, 255, size=(1200, 2000, 3), dtype="uint8")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "big.jpg"
            cv2.imwrite(str(p), big, [cv2.IMWRITE_JPEG_QUALITY, 70])
            raw_size = p.stat().st_size
            data = load_frame_bytes(str(p), tmp)
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertEqual(max(img.shape[:2]), _VLM_MAX_LONG_SIDE)
        self.assertEqual(img.shape[:2], (614, 1024))     # 2000x1200 等比缩到宽 1024
        self.assertLess(len(data), raw_size)             # 顺带把体积也压下去

    def test_load_keeps_small_frame_size(self):
        """没超限的图不缩（只缩不放）。"""
        import cv2
        import numpy as np
        import tempfile
        from apps.vlm.frame_storage import load_frame_bytes

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "small.jpg"
            cv2.imwrite(str(p), np.zeros((480, 640, 3), dtype="uint8"))
            data = load_frame_bytes(str(p), tmp)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (480, 640))

    def test_runner_and_replay_share_same_encoder(self):
        """实时（Runner）与回放（Drainer）必须同一份缩图+编码实现——历史上各写一份，
        回放漏了 resize，代价差 5 倍。这里把"同一份"钉住。"""
        import cv2
        import numpy as np
        import tempfile
        from apps.vlm.frame_storage import load_frame_bytes, encode_vlm_jpeg
        from apps.vlm.runner import PromptRunner

        rng = np.random.default_rng(1)
        arr = rng.integers(0, 255, size=(1200, 2000, 3), dtype="uint8")
        frame = MagicMock()
        frame.ndarray = arr
        frame.ts = 1789391000

        # 实时路径 == 共享编码器
        self.assertEqual(PromptRunner._frame_to_bytes(frame), encode_vlm_jpeg(arr))

        # 回放路径：落盘 q70 → 解码 → 共享编码器，长边同样被限到 1024
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.jpg"
            cv2.imwrite(str(p), arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = load_frame_bytes(str(p), tmp)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(max(img.shape[:2]), 1024)


class LlamaClientSingletonTest(unittest.TestCase):
    """LlamaClient.instance() 单例测试。"""

    def test_singleton(self):
        from apps.vlm.llama_client import LlamaClient
        a = LlamaClient.instance()
        b = LlamaClient.instance()
        self.assertIs(a, b)


class LlamaClientChatLockTest(unittest.TestCase):
    """LlamaClient.chat() 串行化测试（mtmd 不支持并发请求 → 必须全局锁）。

    验证：N 个线程同时调 chat()，任意时刻只有 1 个在执行 HTTP，不重叠。
    """

    def test_concurrent_chat_serialized(self):
        import threading
        import time
        from unittest.mock import patch
        from apps.vlm.llama_client import LlamaClient

        client = LlamaClient.instance()

        in_flight = 0
        max_in_flight = 0
        counter_lock = threading.Lock()

        def fake_post_once(payload, timeout_sec):
            nonlocal in_flight, max_in_flight
            with counter_lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            try:
                # 故意 sleep，让其它线程有时间去抢锁（抢不到才算串行生效）
                time.sleep(0.05)
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            finally:
                with counter_lock:
                    in_flight -= 1

        images = [b"\xff\xd8\xff\xe0fake_jpeg"] * 3

        with patch.object(client, "_post_once", side_effect=fake_post_once):
            threads = [
                threading.Thread(
                    target=lambda: client.chat(images, "p", max_tokens=8, timeout_sec=5),
                )
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10.0)
                self.assertFalse(t.is_alive(), "chat 线程卡死")

        # 关键断言：任意时刻最多 1 个 chat 在跑
        self.assertEqual(
            max_in_flight, 1,
            f"并发 chat 重叠：max_in_flight={max_in_flight}（mtmd 会被打爆）",
        )


# ===========================================================================
# v2 multi-class / multi-model tests
# ===========================================================================
def _feed_fq_multi(
    fq: FrameQueue,
    per_frame: List[dict],
    start_ts: int = 0,
) -> None:
    """v2: push 一组帧，每帧一个 dict ({has_baby=, has_person=, has_cat=})."""
    for i, attrs in enumerate(per_frame):
        fq.push(start_ts + i, ndarray=None, **attrs)


class FrameQueueMarkDetectedKwargTest(unittest.TestCase):
    """v2: mark_detected(ts, **kwargs) writes multiple has_<name> attrs atomically."""

    def test_kwargs_set_multiple_attrs(self):
        fq = FrameQueue(cam_id=1)
        fq.push(1000, ndarray=None, has_baby=None, has_person=None, has_cat=None)
        fq.mark_detected(1000, has_baby=True, has_person=False, has_cat=True)
        item = fq.snapshot_sorted()[0]
        self.assertTrue(item.has_baby)
        self.assertFalse(item.has_person)
        self.assertTrue(item.has_cat)

    def test_kwargs_ignores_unknown_keys(self):
        fq = FrameQueue(cam_id=1)
        fq.push(1000, ndarray=None)
        # has_dog is not defined on FrameItem; should be silently dropped
        fq.mark_detected(1000, has_baby=True, has_dog=True)
        item = fq.snapshot_sorted()[0]
        self.assertTrue(item.has_baby)

    def test_v1_single_kwarg_still_works(self):
        """v1 call form mark_detected(ts, has_baby=True) still works."""
        fq = FrameQueue(cam_id=1)
        fq.push(2000, ndarray=None)
        fq.mark_detected(2000, has_baby=True)
        item = fq.snapshot_sorted()[0]
        self.assertTrue(item.has_baby)
        self.assertIsNone(item.has_person)
        self.assertIsNone(item.has_cat)

    def test_yolo_registry_dict_keys_no_prefix(self):
        """v2 fix: YoloRegistry.detect 返回 Dict[class_name, bool]（无 has_ 前缀），
        mark_detected 必须自动加 has_ 前缀写入 FrameItem 字段。

        回归 bug：mark_detected 之前只 hasattr(item, k)（'baby'），永远 False，
        → FrameItem 字段永远 None → YoloLoop peek_oldest 无限 detect 同帧。
        """
        fq = FrameQueue(cam_id=1)
        fq.push(3000, ndarray=None)
        # 模拟 YoloRegistry.detect 返回的格式（不带 has_ 前缀）
        fq.mark_detected(3000, baby=True, person=True, cat=False)
        item = fq.snapshot_sorted()[0]
        self.assertTrue(item.has_baby)
        self.assertTrue(item.has_person)
        self.assertFalse(item.has_cat)


class FrameSelectorTargetClassesTest(unittest.TestCase):
    """v2: select_three_frames picks frames by target_classes (default ['baby'])."""

    def test_default_target_classes_is_baby(self):
        frames = [
            FrameItem(ts=0, ndarray=None, has_baby=True),
            FrameItem(ts=1, ndarray=None, has_baby=False),
            FrameItem(ts=2, ndarray=None, has_baby=True),
            FrameItem(ts=3, ndarray=None, has_baby=False),
            FrameItem(ts=4, ndarray=None, has_baby=True),
        ]
        picked = select_three_frames(frames, [0, 1, 2, 3, 4])
        self.assertIsNotNone(picked)
        self.assertEqual([f.ts for f in picked], [0, 2, 4])

    def test_target_person_only(self):
        frames = [
            FrameItem(ts=0, ndarray=None, has_baby=False, has_person=True),
            FrameItem(ts=1, ndarray=None, has_baby=False, has_person=False),
            FrameItem(ts=2, ndarray=None, has_baby=False, has_person=True),
            FrameItem(ts=3, ndarray=None, has_baby=False, has_person=True),
            FrameItem(ts=4, ndarray=None, has_baby=False, has_person=False),
        ]
        picked = select_three_frames(frames, [0, 1, 2, 3, 4], ["person"])
        self.assertEqual([f.ts for f in picked], [0, 2, 3])

    def test_target_multi_or_semantics(self):
        frames = [
            FrameItem(ts=0, ndarray=None, has_baby=True, has_person=False),
            FrameItem(ts=1, ndarray=None, has_baby=False, has_person=False),
            FrameItem(ts=2, ndarray=None, has_baby=False, has_person=True),
            FrameItem(ts=3, ndarray=None, has_baby=True, has_person=True),
            FrameItem(ts=4, ndarray=None, has_baby=False, has_person=False),
        ]
        picked = select_three_frames(frames, [0, 1, 2, 3, 4], ["baby", "person"])
        # True count 4 (ts 0,2,3) -> True>=3 3-segment
        self.assertEqual([f.ts for f in picked], [0, 2, 3])

    def test_target_all_none_returns_none(self):
        frames = [
            FrameItem(ts=0, ndarray=None, has_baby=None, has_person=None, has_cat=None),
            FrameItem(ts=1, ndarray=None, has_baby=None, has_person=None, has_cat=None),
            FrameItem(ts=2, ndarray=None, has_baby=None, has_person=None, has_cat=None),
        ]
        self.assertIsNone(select_three_frames(frames, [0, 1, 2], ["baby", "person"]))


class PromptCursorTargetClassesTest(unittest.TestCase):
    """v2: PromptCursor.try_advance gates on target_classes attrs."""

    def test_target_person_only(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq_multi(fq, [
            {"has_baby": None, "has_person": True},
            {"has_baby": None, "has_person": True},
            {"has_baby": None, "has_person": False},
            {"has_baby": None, "has_person": True},
            {"has_baby": None, "has_person": True},
        ])
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        spec = cur.try_advance(fq, ["person"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.ts_list, [0, 1, 2, 3, 4])

    def test_target_multi_all_attrs_required(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq_multi(fq, [
            {"has_baby": True, "has_person": True},
            {"has_baby": True, "has_person": None},
            {"has_baby": True, "has_person": True},
            {"has_baby": True, "has_person": True},
            {"has_baby": True, "has_person": True},
        ])
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq, ["baby", "person"]))

    def test_target_multi_all_ready_finds_window(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq_multi(fq, [
            {"has_baby": True, "has_person": True},
            {"has_baby": False, "has_person": True},
            {"has_baby": True, "has_person": False},
            {"has_baby": True, "has_person": True},
            {"has_baby": False, "has_person": True},
        ])
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        spec = cur.try_advance(fq, ["baby", "person"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.ts_list, [0, 1, 2, 3, 4])

    def test_target_cat_when_cam_lacks_cat_model(self):
        fq = FrameQueue(cam_id=1)
        _feed_fq_multi(fq, [{"has_baby": True, "has_cat": None}] * 5)
        cur = PromptCursor(cam_id=1, prompt_id=10, window_sec=5)
        self.assertIsNone(cur.try_advance(fq, ["cat"]))


class YoloRegistryCacheTest(unittest.TestCase):
    """v2: YoloRegistry caches loaded model; same path only loads once."""

    def test_cache_reuses_same_model(self):
        from apps.yolo_detect.yolo_registry import YoloRegistry

        reg = YoloRegistry.instance()
        reg._models_cache.clear()

        load_calls = []
        mock_model = MagicMock()
        mock_model.predict.return_value = []

        def fake_load(path):
            resolved = reg._resolve_path(path)
            with reg._cache_lock:
                cached = reg._models_cache.get(resolved)
            if cached is not None:
                return cached
            load_calls.append(path)
            with reg._cache_lock:
                reg._models_cache[resolved] = mock_model
            return mock_model

        with patch.object(reg, "_load", side_effect=fake_load):
            from apps.yolo_detect.models import YOLOModel
            m1 = YOLOModel(name="t1", file_path="model/best.pt",
                           class_names=["baby"], coco_class_ids=[], enabled=True)
            m2 = YOLOModel(name="t2", file_path="model/best.pt",
                           class_names=["baby"], coco_class_ids=[], enabled=True)
            reg.detect(None, [m1])
            reg.detect(None, [m2])

        self.assertEqual(load_calls, ["model/best.pt"])

    def test_different_paths_loads_twice(self):
        from apps.yolo_detect.yolo_registry import YoloRegistry

        reg = YoloRegistry.instance()
        reg._models_cache.clear()

        load_calls = []
        mock_model = MagicMock()

        def fake_load(path):
            resolved = reg._resolve_path(path)
            with reg._cache_lock:
                cached = reg._models_cache.get(resolved)
            if cached is not None:
                return cached
            load_calls.append(path)
            with reg._cache_lock:
                reg._models_cache[resolved] = mock_model
            return mock_model

        with patch.object(reg, "_load", side_effect=fake_load):
            from apps.yolo_detect.models import YOLOModel
            m1 = YOLOModel(name="t1", file_path="model/best.pt",
                           class_names=["baby"], coco_class_ids=[], enabled=True)
            m2 = YOLOModel(name="t2", file_path="model/yolov8s.pt",
                           class_names=["person"], coco_class_ids=[0], enabled=True)
            reg.detect(None, [m1])
            reg.detect(None, [m2])

        self.assertEqual(load_calls, ["model/best.pt", "model/yolov8s.pt"])


class ParseTargetClassesTest(unittest.TestCase):
    """v2: runner._parse_target_classes helper."""

    def test_default_baby_when_empty(self):
        from apps.vlm.runner import _parse_target_classes
        self.assertEqual(_parse_target_classes(""), ["baby"])
        self.assertEqual(_parse_target_classes(None), ["baby"])

    def test_valid_single(self):
        from apps.vlm.runner import _parse_target_classes
        self.assertEqual(_parse_target_classes("person"), ["person"])

    def test_valid_multi_sorted_dedup(self):
        from apps.vlm.runner import _parse_target_classes
        self.assertEqual(_parse_target_classes("cat,baby,person"), ["baby", "cat", "person"])
        self.assertEqual(_parse_target_classes("baby,baby,cat"), ["baby", "cat"])

    def test_invalid_filtered_out(self):
        from apps.vlm.runner import _parse_target_classes
        self.assertEqual(_parse_target_classes("baby,dog,cat"), ["baby", "cat"])
        self.assertEqual(_parse_target_classes("dog,fish"), ["baby"])


class YoloLoopCrashResilienceTest(unittest.TestCase):
    """v2: YoloLoop must not die on get_cam_models or detect exception."""

    def test_get_cam_models_exception_does_not_kill_loop(self):
        from apps.yolo_detect.yolo_loop import _YoloLoop
        from apps.yolo_detect.frame_queue import FrameQueueManager

        fq_mgr = FrameQueueManager.instance()
        fq_mgr._queues.clear()
        fq = fq_mgr.get_or_create(999)
        fq.push(1000, ndarray=None)

        yl = _YoloLoop()
        yl._cam_order = [999]
        yl._cam_models = {999: []}
        yl._get_cam_models = MagicMock(side_effect=Exception("DB boom"))
        gpu = MagicMock()
        fq_mgr2 = MagicMock()
        fq_mgr2.get_or_create.return_value = fq

        yl._loop_iteration(fq_mgr2, MagicMock(), gpu)

        gpu.release_yolo_if_idle.assert_called()
        items = fq.snapshot_sorted()
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].has_baby)

    def test_detect_exception_does_not_kill_loop(self):
        from apps.yolo_detect.yolo_loop import _YoloLoop
        from apps.yolo_detect.frame_queue import FrameQueueManager

        fq_mgr = FrameQueueManager.instance()
        fq_mgr._queues.clear()
        fq = fq_mgr.get_or_create(998)
        fq.push(2000, ndarray=None)

        yl = _YoloLoop()
        yl._cam_order = [998]
        yl._cam_models = {998: [MagicMock()]}
        yl._get_cam_models = MagicMock(return_value=[MagicMock(name="m")])
        registry = MagicMock()
        registry.detect.side_effect = Exception("ultralytics boom")
        gpu = MagicMock()
        fq_mgr2 = MagicMock()
        fq_mgr2.get_or_create.return_value = fq

        yl._loop_iteration(fq_mgr2, registry, gpu)

        gpu.release_yolo_if_idle.assert_called()
        items = fq.snapshot_sorted()
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].has_baby)


class RenderNotifyTextTest(unittest.TestCase):
    """通知文案模板渲染（2026-09-14）：status 为空时连它外面那对括号一起去掉。"""

    def test_default_body_without_status_drops_parens(self):
        from apps.vlm.notify import DEFAULT_BODY_TEMPLATE, render_notify_text

        text = render_notify_text(
            DEFAULT_BODY_TEMPLATE,
            prompt_name="测试", cam_name="客厅摄像头",
            time_text="08:30:00", status="",
        )
        self.assertEqual(text, "客厅摄像头 在 08:30:00 检出")

    def test_default_body_with_status_keeps_parens(self):
        from apps.vlm.notify import DEFAULT_BODY_TEMPLATE, render_notify_text

        text = render_notify_text(
            DEFAULT_BODY_TEMPLATE,
            prompt_name="测试", cam_name="客厅摄像头",
            time_text="08:30:00", status="在床",
        )
        self.assertEqual(text, "客厅摄像头 在 08:30:00 检出（在床）")

    def test_custom_template_all_placeholders(self):
        from apps.vlm.notify import render_notify_text

        text = render_notify_text(
            "[{prompt}] {cam}@{time} -> {status}",
            prompt_name="测试", cam_name="客厅", time_text="08:30:00", status="是",
        )
        self.assertEqual(text, "[测试] 客厅@08:30:00 -> 是")

    def test_bare_status_placeholder_becomes_empty(self):
        from apps.vlm.notify import render_notify_text

        text = render_notify_text("{cam} {status} 检出", cam_name="客厅", status="")
        self.assertEqual(text, "客厅 检出")

    def test_default_title(self):
        from apps.vlm.notify import DEFAULT_TITLE_TEMPLATE, render_notify_text

        self.assertEqual(
            render_notify_text(DEFAULT_TITLE_TEMPLATE, prompt_name="测试"),
            "【宝宝告警·测试】",
        )

    def test_blank_or_mock_template_falls_back_to_default(self):
        """模板取值：非字符串（测试替身 Mock）/ 空白 → 回落默认。"""
        from apps.vlm.notify import DEFAULT_BODY_TEMPLATE, _pick_template

        for value in ("", "   ", None, MagicMock()):
            self.assertEqual(
                _pick_template(value, DEFAULT_BODY_TEMPLATE), DEFAULT_BODY_TEMPLATE,
            )
        self.assertEqual(_pick_template("{cam}", DEFAULT_BODY_TEMPLATE), "{cam}")


class FrameQueueDetectedFlagTest(unittest.TestCase):
    """``peek_oldest`` 用 ``detected`` 判定"未推理"（2026-09-14 修 v1 遗留）。

    回归场景：cam 只配了不含 baby 的模型 → ``has_baby`` 永远 None → 旧实现
    （"has_baby is None"）一直返回队头同一帧，YoloLoop 反复 detect 它，每秒实际
    只推进 1 帧；同时依赖 baby 的 prompt 永远凑不出窗口（一条 state 都不写）。
    """

    def _fq(self):
        from apps.yolo_detect.frame_queue import FrameQueue

        return FrameQueue(cam_id=1)

    def test_detected_flag_without_baby(self):
        """只写了 person/cat 也算"已推理"（不再看 has_baby）。"""
        fq = self._fq()
        fq.push(100, ndarray=None)
        fq.mark_detected(100, person=True, cat=False)

        item = fq.snapshot_sorted()[0]
        self.assertIsNone(item.has_baby)
        self.assertTrue(item.detected)
        self.assertIsNone(fq.peek_oldest())

    def test_peek_oldest_returns_first_undetected(self):
        fq = self._fq()
        fq.push(100, ndarray=None)
        fq.push(101, ndarray=None)
        fq.mark_detected(100, person=False)

        self.assertEqual(fq.peek_oldest().ts, 101)

    def test_stats_pending_uses_detected(self):
        fq = self._fq()
        fq.push(100, ndarray=None)
        fq.push(101, ndarray=None)
        fq.mark_detected(101, person=True)

        self.assertEqual(fq.stats()["pending"], 1)


class FrameQueueStoreSizeTest(unittest.TestCase):
    """入队即限长边（2026-09-15）：队列只存下游需要的最宽尺寸。

    动因：摄像头原始帧 2304×1296 = **8.5MB/帧**，90 帧 × 2 路 = 1.53GB；而 VLM 送
    模型前只要长边 1024、YOLO 内部反正 letterbox 到 640 —— 存全尺寸纯浪费。
    """

    def _fq(self):
        from apps.yolo_detect.frame_queue import FrameQueue

        return FrameQueue(cam_id=1)

    def test_long_side_capped_on_push(self):
        import numpy as np

        from apps.yolo_detect.frame_queue import MAX_STORED_LONG_SIDE

        fq = self._fq()
        fq.push(1000, np.zeros((1296, 2304, 3), dtype="uint8"))  # 摄像头原始尺寸
        arr = fq.snapshot_sorted()[0].ndarray
        self.assertLessEqual(max(arr.shape[:2]), MAX_STORED_LONG_SIDE)
        # 等比：宽高比不变（2304:1296 = 16:9）
        self.assertAlmostEqual(arr.shape[1] / arr.shape[0], 2304 / 1296, places=2)

    def test_exact_scale_keeps_aspect(self):
        """2 的幂尺寸 → 精确缩放（2048×1152 → 1024×576）。"""
        import numpy as np

        fq = self._fq()
        fq.push(1000, np.zeros((1152, 2048, 3), dtype="uint8"))
        self.assertEqual(fq.snapshot_sorted()[0].ndarray.shape[:2], (576, 1024))

    def test_small_frame_not_copied(self):
        """未超限 → 原对象透传（不复制、不放大）。"""
        import numpy as np

        fq = self._fq()
        small = np.zeros((480, 640, 3), dtype="uint8")
        fq.push(1000, small)
        self.assertIs(fq.snapshot_sorted()[0].ndarray, small)

    def test_none_frame_is_skipped(self):
        """``ndarray=None``（测试/占位帧）不能崩。"""
        fq = self._fq()
        fq.push(1000, None)
        self.assertIsNone(fq.snapshot_sorted()[0].ndarray)

    def test_vlm_encode_becomes_noop_after_store(self):
        """缩好之后，送 VLM 的那条路径不再重采样（返回同一对象）。"""
        import numpy as np

        from apps.vlm.frame_storage import resize_long_side

        fq = self._fq()
        fq.push(1000, np.zeros((1296, 2304, 3), dtype="uint8"))
        arr = fq.snapshot_sorted()[0].ndarray
        self.assertIs(resize_long_side(arr), arr)


class CamModelsTtlTest(unittest.TestCase):
    """``_cam_models`` 缓存 TTL 自愈（signal 丢失 / 跨进程改绑定的兜底）。"""

    def setUp(self):
        import time

        from apps.yolo_detect.yolo_loop import _YoloLoop

        self.time = time
        self.yl = _YoloLoop()

    def test_fresh_cache_does_not_hit_db(self):
        from unittest.mock import MagicMock, patch

        cached = [MagicMock(name="cached")]
        self.yl._cam_models = {999: cached}
        self.yl._cam_models_at = {999: self.time.monotonic()}

        with patch("apps.yolo_detect.yolo_loop._get_yolo_models_for_cam") as m_read:
            self.assertIs(self.yl._get_cam_models(999), cached)
            m_read.assert_not_called()

    def test_expired_cache_rereads_db(self):
        from unittest.mock import MagicMock, patch

        from apps.yolo_detect.yolo_loop import _CAM_MODELS_TTL_SEC

        self.yl._cam_models = {999: [MagicMock(name="stale")]}
        self.yl._cam_models_at = {
            999: self.time.monotonic() - _CAM_MODELS_TTL_SEC - 1,
        }
        fresh = [MagicMock(name="fresh")]

        with patch("apps.yolo_detect.yolo_loop._get_yolo_models_for_cam",
                   return_value=fresh) as m_read:
            self.assertIs(self.yl._get_cam_models(999), fresh)
            m_read.assert_called_once_with(999)


class RunnerNoWindowLogTest(unittest.TestCase):
    """等不到窗口时按 ``_NO_WINDOW_LOG_SEC`` 限流打 INFO（暴露"某类别一直 None"）。"""

    def test_limited_to_one_per_interval(self):
        from unittest.mock import patch as _patch

        from apps.vlm import runner as mod

        r = mod.PromptRunner.__new__(mod.PromptRunner)
        r.cam_id, r.prompt_id, r.window_sec = 1, 3225, 10
        r.target_classes = ["baby", "person"]
        r._last_no_window_log = 0.0

        with _patch.object(mod, "logger") as m_log:
            r._log_no_window_once()
            r._log_no_window_once()      # 立刻第二次 → 被限流
            m_log.info.assert_called_once()


# ===========================================================================
# v3 fix: add_camera / add_prompt / _bootstrap_existing 必须过滤 prompt.camera_ids
#
# Bug: 原版不过滤 M2M 限定，导致限定 cam 的 prompt 被错误地起在所有 cam 上。
# 典型症状：用户在 UI 把 prompt「人员状态描述」限定到「客厅摄像头 (onvif) #3055」，
# 但「测试视频2 #3045」也触发了 → state.camera=#3045 → 列表显示「测试视频2」。
# ===========================================================================
class CameraIdsFilterTest(unittest.TestCase):
    """RunnerManager.add_camera / add_prompt / _bootstrap 必须按 prompt.camera_ids 过滤。"""

    def setUp(self):
        """直接实例化 manager；绕过 start() 避免拉 llama / async loop。"""
        from apps.vlm.runner import PromptRunnerManager
        from apps.streaming.models import Camera  # noqa: F401
        PromptRunnerManager._instance = None
        self.mgr = PromptRunnerManager()
        self.mgr._started = True
        self.mgr._llama_client = MagicMock()
        self.mgr._executor = MagicMock()
        self.mgr._loop = MagicMock()

        # mock _add_runner_sync 计数
        self._added_pairs = []
        def _fake_add(cam_id, prompt):
            self._added_pairs.append((cam_id, prompt.id))
            return True
        self.mgr._add_runner_sync = _fake_add

    def _make_prompt(self, name, camera_ids=None):
        from apps.vlm.models import VLMPromptConfig
        p = VLMPromptConfig.objects.create(
            name=_test_name(name), prompt="p", positive_keyword="是", enabled=True,
        )
        if camera_ids is not None:
            for cid in camera_ids:
                try:
                    p.camera_ids.add(Camera.objects.get(pk=cid))
                except Camera.DoesNotExist:
                    pass
        return p

    def test_add_camera_prompt_unrestricted_runs_on_cam(self):
        """prompt 不限定 camera_ids → 任何 cam 都该起。"""
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("unrestricted")
        self.mgr.add_camera(cam1.id)
        self.assertIn((cam1.id, p.id), self._added_pairs)

    def test_add_camera_prompt_restricted_to_other_cam_skipped(self):
        """prompt 限定 camera_ids=[other_cam] → 当前 cam 不该起。"""
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("restricted_to_c2", camera_ids=[cam2.id])
        self.mgr.add_camera(cam1.id)
        self.assertNotIn((cam1.id, p.id), self._added_pairs)

    def test_add_camera_prompt_restricted_to_cam_runs(self):
        """prompt 限定 camera_ids=[this_cam] → 当前 cam 该起。"""
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("restricted_to_c1", camera_ids=[cam1.id])
        self.mgr.add_camera(cam1.id)
        self.assertIn((cam1.id, p.id), self._added_pairs)

    def test_add_prompt_unrestricted_uses_all_active_cams(self):
        """prompt 不限定 → 跑所有 active cam（mock Camera.objects.filter 只看本测试 cam）。"""
        from apps.streaming.models import Camera as _Cam
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("unrestricted")
        # 只看本测试 cam（生产 DB 有历史 cam，避免假阳性）
        with patch.object(_Cam.objects, "filter",
                          return_value=MagicMock()) as m_filter:
            m_filter.return_value = [cam1, cam2]
            self.mgr.add_prompt(p.id)
        cams = {c for c, _ in self._added_pairs}
        self.assertEqual(cams, {cam1.id, cam2.id})

    def test_add_prompt_restricted_only_uses_bound_cams(self):
        """prompt 限定 cam1 → 只跑 cam1（mock Camera filter 用 id__in 路径）。"""
        from apps.streaming.models import Camera as _Cam
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("to_c1", camera_ids=[cam1.id])
        # 限定 prompt：add_prompt 走 id__in=bound_ids filter → mock 返回 [cam1]
        with patch.object(_Cam.objects, "filter") as m_filter:
            # 第一次调用（is_active=True, id__in=）→ [cam1]；其它调用 fallback 真 DB
            real_filter = _Cam.objects.filter
            def selective(qs=None, **kwargs):
                if "id__in" in kwargs:
                    return [cam1]
                return real_filter(**kwargs)
            m_filter.side_effect = selective
            self.mgr.add_prompt(p.id)
        cams = {c for c, _ in self._added_pairs}
        self.assertEqual(cams, {cam1.id})

    def test_bootstrap_filters_by_camera_ids(self):
        """_bootstrap_existing 必须按 prompt.camera_ids 过滤（mock Camera filter）。"""
        from apps.streaming.models import Camera as _Cam
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p_unr = self._make_prompt("unr")
        p_to_c1 = self._make_prompt("to_c1", camera_ids=[cam1.id])
        # bootstrap 只看 cam1/cam2（避开历史 cam）
        real_filter = _Cam.objects.filter
        def selective(**kwargs):
            if kwargs.get("is_active") is True and "id__in" not in kwargs:
                return [cam1, cam2]
            return real_filter(**kwargs)
        with patch.object(_Cam.objects, "filter", side_effect=selective):
            self.mgr._bootstrap_existing()
        pairs = set(self._added_pairs)
        # unr → 两 cam 都有
        self.assertIn((cam1.id, p_unr.id), pairs)
        self.assertIn((cam2.id, p_unr.id), pairs)
        # to_c1 → 只有 cam1
        self.assertIn((cam1.id, p_to_c1.id), pairs)
        self.assertNotIn((cam2.id, p_to_c1.id), pairs)

    def test_change_prompt_camera_ids_clears_old_runners(self):
        """改 prompt.camera_ids（post_save 触发 add_prompt）→ 老 cam 上的 Runner 会被 remove_prompt 清掉。"""
        from apps.streaming.models import Camera as _Cam
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p = self._make_prompt("rebind", camera_ids=[cam1.id])
        # mock add_prompt 内 Camera filter（限定走 id__in 路径）→ 第一次返 [cam1]
        real_filter = _Cam.objects.filter
        def selective(**kwargs):
            if "id__in" in kwargs:
                if kwargs["id__in"] == {cam1.id}:
                    return [cam1]
                if cam2.id in kwargs["id__in"]:
                    return [cam2]
            return real_filter(**kwargs)
        with patch.object(_Cam.objects, "filter", side_effect=selective):
            self.mgr.add_prompt(p.id)
            self.assertIn((cam1.id, p.id), self._added_pairs)
            # 改 camera_ids=[cam2]
            p.camera_ids.clear()
            p.camera_ids.add(cam2)
            # remove_prompt 是 sync 走 run_coroutine_threadsafe，替成 mock
            with patch.object(self.mgr, "remove_prompt") as m_remove:
                self.mgr.add_prompt(p.id)
            m_remove.assert_called_once_with(p.id)
        final = {pair for pair in self._added_pairs if pair[1] == p.id}
        self.assertIn((cam2.id, p.id), final)
        # 关键修复：只调过 _add_runner_sync 两次（cam1 一次 + cam2 一次）
        # 修复前 bug：add_prompt 不 remove 老的，cam1 上的 Runner 还在跑
        adds_for_p = [pair for pair in self._added_pairs if pair[1] == p.id]
        self.assertEqual(adds_for_p, [(cam1.id, p.id), (cam2.id, p.id)])

    def test_prompt_targets_cam_helper(self):
        """_prompt_targets_cam 静态方法直接校验。"""
        from apps.vlm.runner import PromptRunnerManager
        cam1 = Camera.objects.create(name=_test_name("c1"), source_type=Camera.SOURCE_FILE, is_active=True)
        cam2 = Camera.objects.create(name=_test_name("c2"), source_type=Camera.SOURCE_FILE, is_active=True)
        p_unr = self._make_prompt("unr")
        p_to_c1 = self._make_prompt("to_c1", camera_ids=[cam1.id])
        # 不限定 → True
        self.assertTrue(PromptRunnerManager._prompt_targets_cam(p_unr, cam1.id))
        self.assertTrue(PromptRunnerManager._prompt_targets_cam(p_unr, cam2.id))
        # 限定 cam1 → cam1 True, cam2 False
        self.assertTrue(PromptRunnerManager._prompt_targets_cam(p_to_c1, cam1.id))
        self.assertFalse(PromptRunnerManager._prompt_targets_cam(p_to_c1, cam2.id))

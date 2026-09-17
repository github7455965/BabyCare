"""熔断器单测（`apps/core/llm_breaker.py`）。

跑法
----
    .venv\\Scripts\\python.exe manage.py test apps.core.tests_llm_breaker -v 2

设计
----
- 状态机测试用**假时钟**（`_Clock`）+ `persist=False` → 不碰 DB、不 sleep；
- 落库行为单独一组（`LLMBreakerPersistTest`），验证"只在状态变化时写"。

为什么这些用例重要：熔断器决定"服务不可用时要不要继续发请求"，
判错的代价是双向的 —— 该熔断不熔断 → 每次请求白等一个超时；
不该熔断乱熔断 → 把一条健康的服务停掉 60s。
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402

if not _django_apps.ready:
    django.setup()

from django.test import SimpleTestCase, TestCase  # noqa: E402

from apps.core.llm_breaker import (  # noqa: E402
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    LLMBreaker,
)
from apps.core.models import LLMHealthState  # noqa: E402

SERVICE = LLMHealthState.SERVICE_ASR


class _Clock:
    """可推进的单调时钟替身。"""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _breaker(clock: _Clock, **kw) -> LLMBreaker:
    kw.setdefault("fail_threshold", 3)
    kw.setdefault("cooldown_sec", 60.0)
    kw.setdefault("probe_timeout_sec", 5.0)
    return LLMBreaker(SERVICE, clock=clock, persist=False, **kw)


# ===========================================================================
# 状态机（无 DB / 无 sleep）
# ===========================================================================
class ClosedStateTest(SimpleTestCase):
    def test_allows_by_default(self):
        b = _breaker(_Clock())
        self.assertTrue(b.allow_attempt())
        self.assertFalse(b.is_blocking())
        self.assertEqual(b.snapshot()["state"], STATE_CLOSED)

    def test_below_threshold_stays_closed(self):
        """连续失败但没到阈值 → 仍然放行（不能一次网络抖动就熔断）。"""
        c = _Clock()
        b = _breaker(c)
        b.record_failure("boom")
        b.record_failure("boom")
        self.assertEqual(b.snapshot()["state"], STATE_CLOSED)
        self.assertTrue(b.allow_attempt())

    def test_threshold_opens(self):
        c = _Clock()
        b = _breaker(c)
        for _ in range(3):
            b.record_failure("connect refused")
        snap = b.snapshot()
        self.assertEqual(snap["state"], STATE_OPEN)
        self.assertEqual(snap["consecutive_failures"], 3)
        self.assertFalse(b.allow_attempt())
        self.assertTrue(b.is_blocking())

    def test_success_resets_counter(self):
        """失败-成功-失败-失败 → 计数从 0 重来，不熔断。"""
        c = _Clock()
        b = _breaker(c)
        b.record_failure("x")
        b.record_failure("x")
        b.record_success()
        self.assertEqual(b.snapshot()["consecutive_failures"], 0)
        b.record_failure("x")
        b.record_failure("x")
        self.assertEqual(b.snapshot()["state"], STATE_CLOSED)


class OpenToHalfOpenTest(SimpleTestCase):
    def _opened(self, c: _Clock) -> LLMBreaker:
        b = _breaker(c)
        for _ in range(3):
            b.record_failure("down")
        assert b.snapshot()["state"] == STATE_OPEN
        return b

    def test_blocked_until_cooldown(self):
        c = _Clock()
        b = self._opened(c)
        c.advance(59.0)
        self.assertFalse(b.allow_attempt())
        self.assertTrue(b.is_blocking())

    def test_cooldown_elapsed_lets_exactly_one_probe(self):
        """冷却到了只放**一条** —— 否则等于把熔断关掉了。"""
        c = _Clock()
        b = self._opened(c)
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        self.assertEqual(b.snapshot()["state"], STATE_HALF_OPEN)
        # 探测还在飞 → 后续一律拦
        self.assertFalse(b.allow_attempt())
        self.assertFalse(b.allow_attempt())
        self.assertTrue(b.is_blocking())

    def test_probe_success_closes(self):
        c = _Clock()
        b = self._opened(c)
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        b.record_success()
        snap = b.snapshot()
        self.assertEqual(snap["state"], STATE_CLOSED)
        self.assertEqual(snap["consecutive_failures"], 0)
        self.assertTrue(b.allow_attempt())
        self.assertFalse(b.is_blocking())

    def test_probe_failure_reopens_and_restarts_cooldown(self):
        """探测失败 → 回 open，**重新计时**（不能沿用旧的时间戳一直放探测）。"""
        c = _Clock()
        b = self._opened(c)
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        b.record_failure("still down")
        self.assertEqual(b.snapshot()["state"], STATE_OPEN)
        # 刚失败过 → 这 60s 内不能再放
        self.assertFalse(b.allow_attempt())
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())

    def test_repeated_probe_failures_keep_cycling(self):
        """连续 3 轮探测失败：每轮都是"等 60s → 放一条 → 失败"，不卡死也不放水。"""
        c = _Clock()
        b = self._opened(c)
        for _ in range(3):
            c.advance(60.0)
            self.assertTrue(b.allow_attempt(), "每轮冷却后应放一条探测")
            self.assertFalse(b.allow_attempt(), "同轮不应放第二条")
            b.record_failure("still down")
            self.assertEqual(b.snapshot()["state"], STATE_OPEN)


class StuckProbeTest(SimpleTestCase):
    def test_stuck_probe_is_recovered(self):
        """拿到 allow_attempt()=True 后调用方崩了 → 超时后允许再放一条，不永久卡死。"""
        c = _Clock()
        b = _breaker(c, probe_timeout_sec=5.0)
        for _ in range(3):
            b.record_failure("down")
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        self.assertFalse(b.allow_attempt())
        c.advance(60.0)  # 远超 max(30s, 5*3)
        self.assertTrue(b.allow_attempt())


class TimeoutTest(SimpleTestCase):
    def test_closed_keeps_normal_timeout(self):
        b = _breaker(_Clock())
        self.assertEqual(b.effective_timeout_sec(30), 30.0)

    def test_open_keeps_normal_timeout(self):
        """open 态根本不会发请求，超时值无所谓 —— 但绝不能返回 0 之类。"""
        b = _breaker(_Clock())
        for _ in range(3):
            b.record_failure("x")
        self.assertEqual(b.effective_timeout_sec(30), 30.0)

    def test_half_open_switches_to_probe_timeout(self):
        """半开必须用短超时：否则探测自己就把链路堵死（ASR 30s×3 / VLM 60s）。"""
        c = _Clock()
        b = _breaker(c, probe_timeout_sec=5.0)
        for _ in range(3):
            b.record_failure("x")
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        self.assertEqual(b.effective_timeout_sec(30), 5.0)
        b.record_success()
        self.assertEqual(b.effective_timeout_sec(30), 30.0)

    def test_open_with_cooldown_elapsed_is_treated_as_probe(self):
        """**回归**：`open` 且冷却已过 → 也算探测窗口，必须给短超时。

        这条路径正是 `is_blocking()` 放行的那一格。`is_blocking()` **刻意无副作用**
        （控制页 / "要不要跳过"门槛也用它），所以它不会把状态迁移成 `half_open`
        —— `effective_timeout_sec()` 必须自己认出"冷却已过"。漏了的话，
        冷却后那条探测会拿到**正常超时**（ASR 最坏 90s / VLM 60s），
        短超时保护形同虚设。
        """
        c = _Clock()
        b = _breaker(c, probe_timeout_sec=5.0)
        for _ in range(3):
            b.record_failure("x")

        self.assertEqual(b.snapshot()["state"], STATE_OPEN, "状态仍停在 open")
        self.assertEqual(b.effective_timeout_sec(30), 30.0, "冷却没过 → 正常超时")
        self.assertFalse(b.is_probing())

        c.advance(60.0)
        self.assertFalse(b.is_blocking(), "冷却已到 → 不该报 blocking")
        self.assertEqual(
            b.snapshot()["state"], STATE_OPEN, "is_blocking() 必须无副作用",
        )
        self.assertTrue(b.is_probing(), "冷却已过 → 这一格就是探测")
        self.assertEqual(b.effective_timeout_sec(30), 5.0)


class DisabledTest(SimpleTestCase):
    def test_disabled_always_allows_and_never_counts(self):
        """关掉熔断 = 回到旧行为；失败了也不计数、不拦截。"""
        b = _breaker(_Clock(), enabled=False)
        for _ in range(10):
            b.record_failure("x")
        snap = b.snapshot()
        self.assertEqual(snap["state"], STATE_CLOSED)
        self.assertEqual(snap["consecutive_failures"], 0)
        self.assertTrue(b.allow_attempt())
        self.assertFalse(b.is_blocking())
        self.assertEqual(b.effective_timeout_sec(30), 30.0)


class IsBlockingHasNoSideEffectTest(SimpleTestCase):
    def test_is_blocking_does_not_consume_the_probe(self):
        """`is_blocking()` 是只读查询（给控制页/回放门槛用），不能把探测名额吃掉。"""
        c = _Clock()
        b = _breaker(c)
        for _ in range(3):
            b.record_failure("x")
        c.advance(60.0)
        self.assertFalse(b.is_blocking(), "冷却已到 → 不该报 blocking")
        self.assertEqual(b.snapshot()["state"], STATE_OPEN, "只读查询不该改状态")
        # 探测名额仍在
        self.assertTrue(b.allow_attempt())


# ===========================================================================
# 落库（可观测性）
# ===========================================================================
class LLMBreakerPersistTest(TestCase):
    def _breaker(self, **kw) -> LLMBreaker:
        kw.setdefault("persist", True)
        kw.setdefault("clock", _Clock())
        return LLMBreaker(SERVICE, **kw)

    def test_open_persists_row(self):
        b = self._breaker()
        for _ in range(3):
            b.record_failure("connect refused")
        row = LLMHealthState.objects.get(service=SERVICE)
        self.assertEqual(row.state, STATE_OPEN)
        self.assertEqual(row.consecutive_failures, 3)
        self.assertIn("connect refused", row.last_error)
        self.assertIsNotNone(row.last_failure_at)

    def test_below_threshold_does_not_write(self):
        """未达阈值不落库：稳态下不能因为偶发失败就反复写表。"""
        b = self._breaker()
        b.record_failure("x")
        self.assertFalse(LLMHealthState.objects.filter(service=SERVICE).exists())

    def test_steady_success_writes_once_then_stops(self):
        """稳态下 `record_success` **最多写一次**，之后零写。

        首次那一次是必须的（用来纠正进程重启后可能残留的陈旧 `open` 行）；
        但每个 VLM 窗都写一次表是不可接受的，所以第二次之后必须停。
        """
        from unittest.mock import patch

        b = self._breaker()
        # 数**真实 DB 写**（不能 patch `_persist_locked`：`_synced` 就是在它里面
        # 置位的，patch 掉之后每次成功都会以为"还没同步过"）
        with patch.object(
            LLMHealthState.objects, "update_or_create",
            wraps=LLMHealthState.objects.update_or_create,
        ) as m_write:
            b.record_success()
            self.assertEqual(m_write.call_count, 1, "首次必须同步（纠正陈旧行）")
            for _ in range(5):
                b.record_success()
            self.assertEqual(m_write.call_count, 1, "之后必须零写")

    def test_first_success_clears_stale_row(self):
        """**回归**：一次陈旧 `open` 行必须被新的成功纠正。

        场景：进程重启后内存是全新的 CLOSED，但 DB 里留着上一轮的 `open`
        —— 若 `record_success` 只在"有变化"时落库，稳态下永远不写，
        那条陈旧行会一直显示"熔断中"骗人。
        """
        LLMHealthState.objects.create(
            service=SERVICE, state=STATE_OPEN,
            consecutive_failures=5, last_error="stale from previous run",
        )
        LLMBreaker(SERVICE, clock=_Clock(), persist=True).record_success()

        row = LLMHealthState.objects.get(service=SERVICE)
        self.assertEqual(row.state, STATE_CLOSED)
        self.assertEqual(row.consecutive_failures, 0)
        self.assertEqual(row.last_error, "")

    def test_success_after_failures_writes_reset(self):
        b = self._breaker()
        b.record_failure("x")
        b.record_failure("x")
        b.record_success()
        row = LLMHealthState.objects.get(service=SERVICE)
        self.assertEqual(row.state, STATE_CLOSED)
        self.assertEqual(row.consecutive_failures, 0)
        self.assertIsNotNone(row.last_success_at)

    def test_half_open_probe_writes_state(self):
        c = _Clock()
        b = self._breaker(clock=c)
        for _ in range(3):
            b.record_failure("x")
        c.advance(60.0)
        self.assertTrue(b.allow_attempt())
        row = LLMHealthState.objects.get(service=SERVICE)
        self.assertEqual(row.state, STATE_HALF_OPEN)
        self.assertIsNotNone(row.last_probe_at)

    def test_two_services_are_independent(self):
        """VLM 熔断不影响 ASR（两条线各有各的状态）。"""
        vlm = LLMBreaker(LLMHealthState.SERVICE_VLM, clock=_Clock(), persist=True)
        asr = LLMBreaker(LLMHealthState.SERVICE_ASR, clock=_Clock(), persist=True)
        for _ in range(3):
            vlm.record_failure("vlm down")
        self.assertEqual(
            LLMHealthState.objects.get(service=LLMHealthState.SERVICE_VLM).state,
            STATE_OPEN,
        )
        self.assertTrue(asr.allow_attempt())
        self.assertFalse(
            LLMHealthState.objects.filter(
                service=LLMHealthState.SERVICE_ASR
            ).exists()
        )


class SnapshotShapeTest(SimpleTestCase):
    def test_snapshot_has_fields_the_control_page_needs(self):
        snap = _breaker(_Clock()).snapshot()
        for key in (
            "service", "state", "enabled", "consecutive_failures",
            "fail_threshold", "cooldown_sec", "probe_timeout_sec", "blocking",
            "last_failure_at", "last_probe_at", "last_success_at", "last_error",
        ):
            self.assertIn(key, snap)


class PerServiceThresholdTest(TestCase):
    """`for_service` 按线读阈值覆盖 —— 两条线的代价不对称（见 config/settings.py）。

    ASR 没有 live 流量（唯一消费者就是 DescribeService）→「攒着」零成本，1 次即熔断；
    VLM 有 live 窗口竞争 → 一次读超时（服务活着但慢）就熔断会把 live 窗口压进队列
    最多 60s，所以沿用通用默认 3。
    """

    def setUp(self):
        LLMBreaker.reset_instances()

    def tearDown(self):
        LLMBreaker.reset_instances()

    def test_asr_is_more_sensitive_than_vlm(self):
        from django.conf import settings

        asr = LLMBreaker.for_service(LLMHealthState.SERVICE_ASR)
        vlm = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)

        self.assertEqual(
            asr.snapshot()["fail_threshold"],
            int(getattr(settings, "BABYCARE_LLM_FAIL_THRESHOLD_ASR", 1)),
        )
        self.assertEqual(
            vlm.snapshot()["fail_threshold"],
            int(getattr(settings, "BABYCARE_LLM_FAIL_THRESHOLD", 3)),
        )
        self.assertLess(
            asr.snapshot()["fail_threshold"],
            vlm.snapshot()["fail_threshold"],
            "ASR 应比 VLM 更敏感：它攒着不花钱，VLM 会把 live 窗口压进队列",
        )

    def test_one_failure_opens_asr_breaker(self):
        """ASR 阈值 1 的直接后果：失败一次即拦截（用户 2026-09-16 定）。"""
        asr = LLMBreaker.for_service(LLMHealthState.SERVICE_ASR)
        self.assertEqual(asr.snapshot()["fail_threshold"], 1)
        asr.record_failure("connect refused")
        self.assertTrue(asr.is_blocking(), "ASR 一次失败就该拦下后续请求")

    def test_vlm_survives_a_single_failure(self):
        """VLM 一次失败**不**熔断 —— live 窗口不该被一次"慢"推进队列。"""
        vlm = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)
        vlm.record_failure("read timeout")
        self.assertFalse(vlm.is_blocking())

    def test_vlm_override_is_honoured(self):
        """`BABYCARE_LLM_FAIL_THRESHOLD_VLM` 能单独拧（不必动通用默认）。"""
        from django.test import override_settings

        with override_settings(BABYCARE_LLM_FAIL_THRESHOLD_VLM=2):
            LLMBreaker.reset_instances()
            vlm = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)
            self.assertEqual(vlm.snapshot()["fail_threshold"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

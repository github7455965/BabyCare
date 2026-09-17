"""Phase 6 单元测试：Prompt 音频三态判定 + through 通知路由 + 迁移回填。

覆盖（spec §11.1 / §7.1 / §7.2）：

- 8 种 `PromptAudioRule.condition` × 覆盖完整 / 不完整；
- `coverage_ok_from_ts` 与 `last_packet_at` 兜底；
- `UNKNOWN` 的全部前置场景（无规则 / 未启用 / 服务未起 / 租约超时 /
  摄像头状态异常 / 无 runtime 行 / 无摄像头）；
- 阳性结论不要求全覆盖（"听到了"就是硬证据）；
- `PromptNotificationDelivery` 审计行（三值 `audio_state`）；
- `VLMCheckStateAudioEvent` 快照；
- 三步迁移的 **raw SQL 回填**（旧表有数据 → `always`；旧表不存在 → no-op）。

用 `django.test.TestCase` 跑（独立 test 库 + 自动回滚）；迁移回填用例要真 DDL，
用 `TransactionTestCase`。
"""

from __future__ import annotations

import importlib
import time
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.apps import apps as django_apps
from django.db import connection
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from apps.audio_detect.models import (
    AudioEvent,
    AudioRuntimeState,
    AudioServiceState,
)
from apps.streaming.models import Camera
from apps.vlm.audio_rule import (
    NOT_SATISFIED,
    SATISFIED,
    UNKNOWN,
    AudioRuleOutcome,
    evaluate_audio_rule,
    record_state_audio_links,
)
from apps.vlm.models import (
    NotifyTarget,
    PromptAudioRule,
    PromptNotificationDelivery,
    PromptNotifyTarget,
    VLMPromptConfig,
    VLMCheckState,
    VLMCheckStateAudioEvent,
)


def _name(base: str) -> str:
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


class _Base(TestCase):
    """公共夹具：摄像头 + Prompt + 音频服务/Runtime 的默认"健康"状态。"""

    def setUp(self):
        self.now = int(time.time())
        self.cam = Camera.objects.create(name=_name("cam"), source_type="onvif")
        self.prompt = VLMPromptConfig.objects.create(
            name=_name("prompt"), prompt="图中有宝宝吗？", positive_keyword="是",
        )

    # ---- 夹具 helpers ----
    def _service(self, desired="on", pid=123, age_sec=0.0):
        AudioServiceState.objects.update_or_create(
            pk=AudioServiceState.SINGLETON_PK,
            defaults={
                "desired": desired,
                "pid": pid,
                "updated_at": timezone.now() - timedelta(seconds=age_sec),
            },
        )

    def _runtime(self, status="ready", coverage_from=None, packet_age=0.0):
        AudioRuntimeState.objects.update_or_create(
            camera=self.cam,
            defaults={
                "status": status,
                "coverage_ok_from_ts": (
                    self.now - 3600 if coverage_from is None else coverage_from
                ),
                "last_packet_at": timezone.now() - timedelta(seconds=packet_age),
            },
        )

    def _healthy(self):
        """服务在跑 + 摄像头 ready + 覆盖完整。"""
        self._service()
        self._runtime()

    def _rule(self, condition=PromptAudioRule.CONDITION_CRY, enabled=True,
              window_sec=60, min_count=1):
        return PromptAudioRule.objects.create(
            prompt_config=self.prompt, enabled=enabled, condition=condition,
            window_sec=window_sec, min_event_count=min_count,
        )

    def _event(self, labels, start_ts=None, end_ts=None, ended=True):
        start = self.now - 10 if start_ts is None else start_ts
        end = self.now - 5 if end_ts is None else end_ts
        return AudioEvent.objects.create(
            camera=self.cam,
            status=AudioEvent.STATUS_PENDING_DESCRIPTION,
            started_at_ts=start,
            ended_at_ts=end if ended else None,
            detected_labels={lbl: {"first_positive_ts": start} for lbl in labels},
        )


# ===========================================================================
# 1. 三态判定：8 种 condition × 覆盖完整 / 不完整
# ===========================================================================
class EvaluateConditionsTest(_Base):

    def test_conditions_fully_covered_no_event(self):
        """覆盖完整 + 无事件：阳性类条件 → NOT_SATISFIED；no_* → SATISFIED。"""
        self._healthy()
        expectations = {
            "any": NOT_SATISFIED,
            "cry": NOT_SATISFIED,
            "speech": NOT_SATISFIED,
            "cry_or_speech": NOT_SATISFIED,
            "no_cry": SATISFIED,
            "no_speech": SATISFIED,
            "count_cry": NOT_SATISFIED,
            "count_speech": NOT_SATISFIED,
        }
        for cond, want in expectations.items():
            with self.subTest(condition=cond):
                rule = self._rule(cond)
                out = evaluate_audio_rule(
                    self.cam.id, self.prompt.id, now_ts=self.now, rule=rule,
                )
                self.assertTrue(out.covered)
                self.assertEqual(out.result, want, out.reason)
                rule.delete()

    def test_conditions_fully_covered_with_cry_and_speech(self):
        """覆盖完整 + 有哭声和说话声：阳性类 → SATISFIED；no_* → NOT_SATISFIED。"""
        self._healthy()
        self._event(["cry"])
        self._event(["speech"])
        expectations = {
            "any": SATISFIED,
            "cry": SATISFIED,
            "speech": SATISFIED,
            "cry_or_speech": SATISFIED,
            "no_cry": NOT_SATISFIED,
            "no_speech": NOT_SATISFIED,
            "count_cry": SATISFIED,
            "count_speech": SATISFIED,
        }
        for cond, want in expectations.items():
            with self.subTest(condition=cond):
                rule = self._rule(cond)
                out = evaluate_audio_rule(
                    self.cam.id, self.prompt.id, now_ts=self.now, rule=rule,
                )
                self.assertEqual(out.result, want, out.reason)
                rule.delete()

    def test_conditions_not_covered_no_event_all_unknown(self):
        """覆盖不完整 + 无事件：全部 8 种 → UNKNOWN（阴性结论必须全覆盖）。"""
        self._healthy()
        self._runtime(coverage_from=self.now)  # coverage_ok_from_ts > window_start
        for cond in (
            "any", "cry", "speech", "cry_or_speech",
            "no_cry", "no_speech", "count_cry", "count_speech",
        ):
            with self.subTest(condition=cond):
                rule = self._rule(cond)
                out = evaluate_audio_rule(
                    self.cam.id, self.prompt.id, now_ts=self.now, rule=rule,
                )
                self.assertFalse(out.covered)
                self.assertEqual(out.result, UNKNOWN, out.reason)
                rule.delete()

    def test_positive_does_not_require_full_coverage(self):
        """覆盖不完整 + 有哭声：阳性类 → SATISFIED（"听到了"本身就是硬证据）。"""
        self._healthy()
        self._runtime(coverage_from=self.now)
        self._event(["cry"])
        expectations = {
            "any": SATISFIED,
            "cry": SATISFIED,
            "cry_or_speech": SATISFIED,
            "count_cry": SATISFIED,
            "speech": UNKNOWN,       # 没听到说话声，但覆盖不完整 → 不能断言"没有"
            "count_speech": UNKNOWN,
            "no_cry": NOT_SATISFIED,  # 确认有哭声 → 不满足"没有哭声"
            "no_speech": UNKNOWN,
        }
        for cond, want in expectations.items():
            with self.subTest(condition=cond):
                rule = self._rule(cond)
                out = evaluate_audio_rule(
                    self.cam.id, self.prompt.id, now_ts=self.now, rule=rule,
                )
                self.assertEqual(out.result, want, out.reason)
                rule.delete()


class EvaluateEventsTest(_Base):
    def test_count_cry_threshold(self):
        """count_cry：事件数 ≥ min_event_count 才满足。"""
        self._healthy()
        self._event(["cry"], start_ts=self.now - 30, end_ts=self.now - 25)
        self._event(["cry"], start_ts=self.now - 20, end_ts=self.now - 15)

        rule_no = self._rule("count_cry", min_count=3)
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule_no)
        self.assertEqual(out.cry_count, 2)
        self.assertEqual(out.result, NOT_SATISFIED)
        rule_no.delete()

        rule_ok = self._rule("count_cry", min_count=2)
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule_ok)
        self.assertEqual(out.result, SATISFIED)
        self.assertEqual(len(out.events), 2)

    def test_in_progress_event_counts(self):
        """进行中事件（ended_at_ts=None，仍在录音）也算命中。"""
        self._healthy()
        self._event(["cry"], start_ts=self.now - 3, ended=False)
        rule = self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        self.assertEqual(out.result, SATISFIED)

    def test_event_outside_window_ignored(self):
        """窗口外的事件不计入。"""
        self._healthy()
        self._event(["cry"], start_ts=self.now - 600, end_ts=self.now - 590)
        rule = self._rule("cry", window_sec=60)
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        self.assertEqual(out.event_count, 0)
        self.assertEqual(out.result, NOT_SATISFIED)

    def test_other_camera_event_ignored(self):
        """别的摄像头的事件不计入。"""
        self._healthy()
        other = Camera.objects.create(name=_name("cam2"), source_type="onvif")
        AudioEvent.objects.create(
            camera=other, status=AudioEvent.STATUS_PENDING_DESCRIPTION,
            started_at_ts=self.now - 5, ended_at_ts=self.now - 2,
            detected_labels={"cry": {}},
        )
        rule = self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        self.assertEqual(out.event_count, 0)
        self.assertEqual(out.result, NOT_SATISFIED)


# ===========================================================================
# 2. UNKNOWN 的全部前置场景（spec §7.1 覆盖表）
# ===========================================================================
class EvaluateUnknownTest(_Base):
    def test_unknown_when_no_camera(self):
        self._healthy()
        out = evaluate_audio_rule(None, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "no_camera")

    def test_unknown_when_no_rule(self):
        self._healthy()
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "no_rule")

    def test_unknown_when_rule_disabled(self):
        self._healthy()
        self._rule("cry", enabled=False)
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "rule_disabled")

    def test_unknown_when_service_desired_off(self):
        self._service(desired="off")
        self._runtime()
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "desired_off")

    def test_unknown_when_no_lease(self):
        self._service(pid=None)
        self._runtime()
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "no_lease")

    def test_unknown_when_lease_timeout(self):
        self._service(age_sec=60)  # > BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC(10)
        self._runtime()
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "lease_timeout")

    def test_unknown_when_no_service_state_row(self):
        AudioServiceState.objects.all().delete()
        self._runtime()
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "no_service_state")

    def test_unknown_when_no_runtime_row(self):
        self._service()
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "no_runtime_state")

    def test_unknown_when_camera_status_bad(self):
        for status in ("no_audio_profile", "capture_error", "model_error",
                       "initializing", "stopped", "disabled"):
            with self.subTest(status=status):
                self._service()
                self._runtime(status=status)
                rule = self._rule("cry")
                out = evaluate_audio_rule(
                    self.cam.id, self.prompt.id, now_ts=self.now, rule=rule,
                )
                self.assertEqual(out.result, UNKNOWN)
                self.assertEqual(out.reason, status)
                rule.delete()

    def test_unknown_when_packet_stalled(self):
        """进程活着但流死了（last_packet_at 兜底）→ 覆盖不完整 → 阴性降级 UNKNOWN。"""
        self._service()
        self._runtime(packet_age=30.0)  # > BABYCARE_AUDIO_STALL_SEC(5)
        self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertEqual(out.reason, "packet_stalled")

    def test_evaluate_never_raises_on_db_error(self):
        """任何内部异常都降级 UNKNOWN，不冒泡。"""
        with patch(
            "apps.audio_detect.models.AudioServiceState.objects.filter",
            side_effect=RuntimeError("db down"),
        ):
            self._rule("cry")
            out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now)
        self.assertEqual(out.result, UNKNOWN)
        self.assertIn(out.reason, ("service_state_error", "evaluate_error"))


# ===========================================================================
# 3. 结果与快照
# ===========================================================================
class OutcomeContractTest(_Base):
    def test_audio_state_mapping(self):
        self.assertEqual(AudioRuleOutcome(result=SATISFIED).audio_state, "satisfied")
        self.assertEqual(
            AudioRuleOutcome(result=NOT_SATISFIED).audio_state, "not_satisfied",
        )
        self.assertEqual(AudioRuleOutcome(result=UNKNOWN).audio_state, "unknown")

    def test_should_send_for_audio_rule(self):
        self.assertTrue(AudioRuleOutcome(result=SATISFIED).should_send_for_audio_rule)
        self.assertTrue(AudioRuleOutcome(result=UNKNOWN).should_send_for_audio_rule)
        self.assertFalse(
            AudioRuleOutcome(result=NOT_SATISFIED).should_send_for_audio_rule,
        )

    def test_snapshot_fields(self):
        self._healthy()
        self._event(["cry"])
        rule = self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        snap = out.snapshot()
        self.assertEqual(snap["result"], SATISFIED)
        self.assertEqual(snap["condition"], "cry")
        self.assertTrue(snap["covered"])
        self.assertTrue(snap["service_available"])
        self.assertEqual(snap["window_sec"], 60)
        self.assertEqual(snap["cry_count"], 1)
        self.assertEqual(len(snap["event_ids"]), 1)

    def test_record_state_audio_links_positive(self):
        self._healthy()
        ev = self._event(["cry"])
        rule = self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        state = VLMCheckState.objects.create(
            camera=self.cam, prompt_config=self.prompt, ts_list=[self.now],
            window_sec=1, hit=True, status="在床",
        )
        n = record_state_audio_links(state, out)
        self.assertEqual(n, 1)
        link = VLMCheckStateAudioEvent.objects.get(vlm_state=state)
        self.assertEqual(link.audio_event_id, ev.id)
        self.assertEqual(link.snapshot_json["result"], SATISFIED)

    def test_record_state_audio_links_negative_writes_snapshot_only(self):
        """阴性 / UNKNOWN 也留一行快照（audio_event=None），保证三态可审计。"""
        self._healthy()
        rule = self._rule("cry")
        out = evaluate_audio_rule(self.cam.id, self.prompt.id, now_ts=self.now, rule=rule)
        self.assertEqual(out.result, NOT_SATISFIED)
        state = VLMCheckState.objects.create(
            camera=self.cam, prompt_config=self.prompt, ts_list=[self.now],
            window_sec=1, hit=True, status="在床",
        )
        record_state_audio_links(state, out)
        link = VLMCheckStateAudioEvent.objects.get(vlm_state=state)
        self.assertIsNone(link.audio_event_id)
        self.assertEqual(link.snapshot_json["result"], NOT_SATISFIED)


# ===========================================================================
# 4. notify_hit：through 路由 + 投递审计（真 DB，HA 发送被 patch）
# ===========================================================================
class NotifyRoutingDbTest(_Base):
    def _state(self):
        return VLMCheckState.objects.create(
            camera=self.cam, prompt_config=self.prompt, ts_list=[self.now],
            window_sec=1, hit=True, status="在床",
        )

    def _link(self, name, kind, condition):
        target = NotifyTarget.objects.create(
            name=_name(name), kind=kind, target_id=f"notify.{_name(name)}",
        )
        PromptNotifyTarget.objects.create(
            prompt_config=self.prompt, notify_target=target, condition=condition,
        )
        return target

    def test_mixed_conditions_on_not_satisfied(self):
        """手机(always) 发、音箱(audio_rule) 不发；两条审计行 audio_state 都是 not_satisfied。"""
        phone = self._link("phone", "mobile_app", PromptNotifyTarget.CONDITION_ALWAYS)
        speaker = self._link("speaker", "speaker", PromptNotifyTarget.CONDITION_AUDIO_RULE)
        self._healthy()
        self._rule("cry")  # 覆盖完整 + 无事件 → NOT_SATISFIED
        state = self._state()

        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._send_mobile_app") as m_mobile, \
             patch("apps.vlm.notify._send_speaker") as m_speaker:
            ok = notify_hit(state)

        self.assertTrue(ok)
        self.assertEqual(m_mobile.call_count, 1)
        self.assertEqual(m_speaker.call_count, 0)

        rows = PromptNotificationDelivery.objects.filter(vlm_state=state)
        self.assertEqual(rows.count(), 2)
        by_target = {r.notify_target_id: r for r in rows}
        self.assertTrue(by_target[phone.id].delivered)
        self.assertFalse(by_target[speaker.id].delivered)
        self.assertEqual(by_target[speaker.id].audio_state, "not_satisfied")
        self.assertEqual(by_target[speaker.id].condition, "audio_rule")

    def test_audio_rule_sent_on_unknown_when_audio_off(self):
        """关闭音频线（UNKNOWN）→ audio_rule 目标**照发**（行为与引入音频前一致）。"""
        phone = self._link("phone", "mobile_app", PromptNotifyTarget.CONDITION_ALWAYS)
        speaker = self._link("speaker", "speaker", PromptNotifyTarget.CONDITION_AUDIO_RULE)
        # 不创建 AudioServiceState → 进程级不可用 → UNKNOWN
        state = self._state()

        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._send_mobile_app") as m_mobile, \
             patch("apps.vlm.notify._send_speaker") as m_speaker:
            ok = notify_hit(state)

        self.assertTrue(ok)
        self.assertEqual(m_mobile.call_count, 1)
        self.assertEqual(m_speaker.call_count, 1)
        rows = {r.notify_target_id: r for r in
                PromptNotificationDelivery.objects.filter(vlm_state=state)}
        self.assertTrue(rows[phone.id].delivered)
        self.assertTrue(rows[speaker.id].delivered)
        self.assertEqual(rows[speaker.id].audio_state, "unknown")

    def test_audio_rule_sent_on_satisfied_records_reference_event(self):
        """SATISFIED 时投递行带参考音频事件。"""
        speaker = self._link("speaker", "speaker", PromptNotifyTarget.CONDITION_AUDIO_RULE)
        self._healthy()
        ev = self._event(["cry"])
        self._rule("cry")
        state = self._state()

        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._send_speaker"):
            ok = notify_hit(state)

        self.assertTrue(ok)
        row = PromptNotificationDelivery.objects.get(
            vlm_state=state, notify_target_id=speaker.id,
        )
        self.assertTrue(row.delivered)
        self.assertEqual(row.audio_state, "satisfied")
        self.assertEqual(row.audio_event_id, ev.id)

    def test_disabled_target_excluded(self):
        target = self._link("speaker", "speaker", PromptNotifyTarget.CONDITION_ALWAYS)
        target.enabled = False
        target.save(update_fields=["enabled"])
        state = self._state()

        from apps.vlm.notify import notify_hit

        with patch("apps.vlm.notify._send_speaker") as m_speaker:
            ok = notify_hit(state)

        self.assertFalse(ok)
        self.assertEqual(m_speaker.call_count, 0)
        self.assertEqual(
            PromptNotificationDelivery.objects.filter(vlm_state=state).count(), 0,
        )


# ===========================================================================
# 5. 三步迁移的 raw SQL 回填（spec §11.1）
# ===========================================================================
class MigrationBackfillTest(TransactionTestCase):
    """旧隐式中间表 → 新 through 表（默认 always）；旧表不存在 → no-op。"""

    OLD_TABLE = "vlm_prompt_config_notify_targets"

    def setUp(self):
        self.mod = importlib.import_module(
            "apps.vlm.migrations.0013_backfill_prompt_notify_targets"
        )
        self.prompt = VLMPromptConfig.objects.create(
            name=_name("prompt"), prompt="x", positive_keyword="是",
        )
        self.target = NotifyTarget.objects.create(
            name=_name("phone"), kind="mobile_app", target_id="notify.x",
        )
        self.addCleanup(self._drop_old_table)

    def _drop_old_table(self):
        with connection.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS `{self.OLD_TABLE}`")

    def _create_old_table(self):
        with connection.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS `{self.OLD_TABLE}`")
            if connection.vendor == "mysql":
                id_and_pk = "`id` bigint NOT NULL AUTO_INCREMENT, PRIMARY KEY (`id`)"
            else:  # SQLite：自增主键必须内联声明
                id_and_pk = "`id` integer PRIMARY KEY AUTOINCREMENT"
            cur.execute(
                f"CREATE TABLE `{self.OLD_TABLE}` ("
                f"{id_and_pk},"
                "`vlmpromptconfig_id` bigint NOT NULL,"
                "`notifytarget_id` bigint NOT NULL)"
            )

    def _run_backfill(self):
        self.mod.backfill(django_apps, SimpleNamespace(connection=connection))

    def test_backfill_copies_rows_with_always(self):
        self._create_old_table()
        with connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO `{self.OLD_TABLE}` "
                "(vlmpromptconfig_id, notifytarget_id) VALUES (%s, %s)",
                [self.prompt.id, self.target.id],
            )

        self._run_backfill()

        link = PromptNotifyTarget.objects.get(
            prompt_config=self.prompt, notify_target=self.target,
        )
        self.assertEqual(link.condition, PromptNotifyTarget.CONDITION_ALWAYS)

    def test_backfill_noop_when_old_table_missing(self):
        self._drop_old_table()
        self._run_backfill()  # 不应抛
        self.assertEqual(PromptNotifyTarget.objects.count(), 0)

    def test_backfill_idempotent(self):
        """重复执行（表还在、行已搬）不炸、不产生重复行。"""
        self._create_old_table()
        with connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO `{self.OLD_TABLE}` "
                "(vlmpromptconfig_id, notifytarget_id) VALUES (%s, %s)",
                [self.prompt.id, self.target.id],
            )
        self._run_backfill()
        self._run_backfill()
        self.assertEqual(
            PromptNotifyTarget.objects.filter(
                prompt_config=self.prompt, notify_target=self.target,
            ).count(),
            1,
        )

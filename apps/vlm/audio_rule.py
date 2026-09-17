"""Prompt 声音规则的**三态判定**（spec §5.5 / §7.1）。

三态
----
``AudioRuleResult ∈ { SATISFIED, NOT_SATISFIED, UNKNOWN }``

核心原则（spec §7.1）
--------------------
- **阳性结论（听到事件）不要求全覆盖** —— "听到了"本身就是硬证据，
  音频进程中途重启也能成立；
- **阴性结论（"没有"）必须要求全覆盖** —— 覆盖不完整时不能断言"没听到"，
  一律降级为 ``UNKNOWN``。

``UNKNOWN`` 覆盖以下**全部**情况（一个都不落，spec §7.1）：

======================  ==================================================
场景                      原因
======================  ==================================================
音频 worker 未启动/手动关  进程级不可用（租约锁 inactive）
摄像头状态非 ready/degraded 摄像头级不可用（no_audio_profile / capture_error / ...）
Prompt 没配规则/未启用      配置级不可用
判定窗口内覆盖不完整        数据级不可用（coverage_ok_from_ts / last_packet_at 兜底）
======================  ==================================================

判定对象是 :class:`apps.audio_detect.models.AudioEvent`（一个连续事件一条），
因此"10 分钟内 3 次哭声"不会被一次持续哭声算成多次（spec §5.5）。

本模块**绝不抛异常**：任何内部错误都降级成 ``UNKNOWN``——音频是辅助维度，
它的故障不能让 Prompt 命中流程或通知流程挂掉。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---- 三态 ----------------------------------------------------------------
SATISFIED = "SATISFIED"
NOT_SATISFIED = "NOT_SATISFIED"
UNKNOWN = "UNKNOWN"

#: 三态 → `PromptNotificationDelivery.audio_state`（小写枚举，spec §7.2）
AUDIO_STATE_MAP = {
    SATISFIED: "satisfied",
    NOT_SATISFIED: "not_satisfied",
    UNKNOWN: "unknown",
}

#: `AudioRuntimeState.status` 中"音频数据可用"的取值（spec §10）
AVAILABLE_CAMERA_STATUSES = frozenset({"ready", "degraded"})

#: 业务标签（spec §4.2）
LABEL_CRY = "cry"
LABEL_SPEECH = "speech"

#: 一次判定最多关联多少个音频事件（防止长 `count_*` 窗口写爆关联表）
MAX_REFERENCED_EVENTS = 10


@dataclass
class AudioRuleOutcome:
    """一次声音规则判定的结果 + 可审计快照。"""

    result: str = UNKNOWN
    condition: str = ""
    rule_enabled: bool = False
    window_sec: int = 0
    window_start_ts: int = 0
    window_end_ts: int = 0
    min_event_count: int = 1
    covered: bool = False
    service_available: bool = False
    camera_status: str = ""
    coverage_ok_from_ts: int = 0
    event_count: int = 0
    cry_count: int = 0
    speech_count: int = 0
    reason: str = ""
    #: 参考到的音频事件（阳性时非空；阴性/UNKNOWN 时为空）
    events: list = field(default_factory=list)

    @property
    def audio_state(self) -> str:
        """`PromptNotificationDelivery.audio_state` 取值。"""
        return AUDIO_STATE_MAP.get(self.result, "unknown")

    @property
    def reference_event(self) -> Any:
        """用于投递审计/关联的单个参考事件（无则 None）。"""
        return self.events[0] if self.events else None

    @property
    def should_send_for_audio_rule(self) -> bool:
        """`condition="audio_rule"` 的目标是否应发送（spec §7.2）。

        `UNKNOWN` **放行**：音频关掉等于这条线不存在，所有目标回到"命中就发"。
        """
        return self.result != NOT_SATISFIED

    def snapshot(self) -> dict:
        """写进 `VLMCheckStateAudioEvent.snapshot_json` 的可审计上下文。"""
        return {
            "result": self.result,
            "condition": self.condition,
            "rule_enabled": self.rule_enabled,
            "window_sec": self.window_sec,
            "window_start_ts": self.window_start_ts,
            "window_end_ts": self.window_end_ts,
            "min_event_count": self.min_event_count,
            "covered": self.covered,
            "service_available": self.service_available,
            "camera_status": self.camera_status,
            "coverage_ok_from_ts": self.coverage_ok_from_ts,
            "event_count": self.event_count,
            "cry_count": self.cry_count,
            "speech_count": self.speech_count,
            "event_ids": [e.pk for e in self.events],
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 可用性判定
# ---------------------------------------------------------------------------
def _service_available() -> tuple[bool, str]:
    """音频 worker 进程级是否可用（spec §7.1 / §9.3）。

    判据 = `AudioServiceState` 单行表 **desired=on 且租约活跃**：
    `pid` 非空 **且** 心跳 `updated_at` 未超过
    `BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC`（两条都要，卡死的实例也算不可用）。
    """
    from django.conf import settings

    from apps.audio_detect.models import AudioServiceState

    try:
        row = AudioServiceState.objects.filter(pk=AudioServiceState.SINGLETON_PK).first()
    except Exception as e:  # noqa: BLE001 - DB 抖动不能让 Prompt 流程挂
        logger.warning("[audio-rule] read AudioServiceState failed: %s", e)
        return False, "service_state_error"

    if row is None:
        return False, "no_service_state"
    if row.desired != AudioServiceState.DESIRED_ON:
        return False, "desired_off"
    if not row.pid:
        return False, "no_lease"

    timeout = int(getattr(settings, "BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC", 10) or 10)
    try:
        from django.utils import timezone

        age = (timezone.now() - row.updated_at).total_seconds()
    except Exception:  # noqa: BLE001
        age = 0.0
    if age > timeout:
        return False, "lease_timeout"
    return True, ""


def _camera_available(camera_id: int) -> tuple[bool, str, Any]:
    """摄像头级音频可用性（spec §7.1 / §10）。返回 (可用, 原因, runtime_state)。"""
    from apps.audio_detect.models import AudioRuntimeState

    st = AudioRuntimeState.objects.filter(camera_id=camera_id).first()
    if st is None:
        return False, "no_runtime_state", None
    if st.status not in AVAILABLE_CAMERA_STATUSES:
        return False, st.status or "unknown_status", st
    return True, "", st


def _coverage_state(runtime, window_start_ts: int) -> tuple[bool, str]:
    """窗口覆盖是否完整（spec §3.3 / §7.1）。

    `coverage_ok_from_ts` 是**唯一依据**；`last_packet_at` 是兜底：
    进程还活着但流已经死了（假活）时也要判覆盖不完整。
    """
    from django.conf import settings
    from django.utils import timezone

    coverage_from = int(getattr(runtime, "coverage_ok_from_ts", 0) or 0)
    if coverage_from > window_start_ts:
        return False, "coverage_starts_later"

    stall = float(getattr(settings, "BABYCARE_AUDIO_STALL_SEC", 5.0) or 5.0)
    last_packet_at = getattr(runtime, "last_packet_at", None)
    if last_packet_at is None:
        return False, "no_recent_packet"
    try:
        if (timezone.now() - last_packet_at).total_seconds() > stall:
            return False, "packet_stalled"
    except Exception:  # noqa: BLE001
        return False, "packet_time_error"
    return True, ""


# ---------------------------------------------------------------------------
# 事件查询
# ---------------------------------------------------------------------------
def _load_events(camera_id: int, window_start_ts: int, window_end_ts: int) -> list:
    """窗口内与该摄像头**相交**的音频事件（新的在前）。

    - 正常事件：`started_at_ts <= window_end` 且 `ended_at_ts >= window_start`；
    - 进行中事件（`ended_at_ts IS NULL`，仍在录音）：只要在窗口结束前开始就算相交。
    """
    from django.db.models import Q

    from apps.audio_detect.models import AudioEvent

    qs = (
        AudioEvent.objects
        .filter(camera_id=camera_id, started_at_ts__lte=window_end_ts)
        .filter(Q(ended_at_ts__gte=window_start_ts) | Q(ended_at_ts__isnull=True))
        .order_by("-started_at_ts", "-id")
    )
    return list(qs[:500])


def _has_label(event, label: str) -> bool:
    labels = getattr(event, "detected_labels", None) or {}
    try:
        return label in labels
    except TypeError:  # 脏数据（不是 dict）
        return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def evaluate_audio_rule(
    camera_id: Optional[int],
    prompt_config_id: Optional[int],
    now_ts: Optional[int] = None,
    rule=None,
) -> AudioRuleOutcome:
    """计算 Prompt 的音频三态（spec §7.1）。**不抛异常**，最差返回 `UNKNOWN`。

    Args:
        camera_id: 摄像头 id；None → UNKNOWN（无法定位音频源）。
        prompt_config_id: Prompt id；用于取 `PromptAudioRule`。
        now_ts: 判定终点（epoch 秒）；缺省 = 当前时间。
        rule: 可选的 `PromptAudioRule` 实例（省一次查询；测试注入用）。

    Returns:
        :class:`AudioRuleOutcome`
    """
    now = int(now_ts if now_ts is not None else time.time())
    out = AudioRuleOutcome(window_end_ts=now)

    try:
        if not camera_id:
            out.reason = "no_camera"
            return out

        if rule is None:
            from apps.vlm.models import PromptAudioRule

            rule = (
                PromptAudioRule.objects
                .filter(prompt_config_id=prompt_config_id)
                .first()
            )
        if rule is None:
            out.reason = "no_rule"
            return out

        out.rule_enabled = bool(rule.enabled)
        out.condition = rule.condition or ""
        out.window_sec = int(rule.window_sec or 0)
        out.min_event_count = int(rule.min_event_count or 1)
        out.window_start_ts = now - out.window_sec

        if not out.rule_enabled:
            out.reason = "rule_disabled"
            return out

        service_ok, service_reason = _service_available()
        out.service_available = service_ok
        if not service_ok:
            out.reason = service_reason
            return out

        cam_ok, cam_reason, runtime = _camera_available(camera_id)
        out.camera_status = cam_reason if not cam_ok else (runtime.status or "")
        if not cam_ok:
            out.reason = cam_reason
            return out

        out.coverage_ok_from_ts = int(getattr(runtime, "coverage_ok_from_ts", 0) or 0)
        covered, cover_reason = _coverage_state(runtime, out.window_start_ts)
        out.covered = covered

        events = _load_events(camera_id, out.window_start_ts, now)
        cry_events = [e for e in events if _has_label(e, LABEL_CRY)]
        speech_events = [e for e in events if _has_label(e, LABEL_SPEECH)]
        out.event_count = len(events)
        out.cry_count = len(cry_events)
        out.speech_count = len(speech_events)

        result, referenced, reason = _decide(
            condition=out.condition,
            covered=covered,
            cover_reason=cover_reason,
            events=events,
            cry_events=cry_events,
            speech_events=speech_events,
            min_event_count=out.min_event_count,
        )
        out.result = result
        out.reason = reason
        out.events = referenced
        return out
    except Exception as e:  # noqa: BLE001 - 三态是辅助维度，绝不冒泡
        logger.warning(
            "[audio-rule] evaluate failed cam=%s prompt=%s: %s",
            camera_id, prompt_config_id, e,
        )
        out.result = UNKNOWN
        out.reason = "evaluate_error"
        out.events = []
        return out


def _decide(
    condition: str,
    covered: bool,
    cover_reason: str,
    events: list,
    cry_events: list,
    speech_events: list,
    min_event_count: int,
) -> tuple[str, list, str]:
    """8 种 condition × 覆盖完整/不完整的具体判定（spec §7.1 表）。

    Returns:
        (三态, 参考事件列表, 原因)
    """
    from apps.vlm.models import PromptAudioRule as R

    def _positive(ref: list) -> tuple[str, list, str]:
        return SATISFIED, ref[:MAX_REFERENCED_EVENTS], ""

    def _negative_or_unknown() -> tuple[str, list, str]:
        # 阴性结论必须全覆盖；不完整 → UNKNOWN
        if covered:
            return NOT_SATISFIED, [], ""
        return UNKNOWN, [], cover_reason or "not_covered"

    if condition == R.CONDITION_ANY:
        if events:
            return _positive(events)
        return _negative_or_unknown()
    if condition == R.CONDITION_CRY:
        if cry_events:
            return _positive(cry_events)
        return _negative_or_unknown()
    if condition == R.CONDITION_SPEECH:
        if speech_events:
            return _positive(speech_events)
        return _negative_or_unknown()
    if condition == R.CONDITION_CRY_OR_SPEECH:
        both = _merge_events(cry_events, speech_events)
        if both:
            return _positive(both)
        return _negative_or_unknown()
    if condition == R.CONDITION_NO_CRY:
        if cry_events:
            return NOT_SATISFIED, [], ""
        if covered:
            return SATISFIED, [], ""
        return UNKNOWN, [], cover_reason or "not_covered"
    if condition == R.CONDITION_NO_SPEECH:
        if speech_events:
            return NOT_SATISFIED, [], ""
        if covered:
            return SATISFIED, [], ""
        return UNKNOWN, [], cover_reason or "not_covered"
    if condition == R.CONDITION_COUNT_CRY:
        if len(cry_events) >= max(1, min_event_count):
            return _positive(cry_events)
        return _negative_or_unknown()
    if condition == R.CONDITION_COUNT_SPEECH:
        if len(speech_events) >= max(1, min_event_count):
            return _positive(speech_events)
        return _negative_or_unknown()
    return UNKNOWN, [], f"bad_condition:{condition}"


def _merge_events(*groups: list) -> list:
    """按 pk 去重合并（保持原顺序）。"""
    seen: set = set()
    out: list = []
    for group in groups:
        for ev in group:
            key = getattr(ev, "pk", id(ev))
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# 落审计
# ---------------------------------------------------------------------------
def record_state_audio_links(vlm_state, outcome: AudioRuleOutcome) -> int:
    """把判定结果写进 `VLMCheckStateAudioEvent`（spec §5.7）。

    - 阳性：每个参考事件一行（最多 `MAX_REFERENCED_EVENTS` 行）；
    - 阴性 / UNKNOWN：写一行 `audio_event=None` 的快照，保证三态都可审计。

    返回写入行数；失败只记日志（不冒泡）。
    """
    try:
        from apps.vlm.models import VLMCheckStateAudioEvent

        # 必须已入库（有真实 pk）；测试里的 mock state 直接跳过，不做无意义写库
        state_pk = getattr(vlm_state, "pk", None)
        if not isinstance(state_pk, int):
            return 0

        snapshot = outcome.snapshot()
        refs = outcome.events if outcome.result == SATISFIED else []
        rows = [
            VLMCheckStateAudioEvent(
                vlm_state_id=vlm_state.pk,
                audio_event_id=ev.pk,
                snapshot_json=snapshot,
            )
            for ev in refs[:MAX_REFERENCED_EVENTS]
        ]
        if not rows:
            rows = [
                VLMCheckStateAudioEvent(
                    vlm_state_id=vlm_state.pk,
                    audio_event=None,
                    snapshot_json=snapshot,
                )
            ]
        VLMCheckStateAudioEvent.objects.bulk_create(rows)
        return len(rows)
    except Exception as e:  # noqa: BLE001
        logger.warning("[audio-rule] record_state_audio_links failed: %s", e)
        return 0

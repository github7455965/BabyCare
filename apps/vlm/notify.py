"""
Step 9 HA 通知 + 防误报静默判定。

设计要点
--------
1. notify_hit(state: VLMCheckState, audio_outcome=None)
   - 命中时（hit=True 且 !auto_silenced）遍历 **through 行**
     `PromptNotifyTarget`（spec §5.6），每行带 condition：
       * `always`     → 命中就发（不看音频）
       * `audio_rule` → 仅当音频三态为 `SATISFIED` 时发；`UNKNOWN` 放行（spec §7.2）
   - 按 target.kind 分发到 HA（mobile_app → notify.mobile_app_xxx；
     speaker → text.xiaomi_xxx）
   - 失败 / 未配置：仅写日志，不抛、不重试
   - 部分成功（≥1 个 target 发送成功）→ 置 notified=True
   - 总开关：prompt.notify_on_hit + 至少一行 through 必须同时有效
   - B5 直发单个 target（不走 .env 全局 targets）
   - 每个 (目标, 判定) 写一条 `PromptNotificationDelivery` 审计行（spec §7.2）
   - 音频三态由调用方（runner）算好传入；缺省时本函数自己算（容错用）

2. compute_auto_silenced(state, silence_count)
   - 用户标误报后，连续 N 次同状态签名触发则静默
   - 静默失效 3 条件（任一打破即失效）：
     a) status 变化（不同警报）
     b) 中间有 hit=True 且 !auto_silenced（即"又报了"）
     c) 连续 auto_silenced=True 计数 ≥ N
   - 中间 hit=False（miss/no_baby/describe/failure）不算中断

3. compute_recent_hit_dedup(cam_id, prompt_id, status, window_sec)
   - 短时同 signature hit 去重（B6：Bug 修复）
   - 同 (cam, prompt, status) 在 window_sec 秒内若已有 hit=True 且 notified=True 记录
     → 本次 hit 应跳过 HA 通知（事件仍写库，state.notified=False）
   - window_sec<=0 → 关闭去重（永远 False）
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from apps.vlm import ha_notice

if TYPE_CHECKING:
    from apps.vlm.models import VLMCheckState

logger = logging.getLogger(__name__)

SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")


def compute_auto_silenced(cam_id: int, prompt_id: int, status: str, silence_count: int) -> bool:
    """本条是否因历史 dismissed_as_false 被静默。

    Args:
        cam_id: Camera.id
        prompt_id: VLMPromptConfig.id
        status: 当前 hit 的 status 文本（用于匹配历史 dismissed 记录）
        silence_count: VLMPromptConfig.silence_count_after_dismiss

    Returns:
        True  → 设为 auto_silenced=True（不报警、不发通知）
        False → 正常 hit
    """
    from apps.vlm.models import VLMCheckState

    if silence_count <= 0:
        return False

    # 找最近一条同 status 的 dismissed_as_false=True 记录
    last_dismiss = (
        VLMCheckState.objects
        .filter(
            camera_id=cam_id,
            prompt_config_id=prompt_id,
            status=status,
            dismissed_as_false=True,
        )
        .order_by("-id")
        .first()
    )
    if last_dismiss is None:
        return False

    # dismiss 之后所有 hit=True 同 status 记录
    later_hits = (
        VLMCheckState.objects
        .filter(
            camera_id=cam_id,
            prompt_config_id=prompt_id,
            status=status,
            hit=True,
            id__gt=last_dismiss.id,
        )
        .order_by("id")
    )

    # 任何一条是 hit=True 且 !auto_silenced → 静默失效
    consecutive_silent = 0
    for prev in later_hits:
        if prev.auto_silenced:
            consecutive_silent += 1
            if consecutive_silent >= silence_count:
                return False  # 已达上限
        else:
            return False  # 又报了 → 静默失效

    # 没找到任何打破条件 → 本次静默
    return True


def compute_recent_hit_dedup(
    cam_id: int,
    prompt_id: int,
    status: str,
    window_sec: int,
) -> bool:
    """短时同 signature hit 去重（B6：Bug 修复）。

    同一 (cam, prompt, status) 在 window_sec 秒内若已有 hit=True 且 notified=True 记录，
    本次 hit 应跳过通知（仍写库，state.notified=False）。

    Args:
        cam_id: Camera.id
        prompt_id: VLMPromptConfig.id
        status: 本次 hit 的 status 文本
        window_sec: 去重窗口秒数（从 settings.BABYCARE_NOTIFY_DEDUP_WINDOW_SEC 读）

    Returns:
        True  → 本次应跳过通知（dedup 命中）
        False → 正常通知
    """
    if window_sec <= 0:
        return False
    from datetime import timedelta

    from django.utils import timezone

    from apps.vlm.models import VLMCheckState

    cutoff = timezone.now() - timedelta(seconds=window_sec)
    return VLMCheckState.objects.filter(
        camera_id=cam_id,
        prompt_config_id=prompt_id,
        status=status,
        hit=True,
        notified=True,
        created_at__gte=cutoff,
    ).exists()


def _load_target_links(prompt) -> list:
    """取 Prompt 的 through 行 → ``[(NotifyTarget, condition), ...]``（spec §5.6）。

    只返回 `NotifyTarget.enabled=True` 的目标（保持 B5 的"停用即不出现在列表"语义）。
    """
    from apps.vlm.models import PromptNotifyTarget

    links = (
        PromptNotifyTarget.objects
        .filter(prompt_config_id=prompt.pk, notify_target__enabled=True)
        .select_related("notify_target")
        .order_by("notify_target__kind", "notify_target__name")
    )
    return [(link.notify_target, link.condition or "") for link in links]


def _should_send(condition: str, audio_result: str) -> bool:
    """按 `PromptNotifyTarget.condition` × 音频三态决定是否发送（spec §7.2）。

    ==============  =============  =================  ==========
    condition       SATISFIED      NOT_SATISFIED      UNKNOWN
    ==============  =============  =================  ==========
    always          发送            发送               发送
    audio_rule      发送            不发送             **发送**
    ==============  =============  =================  ==========

    `UNKNOWN` 一律放行：音频关掉等于这条线不存在，所有目标回到"命中就发"
    （与引入音频功能之前的行为完全一致，spec §7.2）。
    """
    from apps.vlm.models import PromptNotifyTarget

    if condition == PromptNotifyTarget.CONDITION_AUDIO_RULE:
        from apps.vlm.audio_rule import NOT_SATISFIED

        return audio_result != NOT_SATISFIED
    # 未知 condition 一律按 always 处理（保守：宁可多发，不可静默漏发）
    return True


def _deliver(target, title: str, message: str) -> None:
    """按 kind 分发到 HA；未知 kind 抛 ValueError（由调用方记失败）。"""
    if target.kind == "mobile_app":
        _send_mobile_app(target.target_id, title, message)
    elif target.kind == "speaker":
        _send_speaker(target.target_id, message)
    else:
        raise ValueError(f"unknown target kind={target.kind!r}")


def _record_delivery(
    state,
    target,
    condition: str,
    audio_outcome,
    delivered: bool,
    error: str = "",
) -> None:
    """写一条 `PromptNotificationDelivery` 审计行（spec §7.2）。

    **绝不冒泡**：审计写入失败不能影响通知发送结果。
    """
    try:
        from apps.vlm.models import PromptNotificationDelivery

        ref = audio_outcome.reference_event if delivered else None
        PromptNotificationDelivery.objects.create(
            prompt_config_id=state.prompt_config_id,
            vlm_state_id=state.id,
            audio_event=ref,
            notify_target_id=target.id,
            audio_state=audio_outcome.audio_state,
            condition=condition or "",
            delivered=delivered,
            error=(error or "")[:2000],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[notify] record delivery failed state=%s target=%s: %s",
            getattr(state, "id", None), getattr(target, "name", None), e,
        )


#: 通知文案默认模板（Prompt 上留空时用）
DEFAULT_TITLE_TEMPLATE = "【宝宝告警·{prompt}】"
DEFAULT_BODY_TEMPLATE = "{cam} 在 {time} 检出（{status}）"

#: ``{status}`` 被一对括号包住时的整体匹配（空 status → 连括号一起去掉）
_STATUS_PAREN_RE = re.compile(r"[（(]\s*\{status\}\s*[)）]")


def render_notify_text(
    template: str,
    *,
    prompt_name: str = "",
    cam_name: str = "",
    time_text: str = "",
    status: str = "",
) -> str:
    """渲染通知文案模板（spec §7.2）。

    占位符：``{prompt}`` / ``{cam}`` / ``{time}`` / ``{status}``。

    ``status`` 为空（判断型 prompt 只让 VLM 回"是/否"时很常见）时，先把
    ``（{status}）`` 这类"括号包着 status"的整体去掉，再兜底替换裸 ``{status}`` ——
    否则会播出毫无信息量的"（未提供状态）"或一对空括号。
    """
    text = (template or "").strip()
    status_text = (status or "").strip()
    if not status_text:
        text = _STATUS_PAREN_RE.sub("", text)
    rendered = (
        text.replace("{prompt}", prompt_name)
        .replace("{cam}", cam_name)
        .replace("{time}", time_text)
        .replace("{status}", status_text)
    )
    return " ".join(rendered.split())


def _pick_template(value, default: str) -> str:
    """模板取值：只有**非空字符串**才算配置，其余（None / 空串 / 测试替身的 Mock）→ 默认。"""
    return value if isinstance(value, str) and value.strip() else default


def notify_hit(state: "VLMCheckState", audio_outcome=None) -> bool:
    """命中时遍历 through 行，按 `condition` 路由后分发到 HA（spec §7.2）。

    同步发，部分成功（≥1 target 发送成功）→ state.notified=True。
    失败仅写日志，不抛、不重试。

    Args:
        state: 已入库的 VLMCheckState（必须有 id）
        audio_outcome: 事先算好的 `AudioRuleOutcome`；None → 本函数自己算
            （容错：runner 之外的调用方也能直接用）

    Returns:
        True  → 至少 1 个 target 发送成功（state.notified 已被置 True 并 save）
        False → 未发送（hit=False / notify_on_hit=False / 无 enabled target / 全部失败）
    """
    if not state.hit:
        return False

    cam = state.camera
    prompt = state.prompt_config
    if not prompt.notify_on_hit:
        logger.debug("[notify] skipped notify_on_hit=False prompt=%s state=%d",
                     prompt.name, state.id)
        return False

    # Phase 6：按 through 行遍历（每行带 condition）
    links = _load_target_links(prompt)
    if not links:
        logger.debug("[notify] no targets prompt=%s state=%d",
                     prompt.name, state.id)
        return False

    if audio_outcome is None:
        from apps.vlm.audio_rule import evaluate_audio_rule

        audio_outcome = evaluate_audio_rule(
            camera_id=getattr(state, "camera_id", None),
            prompt_config_id=getattr(state, "prompt_config_id", None),
        )

    try:
        ts_cn = datetime.fromtimestamp(state.ts_list[0], tz=SHANGHAI).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        ts_cn = "未知时间"

    # 文案模板（Prompt 页面可配，留空用默认；{status} 为空时自动省略括号）
    render_kwargs = {
        "prompt_name": prompt.name,
        "cam_name": cam.name,
        "time_text": ts_cn,
        "status": state.status or "",
    }
    title = render_notify_text(
        _pick_template(
            getattr(prompt, "notify_title_template", ""), DEFAULT_TITLE_TEMPLATE,
        ),
        **render_kwargs,
    )
    message = render_notify_text(
        _pick_template(
            getattr(prompt, "notify_body_template", ""), DEFAULT_BODY_TEMPLATE,
        ),
        **render_kwargs,
    )

    sent = 0
    skipped = 0
    failed_targets: list[str] = []
    for t, condition in links:
        if not _should_send(condition, audio_outcome.result):
            skipped += 1
            _record_delivery(state, t, condition, audio_outcome, delivered=False)
            logger.info(
                "[notify] SKIP by condition=%s state=%d target=%s audio_state=%s",
                condition, state.id, t.name, audio_outcome.audio_state,
            )
            continue
        try:
            _deliver(t, title, message)
            sent += 1
            _record_delivery(state, t, condition, audio_outcome, delivered=True)
        except (ValueError, ha_notice.HANoticeError) as e:
            logger.warning("[notify] target=%s failed state=%d: %s",
                           t.name, state.id, e)
            failed_targets.append(t.name)
            _record_delivery(
                state, t, condition, audio_outcome, delivered=False, error=str(e),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[notify] target=%s exception state=%d: %s",
                           t.name, state.id, e)
            failed_targets.append(t.name)
            _record_delivery(
                state, t, condition, audio_outcome, delivered=False, error=str(e),
            )

    if sent > 0:
        state.notified = True
        state.save(update_fields=["notified"])
    logger.info(
        "[notify] HIT state=%d prompt=%s cam=%s sent=%d skipped=%d audio=%s failed=%s",
        state.id, prompt.name, cam.name, sent, skipped,
        audio_outcome.audio_state, failed_targets or "无",
    )
    return sent > 0


def _send_mobile_app(notify_service: str, title: str, message: str) -> None:
    """直发单个 mobile_app 通知（不走 .env 全局 targets）。

    Args:
        notify_service: 完整 HA 服务 ID（"notify.mobile_app_xxx"）
    """
    ha_notice.load_env()
    ha_url = os.environ.get("HA_URL", "").rstrip("/")
    token = os.environ.get("HA_TOKEN", "")
    if not ha_url or not token:
        raise ValueError("缺少配置 HA_URL 或 HA_TOKEN")
    service_name = notify_service.removeprefix("notify.")
    payload: dict[str, object] = {"title": title, "message": message}
    ha_notice._post_json(  # noqa: SLF001 - B5 直发复用底层 HTTP 调用
        f"{ha_url}/api/services/notify/{service_name}", token, payload, 10,
    )


def _send_speaker(entity_id: str, text: str) -> None:
    """直发单个 speaker 播报（不走 .env 全局 targets）。

    Args:
        entity_id: 完整 HA 实体 ID（"text.xiaomi_xxx"）
    """
    ha_notice.load_env()
    ha_url = os.environ.get("HA_URL", "").rstrip("/")
    token = os.environ.get("HA_TOKEN", "")
    if not ha_url or not token:
        raise ValueError("缺少配置 HA_URL 或 HA_TOKEN")
    ha_notice._post_json(  # noqa: SLF001 - B5 直发复用底层 HTTP 调用
        f"{ha_url}/api/services/text/set_value", token,
        {"entity_id": entity_id, "value": text}, 10,
    )
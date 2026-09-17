"""共识与判定引擎：两模型分数 → 每窗每标签三态 → 窗口决策 + 落库原因（spec §4.3）。

三态（spec §4.3.4）
------------------
``positive`` 两模型都有分数，按共识策略判定为阳性
``negative`` 两模型都有分数，判定为非阳性
``abstain``  某模型异常或未加载

``abstain ≠ negative``：不推进阳性计数，也不打断结束计时（Phase 3 状态机会用到）。

单模型降级（spec §4.3.3）
-----------------------
``single_model_fallback=False``（默认）：任一模型异常 → ``abstain``，严格模式下
本窗不产生新事件。
``single_model_fallback=True``：存活的那一个模型单独判定，阈值自动上浮
（``×fallback_threshold_scale``），并标记窗口 ``degraded`` 便于审计。

单侧高置信度覆盖（spec §4.3.5）
-------------------------------
``and`` 共识下，两个模型都要过各自阈值才算阳性；但两模型分数分布不同，一侧可能
长期低于阈值（实测 PANNs 的 cry max≈0.19，永远够不到 0.3），于是 ``and`` 实际不工作。
因此加一条补偿规则：**任一模型分数 ≥ ``high_confidence_override``（默认 0.8）→ 即使
另一侧没过阈值，也判该标签窗口为阳性**（``reason = single_side_high_confidence``）。
``0 < x <= 1`` 之外视为关闭。

本模块是**纯计算**，不碰 DB、不碰模型文件，便于单测。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .label_map import MODEL_PANNS, MODEL_YAMNET, MODELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------
STATE_POSITIVE = "positive"
STATE_NEGATIVE = "negative"
STATE_ABSTAIN = "abstain"

DECISION_POSITIVE = "positive"
DECISION_NEGATIVE = "negative"
DECISION_DEGRADED = "degraded"

STRATEGY_AND = "and"
STRATEGY_OR = "or"
STRATEGIES = (STRATEGY_AND, STRATEGY_OR)

# 落库原因（spec §5.1 的 log_reason）
LOG_POSITIVE = "positive"
LOG_NEAR_THRESHOLD = "near_threshold"
LOG_STATE_CHANGE = "state_change"

# 失败原因
REASON_MODEL_ABSTAIN = "model_abstain"
REASON_SINGLE_SIDE_ONLY = "single_side_only"
REASON_ALL_ABSTAIN = "all_models_abstain"
REASON_FALLBACK = "single_model_fallback"
REASON_HIGH_CONFIDENCE = "single_side_high_confidence"


# ---------------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    """per-model 阈值（spec §4.3.1）。

    为什么必须按模型分开：两个模型的分数分布不同，共用一根阈值会让其中一侧
    实际等于"永不判阳"，``and`` 共识将永远无法成立。
    """

    #: ``{model: {business: threshold}}``
    values: Mapping[str, Mapping[str, float]]
    #: 近阈值落库比例（spec §5.1：分数 ≥ 阈值 × 0.7 的窗要落，供事后调阈值）
    near_ratio: float = 0.7
    #: 单模型降级时阈值上浮系数（spec §4.3.3）
    fallback_scale: float = 1.3
    #: 单侧高置信度覆盖阈值（spec §4.3.5）：任一模型分数 ≥ 该值即判阳，``and`` 也用。
    #: ``0 < x <= 1`` 之外视为关闭（默认 0.8）。
    high_confidence_override: float = 0.8

    @classmethod
    def from_settings(cls) -> "Thresholds":
        from django.conf import settings

        return cls(
            values={
                MODEL_YAMNET: {
                    "cry": float(settings.BABYCARE_AUDIO_CRY_THRESHOLD_YAMNET),
                    "speech": float(settings.BABYCARE_AUDIO_SPEECH_THRESHOLD_YAMNET),
                },
                MODEL_PANNS: {
                    "cry": float(settings.BABYCARE_AUDIO_CRY_THRESHOLD_PANNS),
                    "speech": float(settings.BABYCARE_AUDIO_SPEECH_THRESHOLD_PANNS),
                },
            },
            near_ratio=float(settings.BABYCARE_AUDIO_NEAR_THRESHOLD_RATIO),
            fallback_scale=float(settings.BABYCARE_AUDIO_FALLBACK_THRESHOLD_SCALE),
            high_confidence_override=float(
                settings.BABYCARE_AUDIO_HIGH_CONFIDENCE_OVERRIDE
            ),
        )

    def get(self, model: str, business: str) -> float:
        return float((self.values.get(model) or {}).get(business, 1.0))

    def high_confidence(self) -> float | None:
        """单侧高置信度覆盖阈值；不在 ``(0, 1]`` 内 → ``None``（该规则关闭）。"""
        value = float(self.high_confidence_override)
        return value if 0.0 < value <= 1.0 else None

    def snapshot(self) -> dict[str, dict[str, float]]:
        """落进 ``consensus_payload`` 的阈值快照，便于事后解释判定。"""
        return {m: dict(v) for m, v in self.values.items()}


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class LabelVerdict:
    """单个业务标签的窗口判定。"""

    label: str
    scores: dict[str, float | None] = field(default_factory=dict)
    positives: dict[str, bool | None] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    state: str = STATE_ABSTAIN
    reason: str = ""

    @property
    def is_positive(self) -> bool:
        return self.state == STATE_POSITIVE


@dataclass
class WindowVerdict:
    """一个推理窗口的聚合判定。"""

    labels: dict[str, LabelVerdict] = field(default_factory=dict)
    decision: str = DECISION_NEGATIVE
    failure_reason: str = ""
    strategy: str = STRATEGY_AND
    degraded: bool = False

    def positive_labels(self) -> tuple[str, ...]:
        return tuple(l for l, v in self.labels.items() if v.is_positive)

    def as_log(self) -> str:
        parts = []
        for label, v in self.labels.items():
            y = "-" if v.scores.get(MODEL_YAMNET) is None else f"{v.scores[MODEL_YAMNET]:.3f}"
            p = "-" if v.scores.get(MODEL_PANNS) is None else f"{v.scores[MODEL_PANNS]:.3f}"
            # 带 reason：否则"单侧高置信度覆盖"判阳与真正双阳在日志里长得一样，排查时无法区分
            note = f"/{v.reason}" if v.reason else ""
            parts.append(f"{label}[y={y} p={p} -> {v.state}{note}]")
        reason = f" reason={self.failure_reason}" if self.failure_reason else ""
        return f"decision={self.decision} " + " ".join(parts) + reason


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class ConsensusEngine:
    """把两模型分数按策略合成窗口判定。"""

    def __init__(
        self,
        thresholds: Thresholds,
        labels: Sequence[str],
        strategy: str = STRATEGY_AND,
        single_model_fallback: bool = False,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"未知共识策略: {strategy!r}（可选 {STRATEGIES}）")
        self.thresholds = thresholds
        self.labels = tuple(labels)
        self.strategy = strategy
        self.single_model_fallback = bool(single_model_fallback)

    @classmethod
    def from_settings(cls, labels: Sequence[str]) -> "ConsensusEngine":
        from django.conf import settings

        return cls(
            thresholds=Thresholds.from_settings(),
            labels=labels,
            strategy=str(settings.BABYCARE_AUDIO_CONSENSUS),
            single_model_fallback=bool(settings.BABYCARE_AUDIO_SINGLE_MODEL_FALLBACK),
        )

    # ------------------------------------------------------------------
    def evaluate(
        self,
        yamnet_scores: Mapping[str, float] | None,
        panns_scores: Mapping[str, float] | None,
    ) -> WindowVerdict:
        """:param yamnet_scores: ``None`` = 该模型异常 / 未加载（→ ``abstain``）"""
        available = {
            MODEL_YAMNET: yamnet_scores,
            MODEL_PANNS: panns_scores,
        }
        verdict = WindowVerdict(strategy=self.strategy)
        reasons: set[str] = set()

        for label in self.labels:
            lv = self._evaluate_label(label, available)
            verdict.labels[label] = lv
            if lv.reason:
                reasons.add(lv.reason)

        if any(v.is_positive for v in verdict.labels.values()):
            verdict.decision = DECISION_POSITIVE
        elif any(v.state == STATE_ABSTAIN for v in verdict.labels.values()):
            verdict.decision = DECISION_DEGRADED
        else:
            verdict.decision = DECISION_NEGATIVE

        verdict.degraded = any(v.state == STATE_ABSTAIN for v in verdict.labels.values())
        if REASON_ALL_ABSTAIN in reasons:
            verdict.failure_reason = REASON_ALL_ABSTAIN
        elif reasons:
            verdict.failure_reason = ",".join(sorted(reasons))
        return verdict

    # ------------------------------------------------------------------
    def _evaluate_label(
        self,
        label: str,
        available: Mapping[str, Mapping[str, float] | None],
    ) -> LabelVerdict:
        lv = LabelVerdict(
            label=label,
            thresholds={m: self.thresholds.get(m, label) for m in MODELS},
        )
        live: dict[str, float] = {}

        for model in MODELS:
            scores = available.get(model)
            if scores is None or label not in scores:
                lv.scores[model] = None
                lv.positives[model] = None
                continue
            score = float(scores[label])
            lv.scores[model] = score
            live[model] = score

        if not live:
            lv.state = STATE_ABSTAIN
            lv.reason = REASON_ALL_ABSTAIN
            return lv

        # ---- 单模型（降级开关关 → abstain，不产生新事件）----
        if len(live) == 1:
            model, score = next(iter(live.items()))
            if not self.single_model_fallback:
                lv.state = STATE_ABSTAIN
                lv.reason = REASON_MODEL_ABSTAIN
                return lv
            thr = self.thresholds.get(model, label) * self.thresholds.fallback_scale
            lv.thresholds[model] = thr
            positive = score >= thr
            lv.positives[model] = positive
            lv.state = STATE_POSITIVE if positive else STATE_NEGATIVE
            lv.reason = REASON_FALLBACK
            return lv

        # ---- 双模型：按策略判共识 ----
        flags = {
            m: live[m] >= lv.thresholds[m]
            for m in MODELS
        }
        lv.positives = dict(flags)
        if self.strategy == STRATEGY_AND:
            positive = flags[MODEL_YAMNET] and flags[MODEL_PANNS]
        else:
            positive = flags[MODEL_YAMNET] or flags[MODEL_PANNS]

        lv.state = STATE_POSITIVE if positive else STATE_NEGATIVE
        if not positive and self.strategy == STRATEGY_AND:
            # 单侧高置信度覆盖（spec §4.3.5）：任一侧分数 ≥ 覆盖阈值 → 也算阳性
            override = self.thresholds.high_confidence()
            if override is not None and any(s >= override for s in live.values()):
                lv.state = STATE_POSITIVE
                lv.reason = REASON_HIGH_CONFIDENCE
                return lv
        if not positive and (flags[MODEL_YAMNET] != flags[MODEL_PANNS]):
            # 单侧阳性但共识不成立：要落库（供调阈值），但不产出事件
            lv.reason = REASON_SINGLE_SIDE_ONLY
        return lv

    # ------------------------------------------------------------------
    def log_reason(self, verdict: WindowVerdict) -> str:
        """本窗是否值得落库（spec §5.1 稀疏落库策略）。

        返回 ``''`` 表示普通窗，只在内存滚动。
        """
        if verdict.decision == DECISION_POSITIVE:
            return LOG_POSITIVE
        for lv in verdict.labels.values():
            for model in MODELS:
                score = lv.scores.get(model)
                if score is None:
                    continue
                if score >= lv.thresholds.get(model, 1.0) * self.thresholds.near_ratio:
                    return LOG_NEAR_THRESHOLD
        return ""

    # ------------------------------------------------------------------
    def consensus_payload(self, verdict: WindowVerdict) -> dict[str, Any]:
        """落进 ``SoundDetectionLog.consensus_payload`` 的可审计快照。"""
        return {
            "strategy": verdict.strategy,
            "single_model_fallback": self.single_model_fallback,
            "near_ratio": self.thresholds.near_ratio,
            "high_confidence_override": self.thresholds.high_confidence(),
            "decision": verdict.decision,
            "failure_reason": verdict.failure_reason,
            "thresholds": self.thresholds.snapshot(),
            "labels": {
                label: {
                    "state": lv.state,
                    "reason": lv.reason,
                    "scores": lv.scores,
                    "positives": lv.positives,
                    "thresholds": lv.thresholds,
                }
                for label, lv in verdict.labels.items()
            },
        }

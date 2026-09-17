"""标签映射层：业务标签 ↔ 模型原始类（spec §4.1/§4.2）。

两条硬性规则
------------
1. **精确匹配，不做子串匹配**。两个类表里都有语义不同的近似类名：

   ==================== =====================
   要匹配的类名          会被子串误命中的类名
   ==================== =====================
   ``Whimper``          ``Whimper (dog)``
   ``Crying, sobbing``  ``Battle cry``
   ==================== =====================

   子串匹配会把这些误命中的类算进 ``cry`` 分数。

2. **启动强校验**。``raw_labels`` 里任何一个类名在对应模型的类表里找不到 →
   直接抛 :class:`LabelMapError` 让 worker 起不来。**不允许**静默降级成
   ``disabled``——那样表现为"模型永远不判阳"，从日志上根本看不出来。

聚合（spec §4.1）
----------------
本模块只做**第二级**（跨标签取 max）：把模型给出的完整分数向量 + 类名索引
算成业务标签置信度。第一级（跨帧聚合 YAMNet 的 4 个 patch）在 :mod:`detector`。

**必须用完整分数向量按索引取分，不能用 top-K**：K 小的时候会漏掉"不在 top-K
但分数并不低"的标签（spec §4.2）。top-K 只用于日志展示。
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BUSINESS_CRY = "cry"
BUSINESS_SPEECH = "speech"

MODEL_YAMNET = "yamnet"
MODEL_PANNS = "panns"
MODELS = (MODEL_YAMNET, MODEL_PANNS)

#: spec §4.2 的默认映射表。两模型都基于 AudioSet 本体，这 4 个类名字符串一致，
#: 因此默认两边共用一份；结构上仍按模型分开，便于将来某一侧单独调整。
DEFAULT_RAW_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    BUSINESS_CRY: {
        MODEL_YAMNET: ("Crying, sobbing", "Baby cry, infant cry", "Whimper"),
        MODEL_PANNS: ("Crying, sobbing", "Baby cry, infant cry", "Whimper"),
    },
    BUSINESS_SPEECH: {
        MODEL_YAMNET: ("Speech",),
        MODEL_PANNS: ("Speech",),
    },
}

DISPLAY_NAME_COLUMN = "display_name"


class LabelMapError(RuntimeError):
    """映射表校验失败（spec §4.2：必须让启动失败，而不是静默 disabled）。"""


# ---------------------------------------------------------------------------
# 类表读取
# ---------------------------------------------------------------------------
def read_class_map(csv_path: str | Path) -> list[str]:
    """读模型类表 CSV，返回**按索引顺序**排列的 display_name 列表。

    顺序即类别索引，推理时用 ``_index_of`` 定位；不能排序，否则索引全错。
    """
    path = Path(csv_path)
    if not path.exists():
        raise LabelMapError(f"类表不存在: {path}")
    names: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            names.append((row.get(DISPLAY_NAME_COLUMN) or "").strip())
    if not names:
        raise LabelMapError(f"类表为空或缺少 {DISPLAY_NAME_COLUMN} 列: {path}")
    return names


# ---------------------------------------------------------------------------
# 映射表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioLabelMap:
    """业务标签 → 各模型原始类名，以及两侧类表。

    :param raw_labels: ``{business: {model: (raw_name, ...)}}``
    :param class_names: ``{model: [class_name, ...]}``（保持索引顺序）
    """

    raw_labels: Mapping[str, Mapping[str, tuple[str, ...]]]
    class_names: Mapping[str, Sequence[str]] = field(default_factory=dict)
    #: 校验通过后填充：``{model: {raw_name: index}}``
    _index_of: Mapping[str, Mapping[str, int]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls) -> "AudioLabelMap":
        """从 Django settings 读类表路径构造并**强校验**。"""
        from django.conf import settings

        return cls(
            raw_labels=DEFAULT_RAW_LABELS,
            class_names={
                MODEL_YAMNET: read_class_map(settings.BABYCARE_AUDIO_YAMNET_CLASS_MAP),
                MODEL_PANNS: read_class_map(settings.BABYCARE_AUDIO_PANNS_CLASS_MAP),
            },
        ).validated()

    def validated(self) -> "AudioLabelMap":
        """校验并返回一个带索引缓存的实例（frozen，返回新对象）。"""
        self.validate()
        index_of: dict[str, dict[str, int]] = {}
        for model, names in self.class_names.items():
            index_of[model] = {n: i for i, n in enumerate(names)}
        return AudioLabelMap(
            raw_labels=self.raw_labels,
            class_names=self.class_names,
            _index_of=index_of,
        )

    # ------------------------------------------------------------------
    # 校验（spec §4.2）
    # ------------------------------------------------------------------
    def validate(self) -> None:
        problems: list[str] = []

        for model in MODELS:
            if not (self.class_names or {}).get(model):
                problems.append(f"{model}: 缺少类表")

        for business, per_model in self.raw_labels.items():
            if not per_model:
                problems.append(f"{business}: 未声明任何模型映射")
                continue
            for model in MODELS:
                names = per_model.get(model)
                if names is None:
                    problems.append(f"{business}: 缺少 {model} 侧声明")
                    continue
                if len(names) == 0:
                    continue     # 空 → 该标签 disabled（合法，见 enabled_labels）
                known = set(self.class_names.get(model) or ())
                for raw in names:
                    if raw not in known:
                        near = [k for k in known if raw.lower() in k.lower()]
                        hint = f"；近似类名（子串匹配已禁用）: {near}" if near else ""
                        problems.append(
                            f"{model} 类表中不存在 {raw!r}（业务标签 {business}）{hint}"
                        )

        if problems:
            raise LabelMapError(
                "AudioLabelMap 校验失败，拒绝启动（spec §4.2）：\n  - "
                + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    @property
    def all_labels(self) -> tuple[str, ...]:
        return tuple(self.raw_labels)

    def disabled_reason(self, business: str) -> str | None:
        """返回该业务标签被判 ``disabled`` 的原因；可用则返回 ``None``。"""
        per_model = self.raw_labels.get(business) or {}
        missing = [m for m in MODELS if not per_model.get(m)]
        if missing:
            return f"{'/'.join(missing)} 侧未声明原始类名"
        return None

    def is_enabled(self, business: str) -> bool:
        return business in self.raw_labels and self.disabled_reason(business) is None

    def enabled_labels(self) -> tuple[str, ...]:
        return tuple(b for b in self.raw_labels if self.is_enabled(b))

    def disabled_labels(self) -> dict[str, str]:
        return {
            b: (self.disabled_reason(b) or "")
            for b in self.raw_labels
            if not self.is_enabled(b)
        }

    def raw_indices(self, model: str, business: str) -> tuple[int, ...]:
        """该业务标签在 ``model`` 分数向量里的索引（精确匹配，保持声明顺序）。"""
        idx = (self._index_of or {}).get(model) or {}
        return tuple(
            idx[raw] for raw in (self.raw_labels.get(business) or {}).get(model, ())
            if raw in idx
        )

    # ------------------------------------------------------------------
    # 二级聚合：完整分数向量 → 业务标签置信度
    # ------------------------------------------------------------------
    def score_business(
        self,
        frame_scores: Sequence[float],
        model: str,
        labels: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """跨标签聚合（取 max），得到每个业务标签的本模型置信度。

        :param frame_scores: 一维完整分数向量（已做过跨帧聚合）
        """
        out: dict[str, float] = {}
        for business in (labels if labels is not None else self.enabled_labels()):
            indices = self.raw_indices(model, business)
            if not indices:
                continue
            out[business] = max(float(frame_scores[i]) for i in indices)
        return out

    # ------------------------------------------------------------------
    def raw_score_table(
        self,
        frame_scores: Sequence[float],
        model: str,
        labels: Iterable[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """每个业务标签命中的各原始类分数（用于日志审计）。"""
        out: dict[str, dict[str, float]] = {}
        for business in (labels if labels is not None else self.enabled_labels()):
            idx = (self._index_of or {}).get(model) or {}
            raw = (self.raw_labels.get(business) or {}).get(model, ())
            out[business] = {
                name: round(float(frame_scores[idx[name]]), 6)
                for name in raw if name in idx
            }
        return out

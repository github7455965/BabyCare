from django import forms
from django.db import transaction

from apps.vlm.models import (
    NotifyTarget,
    PromptAudioRule,
    PromptNotifyTarget,
    VLMPromptConfig,
)


# v2 多类别：target_classes 校验白名单（与 runner._VALID_TARGET_CLASSES 同步）
_TARGET_CHOICES = [
    ("baby", "baby（宝宝）"),
    ("person", "person（人）"),
    ("cat", "cat（猫）"),
]
# 校验 prompt 文案是否"提到"某个 target 的关键字（粗略启发式，避免 VLM 永远 miss）
_TARGET_KEYWORDS_HINT = {
    "baby":   ["宝宝", "baby", "小孩", "孩子", "婴儿", "娃"],
    "person": ["人", "person", "人物", "他人", "大人"],
    "cat":    ["猫", "cat", "猫咪"],
}


def _missing_targets(target_csv: str, prompt_text: str) -> list[str]:
    """哪些已选目标类别在 prompt 文案里"查不到关键字"。

    只在**多选**（≥2 个）时检查，单选不查：勾一个类别时文案怎么写都不影响该字段的
    语义。纯启发式（只做子串匹配，看不懂语义），所以调用方只把结果当作**警告**，
    不拦保存 —— 用户比关键字表更清楚自己的文案要写什么。
    """
    targets = [t for t in (target_csv or "").split(",") if t]
    if len(targets) <= 1:
        return []
    text = (prompt_text or "").lower()
    return [
        t for t in targets
        if not any(kw.lower() in text for kw in _TARGET_KEYWORDS_HINT.get(t, ()))
    ]


def _target_warning(missing: list[str]) -> str:
    return (
        f"目标类别 {missing} 在 prompt 文案中未提及，VLM 看不到对应目标，"
        f"可能一直 miss。建议文案里带上对应称呼（如 '宝宝/小孩'、'人/大人'、'猫'），"
        f"例如 '图中有 {'/'.join(missing)} 吗？'。"
    )


class VLMPromptConfigForm(forms.ModelForm):
    # 覆盖字段为多选 checkbox（底层 CharField 逗号分隔）
    target_classes = forms.MultipleChoiceField(
        label="目标类别",
        choices=_TARGET_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text=(
            "可多选。多选时 prompt 文案必须涵盖所有 target（如 '图中有宝宝或猫吗？'），"
            "否则 VLM 看到的是未提及的 target，会一直 miss。"
            "不勾 = 默认 baby。"
        ),
    )

    # ----- 声音条件（Phase 6，spec §5.5）-----
    # 判定结果是三态（spec §7.1）：未启用 / 音频不可用 / 覆盖不完整 → UNKNOWN（不筛选）
    audio_rule_enabled = forms.BooleanField(
        label="启用声音条件", required=False,
        help_text=(
            "启用后，Prompt 命中时会按下面的条件计算音频三态，用于筛选 "
            "condition=audio_rule 的目标。未启用 → UNKNOWN（不参与筛选，所有目标照发）。"
        ),
    )
    audio_rule_condition = forms.ChoiceField(
        label="声音条件",
        choices=PromptAudioRule.CONDITION_CHOICES,
        required=False,
        initial=PromptAudioRule.CONDITION_ANY,
    )
    audio_rule_window_sec = forms.IntegerField(
        label="判定窗口(秒)", required=False, min_value=1, max_value=3600, initial=60,
        help_text="Prompt 命中时刻往前看多少秒的 AudioEvent。",
    )
    audio_rule_min_event_count = forms.IntegerField(
        label="最少事件数", required=False, min_value=1, max_value=100, initial=1,
        help_text="仅「哭声事件数达标 / 说话声事件数达标」使用。",
    )

    class Meta:
        model = VLMPromptConfig
        fields = [
            "name", "kind", "target_classes", "prompt", "positive_keyword",
            "camera_ids", "window_sec",
            "time_window_enabled", "time_window_start", "time_window_end",
            "weekdays", "enabled", "notify_on_hit",
            "notify_title_template", "notify_body_template",
        ]
        widgets = {
            "camera_ids": forms.CheckboxSelectMultiple,
            "window_sec": forms.NumberInput(attrs={"min": 5, "max": 30}),
            "time_window_start": forms.TimeInput(attrs={"type": "time"}),
            "time_window_end": forms.TimeInput(attrs={"type": "time"}),
            "weekdays": forms.TextInput(attrs={"placeholder": "1,2,3,4,5,6,7"}),
            "prompt": forms.Textarea(attrs={"rows": 4}),
            "notify_title_template": forms.TextInput(
                attrs={"placeholder": "【宝宝告警·{prompt}】"}
            ),
            "notify_body_template": forms.TextInput(
                attrs={"placeholder": "{cam} 在 {time} 检出（{status}）"}
            ),
        }
        help_texts = {
            "prompt": "系统不会在前后加任何内容，原文透传给 VLM。请写完整指令（含输出格式）。",
            "enabled": "配置层总开关。关闭后 PromptRunner 不再调度本检查项。如需临时手动暂停（误报排查期），请到事件详情页操作。",
            "notify_on_hit": "命中时是否通过 Home Assistant 发推送通知（总开关）。",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["weekdays"].required = False
        # target_classes initial：新建时默认 ["baby"]；编辑时按 instance 解析
        if self.instance and getattr(self.instance, "target_classes", None):
            self.initial["target_classes"] = [
                s for s in self.instance.target_classes.split(",") if s
            ]
        else:
            self.initial["target_classes"] = ["baby"]

        #: 非阻塞警告：不参与 ``is_valid()``，由视图转 messages / 模板直接展示
        self.warnings: list[str] = []
        # 编辑已有配置：进页面就提示，不必先提交一次才发现
        if self.instance and getattr(self.instance, "pk", None):
            stale = _missing_targets(
                getattr(self.instance, "target_classes", "") or "",
                getattr(self.instance, "prompt", "") or "",
            )
            if stale:
                self.warnings.append(_target_warning(stale))

        self._init_notify_targets()
        self._init_audio_rule()

    # ------------------------------------------------------------------
    # 通知目标：每个目标一行「☑ 目标名 + 条件下拉」（spec §5.6 / Phase 6）
    # ------------------------------------------------------------------
    def _init_notify_targets(self) -> None:
        """为每个 NotifyTarget 动态生成一组 (启用 checkbox, 条件下拉) 字段。

        为什么不用 CheckboxSelectMultiple：隐式 M2M 表达不了 per-(Prompt, 目标)
        的 `condition`（spec §5.6），必须走显式 through 表。
        """
        self._targets = list(NotifyTarget.objects.all().order_by("kind", "name"))
        links: dict[int, str] = {}
        if self.instance is not None and self.instance.pk:
            links = {
                link.notify_target_id: link.condition
                for link in PromptNotifyTarget.objects.filter(
                    prompt_config_id=self.instance.pk,
                )
            }
        for t in self._targets:
            self.fields[f"nt_{t.pk}_enabled"] = forms.BooleanField(
                label=t.name, required=False,
                widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
                initial=t.pk in links,
            )
            self.fields[f"nt_{t.pk}_condition"] = forms.ChoiceField(
                label="条件", required=False,
                choices=PromptNotifyTarget.CONDITION_CHOICES,
                initial=links.get(t.pk, PromptNotifyTarget.CONDITION_ALWAYS),
                widget=forms.Select(attrs={"class": "form-select form-select-sm"}),
            )

    @property
    def notify_target_rows(self) -> list[dict]:
        """模板渲染用：``[{target, enabled_field, condition_field}, ...]``。"""
        return [
            {
                "target": t,
                "enabled_field": self[f"nt_{t.pk}_enabled"],
                "condition_field": self[f"nt_{t.pk}_condition"],
            }
            for t in getattr(self, "_targets", [])
        ]

    # ------------------------------------------------------------------
    # 声音规则 initial
    # ------------------------------------------------------------------
    def _init_audio_rule(self) -> None:
        if self.instance is None or not self.instance.pk:
            return
        rule = PromptAudioRule.objects.filter(
            prompt_config_id=self.instance.pk,
        ).first()
        if rule is None:
            return
        self.fields["audio_rule_enabled"].initial = rule.enabled
        self.fields["audio_rule_condition"].initial = rule.condition
        self.fields["audio_rule_window_sec"].initial = rule.window_sec
        self.fields["audio_rule_min_event_count"].initial = rule.min_event_count

    # ------------------------------------------------------------------
    # 保存：模型本体 + through 行 + 声音规则（同一事务）
    # ------------------------------------------------------------------
    def save(self, commit=True):
        # 三步一起原子提交：Prompt 本体 + through 行 + 声音规则。
        # 否则任一步失败会留下"Prompt 改了、通知目标/声音规则没改"的半成品状态。
        with transaction.atomic():
            obj = super().save(commit=commit)
            if commit:
                self._sync_notify_targets(obj)
                self._sync_audio_rule(obj)
        return obj

    def _sync_notify_targets(self, obj) -> None:
        """按提交结果重建 through 行（勾选条件变化也要更新）。"""
        selected: dict[int, str] = {}
        for t in self._targets:
            if not self.cleaned_data.get(f"nt_{t.pk}_enabled"):
                continue
            cond = self.cleaned_data.get(f"nt_{t.pk}_condition") or (
                PromptNotifyTarget.CONDITION_ALWAYS
            )
            selected[t.pk] = cond

        # 1) 删掉取消勾选的
        PromptNotifyTarget.objects.filter(prompt_config=obj).exclude(
            notify_target_id__in=selected.keys(),
        ).delete()
        # 2) 新增 / 更新条件
        existing = {
            link.notify_target_id: link
            for link in PromptNotifyTarget.objects.filter(prompt_config=obj)
        }
        for target_id, cond in selected.items():
            link = existing.get(target_id)
            if link is None:
                PromptNotifyTarget.objects.create(
                    prompt_config=obj, notify_target_id=target_id, condition=cond,
                )
            elif link.condition != cond:
                link.condition = cond
                link.save(update_fields=["condition", "updated_at"])

    def _sync_audio_rule(self, obj) -> None:
        """同步 PromptAudioRule；未启用且从未配过 → 不建行。

        字段缺省（表单没提交该字段 / 提交空值）时：已有行**保留原值**，
        不拿默认值覆盖用户配置；新建行才用默认值。
        """
        enabled = bool(self.cleaned_data.get("audio_rule_enabled"))
        rule = PromptAudioRule.objects.filter(prompt_config=obj).first()
        if rule is None and not enabled:
            return
        is_new = rule is None
        if is_new:
            rule = PromptAudioRule(prompt_config=obj)
        rule.enabled = enabled
        condition = self.cleaned_data.get("audio_rule_condition")
        if condition:
            rule.condition = condition
        elif is_new:
            rule.condition = PromptAudioRule.CONDITION_ANY
        window_sec = self.cleaned_data.get("audio_rule_window_sec")
        if window_sec:
            rule.window_sec = window_sec
        elif is_new:
            rule.window_sec = 60
        min_count = self.cleaned_data.get("audio_rule_min_event_count")
        if min_count:
            rule.min_event_count = min_count
        elif is_new:
            rule.min_event_count = 1
        rule.save()

    def clean_target_classes(self):
        raw = self.cleaned_data.get("target_classes") or []
        # checkbox 全不勾（POST 没传 / 用户清空）→ 兜底 "baby"
        if not raw:
            return "baby"
        valid_keys = {k for k, _ in _TARGET_CHOICES}
        bad = [x for x in raw if x not in valid_keys]
        if bad:
            raise forms.ValidationError(f"未知类别：{bad}")
        # 排序 + 去重 → 存逗号分隔
        return ",".join(sorted(set(raw)))

    def clean(self):
        cleaned = super().clean()
        # 重算警告：绑定表单以本次提交为准（清掉 __init__ 里按 instance 得到的旧警告）
        self.warnings = []
        missing = _missing_targets(
            cleaned.get("target_classes") or "", cleaned.get("prompt") or "",
        )
        if missing:
            self.warnings.append(_target_warning(missing))
        return cleaned

    def clean_weekdays(self):
        v = (self.cleaned_data.get("weekdays") or "").strip()
        if not v:
            return "1,2,3,4,5,6,7"
        parts = [p.strip() for p in v.split(",") if p.strip()]
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise forms.ValidationError("星期必须为 1-7 逗号分隔")
        if any(n < 1 or n > 7 for n in nums):
            raise forms.ValidationError("星期数字必须在 1-7 之间")
        return ",".join(str(n) for n in nums)


class NotifyTargetForm(forms.ModelForm):
    """C5：通知目标 CRUD（修复 B5 死路）。"""

    class Meta:
        model = NotifyTarget
        fields = ["name", "kind", "target_id", "enabled"]
        widgets = {
            "kind": forms.Select,
        }
help_texts = {
            "name": "显示名（如 dad_phone / kitchen_speaker）。",
            "kind": "mobile_app = HA /api/services/notify/<target_id>；speaker = HA /api/services/text/set_value。",
            "target_id": "mobile_app: HA notify service 全名（如 notify.mobile_app_dad，会自动剥前缀拼 URL）；"
                         "speaker: HA 实体 ID（如 text.kitchen_speaker）。",
            "enabled": "关闭后此目标不出现在任何通知列表里。",
        }
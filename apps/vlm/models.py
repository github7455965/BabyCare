"""
Step 5 数据模型：VLMPromptConfig + VLMCheckState。

设计要点（v6）
--------------
1. 不复用 08 的 Event 模型；命中即 1 条 VLMCheckState 记录
2. 不引入 trigger_interval_sec / cooldown_sec 节流
3. **误报静默规则**（用户最终确认版）：
   - 用户标某条 (cam, prompt, status) 为误报 → 后续 N 条同状态签名静默
   - 任一条件打破 → 静默失效：
     a) status 变了（不同警报）
     b) 没触发（即中间有空档没产生该警报）
     c) 连续静默次数已达 N
4. failure_reason 6 值枚举：VLM 调用失败原因
5. 时间用 int 秒（与 FrameQueue 一致）；窗口 ts 列表存 JSONField

字段对照
--------
设计/计划文档：[工程计划.md §5](../../工程计划.md) + [设计.md 数据模型 ER](../../设计.md)
实装文件：apps/vlm/models.py（本文件）

不存 DismissedState 表（v1）：静默状态由 VLMCheckState 时间序列推断，无需单独表。
"""

from __future__ import annotations

from django.db import models


class VLMPromptConfig(models.Model):
    """单个 VLM 检查项的配置（每行 = 一个"判定 prompt" 或 "描述 prompt"）。

    字段含义详见 [设计.md 数据模型 ER](../../设计.md) §2.

    kind 字段（Step 7 增强）：
    - "judge"（默认）：判断型；解析首字符匹配 positive_keyword；命中报警
    - "describe"：描述型；status 存全文；hit=False 不报警；状态跟踪核心
    """

    KIND_CHOICES = [
        ("judge", "判断（命中报警）"),
        ("describe", "描述（状态跟踪）"),
    ]

    # ----- 提示词 -----
    name = models.CharField("名称", max_length=64, unique=True)
    kind = models.CharField(
        "类型", max_length=16, choices=KIND_CHOICES, default="judge",
        help_text="judge=判断型（命中报警）；describe=描述型（状态跟踪）",
    )
    prompt = models.TextField("完整 prompt")
    positive_keyword = models.CharField("命中关键词", max_length=64, default="是")
    image_filename_hint = models.CharField(
        "文件名提示", max_length=64, default="拍摄时间",
        help_text="让 VLM 用文件名当时间参考的提示语",
    )
    result_format = models.CharField(
        "返回格式", max_length=16, default="plain",
        help_text="plain=纯文本命中关键词；json=预留",
    )

    # ----- 触发控制 -----
    enabled = models.BooleanField("启用", default=True)
    manual_paused = models.BooleanField("手动暂停", default=False)
    camera_ids = models.ManyToManyField(
        "streaming.Camera",
        verbose_name="限定摄像头",
        related_name="prompt_configs",
        blank=True,
        help_text="限定生效摄像头；空 = 所有摄像头",
    )
    window_sec = models.IntegerField(
        "窗口秒数", default=10,
        help_text="v6 cursor 必需；范围 5~30",
    )

    # ----- 时间窗 -----
    time_window_enabled = models.BooleanField("启用时段", default=False)
    time_window_start = models.TimeField("时段起点", null=True, blank=True)
    time_window_end = models.TimeField("时段终点", null=True, blank=True)
    weekdays = models.CharField(
        "星期", max_length=32, default="1,2,3,4,5,6,7",
        help_text="1-7 逗号分隔；7=周日",
    )

    # ----- VLM 参数 -----
    timeout_sec_first = models.IntegerField("冷启 timeout(s)", default=60)
    timeout_sec_normal = models.IntegerField("正常 timeout(s)", default=15)
    max_tokens = models.IntegerField(
        "VLM max_tokens", default=512,
        help_text=(
            "VLM 单次回答最大 token 数（默认 512 覆盖状态描述；"
            "判断型可调小，描述型可调大）"
        ),
    )
    target_classes = models.CharField(
        "目标类别",
        max_length=128,
        default="baby",
        help_text=(
            "逗号分隔；可选项 baby / person / cat。"
            "多选时 prompt 文案必须涵盖所有 target（如 '图中有宝宝或猫吗？'），"
            "否则 VLM 看到的是未提及的 target，会一直 miss。"
        ),
    )

    # ----- 误报静默 -----
    silence_count_after_dismiss = models.IntegerField(
        "误报后静默次数", default=5,
        help_text=(
            "用户标某条 (cam, prompt, status) 为误报后，"
            "连续 N 次同状态签名触发则静默。"
            "任一条件打破则静默失效："
            "(a) status 变化；(b) 中间有空档不触发；(c) 连续静默次数达 N。"
        ),
    )

    # ----- 通知（B5 多目标）-----
    notify_on_hit = models.BooleanField(
        "命中时通知（总开关）", default=True,
        help_text="总开关。关闭后即使配了 notify_targets 也不发通知。",
    )
    notify_targets = models.ManyToManyField(
        "vlm.NotifyTarget",
        verbose_name="通知目标",
        related_name="prompts",
        blank=True,
        through="vlm.PromptNotifyTarget",
        help_text=(
            "命中时要通知的目标；空 = 不通知。与 notify_on_hit AND 关系。"
            "每个目标带独立 condition（always / audio_rule），见 PromptNotifyTarget。"
        ),
    )

    # ----- 通知文案模板（页面可配；空 = 内置默认）-----
    notify_title_template = models.CharField(
        "通知标题模板", max_length=128, blank=True, default="",
        help_text=(
            "留空 = 默认「【宝宝告警·{prompt}】」（手机通知的标题）。"
            "占位符：{prompt} 检查项名 / {cam} 摄像头 / {time} 时间 / {status} VLM 回答的状态"
        ),
    )
    notify_body_template = models.CharField(
        "通知正文模板", max_length=255, blank=True, default="",
        help_text=(
            "留空 = 默认「{cam} 在 {time} 检出（{status}）」。同一份文案用于手机正文与音箱播报；"
            "{status} 为空时，它外面那对括号会一起省略（不再出现「未提供状态」）。"
        ),
    )

    # ----- 元数据 -----
    extra = models.JSONField("扩展", default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "VLM 检查项"
        verbose_name_plural = "VLM 检查项"
        ordering = ["id"]
        db_table = "vlm_prompt_config"

    def __str__(self) -> str:
        paused = " [暂停]" if self.manual_paused else ""
        return f"#{self.pk} {self.name}{paused}"


class NotifyTarget(models.Model):
    """通知目标（Step 9+B5）。可被多个 PromptConfig 引用。

    kind=mobile_app → 走 HA notify.mobile_app_xxx 服务
    kind=speaker    → 走 HA text.set_value 实体（小爱音箱等）

    注意：target_id 必须含完整 service/entity 路径（mobile_app: notify.mobile_app_xxx；
    speaker: text.xiaomi_xxx）。
    """

    KIND_CHOICES = [
        ("mobile_app", "手机通知（HA mobile_app）"),
        ("speaker", "音箱播报（HA text.set_value）"),
    ]

    name = models.CharField("别名", max_length=64, unique=True)
    kind = models.CharField("类型", max_length=16, choices=KIND_CHOICES)
    target_id = models.CharField(
        "HA 服务/实体 ID", max_length=128,
        help_text="mobile_app 填 notify.mobile_app_xxx；speaker 填 text.xiaomi_xxx",
    )
    enabled = models.BooleanField("启用", default=True)
    extra = models.JSONField("扩展", default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "vlm_notify_target"
        verbose_name = "通知目标"
        verbose_name_plural = "通知目标"
        ordering = ["kind", "name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.kind})"


class VLMCheckState(models.Model):
    """每次 VLM 调用的状态记录（一行 = 一次 VLM 请求的处理结果）。

    字段含义详见 [设计.md 数据模型 ER](../../设计.md) §2.
    """

    FAILURE_REASON_CHOICES = [
        ("timeout", "超时"),
        ("network_error", "网络错误"),
        ("parse_error", "解析错误"),
        ("no_baby", "窗口内无 baby"),
        ("manual_paused", "已被手动暂停"),
        ("out_of_window", "窗口已被 FrameQueue 淘汰"),
        ("startup_failed", "llama-server 启失败"),
    ]

    camera = models.ForeignKey(
        "streaming.Camera", on_delete=models.SET_NULL, related_name="vlm_states",
        null=True, blank=True,
    )
    prompt_config = models.ForeignKey(
        VLMPromptConfig, on_delete=models.CASCADE, related_name="check_states",
    )

    # ----- 窗口信息（v6） -----
    ts_list = models.JSONField(
        "窗口 ts 列表", default=list,
        help_text="窗口内所有 ts（int 列表，按时间升序；window_start = ts_list[0]）",
    )
    window_sec = models.IntegerField("窗口长度", help_text="冗余便于查；= len(ts_list)")

    detected_at = models.DateTimeField("检测时间", auto_now_add=True)

    # ----- 命中信息 -----
    hit = models.BooleanField("是否命中")
    raw_response = models.TextField("VLM 原始返回", blank=True, default="")
    status = models.TextField(
        "状态描述", blank=True, default="",
        help_text=(
            "判断型：首字符后剩余描述（短）；"
            "描述型：VLM 全文（可能长）。"
            "判断型的 status 用于静默签名 (cam, prompt, status)。"
        ),
    )

    # ----- 图片（按 ts 正序；int 秒 ts） -----
    img1 = models.CharField("最早帧路径", max_length=512, blank=True, default="")
    img2 = models.CharField("中间帧路径", max_length=512, blank=True, default="")
    img3 = models.CharField("最晚帧路径", max_length=512, blank=True, default="")
    img1_ts = models.IntegerField("最早帧 ts", null=True, blank=True)
    img2_ts = models.IntegerField("中间帧 ts", null=True, blank=True)
    img3_ts = models.IntegerField("最晚帧 ts", null=True, blank=True)

    # ----- 流程标记 -----
    has_target_in_window = models.BooleanField(
        "窗口内有目标", default=False,
        help_text="窗口内任一 target_classes 命中为 True；旧名 has_baby_in_window（v2 rename）",
    )
    failure_reason = models.CharField(
        "失败原因", max_length=32, blank=True, default="",
        choices=FAILURE_REASON_CHOICES,
        help_text="VLM 调用失败原因；非空时不报警也不静默",
    )
    retry_count = models.IntegerField("重试次数", default=0)

    # ----- 通知 / 误报 -----
    notified = models.BooleanField("已通知", default=False)
    dismissed_as_false = models.BooleanField(
        "用户标记误报", default=False,
        help_text="用户手动标记为误报；触发后续 N 条同状态静默",
    )
    auto_silenced = models.BooleanField(
        "被静默", default=False,
        help_text="因前一条同状态被标误报，本次触发被静默（不报警）",
    )

    # ----- 元数据 -----
    extra = models.JSONField("扩展", default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "VLM 检查记录"
        verbose_name_plural = "VLM 检查记录"
        ordering = ["-id"]
        db_table = "vlm_check_state"
        # 注：MySQL 会自动给 FK 字段建索引
        # (camera_id, prompt_config_id)；误报静默判定的"按 (cam, prompt, status)"查询
        # 走该索引即可（status 在 SELECT/WHERE 中过滤，量小）。v2 数据量大后再加专用索引。

    def __str__(self) -> str:
        if self.failure_reason:
            mark = f"FAIL[{self.failure_reason}]"
        elif self.auto_silenced:
            mark = "silent"
        elif self.hit:
            mark = "HIT"
        else:
            mark = "miss"
        return (
            f"#{self.pk} {mark} cam={self.camera_id} "
            f"prompt={self.prompt_config_id} status='{self.status}'"
        )


class VLMQueuedTask(models.Model):
    """llama-server 不可用期间缓存的 VLM 请求；Drainer 恢复后逐条回放。

    与 VLMCheckState 关系
    --------------------
    - VLMQueuedTask = 「待处理」队列（llama 关闭期间入队）
    - VLMCheckState  = 「已完成」记录（drainer 处理成功后写一条 + 本行 status='done'）

    Runner 在调 VLM 前若发现 llama-off，会把窗口 (cam, prompt, ts_list, frames) 落到
    MEDIA_ROOT/frames/...（复用 frame_storage.save_frame），再插一行 status=pending 到本表。
    DrainerThread 后台轮询本表 pending 行，依次调 llama + 写 VLMCheckState。
    """

    STATUS_CHOICES = [
        ("pending", "待处理"),
        ("done",    "已完成"),
        ("failed",  "失败（超过重试上限）"),
    ]

    camera = models.ForeignKey(
        "streaming.Camera", on_delete=models.SET_NULL,
        related_name="vlm_queued_tasks", null=True, blank=True,
    )
    prompt_config = models.ForeignKey(
        VLMPromptConfig, on_delete=models.CASCADE,
        related_name="queued_tasks",
    )

    ts_list = models.JSONField(
        "窗口 ts 列表", default=list,
        help_text="窗口内所有 ts（int 列表，按时间升序；window_start = ts_list[0]）",
    )
    window_sec = models.IntegerField("窗口长度")

    img1 = models.CharField("最早帧路径", max_length=512, blank=True, default="")
    img2 = models.CharField("中间帧路径", max_length=512, blank=True, default="")
    img3 = models.CharField("最晚帧路径", max_length=512, blank=True, default="")
    img1_ts = models.IntegerField("最早帧 ts", null=True, blank=True)
    img2_ts = models.IntegerField("中间帧 ts", null=True, blank=True)
    img3_ts = models.IntegerField("最晚帧 ts", null=True, blank=True)

    has_target_in_window = models.BooleanField("窗口内有目标", default=False)

    status = models.CharField(
        "状态", max_length=16, choices=STATUS_CHOICES, default="pending",
        help_text="pending=待处理；done=Drainer 已完成；failed=超过重试上限放弃",
    )
    retry_count = models.IntegerField("已重试次数", default=0)
    last_error = models.TextField("最近错误", blank=True, default="")

    created_at = models.DateTimeField("入队时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "VLM 队列任务"
        verbose_name_plural = "VLM 队列任务"
        ordering = ["id"]
        db_table = "vlm_queued_task"
        indexes = [
            models.Index(fields=["status", "created_at"], name="vlmqueued_status_created_idx"),
        ]

    def __str__(self) -> str:
        return (
            f"#{self.pk} {self.status} cam={self.camera_id} "
            f"prompt={self.prompt_config_id} retry={self.retry_count}"
        )


# ---------------------------------------------------------------------------
# Step 15 Phase 6：Prompt 声音规则 + 通知目标条件（spec §5.5 / §5.6 / §5.7）
# ---------------------------------------------------------------------------
class PromptAudioRule(models.Model):
    """Prompt 的声音条件（OneToOne，spec §5.5）。

    **不持有阈值**：标签判阳阈值统一在 `.env`、且按模型分开配置（spec §4.3.1）。
    本表只表达"用哪些标签 + 判定窗口 + 次数"。

    判定结果是三态 `SATISFIED` / `NOT_SATISFIED` / `UNKNOWN`（spec §7.1），
    由 :func:`apps.vlm.audio_rule.evaluate_audio_rule` 计算；本表只存配置。
    """

    CONDITION_ANY = "any"
    CONDITION_CRY = "cry"
    CONDITION_SPEECH = "speech"
    CONDITION_CRY_OR_SPEECH = "cry_or_speech"
    CONDITION_NO_CRY = "no_cry"
    CONDITION_NO_SPEECH = "no_speech"
    CONDITION_COUNT_CRY = "count_cry"
    CONDITION_COUNT_SPEECH = "count_speech"
    CONDITION_CHOICES = [
        (CONDITION_ANY, "窗口内存在任一声音事件"),
        (CONDITION_CRY, "存在哭声"),
        (CONDITION_SPEECH, "存在说话声"),
        (CONDITION_CRY_OR_SPEECH, "哭声或说话声任一"),
        (CONDITION_NO_CRY, "没有哭声"),
        (CONDITION_NO_SPEECH, "没有说话声"),
        (CONDITION_COUNT_CRY, "哭声事件数达标"),
        (CONDITION_COUNT_SPEECH, "说话声事件数达标"),
    ]

    prompt_config = models.OneToOneField(
        VLMPromptConfig, on_delete=models.CASCADE,
        related_name="audio_rule", verbose_name="Prompt",
    )
    enabled = models.BooleanField(
        "启用声音条件", default=False,
        help_text="关闭 / 未配置 → 音频判为 UNKNOWN，不参与通知筛选（spec §7.1）。",
    )
    condition = models.CharField(
        "声音条件", max_length=24, choices=CONDITION_CHOICES, default=CONDITION_ANY,
    )
    window_sec = models.PositiveIntegerField(
        "判定窗口(秒)", default=60,
        help_text="Prompt 命中时刻往前看多少秒的 AudioEvent。",
    )
    min_event_count = models.PositiveIntegerField(
        "最少事件数", default=1,
        help_text="仅 count_cry / count_speech 使用。",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Prompt 声音规则"
        verbose_name_plural = "Prompt 声音规则"
        db_table = "vlm_prompt_audio_rule"

    def __str__(self) -> str:
        flag = "" if self.enabled else " [未启用]"
        return f"prompt#{self.prompt_config_id} {self.condition}{flag}"


class PromptNotifyTarget(models.Model):
    """``VLMPromptConfig.notify_targets`` 的**显式 through 表**（spec §5.6）。

    旧实现是隐式 M2M，只能表达"通知给谁"，表达不了"在什么条件下通知"。
    这里给每一对 (Prompt, 目标) 增加 ``condition``：

    - ``always``：prompt 命中就发，不看音频；
    - ``audio_rule``：仅当音频三态为 ``SATISFIED`` 时发（``UNKNOWN`` 时放行，spec §7.2）。

    迁移分三步（spec §5.6）：建表 → raw SQL 回填旧中间表（默认 ``always``）→ 切 through。
    """

    CONDITION_ALWAYS = "always"
    CONDITION_AUDIO_RULE = "audio_rule"
    CONDITION_CHOICES = [
        (CONDITION_ALWAYS, "总是发送"),
        (CONDITION_AUDIO_RULE, "声音条件满足时发送"),
    ]

    prompt_config = models.ForeignKey(
        VLMPromptConfig, on_delete=models.CASCADE,
        related_name="notify_target_links", verbose_name="Prompt",
    )
    notify_target = models.ForeignKey(
        NotifyTarget, on_delete=models.CASCADE,
        related_name="prompt_links", verbose_name="通知目标",
    )
    condition = models.CharField(
        "发送条件", max_length=16, choices=CONDITION_CHOICES, default=CONDITION_ALWAYS,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Prompt 通知目标"
        verbose_name_plural = "Prompt 通知目标"
        db_table = "vlm_prompt_notify_target"
        ordering = ["prompt_config", "notify_target"]
        constraints = [
            models.UniqueConstraint(
                fields=["prompt_config", "notify_target"],
                name="vlm_pnt_prompt_target_uq",
            ),
        ]

    def __str__(self) -> str:
        return (
            f"prompt#{self.prompt_config_id} → target#{self.notify_target_id} "
            f"({self.condition})"
        )


class PromptNotificationDelivery(models.Model):
    """通知投递审计行（spec §7.2）：一次 Prompt 命中 × 一个目标的判定/投递结果。

    ``audio_state`` 用**三值**（而不是 bool），事后能区分：

    - ``not_satisfied``：确认"这一刻没有声音"，所以不发；
    - ``unknown``：音频没开 / 覆盖不完整，**不参与筛选、照发**。

    本表**不参与去重、也不抑制后续命中**——只是审计留痕。
    """

    AUDIO_SATISFIED = "satisfied"
    AUDIO_NOT_SATISFIED = "not_satisfied"
    AUDIO_UNKNOWN = "unknown"
    AUDIO_STATE_CHOICES = [
        (AUDIO_SATISFIED, "声音条件满足"),
        (AUDIO_NOT_SATISFIED, "确认不满足"),
        (AUDIO_UNKNOWN, "音频不可用（不筛选）"),
    ]

    prompt_config = models.ForeignKey(
        VLMPromptConfig, on_delete=models.CASCADE,
        related_name="notification_deliveries", verbose_name="Prompt",
    )
    vlm_state = models.ForeignKey(
        VLMCheckState, on_delete=models.CASCADE,
        related_name="notification_deliveries", verbose_name="VLM 事件",
    )
    audio_event = models.ForeignKey(
        "audio_detect.AudioEvent", on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="notification_deliveries", verbose_name="参考音频事件",
    )
    notify_target = models.ForeignKey(
        NotifyTarget, on_delete=models.CASCADE,
        related_name="notification_deliveries", verbose_name="通知目标",
    )
    audio_state = models.CharField(
        "音频三态", max_length=16, choices=AUDIO_STATE_CHOICES,
        default=AUDIO_UNKNOWN,
    )
    condition = models.CharField("目标条件", max_length=16, blank=True, default="")
    delivered = models.BooleanField("已发送", default=False)
    error = models.TextField("发送错误", blank=True, default="")
    created_at = models.DateTimeField("记录时间", auto_now_add=True)

    class Meta:
        verbose_name = "通知投递记录"
        verbose_name_plural = "通知投递记录"
        db_table = "vlm_prompt_notification_delivery"
        ordering = ["-id"]
        indexes = [
            models.Index(
                fields=["prompt_config", "created_at"],
                name="vlm_pnd_prompt_created_idx",
            ),
            models.Index(fields=["vlm_state"], name="vlm_pnd_state_idx"),
        ]

    def __str__(self) -> str:
        return (
            f"state#{self.vlm_state_id} target#{self.notify_target_id} "
            f"{self.audio_state} delivered={self.delivered}"
        )


class VLMCheckStateAudioEvent(models.Model):
    """VLM 事件 ↔ 参考音频事件 的关联 + 判定快照（spec §5.7 / §7.1）。

    不在 ``VLMCheckState.status`` 里拼大段音频描述；只在关联表和 ``snapshot_json``
    里保存可审计上下文（三态结果、判定窗口、覆盖情况、命中规则）。

    ``audio_event`` 可空：阴性 / UNKNOWN 结论没有"参考事件"，但仍写一行快照留痕。
    """

    vlm_state = models.ForeignKey(
        VLMCheckState, on_delete=models.CASCADE,
        related_name="audio_event_links", verbose_name="VLM 事件",
    )
    audio_event = models.ForeignKey(
        "audio_detect.AudioEvent", on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="vlm_state_links", verbose_name="音频事件",
    )
    snapshot_json = models.JSONField("判定快照", default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "VLM 事件音频关联"
        verbose_name_plural = "VLM 事件音频关联"
        db_table = "vlm_check_state_audio_event"
        ordering = ["-id"]

    def __str__(self) -> str:
        return f"state#{self.vlm_state_id} ← audio#{self.audio_event_id}"

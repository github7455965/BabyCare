"""音频检测数据模型。

Phase 1 落 `AudioRuntimeState`（采集健康状态）+ Phase 2 落 `SoundDetectionLog`
（推理窗口，稀疏落库）；后续阶段增量补 `AudioEvent`（Phase 3）/
`AudioServiceState`（Phase 5）。

时间约定
--------
项目 `USE_TZ = False`（全本地 naive 时间）。覆盖完整性判定与窗口时间戳统一用
**epoch 秒**（`coverage_ok_from_ts` / `window_start_ts`），与 VLM 侧 `ts_list`
同一时间基（spec §5.2/§7.1）。
"""

from __future__ import annotations

from django.db import models
from django.utils import timezone


class AudioRuntimeState(models.Model):
    """每个摄像头的音频采集健康状态（spec §5.4）。

    写入方：音频 worker（独立进程 + 独立 venv）
    读取方：web 侧只读（控制页 / 健康检查）

    心跳语义（spec §9.3）
    -------------------
    worker 每 `BABYCARE_AUDIO_HEARTBEAT_SEC`（默认 2s）更新一次 `updated_at`；
    web 侧 `now - updated_at > BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC`（默认 10s）
    即判定"进程死了或卡住了"。

    覆盖完整性（spec §3.3）
    ---------------------
    `coverage_ok_from_ts` = "从这一刻起音频持续正常，中间没断过"。
    注意：**不能**用这张表的 `updated_at` 来判断覆盖——它是心跳，不是连续性凭证。
    """

    STATUS_DISABLED = "disabled"
    STATUS_INITIALIZING = "initializing"
    STATUS_READY = "ready"
    STATUS_NO_AUDIO_PROFILE = "no_audio_profile"
    STATUS_CAPTURE_ERROR = "capture_error"
    STATUS_MODEL_ERROR = "model_error"
    STATUS_DEGRADED = "degraded"
    STATUS_STOPPED = "stopped"

    STATUS_CHOICES = [
        (STATUS_DISABLED, "已禁用"),
        (STATUS_INITIALIZING, "初始化中"),
        (STATUS_READY, "采集中"),
        (STATUS_NO_AUDIO_PROFILE, "无音频轨"),
        (STATUS_CAPTURE_ERROR, "采集异常"),
        (STATUS_MODEL_ERROR, "模型异常"),
        (STATUS_DEGRADED, "降级运行"),
        (STATUS_STOPPED, "已停止"),
    ]

    camera = models.OneToOneField(
        "streaming.Camera",
        on_delete=models.CASCADE,
        related_name="audio_runtime_state",
        verbose_name="摄像头",
    )
    status = models.CharField(
        "状态", max_length=32, choices=STATUS_CHOICES, default=STATUS_DISABLED,
    )
    last_packet_at = models.DateTimeField("最近收到音频", null=True, blank=True)
    coverage_ok_from_ts = models.BigIntegerField(
        "覆盖正常起点(epoch秒)", default=0,
        help_text="从该时刻起音频采集持续正常、中间没断过；用于覆盖完整性判定。",
    )
    last_error = models.TextField("最近错误", blank=True, default="")
    # Phase 2 加载模型后回填；Phase 1 恒为空
    yamnet_version = models.CharField("YAMNet 版本", max_length=64, blank=True, default="")
    panns_version = models.CharField("PANNs 版本", max_length=64, blank=True, default="")
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "音频运行状态"
        verbose_name_plural = "音频运行状态"
        db_table = "vlm_audio_runtime_state"

    def __str__(self) -> str:
        return f"cam#{self.camera_id} {self.status}"


class SoundDetectionLog(models.Model):
    """一个推理窗口的记录（spec §5.1）。

    **不是每个窗口都落库。** 完整落库 = 1 行/秒/摄像头（3 路约 26 万行/天），
    加上三个 JSON 字段，数据库撑不住。落库策略（:meth:`log_reason`）：

    ============================== ===================================
    条件                            说明
    ============================== ===================================
    窗口为共识阳性                  必须落
    近阈值窗（分数 ≥ 阈值 × 0.7）     落，用于事后调阈值（P0-5）
    状态变化窗                      落，用于审计
    其它普通窗                      **只在内存滚动，不落库**
    ============================== ===================================

    因此"覆盖完整性"**不能依赖本表**，必须用 :class:`AudioRuntimeState`
    的 ``coverage_ok_from_ts``（spec §3.3/§5.1）。
    """

    DECISION_POSITIVE = "positive"
    DECISION_NEGATIVE = "negative"
    DECISION_DEGRADED = "degraded"
    DECISION_CHOICES = [
        (DECISION_POSITIVE, "共识阳性"),
        (DECISION_NEGATIVE, "非阳性"),
        (DECISION_DEGRADED, "降级"),
    ]

    LOG_POSITIVE = "positive"
    LOG_NEAR_THRESHOLD = "near_threshold"
    LOG_STATE_CHANGE = "state_change"
    LOG_REASON_CHOICES = [
        (LOG_POSITIVE, "共识阳性"),
        (LOG_NEAR_THRESHOLD, "近阈值"),
        (LOG_STATE_CHANGE, "状态变化"),
    ]

    camera = models.ForeignKey(
        "streaming.Camera",
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="sound_detection_logs",
        verbose_name="摄像头",
        help_text="摄像头删除后保留日志（spec §5.1）。",
    )
    window_start_ts = models.BigIntegerField("窗口开始(epoch秒)")
    window_end_ts = models.BigIntegerField("窗口结束(epoch秒)")
    yamnet_payload = models.JSONField("YAMNet 输出", default=dict, blank=True)
    panns_payload = models.JSONField("PANNs 输出", default=dict, blank=True)
    consensus_payload = models.JSONField("共识结果", default=dict, blank=True)
    decision = models.CharField(
        "窗口决策", max_length=16, choices=DECISION_CHOICES, default=DECISION_NEGATIVE,
        help_text="多标签时：任一标签 positive → positive。",
    )
    log_reason = models.CharField(
        "落库原因", max_length=24, choices=LOG_REASON_CHOICES, blank=True, default="",
    )
    failure_reason = models.CharField("失败原因", max_length=64, blank=True, default="")
    created_at = models.DateTimeField("写入时间", auto_now_add=True)

    class Meta:
        verbose_name = "声音检测日志"
        verbose_name_plural = "声音检测日志"
        db_table = "vlm_sound_detection_log"
        ordering = ["-window_end_ts", "-id"]
        indexes = [
            models.Index(fields=["camera", "window_end_ts"], name="vlm_sdl_cam_end_idx"),
            models.Index(fields=["camera", "created_at"], name="vlm_sdl_cam_created_idx"),
        ]

    def __str__(self) -> str:
        return f"cam#{self.camera_id} {self.window_end_ts} {self.decision}"


class AudioEvent(models.Model):
    """一个完整声音事件（spec §5.2/§6.2，Phase 3）。

    生命周期：状态机启动事件时**先落一行 `recording`**，事件关闭时回填
    `ended_at_ts` / 录音路径 / 静音区间 → `pending_description`（Phase 4 再
    推进 describing/completed/failed）。worker 崩溃会留下 recording 行，
    由 `EventAssembler` 启动时回收成 failed。

    时间约定
    --------
    - 判定、比较、关联一律用 `started_at_ts` / `ended_at_ts`（**epoch 秒**，
      与 `SoundDetectionLog` / `coverage_ok_from_ts` 同基）；
    - `started_at` / `ended_at`（DateTimeField）是**仅供页面展示的冗余**。
    - **事件时长 = 真实声音 + 3s 后录尾巴**；4B 分段必须用真实起止，不能拿
      录音时长当内容边界（spec §6.2）。

    录音区间 vs 事件区间
    -------------------
    录音从"启动时刻前 pre_roll 5s"开始（必然覆盖事件起点，P0-7 结论 4），
    `recording_lead_sec` = 录音起点比 `started_at_ts` 早多少秒——分段/展示时
    用它把录音内偏移换算成事件内偏移。
    """

    STATUS_RECORDING = "recording"
    STATUS_PENDING_DESCRIPTION = "pending_description"
    STATUS_DESCRIBING = "describing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_RECORDING, "录制中"),
        (STATUS_PENDING_DESCRIPTION, "待描述"),
        (STATUS_DESCRIBING, "描述中"),
        (STATUS_COMPLETED, "已完成"),
        (STATUS_FAILED, "失败"),
    ]

    DESC_PENDING = "pending"
    DESC_PROCESSING = "processing"
    DESC_COMPLETED = "completed"
    DESC_FAILED = "failed"
    DESC_STATUS_CHOICES = [
        (DESC_PENDING, "待处理"),
        (DESC_PROCESSING, "处理中"),
        (DESC_COMPLETED, "已完成"),
        (DESC_FAILED, "失败"),
    ]

    camera = models.ForeignKey(
        "streaming.Camera",
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="audio_events",
        verbose_name="摄像头",
        help_text="摄像头删除后保留事件（spec §5.2）。",
    )
    status = models.CharField(
        "事件状态", max_length=24, choices=STATUS_CHOICES, default=STATUS_RECORDING,
    )
    started_at_ts = models.BigIntegerField("事件开始(epoch秒)")
    ended_at_ts = models.BigIntegerField(
        "事件结束(epoch秒)", null=True, blank=True,
        help_text="事件关闭时回填；= 最后一个阳性窗结束 + 后录 3s。",
    )
    # 冗余展示字段（USE_TZ=False，本地 naive）
    started_at = models.DateTimeField("开始(展示用)", null=True, blank=True)
    ended_at = models.DateTimeField("结束(展示用)", null=True, blank=True)
    duration_sec = models.FloatField("录音时长(秒)", default=0.0)
    audio_path = models.CharField(
        "音频文件绝对路径", max_length=512, blank=True, default="",
        help_text="与 VLMCheckState.img1/2/3 约定一致，存绝对路径不存 URL。",
    )
    audio_format = models.CharField("音频格式", max_length=8, default="flac")
    #: ``{label: {first_positive_ts, last_positive_ts, positive_windows: [[s,e],...]}}``
    detected_labels = models.JSONField("检测标签及时间线", default=dict, blank=True)
    #: ``[[offset_start, offset_end], ...]``，相对事件起点（started_at_ts）的秒偏移
    silence_ranges = models.JSONField("静音区间", default=list, blank=True)
    degraded = models.BooleanField(
        "降级标记", default=False,
        help_text="事件期间发生采集中断/模型异常（capture_gap），非静音结束。",
    )
    recording_lead_sec = models.FloatField(
        "录音前导(秒)", default=0.0,
        help_text="录音起点比事件起点早多少秒；事件内偏移 = 录音内偏移 - 该值。",
    )
    description_status = models.CharField(
        "描述状态", max_length=16, choices=DESC_STATUS_CHOICES, default=DESC_PENDING,
    )
    description_json = models.JSONField("结构化描述", default=dict, blank=True)
    description_raw = models.TextField("4B 原始返回", blank=True, default="")
    moss_model = models.CharField("描述模型", max_length=64, blank=True, default="")
    worker_epoch = models.CharField(
        "产出实例", max_length=32, blank=True, default="", db_index=True,
        help_text="产出该事件的 worker 租约 epoch；recover 据此限定回收范围。",
    )
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "音频事件"
        verbose_name_plural = "音频事件"
        db_table = "vlm_audio_event"
        ordering = ["-started_at_ts", "-id"]
        indexes = [
            # §7.1 按窗口查事件是 Prompt 命中的关键路径
            models.Index(fields=["camera", "started_at_ts"], name="vlm_ae_cam_start_idx"),
            models.Index(fields=["camera", "ended_at_ts"], name="vlm_ae_cam_end_idx"),
        ]

    def __str__(self) -> str:
        return f"cam#{self.camera_id} {self.started_at_ts}-{self.ended_at_ts} {self.status}"


class AudioEventSegment(models.Model):
    """事件超过单次 4B 描述长度时拆的片段（spec §5.3，Phase 3 建模型、Phase 4 使用）。

    分段规则（spec §6.4）：目标 10–60s，优先在静音点切；无静音点超 90s 按 60s
    强制切；<1s 丢弃。**分段失败时整个事件不能标记完成**，保留错误状态支持重试。
    """

    audio_event = models.ForeignKey(
        AudioEvent,
        on_delete=models.CASCADE,
        related_name="segments",
        verbose_name="音频事件",
    )
    sequence = models.PositiveIntegerField("片段序号")
    start_offset = models.FloatField(
        "起点偏移(秒)", help_text="相对事件 started_at_ts。",
    )
    end_offset = models.FloatField("终点偏移(秒)")
    audio_path = models.CharField(
        "分段音频绝对路径", max_length=512, blank=True, default="",
        help_text="FLAC 落盘；请求 4B 前在内存转 WAV，不落中间文件。",
    )
    has_silence_before = models.BooleanField("分段前有静音点", default=False)
    description_status = models.CharField(
        "描述状态", max_length=16, choices=AudioEvent.DESC_STATUS_CHOICES,
        default=AudioEvent.DESC_PENDING,
    )
    description_json = models.JSONField("片段描述", default=dict, blank=True)
    retry_count = models.IntegerField("重试次数", default=0)
    last_error = models.TextField("最近错误", blank=True, default="")
    worker_epoch = models.CharField(
        "产出实例", max_length=32, blank=True, default="", db_index=True,
        help_text="产出该分段的 worker 租约 epoch；recover 据此限定回收范围。",
    )
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "音频事件分段"
        verbose_name_plural = "音频事件分段"
        db_table = "vlm_audio_event_segment"
        ordering = ["audio_event", "sequence"]
        constraints = [
            models.UniqueConstraint(
                fields=["audio_event", "sequence"], name="vlm_aes_event_seq_uq",
            ),
        ]

    def __str__(self) -> str:
        return f"event#{self.audio_event_id} seg{self.sequence} {self.start_offset}-{self.end_offset}s"


class AudioServiceState(models.Model):
    """音频线**全局单行**状态（spec §5.4，Phase 5）：用户期望状态 + worker 实例租约。

    固定只有一行（``pk = SINGLETON_PK``）。

    为什么需要它
    ------------
    1. **用户期望状态**：用户手动关掉音频线后，重启 daphne 不能又自动拉起
       （``desired`` 持久化，spec §9.3）；
    2. **实例互斥**：音频线是独立进程，同一时刻只允许一个实例。本行的
       ``pid`` / ``worker_epoch`` / ``updated_at`` 构成一把租约锁，第二个实例
       启动时直接拒绝（见 :mod:`apps.audio_detect.worker_lock`）；
    3. **归属审计**：``worker_epoch`` 同时写进 :class:`AudioEvent` /
       :class:`AudioEventSegment`；recover 只回收"本次接管前那个租约"的遗留行，
       不属于该租约的行（例如另一个异常存活的实例）一律不动。
    """

    SINGLETON_PK = 1

    DESIRED_ON = "on"
    DESIRED_OFF = "off"
    DESIRED_CHOICES = [(DESIRED_ON, "开启"), (DESIRED_OFF, "关闭")]

    REASON_MANUAL = "manual"
    REASON_AUTOSTART = "autostart"
    REASON_CRASH = "crash"
    REASON_CHOICES = [
        (REASON_MANUAL, "用户手动"),
        (REASON_AUTOSTART, "自动启动"),
        (REASON_CRASH, "异常退出"),
    ]

    desired = models.CharField(
        "用户期望状态", max_length=8, choices=DESIRED_CHOICES, default=DESIRED_OFF,
    )
    reason = models.CharField(
        "最近变更来源", max_length=16, choices=REASON_CHOICES, blank=True, default="",
    )
    pid = models.IntegerField(
        "worker PID", null=True, blank=True,
        help_text="非空 = 当前有活跃租约，释放时清空；心跳超时即可被接管。",
    )
    worker_epoch = models.CharField(
        "活跃实例标识", max_length=32, blank=True, default="", db_index=True,
        help_text="最近一次租约的 epoch；worker 每次接管时生成新的，释放时保留（供 recover 界定范围）。",
    )
    updated_at = models.DateTimeField(
        "租约更新时间", default=timezone.now,
        help_text="worker 心跳刷新；超过心跳超时即判定旧实例已死/卡死，允许接管。",
    )

    class Meta:
        verbose_name = "音频服务状态"
        verbose_name_plural = "音频服务状态"
        db_table = "vlm_audio_service_state"

    def __str__(self) -> str:
        return f"desired={self.desired} pid={self.pid} epoch={self.worker_epoch or '-'}"

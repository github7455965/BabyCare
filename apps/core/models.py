"""apps.core 的持久化模型。

目前只有一个：:class:`LLMHealthState` —— 两条 llama 服务（VLM :8082 / ASR :8136）
的**熔断状态**。业务逻辑在 :mod:`apps.core.llm_breaker`。
"""

from __future__ import annotations

from django.db import models


class LLMHealthState(models.Model):
    """LLM 服务的熔断状态。**每个 service 一行**。

    为什么落库（而不是只放内存）
    ---------------------------
    两条线的熔断状态其实都只在**单进程**内使用：

    - VLM(:8082) 只被 daphne 进程里的 `PromptRunner` / `DrainerThread` 调；
    - ASR(:8136) 只被 `audio_worker` 进程里的 `DescribeService` 调。

    所以进程内单例就够用，落库**不是**为了跨进程一致性。真正的理由是
    **可观测性**：控制页跑在 daphne 进程里，但 ASR 的状态在 `audio_worker`
    进程里 —— 不落库，页面就永远看不到"ASR 已经熔断 12 分钟、攒了 340 条"。

    顺带解决"进程重启后计数清零、又立刻连试 N 次"（代价本来也小，能省就省）。

    双写与容错
    ----------
    状态以**内存为准**（读写都在锁内，快且不依赖 DB），DB 只做"尽力而为"的镜像：
    写库失败只记日志，**绝不影响**采集/检测/通知链路（spec §7.4 的同款原则）。
    """

    SERVICE_VLM = "vlm"
    SERVICE_ASR = "asr"
    SERVICE_CHOICES = [
        (SERVICE_VLM, "VLM 视觉（:8082）"),
        (SERVICE_ASR, "ASR 音频描述（:8136）"),
    ]

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"
    STATE_CHOICES = [
        (STATE_CLOSED, "正常"),
        (STATE_OPEN, "熔断中"),
        (STATE_HALF_OPEN, "探测中"),
    ]

    service = models.CharField(
        "服务", max_length=16, unique=True, choices=SERVICE_CHOICES,
    )
    state = models.CharField(
        "熔断状态", max_length=16, choices=STATE_CHOICES, default=STATE_CLOSED,
    )
    consecutive_failures = models.IntegerField("连续连接失败次数", default=0)
    last_failure_at = models.DateTimeField("最近一次失败", null=True, blank=True)
    last_probe_at = models.DateTimeField(
        "最近一次探测发出", null=True, blank=True,
        help_text="半开态放出的那条探测请求的发出时间。",
    )
    last_success_at = models.DateTimeField("最近一次成功", null=True, blank=True)
    last_error = models.TextField("最近错误", blank=True, default="")
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        verbose_name = "LLM 服务熔断状态"
        verbose_name_plural = "LLM 服务熔断状态"

    def __str__(self) -> str:  # pragma: no cover - 仅调试用
        return f"{self.service}:{self.state}(fail={self.consecutive_failures})"

"""LLM 服务熔断器（分体部署，见 `docs/superpowers/specs/2026-09-16-split-deploy-plan.md` Phase 1）。

解决什么问题
------------
推理机不是 7×24 常开，两条 llama 线（VLM :8082 / ASR :8136）会**长时间不可达**。
现状下这有两种坏结果：

- **VLM 线**：网络错走 `LlamaNetworkError` 分支（`apps/vlm/runner.py:334-348`），
  写 `llama_unreachable` 后 `mark_done` —— **窗口被直接丢弃**，不入队；
- **ASR 线**：`_describe_segment` 把所有异常记进同一个 `retry_count`
  （`apps/audio_detect/describer.py:838-844`），8 分钟就耗尽 → **事件永久 failed**。

熔断器的职责只有一件：**知道"服务现在不可用"，让调用方别再白试**。
攒队列、回放、通知都由各自的业务模块负责（本模块不认识 `VLMQueuedTask` / `AudioEvent`）。

三态语义
--------
= ============ ==================================================================
  状态          行为
= ============ ==================================================================
``closed``    正常放行。连续失败达阈值 → ``open``
``open``      一律拦截，**不发送请求**。距上次失败满 ``cooldown_sec`` → ``half_open``
``half_open`` **只放一条**探测（置 inflight 防重复放行）。成功 → ``closed``；
              失败 → 回 ``open`` 并重新计时
= ============ ==================================================================

**探活就是真实请求**（用户定）：没有独立的 health 轮询线程，半开时放出去的那条
正常业务请求同时充当探测。

两条必须遵守的调用约定
----------------------
1. **只对"连接类失败"调 :meth:`record_failure`** —— timeout / connect error / 5xx。
   解析错、校验错、4xx 属于**内容问题**，既不是服务不可用，也不该触发熔断
   （否则会把 bug 藏起来，还白停一条可用的服务）。
2. 半开态要**短超时 + 不重试**：调用方在发请求前问 :meth:`effective_timeout_sec`。
   否则探测自己就把自己堵死 —— ASR 侧 `30s × (2+1) = 最坏 90s`，
   VLM 侧 `timeout_sec_first=60s` 且 `LlamaClient._chat_lock` 全局串行。

可测性
------
`clock` 可注入（默认 `time.monotonic`，不受墙钟跳变影响）；`persist=False`
可关掉落库。因此状态机单测不需要 DB、也不需要 sleep。

与 spec §7.4 的关系
-------------------
熔断是**降级能力**，不能反过来把主链路拖死：所以落库是"尽力而为"，
任何 DB 异常只记日志（:meth:`_persist`）。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

#: 熔断状态（与 `LLMHealthState.STATE_*` 一致，这里不 import 模型以便无 DB 使用）
STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

#: 半开探测"发出后迟迟无回报"的兜底：超过这么久仍没 record_* → 允许再放一条。
#: 防的是调用方拿到 allow_attempt()=True 后崩了/没发出去，把 inflight 永久占住。
_STUCK_PROBE_MIN_SEC = 30.0


class LLMBreaker:
    """单个服务的熔断器（进程内单例，见 :meth:`for_service`）。"""

    _instances: dict[str, "LLMBreaker"] = {}
    _cls_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 构造 / 单例
    # ------------------------------------------------------------------
    def __init__(
        self,
        service: str,
        fail_threshold: int = 3,
        cooldown_sec: float = 60.0,
        probe_timeout_sec: float = 5.0,
        enabled: bool = True,
        persist: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            service: ``LLMHealthState.SERVICE_VLM`` / ``SERVICE_ASR``
            fail_threshold: 连续连接类失败多少次 → 熔断打开
            cooldown_sec: 打开后多久放一条探测
            probe_timeout_sec: 半开态下给调用方用的短超时
            enabled: ``False`` → 完全旁路（永远放行、不计数）
            persist: 是否把状态镜像到 `LLMHealthState`（单测可关）
            clock: 单调时钟（可注入假时钟；**不要**用 `time.time`）
        """
        self._service = service
        self._fail_threshold = max(1, int(fail_threshold))
        self._cooldown = float(cooldown_sec)
        self._probe_timeout_sec = float(probe_timeout_sec)
        self._enabled = bool(enabled)
        self._persist_enabled = bool(persist)
        self._now = clock

        self._lock = threading.Lock()
        self._state = STATE_CLOSED
        self._consecutive = 0
        self._probe_inflight = False
        #: 本实例是否已经把状态同步过 DB（首次成功/失败时强制写一次）。
        #: 没有它的话：进程重启后内存是全新的 CLOSED，而 DB 里可能留着上一轮的
        #: `open` 行 —— `record_success` 只在"有变化"时落库，稳态下永远不写，
        #: 那条陈旧行就**永远不会被纠正**，控制页一直显示"熔断中"骗人。
        self._synced = False
        # 单调时刻（判断间隔用）
        self._last_failure_mono = 0.0
        self._last_probe_mono = 0.0
        # 墙钟（给页面看，落库用）
        self._last_failure_at = None
        self._last_probe_at = None
        self._last_success_at = None
        self._last_error = ""

    @classmethod
    def for_service(cls, service: str) -> "LLMBreaker":
        """取（或建）某服务的进程内单例；配置从 Django settings 读一次。

        第一次调用会读 `BABYCARE_LLM_BREAKER_*`；之后即使 settings 变了也不重建
        （与 `LlamaClient.instance()` 同款约定：启动期配置固定）。
        """
        inst = cls._instances.get(service)
        if inst is not None:
            return inst
        with cls._cls_lock:
            inst = cls._instances.get(service)
            if inst is not None:
                return inst
            from django.conf import settings

            # 每条线可以有自己的阈值（代价不对称，见 config/settings.py 的注释）；
            # 没配就回落到通用默认 `BABYCARE_LLM_FAIL_THRESHOLD`。
            threshold = getattr(
                settings, f"BABYCARE_LLM_FAIL_THRESHOLD_{service.upper()}", None,
            )
            if threshold is None:
                threshold = getattr(settings, "BABYCARE_LLM_FAIL_THRESHOLD", 3)
            inst = cls(
                service=service,
                fail_threshold=int(threshold),
                cooldown_sec=float(
                    getattr(settings, "BABYCARE_LLM_BREAKER_COOLDOWN_SEC", 60)
                ),
                probe_timeout_sec=float(
                    getattr(settings, "BABYCARE_LLM_PROBE_TIMEOUT_SEC", 5)
                ),
                enabled=bool(
                    getattr(settings, "BABYCARE_LLM_BREAKER_ENABLED", True)
                ),
            )
            cls._instances[service] = inst
            return inst

    @classmethod
    def reset_instances(cls) -> None:
        """清掉单例缓存（单测用；也便于将来"改完配置不强重启"）。"""
        with cls._cls_lock:
            cls._instances.clear()

    # ------------------------------------------------------------------
    # 属性（只读）
    # ------------------------------------------------------------------
    @property
    def service(self) -> str:
        return self._service

    @property
    def probe_timeout_sec(self) -> float:
        return self._probe_timeout_sec

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # 门禁
    # ------------------------------------------------------------------
    def allow_attempt(self) -> bool:
        """要不要发这次请求。

        - ``closed`` → ``True``
        - ``open`` 且冷却未到 → ``False``
        - ``open`` 且冷却已到 → 转 ``half_open``、**放这一条**，``True``
        - ``half_open`` 已有探测在飞 → ``False``（严格单条）
        """
        with self._lock:
            if not self._enabled:
                return True
            now = self._now()
            if self._state == STATE_CLOSED:
                return True
            if self._state == STATE_OPEN:
                if now - self._last_failure_mono >= self._cooldown:
                    self._enter_half_open_locked(now)
                    return True
                return False
            # half_open
            if not self._probe_inflight:
                self._probe_inflight = True
                self._last_probe_mono = now
                self._touch_probe_at_locked()
                return True
            stuck_after = max(_STUCK_PROBE_MIN_SEC, self._probe_timeout_sec * 3)
            if now - self._last_probe_mono >= stuck_after:
                logger.warning(
                    "[breaker] %s 探测无回报已 %.0fs → 重新放一条",
                    self._service, now - self._last_probe_mono,
                )
                self._last_probe_mono = now
                self._touch_probe_at_locked()
                return True
            return False

    def effective_timeout_sec(self, normal_timeout: float) -> float:
        """当前这一格如果是探测，就把超时压到 ``probe_timeout_sec``。

        调用方发请求前问一次即可。**探测不复用正常超时** —— 探测必须便宜，
        否则它自己就会把链路堵住（见模块 docstring 的调用约定 2）。
        """
        with self._lock:
            if self._probe_window_locked():
                return float(self._probe_timeout_sec)
            return float(normal_timeout)

    def is_probing(self) -> bool:
        """当前这一次尝试算不算"半开探测"（调用方据此决定要不要省掉重试）。

        探测必须**便宜** —— 既压短超时（:meth:`effective_timeout_sec`），
        也要**不重试**（否则默认 `30s × (1+2) = 最坏 90s`，压了超时也白搭）。
        """
        with self._lock:
            return self._probe_window_locked()

    def _probe_window_locked(self) -> bool:
        """当前是否处于"这一格是探测"的窗口（调用方须持锁）。

        两种入口都算：

        - ``half_open``：标准探测态（:meth:`allow_attempt` 已把状态迁移过来）；
        - ``open`` **且冷却已过**：`is_blocking()` 放行的那一格。

        第二种必须自己认出来，不能指望状态已变成 ``half_open``：`is_blocking()`
        **刻意无副作用**（控制页、"要不要跳过"门槛也用它，不能在查询里改状态），
        所以它不会替调用方完成 `open → half_open` 的迁移。漏了这个分支的话，
        冷却后那条探测会拿到**正常超时**（ASR 最坏 90s / VLM 60s），
        短超时保护形同虚设。
        """
        if not self._enabled:
            return False
        if self._state == STATE_HALF_OPEN:
            return True
        if self._state == STATE_OPEN:
            return (self._now() - self._last_failure_mono) >= self._cooldown
        return False

    def is_blocking(self) -> bool:
        """当前是否会拦截请求（**无副作用**：不会放探测、不改状态）。

        给"只想看状态"的调用方（控制页、回放门槛）用；要真发请求请用
        :meth:`allow_attempt`。
        """
        with self._lock:
            if not self._enabled:
                return False
            if self._state == STATE_OPEN:
                return (self._now() - self._last_failure_mono) < self._cooldown
            if self._state == STATE_HALF_OPEN:
                return self._probe_inflight
            return False

    # ------------------------------------------------------------------
    # 回报
    # ------------------------------------------------------------------
    def record_success(self) -> None:
        """调用成功后调。**只在状态/计数有变化时落库**，稳态下零 DB 写。"""
        with self._lock:
            if not self._enabled:
                return
            from django.utils import timezone

            self._last_success_at = timezone.now()
            changed = self._state != STATE_CLOSED or self._consecutive != 0
            if self._state == STATE_HALF_OPEN:
                logger.info("[breaker] %s 探测成功 → 熔断关闭", self._service)
            elif self._state == STATE_OPEN:
                logger.info("[breaker] %s 恢复（未见探测）→ 熔断关闭", self._service)
            self._state = STATE_CLOSED
            self._consecutive = 0
            self._probe_inflight = False
            self._last_error = ""
            # `not self._synced` → 本实例的第一次成功一定要写：把可能残留的
            # 陈旧 `open` 行纠正回来（否则那条行永远不会被覆盖）
            if changed or not self._synced:
                self._persist_locked()

    def record_failure(self, error: object = "") -> None:
        """**连接类**失败后调（timeout / connect error / 5xx）。

        内容类失败（解析错 / 校验错 / 4xx）**不要**调这里，见模块 docstring。
        """
        with self._lock:
            if not self._enabled:
                return
            from django.utils import timezone

            now = self._now()
            self._consecutive += 1
            self._last_failure_mono = now
            self._last_failure_at = timezone.now()
            self._last_error = str(error)[:2000]
            was_half_open = self._state == STATE_HALF_OPEN
            self._probe_inflight = False

            if was_half_open:
                # 探测失败 → 回 open，重新计冷却
                self._state = STATE_OPEN
                logger.warning(
                    "[breaker] %s 探测失败 → 继续熔断（再等 %.0fs）: %s",
                    self._service, self._cooldown, self._last_error,
                )
                self._persist_locked()
                return

            if self._consecutive >= self._fail_threshold:
                just_opened = self._state != STATE_OPEN
                self._state = STATE_OPEN
                if just_opened:
                    logger.warning(
                        "[breaker] %s 连续失败 %d 次 → 熔断打开（冷却 %.0fs）: %s",
                        self._service, self._consecutive, self._cooldown,
                        self._last_error,
                    )
                self._persist_locked()
            # 未达阈值：还在 closed，不落库（避免每个失败都写一次）

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """给控制页/状态接口用的只读快照。"""
        with self._lock:
            return {
                "service": self._service,
                "state": self._state,
                "enabled": self._enabled,
                "consecutive_failures": self._consecutive,
                "fail_threshold": self._fail_threshold,
                "cooldown_sec": self._cooldown,
                "probe_timeout_sec": self._probe_timeout_sec,
                "blocking": self._blocking_locked(),
                "last_failure_at": _iso(self._last_failure_at),
                "last_probe_at": _iso(self._last_probe_at),
                "last_success_at": _iso(self._last_success_at),
                "last_error": self._last_error,
            }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _blocking_locked(self) -> bool:
        if not self._enabled:
            return False
        if self._state == STATE_OPEN:
            return (self._now() - self._last_failure_mono) < self._cooldown
        if self._state == STATE_HALF_OPEN:
            return self._probe_inflight
        return False

    def _enter_half_open_locked(self, now: float) -> None:
        self._state = STATE_HALF_OPEN
        self._probe_inflight = True
        self._last_probe_mono = now
        self._touch_probe_at_locked()
        logger.info("[breaker] %s 冷却结束 → 放一条探测", self._service)

    def _touch_probe_at_locked(self) -> None:
        from django.utils import timezone

        self._last_probe_at = timezone.now()
        self._persist_locked()

    def _persist_locked(self) -> None:
        """把内存状态镜像到 `LLMHealthState`（尽力而为，失败只记日志）。"""
        if not self._persist_enabled:
            return
        self._synced = True
        try:
            from .models import LLMHealthState

            LLMHealthState.objects.update_or_create(
                service=self._service,
                defaults={
                    "state": self._state,
                    "consecutive_failures": self._consecutive,
                    "last_failure_at": self._last_failure_at,
                    "last_probe_at": self._last_probe_at,
                    "last_success_at": self._last_success_at,
                    "last_error": self._last_error,
                },
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "[breaker] %s 状态落库失败（忽略，不影响链路）", self._service,
            )


def _iso(dt) -> str | None:
    return dt.isoformat() if dt is not None else None


__all__ = [
    "STATE_CLOSED",
    "STATE_HALF_OPEN",
    "STATE_OPEN",
    "LLMBreaker",
]

"""
DrainerThread（Step 14）：常驻模式下的后台回放线程。

职责
----
- 轮询 VLMQueuedTask.status='pending'（旧→新 FIFO）
- llama 没被 force_off 且子进程在跑 且 GPU 闲 → 拿出 1 条
- 读 3 帧 → 调 LlamaClient.chat → 写 VLMCheckState → 标 task done
- 失败 → retry_count++；超 BABYCARE_LLM_QUEUE_MAX_RETRIES → task='failed'

轮询节奏（吞吐关键）
--------------------
``_drain_one()`` 返回一个结果码（``DRAIN_*``），主循环按结果决定等多久：

- ``DRAIN_DONE``  刚成功消费一条 → **不等待**，立刻抢下一条（drainer 自己就能连着跑满）；
- ``DRAIN_RETRY`` 取到了但处理失败（那条仍是 pending）→ 等一个完整间隔，
  免得把 ``MAX_RETRIES`` 次重试在毫秒内烧光；
- ``DRAIN_BUSY``  让位给 live 流量 → 短睡（``_BUSY_RETRY_SEC``）后再探；
- ``DRAIN_IDLE``  队列空 / llama 不可用 → 等一个完整间隔
  （``BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC``）。

**为什么不能"每次尝试都固定 sleep 一个间隔"**：老实现无论成败都睡 5s，连"让位跳过"
也要白等 5s，于是采样粒度 = 5s——只要采样瞬间撞上 live 请求就整轮作废，实测只能消化
2~8 条/分钟，而 llama 单次调用只有 ~1s（产能约 60 次/分钟）、GPU 利用率 52%。
把"成功→立刻继续"和"让位→短睡再探"分开后，drainer 才能吃满 live 流量之间的空隙。

**为什么连抢 N 条后要主动让一次**：drainer 与 live Runner 共用 GpuManager 的
``_vlm_request_pending`` **布尔**旗标，而 drainer 的 ``release_vlm_if_idle()`` 会把
live 侧同时置起的 pending 一并清掉——也就是说自己的 release 之后，那个旗标不再是
可信的"live 还在等"信号。所以不能无限连抢，用一个 burst 上限给 live 窗口留缝
（``_MAX_BURST`` 条 ≈ 10s 才让 0.5s，吞吐几乎不受影响）。

不调 VLM 条件（短路 _drain_one 直接返回 DRAIN_IDLE / DRAIN_BUSY）
------------------------------------------------------------------
- LlamaManager.is_forced_off()（手动关 / 自动重启窗口期）
- LlamaManager 子进程没在跑（_proc.poll() != None 或 None）
- GpuManager 有 vlm/yolo request_pending（让位给正常 Runner）

启动 / 关闭
-----------
- apps.vlm.apps.ready() 在 BABYCARE_LLM_RESIDENT=true 时调 start()
- signal handler / atexit 调 stop()
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from typing import Optional

from django.conf import settings


logger = logging.getLogger(__name__)


#: ``_drain_one`` 结果码（主循环据此决定等多久；语义见模块 docstring）
DRAIN_DONE = "done"      # 成功消费 1 条 → 立刻抢下一条，不等待
DRAIN_RETRY = "retry"    # 取到了但失败（那条仍是 pending）→ 等一个完整间隔
DRAIN_BUSY = "busy"      # 让位给 live 流量 → 短睡后再探
DRAIN_IDLE = "idle"      # 队列空 / llama 不可用 → 等一个完整间隔

#: 让位后的重探间隔（秒）。必须远小于 live 请求的占空比：老实现用整间隔重探，
#: 撞上 live 请求就白等一整轮，这正是队列消化慢的主因。
_BUSY_RETRY_SEC = 0.5

#: 连续消费多少条后主动让位一次（见模块 docstring「为什么连抢 N 条后要主动让一次」）
_MAX_BURST = 10

#: 超期收割的最小间隔（秒）。它是一次全表 UPDATE，不能每轮都跑
#: （`QUEUE_MAX_AGE_SEC <= 0` 时压根不跑，见 :meth:`DrainerThread._reap_expired`）。
_REAP_INTERVAL_SEC = 60.0


class DrainerThread:
    """常驻模式下后台轮询 + 回放 VLMQueuedTask 的单例。"""

    _instance: Optional["DrainerThread"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._stop: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started: bool = False
        self._lock = threading.Lock()
        #: 上次超期收割的 monotonic 时刻（限流用）
        self._last_reap: float = 0.0

    @classmethod
    def instance(cls) -> "DrainerThread":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="vlm-drainer", daemon=True,
            )
            self._thread.start()
            self._started = True
            logger.info("[drainer] started")

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._started = False
        logger.info("[drainer] stopped")

    def _loop(self) -> None:
        """按 ``_drain_one`` 的结果码决定等待时长（见模块 docstring「轮询节奏」）。"""
        interval = float(
            getattr(settings, "BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC", 5)
        )
        burst = 0
        while not self._stop.is_set():
            try:
                result = self._drain_one()
            except Exception:
                logger.exception("[drainer] loop error")
                result = DRAIN_IDLE

            if result == DRAIN_DONE:
                burst += 1
                if burst >= _MAX_BURST:
                    # 连抢了 _MAX_BURST 条 → 让一次再继续（live 流量共用布尔 pending 旗标）
                    burst = 0
                    self._stop.wait(timeout=_BUSY_RETRY_SEC)
                continue

            burst = 0
            if result == DRAIN_BUSY:
                self._stop.wait(timeout=_BUSY_RETRY_SEC)
            else:
                self._stop.wait(timeout=interval)

    def _drain_one(self) -> str:
        from apps.vlm.llama_manager import LlamaManager
        from apps.yolo_detect.gpu_manager import GpuManager
        from apps.vlm.models import VLMQueuedTask

        # 0) 超期收割 —— **必须排在所有"服务不可用"闸门之前**。
        #
        #    它最该生效的场景恰恰是"服务长期不在"（队列只增不减、MEDIA_ROOT
        #    一直涨，见 checklist §8.8）；而下面几个 return 正是在那种场景下
        #    提前跳过的。放闸门之后就等于**服务越不可用越不收割**，与设计意图相反。
        self._reap_expired()

        # 1) llama 不可用 → 跳过
        #    手动关（旗标）/ 远端不通（外部模式下 is_running 就是远端 health）/
        #    熔断打开（连撞 N 次连接类失败，冷却未到）。
        #    **半开态不拦** —— 那一格就是要放一条真实请求当探测，回放正好用得上。
        lm = LlamaManager.instance()
        if lm.is_forced_off() or not lm.is_running():
            return DRAIN_IDLE
        if self._breaker().is_blocking():
            return DRAIN_IDLE

        # 2) GPU 忙（normal Runner in-flight）→ 让位
        gpu = GpuManager.instance()
        with gpu._state_cond:
            if gpu._vlm_request_pending or gpu._yolo_request_pending:
                return DRAIN_BUSY

        # 3) 取 1 条 pending —— **newest-first**（用户 2026-09-16 定：
        #    告警时效 > 完整性）。旧行为是 FIFO（`order_by("id")`）：停 3 小时后
        #    回来先补 3 小时前的窗口，最新、还可能有意义的排最后。
        #    取不到的老任务由 `_reap_expired` 给显式出口。
        try:
            task = (
                VLMQueuedTask.objects
                .select_related("prompt_config")
                .filter(status="pending")
                .order_by("-id")
                .first()
            )
        except Exception:
            logger.exception("[drainer] query task failed")
            return DRAIN_IDLE
        if task is None:
            return DRAIN_IDLE

        # 4) acquire_vlm（resident 模式仅置 pending；exclusive 模式会走状态机）
        try:
            gpu.acquire_vlm()
        except Exception as e:
            logger.warning("[drainer] acquire_vlm failed task=%d: %s", task.id, e)
            self._mark_retry_or_fail(task, f"acquire_vlm: {e}")
            return DRAIN_RETRY

        # 5) acquire 完再判一次 llama（防 acquire 期间被 force_off）
        if lm.is_forced_off() or not lm.is_running():
            try:
                gpu.release_vlm_if_idle()
            except Exception:
                pass
            return DRAIN_IDLE

        try:
            ok = self._process_task(task)
        finally:
            try:
                gpu.release_vlm_if_idle()
            except Exception:
                logger.exception("[drainer] release_vlm_if_idle failed")
        return DRAIN_DONE if ok else DRAIN_RETRY

    def _reap_expired(self) -> int:
        """把老得不可能再回放的 pending 任务标 `failed`（`QUEUE_MAX_AGE_SEC`）。

        为什么必须有它
        -------------
        newest-first 之后，`_drain_one` 每次只取**最新**的一条：只要持续有新任务
        进来，**老任务永远取不到**，会永远挂在 `pending` —— 控制页"待回放"数字
        永不归零，也分不清"还在攒"还是"卡死"。这里给它们一个显式出口。

        状态复用现成的 `failed`（`VLMQueuedTask.STATUS_CHOICES` 只有
        pending/done/failed），原因写在 `last_error` 里，**不动状态机**。

        `QUEUE_MAX_AGE_SEC <= 0`（默认，用户定"先给 0，上线再调"）→ 不收割。
        非零时也限流到 `_REAP_INTERVAL_SEC` 一次 —— 它是全表 UPDATE。
        """
        max_age = float(
            getattr(settings, "BABYCARE_LLM_QUEUE_MAX_AGE_SEC", 0) or 0
        )
        if max_age <= 0:
            return 0
        now = time.monotonic()
        if now - self._last_reap < _REAP_INTERVAL_SEC:
            return 0
        self._last_reap = now

        from django.utils import timezone

        from apps.vlm.models import VLMQueuedTask

        cutoff = timezone.now() - timedelta(seconds=int(max_age))
        n = VLMQueuedTask.objects.filter(
            status="pending", created_at__lt=cutoff,
        ).update(
            status="failed",
            last_error=f"expired: 入队超过 {int(max_age)}s 仍未回放",
        )
        if n:
            logger.info(
                "[drainer] 超期收割 %d 条（created_at < %s）", n, cutoff,
            )
        return n

    def _breaker(self):
        """本进程的 VLM 熔断器（与 Runner 共用同一个进程内单例）。"""
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState

        return LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)

    def _process_task(self, task) -> bool:
        """处理一条任务。

        返回 ``True`` = 已写 VLMCheckState 且 task 标 done；``False`` = 失败（已计重试，
        那条仍是 pending）。调用方（``_drain_one``）据此决定主循环要不要立刻抢下一条。
        """
        from apps.vlm.frame_storage import load_frame_bytes
        from apps.vlm.llama_client import (
            LlamaClient,
            LlamaNetworkError,
            LlamaTimeoutError,
        )
        from apps.vlm.models import VLMCheckState
        from apps.vlm.notify import compute_auto_silenced

        media_root = settings.MEDIA_ROOT
        try:
            images = [
                load_frame_bytes(task.img1, media_root),
                load_frame_bytes(task.img2, media_root),
                load_frame_bytes(task.img3, media_root),
            ]
        except Exception as e:
            self._mark_retry_or_fail(task, f"load_frame: {e}")
            return False

        try:
            text = LlamaClient.instance().chat(
                images,
                task.prompt_config.prompt,
                task.prompt_config.max_tokens,
                # 半开态压短超时：探测不能自己把链路堵住（同 Runner，见 llm_breaker）
                timeout_sec=self._breaker().effective_timeout_sec(
                    task.prompt_config.timeout_sec_normal
                ),
            )
        except (LlamaNetworkError, LlamaTimeoutError) as e:
            # **连接类**失败 → 记熔断器。连撞 N 次后 `_drain_one` 会整段停下来等冷却，
            # 而不是把每条任务的 MAX_RETRIES 在毫秒内烧光（那会把整个队列判死）
            self._breaker().record_failure(f"chat: {e}")
            self._mark_retry_or_fail(task, f"chat: {e}")
            return False
        except Exception as e:
            # 内容/编排类失败（解析错等）**不记熔断器** —— 那不是服务不可用
            self._mark_retry_or_fail(task, f"chat: {e}")
            return False

        # 成功 → 关掉熔断 / 清零连续失败计数（稳态下零 DB 写）
        self._breaker().record_success()

        try:
            hit, status = LlamaClient.instance().parse(
                text,
                task.prompt_config.positive_keyword,
                task.prompt_config.result_format,
                task.prompt_config.kind,
            )
        except Exception as e:
            self._mark_retry_or_fail(task, f"parse: {e}")
            return False

        silence_count = getattr(
            task.prompt_config, "silence_count_after_dismiss", 0
        ) or 0
        auto_silenced = False
        if hit:
            auto_silenced = compute_auto_silenced(
                cam_id=task.camera_id,
                prompt_id=task.prompt_config_id,
                status=status,
                silence_count=silence_count,
            )

        VLMCheckState.objects.create(
            camera_id=task.camera_id,
            prompt_config_id=task.prompt_config_id,
            ts_list=list(task.ts_list),
            window_sec=task.window_sec,
            hit=hit,
            raw_response=text,
            status=status,
            img1=task.img1, img1_ts=task.img1_ts,
            img2=task.img2, img2_ts=task.img2_ts,
            img3=task.img3, img3_ts=task.img3_ts,
            has_target_in_window=task.has_target_in_window,
            failure_reason="",
            retry_count=0,
            auto_silenced=auto_silenced,
        )

        task.status = "done"
        task.save(update_fields=["status", "updated_at"])
        logger.info(
            "[drainer] processed task=%d cam=%d prompt=%d hit=%s",
            task.id, task.camera_id, task.prompt_config_id, hit,
        )
        return True

    def _mark_retry_or_fail(self, task, err: str) -> None:
        from apps.vlm.models import VLMQueuedTask

        max_retry = int(
            getattr(settings, "BABYCARE_LLM_QUEUE_MAX_RETRIES", 3)
        )
        task.retry_count += 1
        task.last_error = err[:500]
        if task.retry_count >= max_retry:
            task.status = "failed"
            task.save(update_fields=[
                "status", "retry_count", "last_error", "updated_at",
            ])
            logger.warning(
                "[drainer] task=%d failed after %d retries: %s",
                task.id, task.retry_count, err,
            )
        else:
            task.save(update_fields=[
                "retry_count", "last_error", "updated_at",
            ])
            logger.info(
                "[drainer] task=%d retry %d/%d: %s",
                task.id, task.retry_count, max_retry, err,
            )
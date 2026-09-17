"""
PromptRunner + PromptRunnerManager（Step 8）

职责
----
- PromptRunner: 单个 (cam, prompt) 的 asyncio task
    主循环:
        window = cursor.try_advance(frame_queue)
        if window is None: sleep 1s; continue
        if cursor.fail_count >= 2: skip_window; continue
        frames = FrameSelectorV2.select(window_ts_list, frame_queue)
        if frames is None: skip_window; continue   # 全窗口无 baby
        cold = cursor.consume_cold_start()
        timeout = first if cold else normal
        try:
            text = await loop.run_in_executor(None,
                lambda: llama_client.chat(images, prompt, max_tokens, timeout))
        except LlamaTimeoutError / LlamaNetworkError:
            cursor.inc_fail_count(); log; continue
        except LlamaParseError:
            log; save VLMCheckState(failure_reason=parse_error); continue
        cursor.reset_fail_count()
        hit, status = llama_client.parse(text, positive_keyword, result_format, kind)
        cursor.mark_done(window.max_ts)
        save VLMCheckState(hit, status, raw_response, ts_list, img1/2/3, has_baby)

- PromptRunnerManager: 单例；per (cam, prompt) 一份 Runner
    start() / stop() / add_camera(cam_id) / remove_camera(cam_id)
    add_prompt(prompt_id) / remove_prompt(prompt_id)
    stats()

设计要点
--------
- Manager 自带独立 daemon 线程 + asyncio event loop（run_forever）
  → 调用方（apps.ready / signal handler / HTTP 启停 API）完全同步
  → 与 Daphne 主 loop 解耦，不共享 loop
- 所有 Runner 的 task 通过 asyncio.run_coroutine_threadsafe(...) 提交到 Manager 的 loop
- 每个 (cam, prompt) 一个 task；N 个 task 并发跑，互不阻塞
- fail_count 是 cursor 的字段，Runner 只读 + 调 inc/reset（不存本地）
- LlamaClient 是单例全局（manager 自己持有一个），所有 Runner 共享
- VLMCheckState 写库走 Django ORM（同步；ORM 在线程池里调也是同步的）
  → 写库丢到 self.executor（ThreadPoolExecutor）

不在 Step 8 范围
-----------------
- 不做报警推送（Q2 选 A：只写库 + 日志）
- 不做静默签名判定（VLMCheckState.dismissed_as_false / auto_silenced 字段保留，留给后续 Step）
- 不做 time_window_enabled 时段检查（保留字段，留给后续 Step）
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from apps.vlm.cursor import PromptCursor, PromptCursorManager, WindowSpec
from apps.vlm.frame_selector import select_three_frames
from apps.vlm.frame_storage import encode_vlm_jpeg
from apps.vlm.llama_client import (
    LlamaClient,
    LlamaLoadingError,
    LlamaNetworkError,
    LlamaParseError,
    LlamaTimeoutError,
)
from apps.vlm.llama_manager import LlamaStartError


logger = logging.getLogger(__name__)


# Runner 主循环空轮询间隔（秒）；用户确认 B 方案
_POLL_INTERVAL_SEC = 1.0

# fail_count 阈值；>= 此值 Runner 跳过当前 window 不调 VLM
_FAIL_COUNT_SKIP_THRESHOLD = 2

# 共享线程池大小（llama.chat + ORM 写库都用这个）
_DEFAULT_POOL_SIZE = 8

# Runner stop / add 异步操作的同步等待上限（秒）
_ASYNC_OP_TIMEOUT_SEC = 10.0

# "等不到窗口"日志的最小间隔（秒）。
# 窗口凑不齐的典型原因是某个 target 的 has_<class> 一直是 None（YOLO 没推理到该类别，
# 如 cam 没绑对应模型 / 模型缓存陈旧），此前完全静默、只能靠 DB 里 0 条 state 猜。
_NO_WINDOW_LOG_SEC = 60.0

# 允许的 target_classes 取值（admin checkbox + 校验依据）
_VALID_TARGET_CLASSES = ("baby", "person", "cat")


def _parse_target_classes(raw: str) -> List[str]:
    """VLMPromptConfig.target_classes（逗号分隔字符串）→ 去重 + 排序 + 过滤非法值。

    兼容老数据（空串 / None → ["baby"] 兜底）。
    """
    if not raw:
        return ["baby"]
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    valid = sorted({p for p in parts if p in _VALID_TARGET_CLASSES})
    return valid or ["baby"]


# ---------------------------------------------------------------------------
# Runner 单实例数据
# ---------------------------------------------------------------------------
@dataclass
class _RunnerStats:
    """单个 Runner 的运行统计。"""
    invoke_count: int = 0          # 调过几次 llama
    success_count: int = 0         # 成功次数
    fail_count_total: int = 0      # 累计失败次数（不含被 skip_window 的）
    skip_window_count: int = 0     # 跳过 window 次数
    no_baby_count: int = 0         # 全窗口无 baby 次数
    last_invoke_ts: int = 0        # 上次调 llama 的 wall-clock 时间戳
    last_error: str = ""           # 上次错误信息（截断）


# ---------------------------------------------------------------------------
# PromptRunner：单个 (cam, prompt) 的 asyncio task
# ---------------------------------------------------------------------------
class PromptRunner:
    """单个 (cam, prompt) 的调度循环。

    由 PromptRunnerManager 拥有；外部不直接 new。
    """

    def __init__(
        self,
        cam_id: int,
        prompt_id: int,
        prompt_text: str,
        positive_keyword: str,
        result_format: str,
        kind: str,
        window_sec: int,
        max_tokens: int,
        timeout_sec_first: int,
        timeout_sec_normal: int,
        cursor: PromptCursor,
        frame_queue,                 # FrameQueue 实例
        llama_client: LlamaClient,
        executor: ThreadPoolExecutor,
        prompt_obj=None,                 # VLMPromptConfig / _FakePrompt（notify 用 silence_count / notify_on_hit）
        target_classes: Optional[List[str]] = None,
        poll_interval: float = _POLL_INTERVAL_SEC,
    ):
        self.cam_id = cam_id
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text
        self.positive_keyword = positive_keyword
        self.result_format = result_format
        self.kind = kind                  # "judge" | "describe"
        self.window_sec = window_sec
        self.max_tokens = max_tokens
        self.timeout_sec_first = timeout_sec_first
        self.timeout_sec_normal = timeout_sec_normal
        self.target_classes = target_classes or ["baby"]
        self._prompt_obj = prompt_obj     # notify / auto_silence 需要 silence_count + notify_on_hit

        self.cursor = cursor
        self.frame_queue = frame_queue
        self.llama_client = llama_client
        self.executor = executor
        self.poll_interval = poll_interval

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._started = False
        self.stats = _RunnerStats()
        #: 上次打"等不到窗口"日志的 monotonic 时刻（限流）
        self._last_no_window_log = 0.0

    # ------------------------------------------------------------------
    # 生命周期（由 Manager 在 Manager 的 loop 上调）
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """启动 Runner 的 asyncio task（必须在 Manager loop 上调）。"""
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._task = asyncio.create_task(
            self._run(), name=f"PromptRunner[{self.cam_id},{self.prompt_id}]"
        )
        logger.info("[runner] started cam=%d prompt=%d", self.cam_id, self.prompt_id)

    async def stop(self) -> None:
        """停止 Runner（设标志 + 等待 task 退出）。"""
        if not self._started:
            return
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[runner] stop timeout cam=%d prompt=%d; cancel",
                               self.cam_id, self.prompt_id)
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._started = False
        logger.info("[runner] stopped cam=%d prompt=%d", self.cam_id, self.prompt_id)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def _run(self) -> None:
        """Runner 主循环（Step 8 算法）。"""
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                # 1) cursor 决定该不该触发、看哪段窗口（v2: target_classes 决定就绪判定）
                window = self.cursor.try_advance(
                    self.frame_queue, self.target_classes,
                )
                if window is None:
                    self._log_no_window_once()
                    await self._sleep(self.poll_interval)
                    continue

                # 2) fail_count 超阈值 → 跳过当前 window（mark_done 推进 cursor + fail_count -1 衰减）
                #    语义：fail_count 是个"连续失败配额"——每次跳过消耗 1 次。
                #    mark_done 让 try_advance 找下一 window（避免反复在同一 window 上循环衰减）；
                #    衰减到 0 后下次窗口（已推进到新 ts）正常尝试；若 llama 已恢复 → 成功 →
                #    reset_fail_count 永久解锁；否则 inc_fail_count 再触发新一轮 skip。
                if self.cursor.fail_count >= _FAIL_COUNT_SKIP_THRESHOLD:
                    self.stats.skip_window_count += 1
                    self.cursor.fail_count = max(0, self.cursor.fail_count - 1)
                    self.cursor.mark_done(window.ts_list[-1])
                    logger.warning(
                        "[runner] skip window cam=%d prompt=%d fail_count=%d (start_ts=%d)",
                        self.cam_id, self.prompt_id, self.cursor.fail_count, window.start_ts,
                    )
                    await self._sleep(self.poll_interval)
                    continue

                # 3) 选 3 帧（None = 全窗口无 baby，不调 VLM）
                frames = self._select_frames(window)
                if frames is None:
                    # 全窗口无 baby：不调 VLM；写一条 no_baby 记录 + mark_done
                    self.stats.no_baby_count += 1
                    await loop.run_in_executor(
                        self.executor, self._save_no_baby, window,
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue

                # 4) 调 llama（带 cold_start 切 timeout）
                cold = self.cursor.consume_cold_start()
                timeout_sec = self.timeout_sec_first if cold else self.timeout_sec_normal
                # 半开态（熔断器刚放出的那条探测）必须**短超时**：正常是
                # 15s/60s，而 `LlamaClient._chat_lock` 是全局串行的 —— 探测不压
                # 超时的话，它自己就能把后面所有请求堵住（见 llm_breaker 模块说明）
                timeout_sec = self._breaker().effective_timeout_sec(timeout_sec)
                self.stats.invoke_count += 1
                self.stats.last_invoke_ts = int(time.time())

                # 4a) 现在不该发请求（手动关 / 熔断打开 / 远端不通）→ 入库不调
                if self._llama_is_off():
                    await loop.run_in_executor(
                        self.executor, self._enqueue, window, frames,
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue

                try:
                    # 重要：调 VLM 前必须 acquire_vlm（GPU 模式 exclusive：卸 YOLO + lazy 启
                    # llama-server）。这是 llama lazy 启动的唯一入口——之前 Runner 漏调导致
                    # 永远只能手动 POST /api/vlm/restart/。
                    await loop.run_in_executor(self.executor, self._acquire_vlm)
                    images = [self._frame_to_bytes(f) for f in frames]
                    try:
                        text = await loop.run_in_executor(
                            self.executor,
                            lambda: self.llama_client.chat(
                                images, self.prompt_text, self.max_tokens,
                                timeout_sec=timeout_sec,
                            ),
                        )
                    finally:
                        # 释放 VLM（GPU 模式切回 IDLE / parallel 清 pending）
                        await loop.run_in_executor(self.executor, self._release_vlm)
                    # 成功 → 关掉熔断 / 清零连续失败计数（稳态下零 DB 写）
                    self._breaker().record_success()
                except LlamaLoadingError as e:
                    # 503 Loading 是临时状态：mark_done 推进 cursor（避免同 window 死循环
                    # 重试）+ sleep 5s 后下个 cycle 再试；不写库、不计 fail_count。
                    logger.info(
                        "[runner] llama_loading cam=%d prompt=%d err=%s",
                        self.cam_id, self.prompt_id, e,
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(5.0)
                    continue
                except LlamaStartError as e:
                    # llama-server 起不来（路径错 / 端口冲突 / 健康探测超时），
                    # 或者**外部模式下远端不可达**（`_ensure_external` 抛的）。
                    # 不计 fail_count：这不是 VLM 调用失败，是可用性问题；
                    # 下次 Runner 触发还会重试 acquire_vlm。
                    #
                    # **行为变更（Phase 3）**：不再写 `startup_failed` 把窗口丢掉，
                    # 改成**入队等回放** —— 旧行为下 llama 长期起不来时，窗口会一直
                    # 静默蒸发（checklist §2 #8）。
                    self._breaker().record_failure(f"startup: {e}")
                    self.stats.fail_count_total += 1
                    self.stats.last_error = f"startup: {e}"[:200]
                    logger.warning(
                        "[runner] llama_startup_failed cam=%d prompt=%d err=%s → 入队等回放",
                        self.cam_id, self.prompt_id, e,
                    )
                    await loop.run_in_executor(
                        self.executor, self._enqueue, window, frames,
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue
                except LlamaTimeoutError as e:
                    # 超时也算"服务不可用"信号：记进熔断器（连撞 N 次后，后续窗口
                    # 会在 4a 被拦进队列，不再一个个白等超时）。
                    # 窗口本身仍按原有 fail_count 衰减策略处理（不改成入队）：
                    # 超时可能是"服务活着但慢"，反复重放同一窗会放大拥塞。
                    self._breaker().record_failure(f"timeout: {e}")
                    self.cursor.inc_fail_count()
                    self.stats.fail_count_total += 1
                    self.stats.last_error = f"timeout: {e}"[:200]
                    logger.warning(
                        "[runner] timeout cam=%d prompt=%d fail_count=%d err=%s",
                        self.cam_id, self.prompt_id, self.cursor.fail_count, e,
                    )
                    await loop.run_in_executor(
                        self.executor, self._save_failure,
                        window, frames, "llama_timeout", str(e),
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue
                except LlamaNetworkError as e:
                    # 连不上 / 5xx —— **连接类**失败，记进熔断器（连续 N 次 → 熔断打开，
                    # 后续窗口直接走 4a 的入队分支，不再白等超时）。
                    #
                    # **行为变更（Phase 3）**：窗口不再丢弃，改成**入队等回放**。
                    # 旧行为（写 `llama_unreachable` + mark_done）在分体部署下
                    # 会让"推理机没开"期间的窗口全部丢失（checklist §2 #8 / §8.2）。
                    self._breaker().record_failure(f"network: {e}")
                    self.stats.fail_count_total += 1
                    self.stats.last_error = f"network: {e}"[:200]
                    logger.warning(
                        "[runner] network_error cam=%d prompt=%d err=%s → 入队等回放",
                        self.cam_id, self.prompt_id, e,
                    )
                    await loop.run_in_executor(
                        self.executor, self._enqueue, window, frames,
                    )
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue
                except LlamaParseError as e:
                    # 解析错：不计入 fail_count（重试也救不了）；写 failure_reason
                    self.stats.fail_count_total += 1
                    self.stats.last_error = f"parse_error: {e}"[:200]
                    logger.warning(
                        "[runner] parse_error cam=%d prompt=%d err=%s",
                        self.cam_id, self.prompt_id, e,
                    )
                    await loop.run_in_executor(
                        self.executor, self._save_failure,
                        window, frames, "parse_error", str(e),
                    )
                    # 推进 cursor（这窗处理过了）
                    self.cursor.mark_done(window.ts_list[-1])
                    await self._sleep(self.poll_interval)
                    continue

                # 5) 解析 + 写库
                hit, status = self.llama_client.parse(
                    text, self.positive_keyword, self.result_format, self.kind,
                )
                self.cursor.reset_fail_count()
                self.stats.success_count += 1

                await loop.run_in_executor(
                    self.executor, self._save_success,
                    window, frames, hit, status, text,
                )
                self.cursor.mark_done(window.ts_list[-1])

                # 报警命中日志（Step 8 不推送）
                if hit:
                    logger.info(
                        "[runner] HIT cam=%d prompt=%d status='%s'",
                        self.cam_id, self.prompt_id, status,
                    )
                else:
                    logger.debug(
                        "[runner] miss cam=%d prompt=%d status='%s'",
                        self.cam_id, self.prompt_id, status,
                    )

                await self._sleep(self.poll_interval)

            except asyncio.CancelledError:
                raise
            except RuntimeError as e:
                # ThreadPoolExecutor 被 Manager.stop() 关掉后 submit 抛
                # 'cannot schedule new futures after shutdown'。此时 Runner 已经没有 loop /
                # 资源可用——再 sleep+continue 会无限报错且无法自救。立即退出循环。
                if "shutdown" in str(e):
                    logger.info(
                        "[runner] executor shutdown detected cam=%d prompt=%d; exiting loop",
                        self.cam_id, self.prompt_id,
                    )
                    return
                logger.exception("[runner] unexpected RuntimeError cam=%d prompt=%d: %s",
                                 self.cam_id, self.prompt_id, e)
                await self._sleep(self.poll_interval)
            except Exception as e:  # noqa: BLE001
                logger.exception("[runner] unexpected error cam=%d prompt=%d: %s",
                                 self.cam_id, self.prompt_id, e)
                await self._sleep(self.poll_interval)

    async def _sleep(self, sec: float) -> None:
        """sleep 但能被 stop() 快速唤醒。"""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=sec)
        except asyncio.TimeoutError:
            pass

    def _log_no_window_once(self) -> None:
        """周期记录"仍在等窗口"（限流 ``_NO_WINDOW_LOG_SEC``）。

        持续出现这条 = 某个 target 一直没被推理（见 ``PromptCursor.try_advance``：
        要求 window_sec 个连续 ts 且**每个** target 的 has_<class> 都 ≠ None）。
        """
        now = time.monotonic()
        if now - self._last_no_window_log < _NO_WINDOW_LOG_SEC:
            return
        self._last_no_window_log = now
        logger.info(
            "[runner] no window cam=%d prompt=%d window=%ds targets=%s "
            "(等 YOLO 推理：%s 全部 != None 才能凑窗)",
            self.cam_id, self.prompt_id, self.window_sec,
            ",".join(self.target_classes),
            "/".join(f"has_{c}" for c in self.target_classes),
        )

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    def _select_frames(self, window: WindowSpec) -> Optional[list]:
        """从 FrameQueue 取窗口内的帧 + 调 FrameSelector 选 3 帧（v2 按 target_classes）。"""
        ts_set = set(window.ts_list)
        frames = self.frame_queue.get_in_range(ts_set)
        return select_three_frames(frames, window.ts_list, self.target_classes)

    @staticmethod
    def _frame_to_bytes(frame) -> bytes:
        """FrameItem.ndarray → 送给 VLM 的 JPEG bytes（长边限到 1024）。

        缩图 + 编码统一走 :func:`apps.vlm.frame_storage.encode_vlm_jpeg`——Drainer
        回放已落盘的帧也调它，**两条路径必须同一份实现**（历史上回放那份漏了 resize，
        代价差 5 倍，见 frame_storage 模块 docstring）。

        如果 ndarray 是 None（测试），返回 1x1 黑色 jpg。
        """
        arr = frame.ndarray
        if arr is None:
            # 测试场景 / 早期帧：返回 1x1 黑色 jpg
            arr = np.zeros((1, 1, 3), dtype="uint8")
        try:
            return encode_vlm_jpeg(arr)
        except RuntimeError as e:
            raise RuntimeError(f"cv2.imencode failed for ts={frame.ts}: {e}") from e

    # ------------------------------------------------------------------
    # 写库（同步；由 executor 调）
    # ------------------------------------------------------------------
    def _save_success(
        self,
        window: WindowSpec,
        frames: list,
        hit: bool,
        status: str,
        raw_response: str,
    ) -> None:
        """成功调 VLM 后写一条 VLMCheckState 记录；命中且非静默时调 HA。"""
        from apps.vlm.models import VLMCheckState
        from apps.vlm.notify import compute_auto_silenced, notify_hit

        img_paths = self._resolve_img_paths(frames, self.cam_id)
        silence_count = getattr(self._prompt_obj, "silence_count_after_dismiss", 0) or 0

        # auto_silenced 仅在 hit=True 时计算；hit=False 记录直接 auto_silenced=False
        auto_silenced = False
        if hit:
            auto_silenced = compute_auto_silenced(
                cam_id=self.cam_id,
                prompt_id=self.prompt_id,
                status=status,
                silence_count=silence_count,
            )

        state = VLMCheckState.objects.create(
            camera_id=self.cam_id,
            prompt_config_id=self.prompt_id,
            ts_list=list(window.ts_list),
            window_sec=self.window_sec,
            hit=hit,
            raw_response=raw_response,
            status=status,
            img1=img_paths[0][0], img1_ts=img_paths[0][1],
            img2=img_paths[1][0], img2_ts=img_paths[1][1],
            img3=img_paths[2][0], img3_ts=img_paths[2][1],
            has_target_in_window=True,
            failure_reason="",
            retry_count=0,
            auto_silenced=auto_silenced,
        )

        # 命中 + 未静默 + prompt.notify_on_hit=True → 调 HA（失败仅日志）
        if hit and not auto_silenced and getattr(self._prompt_obj, "notify_on_hit", False):
            from django.conf import settings

            from apps.vlm.audio_rule import evaluate_audio_rule, record_state_audio_links
            from apps.vlm.notify import compute_recent_hit_dedup

            # Phase 6：先算音频三态（spec §7.1），落审计快照，再按 condition 路由。
            # evaluate_audio_rule / record_state_audio_links 自身不抛异常：
            # 音频维度故障不能让 Prompt 命中流程或通知流程挂掉。
            audio_outcome = evaluate_audio_rule(
                camera_id=self.cam_id,
                prompt_config_id=self.prompt_id,
            )
            record_state_audio_links(state, audio_outcome)

            dedup_win = int(getattr(settings, "BABYCARE_NOTIFY_DEDUP_WINDOW_SEC", 30))
            if compute_recent_hit_dedup(
                self.cam_id, self.prompt_id, status, dedup_win,
            ):
                logger.info(
                    "[notify] DEDUP skip cam=%d prompt=%d state_id=%d (within %ds)",
                    self.cam_id, self.prompt_id, state.id, dedup_win,
                )
            else:
                notify_hit(state, audio_outcome)

    def _save_no_baby(self, window: WindowSpec) -> None:
        """全窗口无 baby：写一条 failure_reason='no_baby' 的记录。"""
        from apps.vlm.models import VLMCheckState

        VLMCheckState.objects.create(
            camera_id=self.cam_id,
            prompt_config_id=self.prompt_id,
            ts_list=list(window.ts_list),
            window_sec=self.window_sec,
            hit=False,
            raw_response="",
            status="",
            img1="", img1_ts=None,
            img2="", img2_ts=None,
            img3="", img3_ts=None,
            has_target_in_window=False,
            failure_reason="no_baby",
            retry_count=0,
        )

    def _acquire_vlm(self) -> None:
        """同步阻塞：GPU 模式切到 VLM_RUNNING（含 llama-server lazy 启）。"""
        from apps.yolo_detect.gpu_manager import GpuManager
        GpuManager.instance().acquire_vlm()

    def _breaker(self):
        """本进程的 VLM 熔断器（进程内单例，状态镜像到 LLMHealthState）。"""
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState

        return LLMBreaker.for_service(LLMHealthState.SERVICE_VLM)

    def _llama_is_off(self) -> bool:
        """现在**该不该跳过调用、把窗口入队**。

        三种情况都算"别发请求"：

        1. **手动关闭 / 自动重启窗口期** —— `_user_forced_off` 旗标（原有判据）；
        2. **熔断器打开** —— 连续 N 次连接类失败，冷却未到（见 `llm_breaker`）。
           半开态**不算**：那一格就是要放一条真实请求当探测；
        3. **外部模式下远端 health 不通** —— 分体部署时"推理机没开"是日常状态。

        第 3 条不能靠熔断器代劳：熔断要连撞 3 次才会打开，而分体下这个状态
        **从启动就存在** —— 不提前拦，前两个窗口会被白丢（各等一个超时）。

        返回 True 时 Runner 不调 VLM，把窗口入 VLMQueuedTask 等 Drainer 回放。
        """
        from apps.vlm.llama_manager import LlamaManager

        lm = LlamaManager.instance()
        if lm.is_forced_off():
            return True
        if self._breaker().is_blocking():
            return True
        return lm.is_external() and not lm.is_running()

    def _enqueue(self, window: "WindowSpec", frames: list) -> None:
        """llama-off 期间把窗口入队：落盘 3 帧 + 写 VLMQueuedTask(status=pending)。

        与 _save_success / _save_failure 共用 _resolve_img_paths（与 VLMCheckState 一致）。
        """
        from apps.vlm.models import VLMQueuedTask

        img_paths = self._resolve_img_paths(frames, self.cam_id)
        VLMQueuedTask.objects.create(
            camera_id=self.cam_id,
            prompt_config_id=self.prompt_id,
            ts_list=list(window.ts_list),
            window_sec=self.window_sec,
            img1=img_paths[0][0], img1_ts=img_paths[0][1],
            img2=img_paths[1][0], img2_ts=img_paths[1][1],
            img3=img_paths[2][0], img3_ts=img_paths[2][1],
            has_target_in_window=True,
            status="pending",
            retry_count=0,
        )
        logger.info(
            "[runner] queued (llama off) cam=%d prompt=%d window_start=%d frames=%d",
            self.cam_id, self.prompt_id, window.ts_list[0], len(frames),
        )

    def _release_vlm(self) -> None:
        """同步：调 VLM 后释放 GPU（切 IDLE / 清 pending）。"""
        from apps.yolo_detect.gpu_manager import GpuManager
        GpuManager.instance().release_vlm_if_idle()

    def _save_failure(
        self,
        window: WindowSpec,
        frames: list,
        failure_reason: str,
        err_msg: str,
    ) -> None:
        """VLM 解析失败：写一条带 failure_reason 的记录（保留 raw_response 给排错用）。"""
        from apps.vlm.models import VLMCheckState

        img_paths = self._resolve_img_paths(frames, self.cam_id)
        VLMCheckState.objects.create(
            camera_id=self.cam_id,
            prompt_config_id=self.prompt_id,
            ts_list=list(window.ts_list),
            window_sec=self.window_sec,
            hit=False,
            raw_response=err_msg[:500],
            status="",
            img1=img_paths[0][0], img1_ts=img_paths[0][1],
            img2=img_paths[1][0], img2_ts=img_paths[1][1],
            img3=img_paths[2][0], img3_ts=img_paths[2][1],
            has_target_in_window=True,
            failure_reason=failure_reason,
            retry_count=0,
        )

    def _resolve_img_paths(
        self, frames: list, cam_id: int,
    ) -> List[Tuple[str, Optional[int]]]:
        """3 帧 → [(abs_path, ts), ...]。

        Step 12：把每帧真落盘到 MEDIA_ROOT/frames/<YYYY-MM-DD>/<cam_id>_<ts>.jpg。
        - 函数内 import save_frame + settings：便于测试 patch（mock save_frame 即可）
        - 同一 (cam_id, ts) 第二次调自动去重（frame_storage.save_frame 内部判 exists）
        - ndarray 为 None（测试场景 / 早期帧）→ 落盘失败 → path 留空串
        - 全部失败时 img1/2/3 全是空串（保持 v1 兼容）
        """
        from django.conf import settings

        from apps.vlm.frame_storage import save_frame

        media_root = settings.MEDIA_ROOT
        out: List[Tuple[str, Optional[int]]] = []
        for f in frames:
            try:
                p = save_frame(
                    frame_ndarray=f.ndarray,
                    cam_id=cam_id,
                    ts=f.ts,
                    media_root=media_root,
                )
            except Exception as e:  # noqa: BLE001
                # 兜底：异常 → path 空串；避免落盘失败炸写库主流程
                logger.warning(
                    "[runner] _resolve_img_paths failed cam=%d ts=%d: %s",
                    cam_id, f.ts, e,
                )
                p = None
            out.append((str(p) if p is not None else "", f.ts))
        return out


# ---------------------------------------------------------------------------
# PromptRunnerManager：单例，自带独立 loop 线程
# ---------------------------------------------------------------------------
class PromptRunnerManager:
    """全局单例：管理所有 PromptRunner + 独立 asyncio loop 线程。

    - per (cam_id, prompt_id) 一个 PromptRunner
    - 提供 start() / stop() / add_camera / remove_camera / add_prompt / remove_prompt / stats()
    - 共享 LlamaClient（一个进程一个 llama 连接池）
    - 共享 ThreadPoolExecutor（chat + ORM 写库都用）
    - 自带 daemon loop 线程（run_forever），调用方全部同步
    """

    _instance: Optional["PromptRunnerManager"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._runners: Dict[Tuple[int, int], PromptRunner] = {}
        self._cursors = PromptCursorManager()
        self._llama_client: Optional[LlamaClient] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._started = False
        # Camera → enabled prompts（用于 add_camera 时批量起）
        self._cam_prompts: Dict[int, Set[int]] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "PromptRunnerManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动 manager：建 executor + llama_client + 独立 loop 线程 + 加载 enabled (cam, prompt) 组合。

        同步接口；与 Django apps.ready() / signal handler / HTTP 启停 API 调用上下文兼容。
        """
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._executor = ThreadPoolExecutor(
                max_workers=_DEFAULT_POOL_SIZE,
                thread_name_prefix="vlm-worker",
            )
            self._llama_client = LlamaClient.instance()  # base_url 从 .env 读
            # 独立 loop 线程（daemon；进程退出时自动回收）
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_loop_forever,
                name="vlm-runner-loop",
                daemon=True,
            )
            self._loop_thread.start()
            self._started = True
            logger.info("[runner-mgr] started; loading existing cameras/prompts")

        # 启动时全量加载：所有 enabled prompt × 所有 enabled camera
        self._bootstrap_existing()

    def _run_loop_forever(self) -> None:
        """loop 线程入口：跑 run_forever 直到 loop.stop()。"""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            logger.info("[runner-mgr] loop thread exiting")

    def stop(self) -> None:
        """停止 manager：停所有 Runner + 停 loop + join 线程 + 关 executor。"""
        if not self._started:
            return
        with self._lock:
            runners = list(self._runners.values())
            self._runners.clear()
            loop = self._loop
            loop_thread = self._loop_thread
            executor = self._executor
            self._executor = None
            self._loop = None
            self._loop_thread = None
            self._started = False

        # 1) 停所有 Runner（同步等待 loop 上的 stop 完成）
        if loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._astop_all(runners), loop
                )
                future.result(timeout=_ASYNC_OP_TIMEOUT_SEC)
            except Exception as e:  # noqa: BLE001
                logger.warning("[runner-mgr] async stop runners failed: %s", e)

            # 2) 停 loop + join 线程
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception as e:  # noqa: BLE001
                logger.warning("[runner-mgr] loop.call_soon_threadsafe(stop) failed: %s", e)

        if loop_thread is not None:
            loop_thread.join(timeout=_ASYNC_OP_TIMEOUT_SEC)

        # 3) 关 executor（不管上面成不成功都关）
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        logger.info("[runner-mgr] stopped")

    async def _astop_all(self, runners: List[PromptRunner]) -> None:
        await asyncio.gather(*[r.stop() for r in runners], return_exceptions=True)

    # ------------------------------------------------------------------
    # 动态增删（signal 调用，同步接口）
    # ------------------------------------------------------------------
    def add_camera(self, cam_id: int) -> None:
        """加一个 cam：对所有 enabled prompt 起 Runner（按 prompt.camera_ids 过滤）。

        v3 fix：原版不过滤 camera_ids，导致限定 cam 的 prompt 错误地被起在所有 cam 上
        （典型症状：用户限定 prompt 只跑客厅摄像头，结果测试视频 cam 也触发）。
        """
        if not self._started or self._loop is None:
            return
        from apps.vlm.models import VLMPromptConfig

        prompts = list(
            VLMPromptConfig.objects
            .filter(enabled=True, manual_paused=False)
            .prefetch_related("camera_ids")
        )
        added: Set[int] = set()
        for p in prompts:
            if not self._prompt_targets_cam(p, cam_id):
                continue
            if self._add_runner_sync(cam_id, p):
                added.add(p.id)
        if added:
            with self._lock:
                self._cam_prompts.setdefault(cam_id, set()).update(added)
        logger.info(
            "[runner-mgr] add_camera cam=%d prompts=%d (of %d enabled, after camera_ids filter)",
            cam_id, len(added), len(prompts),
        )

    def remove_camera(self, cam_id: int) -> None:
        """删一个 cam：停掉所有该 cam 的 Runner。"""
        if not self._started or self._loop is None:
            return
        with self._lock:
            keys = [k for k in self._runners if k[0] == cam_id]
            runners = [self._runners.pop(k) for k in keys]
            self._cam_prompts.pop(cam_id, None)

        # 同步停 Runner
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._astop_all(runners), self._loop
            )
            future.result(timeout=_ASYNC_OP_TIMEOUT_SEC)
        except Exception as e:  # noqa: BLE001
            logger.warning("[runner-mgr] remove_camera async stop failed: %s", e)

        # 删 cursor
        self._cursors.remove_cam(cam_id)
        logger.info("[runner-mgr] remove_camera cam=%d runners=%d", cam_id, len(runners))

    def add_prompt(self, prompt_id: int) -> None:
        """加一个 prompt：对适用的 cam 起 Runner（按 prompt.camera_ids 过滤）。

        v3 fix：原版遍历所有 active cam 不检查 prompt.camera_ids 限定，
        错把限定 cam 的 prompt 跑到不该跑的 cam 上。

        改 camera_ids 也走这条路（post_save 触发）：先 remove_prompt 清干净，
        再按新 camera_ids 集合启 Runner（cursor 状态随 remove 丢，可接受）。
        """
        if not self._started or self._loop is None:
            return
        # 先清掉旧的（camera_ids 改了的话，老 cam 上的 Runner 不该再跑）
        self.remove_prompt(prompt_id)

        from apps.vlm.models import VLMPromptConfig
        from apps.streaming.models import Camera

        prompt = (
            VLMPromptConfig.objects
            .filter(pk=prompt_id, enabled=True, manual_paused=False)
            .prefetch_related("camera_ids")
            .first()
        )
        if prompt is None:
            return

        bound_ids = {c.id for c in prompt.camera_ids.all()}
        if bound_ids:
            cams = list(Camera.objects.filter(is_active=True, id__in=bound_ids))
        else:
            cams = list(Camera.objects.filter(is_active=True))
        added: Set[int] = set()
        for cam in cams:
            if self._add_runner_sync(cam.id, prompt):
                added.add(cam.id)
        if added:
            with self._lock:
                for cam_id in added:
                    self._cam_prompts.setdefault(cam_id, set()).add(prompt.id)
        logger.info(
            "[runner-mgr] add_prompt prompt=%d cams=%d (after camera_ids filter)",
            prompt_id, len(cams),
        )

    def remove_prompt(self, prompt_id: int) -> None:
        """删一个 prompt：停掉所有该 prompt 的 Runner。"""
        if not self._started or self._loop is None:
            return
        with self._lock:
            keys = [k for k in self._runners if k[1] == prompt_id]
            runners = [self._runners.pop(k) for k in keys]
            for cam_id, pids in self._cam_prompts.items():
                pids.discard(prompt_id)

        if self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._astop_all(runners), self._loop
                )
                future.result(timeout=_ASYNC_OP_TIMEOUT_SEC)
            except Exception as e:  # noqa: BLE001
                logger.warning("[runner-mgr] remove_prompt async stop failed: %s", e)
        logger.info("[runner-mgr] remove_prompt prompt=%d runners=%d", prompt_id, len(runners))

    @staticmethod
    def _prompt_targets_cam(prompt, cam_id: int) -> bool:
        """prompt 是否作用于 cam_id。

        - prompt.camera_ids 为空 → 适用于所有 cam（True）
        - prompt.camera_ids 非空 → 仅当 cam_id 在其中时 True
        """
        bound_ids = {c.id for c in prompt.camera_ids.all()}
        return not bound_ids or cam_id in bound_ids

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _bootstrap_existing(self) -> None:
        """启动时全量加载：所有 enabled prompt × 适用的 enabled camera。"""
        from apps.vlm.models import VLMPromptConfig
        from apps.streaming.models import Camera

        prompts = list(
            VLMPromptConfig.objects
            .filter(enabled=True, manual_paused=False)
            .prefetch_related("camera_ids")
        )
        cams = list(Camera.objects.filter(is_active=True))
        for cam in cams:
            for p in prompts:
                if not self._prompt_targets_cam(p, cam.id):
                    continue
                self._add_runner_sync(cam.id, p)
        for cam in cams:
            cam_prompt_ids = [
                p.id for p in prompts if self._prompt_targets_cam(p, cam.id)
            ]
            if cam_prompt_ids:
                with self._lock:
                    self._cam_prompts.setdefault(cam.id, set()).update(cam_prompt_ids)
        logger.info(
            "[runner-mgr] bootstrap: %d cams × %d enabled prompts = %d runners (after camera_ids filter)",
            len(cams), len(prompts), len(self._runners),
        )

    def _add_runner_sync(self, cam_id: int, prompt) -> bool:
        """加一个 (cam_id, prompt) 的 Runner（同步包装：把 async start 提交到 Manager loop）。

        Returns:
            True  - 新增成功（或已存在）
            False - Manager 未就绪 / 启动失败

        锁设计
        ------
        整段（构造 + 塞 dict）在同一把锁内，避免并发 add 同一 (cam, prompt) 时
        创建 2 个 Runner。start() 在锁外提交 + 等结果；start 失败时回滚 dict。
        """
        if not self._started or self._loop is None or self._llama_client is None or self._executor is None:
            return False
        key = (cam_id, prompt.id)
        with self._lock:
            if key in self._runners:
                return True  # 已存在

            from apps.yolo_detect.frame_queue import FrameQueueManager

            fq = FrameQueueManager.instance().get_or_create(cam_id)
            cursor = self._cursors.get(cam_id, prompt.id, prompt.window_sec)

            runner = PromptRunner(
                cam_id=cam_id,
                prompt_id=prompt.id,
                prompt_text=prompt.prompt,
                positive_keyword=prompt.positive_keyword,
                result_format=prompt.result_format,
                kind=prompt.kind,
                window_sec=prompt.window_sec,
                max_tokens=prompt.max_tokens,
                timeout_sec_first=prompt.timeout_sec_first,
                timeout_sec_normal=prompt.timeout_sec_normal,
                cursor=cursor,
                frame_queue=fq,
                prompt_obj=prompt,
                llama_client=self._llama_client,
                executor=self._executor,
                target_classes=_parse_target_classes(prompt.target_classes),
            )
            self._runners[key] = runner

        # 锁外 start（run_coroutine_threadsafe 立即 schedule；start 内 create_task 后立即返回）
        try:
            future = asyncio.run_coroutine_threadsafe(runner.start(), self._loop)
            future.result(timeout=_ASYNC_OP_TIMEOUT_SEC)
        except Exception as e:  # noqa: BLE001
            logger.error("[runner-mgr] add_runner start failed cam=%d prompt=%d: %s",
                         cam_id, prompt.id, e)
            with self._lock:
                self._runners.pop(key, None)
            return False
        return True

    # 兼容旧名字（外部如果直接调 add_runner 也能用）
    def add_runner(self, cam_id: int, prompt) -> bool:
        return self._add_runner_sync(cam_id, prompt)

    # ------------------------------------------------------------------
    # 监控
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            runners = list(self._runners.values())
            loop_alive = (
                self._loop is not None
                and self._loop_thread is not None
                and self._loop_thread.is_alive()
            )
        return {
            "started": self._started,
            "loop_alive": loop_alive,
            "runner_count": len(runners),
            "cursor_count": self._cursors.stats()["cursor_count"],
            "keys": [f"{c},{p}" for c, p in self._runners.keys()],
            "runners": [
                {
                    "cam_id": r.cam_id,
                    "prompt_id": r.prompt_id,
                    "kind": r.kind,
                    "invoke_count": r.stats.invoke_count,
                    "success_count": r.stats.success_count,
                    "fail_count_total": r.stats.fail_count_total,
                    "skip_window_count": r.stats.skip_window_count,
                    "no_baby_count": r.stats.no_baby_count,
                    "cursor_max_read_ts": r.cursor.max_read_ts,
                    "cursor_fail_count": r.cursor.fail_count,
                    "cursor_is_cold_start": r.cursor.is_cold_start,
                    "last_error": r.stats.last_error,
                }
                for r in runners
            ],
        }

    # ------------------------------------------------------------------
    # Cursor 查询（Step 10 HTTP API 调试用）
    # ------------------------------------------------------------------
    def cursor_keys_for_cam(self, cam_id: int) -> list[int]:
        """返回该 cam 的所有 cursor 的 prompt_id 列表（顺序：按键添加顺序）。"""
        with self._lock:
            keys = self._cursors.stats().get("keys", []) or []
        return [pid for (cid, pid) in keys if cid == cam_id]

    def cursor_get(self, cam_id: int, prompt_id: int, window_sec: int):
        """取 (cam, prompt) 的 cursor；不存在返回 None。"""
        try:
            return self._cursors.get(cam_id, prompt_id, window_sec)
        except Exception:
            return None

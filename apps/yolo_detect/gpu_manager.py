"""
GpuManager（Step 4 实装）：YOLO / VLM 显存互斥协调。

设计（设计.md §5.3）
-------------------
- 状态机：IDLE / YOLO_RUNNING / VLM_RUNNING
- exclusive 模式（默认，6G 显存）：状态切换 + 阻塞 acquire + release_if_idle 建议
- parallel 模式（22G+ 显存）：状态机 no-op；acquire/release 直接返回
  - 但 acquire_vlm 仍会触发 LlamaManager.ensure_running()（子进程该启还得启）
- 锁：threading.Condition；通知用 notify_all

API 概览
--------
- acquire_yolo()         阻塞直到 YOLO 可用（parallel 直接返回）
- release_yolo_if_idle() "我没事干了"；对面也没活 → unload + 切 IDLE
- acquire_vlm()          阻塞直到 VLM 可用（含 llama-server 启动；parallel 跳过状态切换但仍 ensure_running）
- release_vlm_if_idle()  同上
- status()               监控用 dict

依赖注入
--------
- LlamaManager 通过 attach_llama_manager() 注入
- BabyDetector 在 acquire_yolo 时 lazy 调用 ensure_loaded()，release 时 unload()
- 两个 detector 的延迟 import 在 acquire/release 内部做，避免启动期循环依赖
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


class GpuState:
    """GPU 状态常量。"""
    IDLE = "IDLE"
    YOLO_RUNNING = "YOLO_RUNNING"
    VLM_RUNNING = "VLM_RUNNING"


class GpuManager:
    """全局单例：YOLO/VLM 显存互斥（exclusive）或直通（parallel）。"""

    _instance: Optional["GpuManager"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._mode: str = "exclusive"
        # 状态机
        self._state: str = GpuState.IDLE
        # 条件变量（所有状态切换/等待都在它下面）
        self._state_cond = threading.Condition()
        # 对侧是否在排队（用来判定 release 后是否真的 unload）
        self._yolo_request_pending: bool = False
        self._vlm_request_pending: bool = False
        # 注入的 LlamaManager（attach_llama_manager 时设）
        self._llama_manager: Optional[Any] = None
        # 常驻模式旗标（apps.ready() 读 .env 后调 set_resident）
        # True 时 acquire_*/release_* 都仅操作 pending，不动 state / 不 unload
        self._resident: bool = False

    @classmethod
    def instance(cls) -> "GpuManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """设互斥模式。

        - ``exclusive``：6G 显存，YOLO/VLM 互斥（本机方案默认）
        - ``parallel``：22G+，两者常驻
        - ``off``：**不做本机仲裁**（分体部署用）—— YOLO 与 VLM 在不同机器上，
          没有显存竞争，状态机纯属负担；更关键的是它的 ``unload()`` 会去动
          **远端**服务（checklist §2 #4）。
        """
        if mode not in {"exclusive", "parallel", "off"}:
            raise ValueError(
                f"GpuManager mode 必须是 exclusive|parallel|off，当前: {mode}"
            )
        self._mode = mode
        logger.info("[gpu_manager] mode=%s", mode)

    def _arbitrate(self) -> bool:
        """要不要走本机显存状态机。

        ``resident`` / ``parallel`` / ``off`` 三种情况都**不走**（只置 pending）。
        收在一处是为了让 acquire/release 四个入口的判据永远一致 ——
        之前 ``acquire_vlm`` 把 parallel 写在状态机之外，而 ``release_vlm_if_idle``
        又单独判一次，改一处漏一处（checklist §2 #10）。
        """
        return not (self._resident or self._mode in ("parallel", "off"))

    def _llama_is_external(self) -> bool:
        """llama-server 是否在别的机器上（注入的 manager 说了算）。"""
        lm = self._llama_manager
        if lm is None:
            return False
        fn = getattr(lm, "is_external", None)
        return bool(fn()) if callable(fn) else False

    def attach_llama_manager(self, llama_manager: Any) -> None:
        """apps.vlm.apps.ready() 调：注入 LlamaManager 引用。

        注：parallel 模式下 llama_manager 必须存在（否则 acquire_vlm 会 raise）。
        """
        self._llama_manager = llama_manager
        logger.info("[gpu_manager] llama_manager attached: %r", llama_manager)

    def set_resident(self, enabled: bool) -> None:
        """apps.ready() 调：根据 BABYCARE_LLM_RESIDENT 设旗标。

        True 时 acquire_yolo / acquire_vlm / release_*_if_idle 都仅操作 pending；
        state 切换 + load/unload 都跳过。
        """
        self._resident = bool(enabled)
        logger.info("[gpu_manager] resident=%s", enabled)

    # ------------------------------------------------------------------
    # YOLO 侧（YoloLoop 调）
    # ------------------------------------------------------------------
    def acquire_yolo(self) -> None:
        """阻塞直到 YOLO 可用。

        parallel / resident / off：直接返回（仅置 _yolo_request_pending=True）。
        exclusive：若当前是 VLM_RUNNING，则等对面 release；之后切 YOLO_RUNNING 并 ensure_loaded。
        """
        if not self._arbitrate():
            with self._state_cond:
                self._yolo_request_pending = True
            return

        # exclusive
        with self._state_cond:
            self._yolo_request_pending = True
            while self._state == GpuState.VLM_RUNNING:
                logger.info("[gpu] yolo waiting: state=VLM_RUNNING")
                self._state_cond.wait(timeout=5.0)
                # 醒来再判；防止 spurious wake

            if self._state == GpuState.IDLE:
                # 加载 YOLO（延迟 import 防启动期循环依赖）
                try:
                    from .detector import BabyDetector
                    BabyDetector.instance().ensure_loaded()
                except Exception:
                    logger.exception("[gpu] yolo ensure_loaded failed")
                    # 加载失败也要清掉 pending，避免状态卡死
                    self._yolo_request_pending = False
                    raise
                self._state = GpuState.YOLO_RUNNING
                logger.info("[gpu] -> YOLO_RUNNING (yolo loaded)")
            # 已经是 YOLO_RUNNING：直接用

    def release_yolo_if_idle(self) -> None:
        """YoloLoop 完成一批 detect 后调；如果对面（VLM）没在排队，立即切 IDLE。

        parallel / resident / off：仅清 pending（不卸载）。
        exclusive：检查 _vlm_request_pending；
                   - False → unload YOLO + 切 IDLE
                   - True → 保留 YOLO_RUNNING（让 VLM 抢 GPU 时再卸）
        """
        if not self._arbitrate():
            with self._state_cond:
                self._yolo_request_pending = False
            return

        with self._state_cond:
            self._yolo_request_pending = False
            if self._state != GpuState.YOLO_RUNNING:
                return
            if not self._vlm_request_pending:
                # VLM 没活 → 卸 YOLO + 切 IDLE
                self._unload_yolo()
                self._state = GpuState.IDLE
                self._state_cond.notify_all()
                logger.info("[gpu] -> IDLE (yolo released, no vlm pending)")

    # ------------------------------------------------------------------
    # VLM 侧（VlmWorker / PromptRunner 调，Step 8 才有调用方）
    # ------------------------------------------------------------------
    def acquire_vlm(self) -> None:
        """阻塞直到 VLM 可用（含 llama-server 启动）。

        parallel / resident：直接返回（仅置 pending），但仍 ``ensure_running``
          （子进程该启还得启）。
        off：仅置 pending，**不** ensure（分体部署：远端服务不归本进程启）。
        exclusive：若当前是 YOLO_RUNNING，等对面 release（对面若没活则主动卸）；
                   之后启 llama-server + 切 VLM_RUNNING。

        外部模式（`BABYCARE_LLAMA_EXTERNAL=1`）下**一律不 ensure_running** ——
        这是分体的核心差异：可用性交给 health + 熔断器判，本进程只探活（checklist §2 #3）。
        """
        external = self._llama_is_external()
        if not self._arbitrate():
            with self._state_cond:
                self._vlm_request_pending = True
            if not external:
                self._ensure_llama_running()
            return

        # exclusive
        with self._state_cond:
            self._vlm_request_pending = True
            while self._state == GpuState.YOLO_RUNNING:
                # 关键路径：YOLO_RUNNING 期间 VLM 要进来
                if not self._yolo_request_pending:
                    # YOLO 没活了 → 主动卸，让位
                    self._unload_yolo()
                    self._state = GpuState.IDLE
                    self._state_cond.notify_all()
                    break
                # YOLO 还在忙 → 等
                logger.info("[gpu] vlm waiting: state=YOLO_RUNNING (yolo busy)")
                self._state_cond.wait(timeout=5.0)
                # 醒后再判

            # 此时 _state 应该是 IDLE 或 VLM_RUNNING（之前已 VLM_RUNNING 时直接复用）
            if self._state == GpuState.IDLE:
                # 启 llama-server（在锁外做 I/O？这里为简单先放锁内，加注释；Phase 2 优化可拆出去）
                # 注：ensure_running 内部阻塞最坏 60s，但持有 cond 期间其它 acquire 会排队，
                #     这是有意的——避免 VLM 并发启多个子进程。
                try:
                    self._ensure_llama_running()
                except Exception:
                    self._vlm_request_pending = False
                    raise
                self._state = GpuState.VLM_RUNNING
                logger.info("[gpu] -> VLM_RUNNING (llama-server up)")
            # 已经是 VLM_RUNNING：复用

    def release_vlm_if_idle(self) -> None:
        """VlmWorker 完成所有请求后调；停 llama-server 释放显存。

        parallel / resident / off：仅清 pending（不卸载）。
        exclusive：无条件切回 IDLE + unload llama + notify_all。

        修复死锁：之前看 _yolo_request_pending 决定"保留 VLM_RUNNING 让 YOLO 抢"，但
        YoloLoop 的 acquire_yolo 只 wait state 变 IDLE，不会主动抢 → 两者互等 →
        永久卡住。改成无条件切回 IDLE + 卸 llama，YoloLoop 拿到 GPU 加载 YOLO。

        **`off` 是分体部署的安全阀**：这条路径会调 `_unload_llama()`，而在外部模式下
        那个 `unload()` 会 POST 远端 `/unload`（关掉对面正在用的服务）。
        `_arbitrate()` 返回 False 就永远走不到下面（见 `_unload_llama` 的二次防护）。
        """
        if not self._arbitrate():
            with self._state_cond:
                self._vlm_request_pending = False
            return

        with self._state_cond:
            self._vlm_request_pending = False
            if self._state != GpuState.VLM_RUNNING:
                return
            # 卸 VLM + 切 IDLE + notify（让等着的 YoloLoop / 下一个请求拿 GPU）
            self._unload_llama()
            self._state = GpuState.IDLE
            self._state_cond.notify_all()
            logger.info("[gpu] -> IDLE (llama-server stopped)")

    # ------------------------------------------------------------------
    # 内部：实际 load/unload（锁内调用）
    # ------------------------------------------------------------------
    def _unload_yolo(self) -> None:
        try:
            from .detector import BabyDetector
            BabyDetector.instance().unload()
        except Exception:
            logger.exception("[gpu] yolo unload failed")

    def _unload_llama(self) -> None:
        if self._llama_manager is None:
            logger.warning("[gpu] _unload_llama but llama_manager is None")
            return
        # 二次防护：外部模式绝不允许走到"卸远端服务"（checklist §2 #4）。
        # 正常路径上 `_arbitrate()` 已经拦住了，这里防的是将来有人改回按 mode 判断。
        if self._llama_is_external():
            logger.warning("[gpu] 外部模式：拒绝 _unload_llama（不卸远端服务）")
            return
        try:
            self._llama_manager.unload()
        except Exception:
            logger.exception("[gpu] unload failed")

    def _ensure_llama_running(self) -> None:
        if self._llama_manager is None:
            raise RuntimeError(
                "GpuManager.acquire_vlm 调用了但 _llama_manager 未注入；"
                "请检查 apps.vlm.apps.ready() 是否先于首次 VLM 调用执行"
            )
        # 读 settings（启动期已加载）
        try:
            from django.conf import settings
            timeout_sec = getattr(settings, "BABYCARE_LLAMA_START_TIMEOUT_SEC", 60)
            max_retries = getattr(settings, "BABYCARE_LLAMA_START_RETRIES", 1)
        except Exception:
            timeout_sec = 60
            max_retries = 1
        self._llama_manager.ensure_running(timeout_sec=timeout_sec, max_retries=max_retries)

    # ------------------------------------------------------------------
    # 监控
    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._state_cond:
            return {
                "mode": self._mode,
                "state": self._state,
                "yolo_request_pending": self._yolo_request_pending,
                "vlm_request_pending": self._vlm_request_pending,
                "llama_manager_attached": self._llama_manager is not None,
                "step": 4,
            }
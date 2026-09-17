"""推理服务：每 2 秒窗 / 1 秒 hop → 共识判定 → 稀疏落库（Phase 2，spec §4.3/§5.1）。

与采集的关系
------------
采集（:mod:`capture`）只负责把 PCM 搬进环形缓冲，不碰模型；本模块按 hop 从
每路的环形缓冲**取最近 window 秒**做推理。这样 FFmpeg 的管道反压不会传导到
推理，推理慢也不会卡住采集。

积压丢弃（spec §4.4）
--------------------
本模块是单线程串行推理。若一次 tick 超过了 hop，说明算不过来：**丢弃落后的
决策窗并记日志**，而不是排队——排队只会越积越多，最后所有结论都过期。
不能静默丢帧，否则表现为"覆盖完整却没检测到"。

线程数限制
----------
模型侧统一由 :class:`~apps.audio_detect.detector.DualModelDetector` 限制
（TF=1 / torch=1，P0-4 结论）。此处不再开线程池，避免又引入自旋。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .assembler import EventAssembler
from .capture import AudioCapture
from .consensus import LOG_STATE_CHANGE, ConsensusEngine, WindowVerdict
from .detector import DualModelDetector
from .label_map import AudioLabelMap
from .models import AudioRuntimeState, SoundDetectionLog

logger = logging.getLogger(__name__)

# 心跳循环里每隔多少次做一次 DB 连接回收（MySQL wait_timeout 兜底）
_CLOSE_CONN_EVERY = 30


@dataclass
class InferenceConfig:
    """推理调度参数（全部来自 settings，允许测试覆写）。"""

    window_sec: float = 2.0
    hop_sec: float = 1.0
    #: 窗口内音频不足该比例视为"缓冲没喂满"，本窗跳过（不算阴性结论）
    min_audio_ratio: float = 0.9
    #: INFO 状态日志间隔秒；0 = 每窗都打（Phase 2 验收要"每窗可见"）
    log_every_sec: float = 5.0

    @classmethod
    def from_settings(cls) -> "InferenceConfig":
        from django.conf import settings

        return cls(
            window_sec=float(settings.BABYCARE_AUDIO_DECISION_WINDOW_SEC),
            hop_sec=float(settings.BABYCARE_AUDIO_DECISION_HOP_SEC),
            min_audio_ratio=float(settings.BABYCARE_AUDIO_MIN_AUDIO_RATIO),
            log_every_sec=float(settings.BABYCARE_AUDIO_DETECT_LOG_SEC),
        )


@dataclass
class WindowOutcome:
    """一个窗口的完整结果（也便于单测断言）。"""

    camera_id: int
    window_start_ts: int
    window_end_ts: int
    verdict: WindowVerdict
    log_reason: str
    yamnet_payload: dict[str, Any]
    panns_payload: dict[str, Any]


class InferenceService:
    """常驻双模型 + 多路窗口调度 + 稀疏落库。"""

    def __init__(
        self,
        captures: Mapping[int, AudioCapture],
        detector: DualModelDetector,
        engine: ConsensusEngine,
        label_map: AudioLabelMap,
        config: InferenceConfig | None = None,
        enable_events: bool = True,
        media_root: str = "",
        worker_epoch: str = "",
    ):
        self._captures = captures
        self.detector = detector
        self.engine = engine
        self.label_map = label_map
        self._cfg = config or InferenceConfig.from_settings()
        self._enable_events = bool(enable_events)
        self._media_root = media_root
        self._worker_epoch = worker_epoch
        self._assemblers: dict[int, EventAssembler] = {}

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._windows = 0
        self._skipped_starved = 0
        self._last_log_at = 0.0
        self._camera_status: dict[int, str] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._loop, name="audio-infer", daemon=True)
        self._thread = t
        t.start()
        logger.info(
            "[audio] inference started: window=%.1fs hop=%.1fs strategy=%s labels=%s",
            self._cfg.window_sec, self._cfg.hop_sec,
            self.engine.strategy, ",".join(self.label_map.enabled_labels()),
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        # 收尾所有进行中的事件（摘 tap + 落盘 + 落库），避免遗留 recording 行
        for asm in self._assemblers.values():
            try:
                asm.stop()
            except Exception:  # noqa: BLE001
                logger.exception("[audio cam#%s] assembler stop failed", asm.camera_id)
        logger.info(
            "[audio] inference stopped: windows=%d dropped=%d starved=%d",
            self._windows, self._dropped, self._skipped_starved,
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        hop = max(self._cfg.hop_sec, 0.05)
        t0 = time.monotonic()
        deadline = t0 + hop
        tick = 0
        while not self._stop.is_set():
            wait = deadline - time.monotonic()
            if wait > 0 and self._stop.wait(wait):
                break
            now = time.monotonic()

            # 落后超过一个 hop → 丢弃落后的决策窗（不是排队）
            lag = now - deadline
            if lag > hop:
                self._dropped += int(lag // hop)
                logger.warning(
                    "[audio] 推理落后 %.2fs，丢弃 %d 个决策窗（spec §4.4）",
                    lag, int(lag // hop),
                )
                # 重新对齐节奏：从"现在"起算下一拍（不能往回补，补就是排队）
                tick += 1
                deadline = now + hop
            else:
                tick += 1
                # 用 t0 + n*hop 而非累加：浮点逐次相加会漂移，窗口间隔会不均
                deadline = t0 + (tick + 1) * hop
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] inference tick failed")
            if tick % _CLOSE_CONN_EVERY == 0:
                try:
                    from django.db import close_old_connections

                    close_old_connections()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    def tick(self, now_ts: float | None = None) -> list[WindowOutcome]:
        """对所有在跑的采集做一轮窗口推理（可被单测直接调用）。"""
        now_ts = time.time() if now_ts is None else now_ts
        window_end = int(now_ts)
        window_start = int(now_ts - self._cfg.window_sec)
        outcomes: list[WindowOutcome] = []

        for cam_id, cap in list(self._captures.items()):
            snap = cap.snapshot()
            if snap.status != AudioRuntimeState.STATUS_READY:
                continue
            audio = cap.last_samples(self._cfg.window_sec)
            need = int(self._cfg.window_sec * cap.sample_rate * self._cfg.min_audio_ratio)
            if audio is None or len(audio) < need:
                self._skipped_starved += 1
                logger.debug(
                    "[audio cam#%s] 窗口音频不足（%d/%d 样本），跳过本窗",
                    cam_id, 0 if audio is None else len(audio), need,
                )
                continue
            try:
                outcome = self.process_window(cam_id, audio, window_start, window_end)
                if self._enable_events:
                    self._on_window_for_assembler(cam_id, cap, audio, outcome)
                outcomes.append(outcome)
            except Exception:  # noqa: BLE001
                logger.exception("[audio cam#%s] 窗口处理失败", cam_id)
        return outcomes

    def _on_window_for_assembler(
        self,
        cam_id: int,
        cap: AudioCapture,
        audio: Any,
        outcome: WindowOutcome,
    ) -> None:
        """把窗口结果喂给事件状态机；摄像头动态加入时按需建 assembler。"""
        asm = self._assemblers.get(cam_id)
        if asm is None:
            asm = EventAssembler(
                cam_id, cap, media_root=self._media_root,
                worker_epoch=self._worker_epoch,
            )
            self._assemblers[cam_id] = asm
            logger.info("[audio cam#%s] event assembler created", cam_id)
        asm.on_window(
            outcome.window_start_ts, outcome.window_end_ts, outcome.verdict, audio,
        )

    # ------------------------------------------------------------------
    def process_window(
        self,
        camera_id: int,
        audio: Any,
        window_start_ts: int,
        window_end_ts: int,
    ) -> WindowOutcome:
        """单窗：推理 → 共识 → 按策略落库。"""
        self._windows += 1
        out_y, out_p = self.detector.infer(audio)
        verdict = self.engine.evaluate(
            out_y.scores if out_y.ok else None,
            out_p.scores if out_p.ok else None,
        )
        log_reason = self.engine.log_reason(verdict)
        outcome = WindowOutcome(
            camera_id=camera_id,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            verdict=verdict,
            log_reason=log_reason,
            yamnet_payload=_model_payload(out_y),
            panns_payload=_model_payload(out_p),
        )

        line = (
            f"[audio cam#{camera_id}] window={window_start_ts}-{window_end_ts} "
            f"{verdict.as_log()} "
            f"cost=y{out_y.elapsed_ms:.0f}ms/p{out_p.elapsed_ms:.0f}ms"
        )
        if log_reason:
            logger.info("%s log=%s", line, log_reason)
        elif self._should_log_tick():
            logger.info("%s", line)
        else:
            logger.debug("%s", line)

        if log_reason:
            self._persist(camera_id, outcome)
        return outcome

    def _should_log_tick(self) -> bool:
        every = self._cfg.log_every_sec
        if every <= 0:
            return True
        now = time.monotonic()
        if now - self._last_log_at >= every:
            self._last_log_at = now
            return True
        return False

    # ------------------------------------------------------------------
    def _persist(self, camera_id: int, outcome: WindowOutcome) -> None:
        try:
            SoundDetectionLog.objects.create(
                camera_id=camera_id,
                window_start_ts=outcome.window_start_ts,
                window_end_ts=outcome.window_end_ts,
                yamnet_payload=outcome.yamnet_payload,
                panns_payload=outcome.panns_payload,
                consensus_payload=self.engine.consensus_payload(outcome.verdict),
                decision=outcome.verdict.decision,
                log_reason=outcome.log_reason,
                failure_reason=outcome.verdict.failure_reason,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[audio cam#%s] SoundDetectionLog 写入失败", camera_id)

    # ------------------------------------------------------------------
    # 状态变化落库（spec §5.1 的第三类落库条件）
    # ------------------------------------------------------------------
    def notify_state_change(self, camera_id: int, status: str, detail: str = "") -> None:
        """采集/模型状态变化时落一条审计记录（模型加载、标签停用、就绪切换…）。"""
        if self._camera_status.get(camera_id) == status:
            return
        self._camera_status[camera_id] = status
        payload = {
            "status": status,
            "detail": detail,
            "models": self.detector.status,
            "labels_enabled": list(self.label_map.enabled_labels()),
            "labels_disabled": self.label_map.disabled_labels(),
            "strategy": self.engine.strategy,
        }
        try:
            SoundDetectionLog.objects.create(
                camera_id=camera_id,
                window_start_ts=int(time.time()),
                window_end_ts=int(time.time()),
                yamnet_payload={"version": self.detector.yamnet.version},
                panns_payload={"version": self.detector.panns.version},
                consensus_payload=payload,
                decision=SoundDetectionLog.DECISION_DEGRADED,
                log_reason=LOG_STATE_CHANGE,
                failure_reason="",
            )
        except Exception:  # noqa: BLE001
            logger.exception("[audio cam#%s] 状态变化落库失败", camera_id)

    def forget_camera(self, camera_id: int) -> None:
        """摄像头被移除时清理状态缓存，避免重新加入后漏报状态变化。"""
        self._camera_status.pop(camera_id, None)
        asm = self._assemblers.pop(camera_id, None)
        if asm is not None:
            try:
                asm.stop()
            except Exception:  # noqa: BLE001
                logger.exception("[audio cam#%s] assembler stop failed", camera_id)

    def announce_models_ready(self) -> None:
        """模型加载完成：为每路已存在的采集写一条状态变化记录。"""
        disabled = self.label_map.disabled_labels()
        if disabled:
            logger.warning("[audio] 以下业务标签 disabled，不参与检测: %s", disabled)
        st = self.detector.status
        logger.info(
            "[audio] 模型就绪: yamnet=%s(loaded=%s) panns=%s(loaded=%s, device=%s) "
            "strategy=%s enabled=%s",
            st["yamnet"]["version"], st["yamnet"]["loaded"],
            st["panns"]["version"], st["panns"]["loaded"], st["panns"]["device"],
            self.engine.strategy, ",".join(self.label_map.enabled_labels()),
        )
        for cam_id in list(self._captures):
            self.notify_state_change(cam_id, "models_ready")

    # ------------------------------------------------------------------
    @property
    def status(self) -> dict[str, Any]:
        return {
            "windows": self._windows,
            "dropped": self._dropped,
            "starved": self._skipped_starved,
            "strategy": self.engine.strategy,
        }


# ---------------------------------------------------------------------------
def _model_payload(out: Any) -> dict[str, Any]:
    return {
        "model": out.model,
        "ok": out.ok,
        "version": out.version,
        "elapsed_ms": round(float(out.elapsed_ms), 2),
        "scores": {k: round(float(v), 6) for k, v in (out.scores or {}).items()},
        "raw": out.raw_scores or {},
        "top_k": out.top_k or [],
        "error": out.error or "",
    }


__all__ = ["InferenceConfig", "InferenceService", "WindowOutcome"]

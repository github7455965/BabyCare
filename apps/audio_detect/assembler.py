"""事件组装器：阳性窗序列 → 可回放的音频事件（Phase 3，spec §4.3.4/§6.2/§6.3）。

状态机（照 `scripts/tmp/p0_7_state_machine.py` 参考实现，单参数版）
---------------------------------------------------------------
时间轴约定：窗 t 覆盖 ``[window_end - window, window_end]``；inference 每 tick
调 :meth:`on_window`，把该窗的三态结果喂进来。

规则（P0-7 用 8 个合成场景验证过，含边界）：

1. 开启新事件前必须满足"最近 ``start_n`` 个窗里至少 ``start_k`` 个**共识阳性**"；
   事件真实起点回溯到该窗口内的**第一个阳性窗起点**。
2. 事件一旦开启，后续阳性窗直接并入（同一事件可同时含 cry 和 speech）。
3. 距最后一个阳性窗结束超过 ``event_gap_sec`` → 判定结束；
   结束时间 = 最后一个阳性窗结束 + ``post_roll_sec``（尾巴）。
4. ``abstain`` 不推进也不打断（等价于"这一窗不存在"）。

录音（spec §6.2）
----------------
- 启动时把采集环形缓冲里最近 ``pre_roll`` 秒复制进事件 tap（P0-7 结论 4：
  启动延迟 2~3s < 5s，必然覆盖声音真实起点）；
- 活跃期间由 capture 读线程**同步追加**（tap，见 :mod:`capture`）；
- 判定结束后再等 ``post_roll`` 秒真正落盘（尾巴攒够）；
- 事件时长 = 真实声音 + 3s 尾巴；4B 分段必须用真实起止，不能拿录音时长当
  内容边界。

degraded / capture_gap
---------------------
模型异常或采集中断**不能伪装成静音结束**（spec §6.2）：窗口 abstain、或窗与窗
之间出现时间不连续（capture 断流导致 tick 缺失）都记 ``capture_gap``，事件
标记 ``degraded=True``。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .capture import AudioCapture
from .consensus import DECISION_DEGRADED, DECISION_POSITIVE, WindowVerdict
from .models import AudioEvent
from .silence import EnergyTracker, find_silence_ranges

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class AssemblerConfig:
    """事件组装参数（全部来自 settings，允许测试覆写）。"""

    hop_sec: float = 1.0
    window_sec: float = 2.0
    start_n: int = 3            # 启动判定：最近 n 个窗
    start_k: int = 2            # 其中至少 k 个阳性（2/3 规则）
    event_gap_sec: float = 3.0  # 单一间隔参数（P0-7 结论）
    pre_roll_sec: float = 5.0
    post_roll_sec: float = 3.0
    sample_rate: int = 16000
    silence_window_sec: float = 30.0
    silence_percentile: float = 20.0
    silence_min_ms: int = 500
    max_event_sec: float = 3600.0   # 防御：超长事件强制关闭（1h）

    @classmethod
    def from_settings(cls) -> "AssemblerConfig":
        from django.conf import settings

        return cls(
            hop_sec=float(settings.BABYCARE_AUDIO_DECISION_HOP_SEC),
            window_sec=float(settings.BABYCARE_AUDIO_DECISION_WINDOW_SEC),
            start_n=int(settings.BABYCARE_AUDIO_EVENT_START_N),
            start_k=int(settings.BABYCARE_AUDIO_EVENT_START_K),
            event_gap_sec=float(settings.BABYCARE_AUDIO_EVENT_GAP_SEC),
            pre_roll_sec=float(settings.BABYCARE_AUDIO_PRE_ROLL_SEC),
            post_roll_sec=float(settings.BABYCARE_AUDIO_POST_ROLL_SEC),
            sample_rate=int(settings.BABYCARE_AUDIO_SAMPLE_RATE),
            silence_window_sec=float(settings.SILENCE_ADAPTIVE_WINDOW_SEC),
            silence_percentile=float(settings.SILENCE_ADAPTIVE_PERCENTILE),
            silence_min_ms=int(settings.SILENCE_MIN_MS),
        )


# ---------------------------------------------------------------------------
# 只追加不丢弃的事件录音缓冲（tap 目标）
# ---------------------------------------------------------------------------
class AppendBuffer:
    """事件录音缓冲：只追加、不丢弃（事件必须完整，不能像 pre_roll 那样滑窗）。

    线程安全：append 来自 capture 读线程，读取来自 assembler（finalize）。
    """

    def __init__(self):
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._buf.extend(data)

    def get_bytes(self) -> bytes:
        with self._lock:
            return bytes(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def default_flac_writer(path: Path, pcm: Any, sample_rate: int) -> None:
    """默认 FLAC 落盘（音频 venv 的 soundfile；主 venv 测试用注入替身）。"""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sample_rate, format="FLAC", subtype="PCM_16")


def ts_to_dt(ts: int | float | None) -> datetime | None:
    """epoch 秒 → 本地 naive datetime（展示字段用）。

    失败（负值 / 越界；Windows 的 fromtimestamp 对负时间戳直接 OSError）→ None：
    展示字段允许为空，**不能因此把事件流程打断**。
    """
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts)
    except (OSError, ValueError, OverflowError):
        return None


def event_audio_path(media_root: str | Path, cam_id: int, started_ts: int) -> Path:
    """事件 FLAC 的落盘路径：MEDIA_ROOT/audio_events/<YYYY-MM-DD>/<cam>_<ts>.flac。

    与 frame_storage 的 ``frames/<date>/<cam>_<ts>.jpg`` 约定对齐
    （spec §6.1：绝对路径，复用 dashboard 的 _media_url_for）。
    """
    dt = ts_to_dt(started_ts)
    date_dir = dt.strftime("%Y-%m-%d") if dt else "unknown-date"
    return Path(media_root) / "audio_events" / date_dir / f"{cam_id}_{started_ts}.flac"


# ---------------------------------------------------------------------------
# 进行中的事件
# ---------------------------------------------------------------------------
@dataclass
class _PendingEvent:
    started_ts: int
    last_pos_end: int
    db_id: int | None = None
    tap_key: str = ""
    #: ``{label: [[s,e], ...]}`` 各标签阳性窗（含事件起点回溯窗）
    label_windows: dict[str, list[list[int]]] = field(default_factory=dict)
    degraded: bool = False
    capture_gaps: list[list[int]] = field(default_factory=list)
    recording_lead_sec: float = 0.0
    #: 判定结束后的等待尾巴期：close_target_ts 之后才真正落盘
    closing: bool = False
    close_target_ts: int = 0


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------
class EventAssembler:
    """一路摄像头的事件状态机 + 录音 + 落盘 + 落库。"""

    def __init__(
        self,
        camera_id: int,
        capture: AudioCapture,
        config: AssemblerConfig | None = None,
        media_root: str | Path = "",
        flac_writer: Callable[[Path, Any, int], None] = default_flac_writer,
        worker_epoch: str = "",
    ):
        self.camera_id = camera_id
        self._capture = capture
        self._cfg = config or AssemblerConfig.from_settings()
        self._media_root = media_root
        self._write_flac = flac_writer
        #: 产出事件的 worker 租约 epoch（写进事件行，供 recover 限定回收范围）
        self._worker_epoch = worker_epoch

        #: 最近 start_n 个决策窗：``[(x, win_start, win_end, positive_labels)]``
        #: x ∈ True(阳性) / False(阴性) / None(abstain)
        self._recent: deque[tuple] = deque(maxlen=max(self._cfg.start_n, 1))
        self._pending: _PendingEvent | None = None
        self._closing: list[_PendingEvent] = []   # 判定结束、等尾巴的事件
        # 状态锁：on_window 走 inference 线程，stop() 走 manager 线程——
        # 正常情况下 inference.stop() 先 join 再 asm.stop() 保证串行；
        # 但 join 超时的边缘情况下两者会并发（code review 发现的竞态）。
        self._state_lock = threading.Lock()
        self._energy = EnergyTracker(
            window_sec=self._cfg.silence_window_sec,
            percentile=self._cfg.silence_percentile,
            hop_sec=self._cfg.hop_sec,
        )
        self._last_win_end: int | None = None
        self._event_seq = 0

    # ------------------------------------------------------------------
    # 对外：每个推理窗喂一次
    # ------------------------------------------------------------------
    def on_window(
        self,
        window_start_ts: int,
        window_end_ts: int,
        verdict: WindowVerdict,
        audio: Any,
    ) -> list[int]:
        """喂入一个决策窗；返回本轮真正落盘完成的事件 id 列表。"""
        self._energy.push(audio)

        # 时间不连续 = capture 断流期间 tick 缺失 → capture_gap（spec §6.2）
        if (
            self._last_win_end is not None
            and window_start_ts - self._last_win_end > self._cfg.hop_sec + 0.5
        ):
            gap = [self._last_win_end, window_start_ts]
            logger.warning(
                "[audio cam#%s] 事件窗口不连续 %s~%ss（capture_gap）",
                self.camera_id, gap[0], gap[1],
            )
            if self._pending is not None and not self._pending.closing:
                self._pending.degraded = True
                self._pending.capture_gaps.append(gap)
        self._last_win_end = window_end_ts

        if verdict.decision == DECISION_POSITIVE:
            x = True
        elif verdict.decision == DECISION_DEGRADED:
            x = None      # abstain：不推进也不打断（spec §4.3.4）
        else:
            x = False
        positive_labels = tuple(verdict.positive_labels())

        # abstain 窗：事件期间 → degraded（不能伪装成静音结束）
        if x is None and self._pending is not None and not self._pending.closing:
            self._pending.degraded = True
            self._pending.capture_gaps.append([window_start_ts, window_end_ts])

        self._recent.append((x, window_start_ts, window_end_ts, positive_labels))

        with self._state_lock:
            self._start_or_extend(x, window_start_ts, window_end_ts, positive_labels)
            self._maybe_judge_close(window_end_ts)
            return self._flush_closed(window_end_ts)

    # ------------------------------------------------------------------
    # 停止（worker 退出）：所有进行中/等尾巴的事件立即收尾
    # ------------------------------------------------------------------
    def stop(self) -> list[int]:
        with self._state_lock:
            ids: list[int] = []
            if self._pending is not None:
                ev = self._pending
                self._pending = None
                ids.extend(self._finalize(ev))
            while self._closing:
                ev = self._closing.pop(0)
                ids.extend(self._finalize(ev))
            return ids

    # ------------------------------------------------------------------
    # 状态机主体（照 p0_7 参考实现）
    # ------------------------------------------------------------------
    def _start_or_extend(
        self,
        x: bool | None,
        win_start: int,
        win_end: int,
        positive_labels: tuple[str, ...],
    ) -> None:
        if x is not True:
            return

        if self._pending is None:
            # 启动判定：最近 start_n 个窗里至少 start_k 个阳性
            n_pos = sum(1 for t in self._recent if t[0] is True)
            if n_pos < self._cfg.start_k:
                return
            # 事件起点回溯到窗口内第一个阳性窗起点
            first = next(t for t in self._recent if t[0] is True)
            # 回溯的阳性窗也计入 label_windows（含事件起点窗）
            ev = _PendingEvent(started_ts=first[1], last_pos_end=win_end)
            for t in self._recent:
                if t[0] is True:
                    for label in t[3]:
                        ev.label_windows.setdefault(label, []).append([t[1], t[2]])
            self._open_event(ev, now_ts=win_end)
            self._pending = ev
            logger.info(
                "[audio cam#%s] 事件启动：起点回溯到 %s（最近 %d 窗 %d 阳性，标签 %s）",
                self.camera_id, ev.started_ts, self._cfg.start_n, n_pos,
                sorted(ev.label_windows),
            )
        else:
            # 已在事件中：阳性窗并入
            ev = self._pending
            ev.last_pos_end = win_end
            for label in positive_labels:
                ev.label_windows.setdefault(label, []).append([win_start, win_end])

    def _open_event(self, ev: _PendingEvent, now_ts: int) -> None:
        """复制 pre_roll、挂 tap、落 recording 行。"""
        # 1) 录音缓冲：先把 ring 里最近 pre_roll 秒复制进去（spec §6.2）
        tap = AppendBuffer()
        pre_bytes = self._capture.pre_roll_snapshot()
        tap.append(pre_bytes)
        # 录音起点 ≈ now - 已有字节数；lead = 事件起点比录音起点晚多少
        recording_start = now_ts - len(pre_bytes) / (self._cfg.sample_rate * 2)
        ev.recording_lead_sec = max(ev.started_ts - recording_start, 0.0)

        # 2) 挂 tap：读线程持续追加
        self._event_seq += 1
        ev.tap_key = f"event-{self.camera_id}-{ev.started_ts}-{self._event_seq}"
        self._capture.add_tap(ev.tap_key, tap)

        # 3) 落 recording 行（崩溃可追溯，由 recover_stale_events 回收）
        try:
            row = AudioEvent.objects.create(
                camera_id=self.camera_id,
                status=AudioEvent.STATUS_RECORDING,
                started_at_ts=ev.started_ts,
                started_at=ts_to_dt(ev.started_ts),
                recording_lead_sec=round(ev.recording_lead_sec, 3),
                worker_epoch=self._worker_epoch,
            )
            ev.db_id = row.id
        except Exception:  # noqa: BLE001
            logger.exception("[audio cam#%s] AudioEvent(recording) 落库失败", self.camera_id)

    # ------------------------------------------------------------------
    # 关闭判定：每个窗都查（与是否阳性无关，时钟在走）
    # ------------------------------------------------------------------
    def _maybe_judge_close(self, now_ts: int) -> None:
        ev = self._pending
        if ev is None or ev.closing:
            return
        gap = now_ts - ev.last_pos_end
        too_long = (now_ts - ev.started_ts) > self._cfg.max_event_sec
        if gap > self._cfg.event_gap_sec or too_long:
            ev.closing = True
            ev.close_target_ts = ev.last_pos_end + int(self._cfg.post_roll_sec)
            self._closing.append(ev)
            self._pending = None
            logger.info(
                "[audio cam#%s] 事件判定结束：起点 %s 距末阳性 %.1fs"
                "（gap=%.0fs%s）→ 等 %.0fs 尾巴后落盘",
                self.camera_id, ev.started_ts, gap, self._cfg.event_gap_sec,
                "，超长强制关闭" if too_long else "", self._cfg.post_roll_sec,
            )

    def _flush_closed(self, now_ts: int) -> list[int]:
        """等尾巴的事件攒够了 → 真正落盘。"""
        done: list[int] = []
        remaining: list[_PendingEvent] = []
        for ev in self._closing:
            if now_ts >= ev.close_target_ts:
                done.extend(self._finalize(ev))
            else:
                remaining.append(ev)
        self._closing = remaining
        return done

    # ------------------------------------------------------------------
    # 落盘 + 落库
    # ------------------------------------------------------------------
    def _finalize(self, ev: _PendingEvent) -> list[int]:
        """结束事件：摘 tap → 静音区间 → FLAC 落盘 → 更新 AudioEvent。"""
        tap = self._capture.remove_tap(ev.tap_key)
        if tap is None:
            logger.error("[audio cam#%s] tap 已丢失（key=%s）", self.camera_id, ev.tap_key)
            return []

        ended_ts = ev.last_pos_end + int(self._cfg.post_roll_sec)
        try:
            import numpy as np

            data = tap.get_bytes()
            pcm = (
                np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
                if data else np.zeros(0, dtype="float32")
            )
            duration = len(pcm) / float(self._cfg.sample_rate)

            # 静音区间：录音内偏移 → 事件内偏移（减 lead）
            raw_ranges = find_silence_ranges(
                pcm,
                sample_rate=self._cfg.sample_rate,
                threshold_db=self._energy.threshold_db,
                min_ms=self._cfg.silence_min_ms,
            )
            lead = ev.recording_lead_sec
            silence = [
                [round(max(s - lead, 0.0), 3), round(e - lead, 3)]
                for s, e in raw_ranges
                if e > lead
            ]

            audio_path = ""
            if len(pcm) > 0:
                path = event_audio_path(
                    self._media_root, self.camera_id, ev.started_ts,
                )
                try:
                    self._write_flac(path, pcm, self._cfg.sample_rate)
                    audio_path = str(path)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[audio cam#%s] FLAC 落盘失败（事件保留，audio_path 留空）",
                        self.camera_id,
                    )

            detected = {
                label: {
                    "first_positive_ts": wins[0][0],
                    "last_positive_ts": wins[-1][1],
                    "positive_windows": wins,
                }
                for label, wins in sorted(ev.label_windows.items())
            }

            if ev.db_id is not None:
                AudioEvent.objects.filter(id=ev.db_id).update(
                    status=AudioEvent.STATUS_PENDING_DESCRIPTION,
                    ended_at_ts=ended_ts,
                    ended_at=ts_to_dt(ended_ts),
                    duration_sec=round(duration, 3),
                    audio_path=audio_path,
                    audio_format="flac",
                    detected_labels=detected,
                    silence_ranges=silence,
                    degraded=ev.degraded,
                    recording_lead_sec=round(lead, 3),
                )
            logger.info(
                "[audio cam#%s] 事件完成：[%s ~ %s] %.1fs（含后录），标签 %s，"
                "静音 %d 段，degraded=%s，flac=%s",
                self.camera_id, ev.started_ts, ended_ts, duration,
                sorted(detected), len(silence), ev.degraded,
                audio_path or "-",
            )
            return [ev.db_id] if ev.db_id is not None else []
        except Exception:  # noqa: BLE001
            logger.exception("[audio cam#%s] 事件收尾失败", self.camera_id)
            if ev.db_id is not None:
                AudioEvent.objects.filter(id=ev.db_id).update(
                    status=AudioEvent.STATUS_FAILED, degraded=True,
                )
            return []

    # ------------------------------------------------------------------
    # 遗留 recording 行回收（worker 崩溃后重启）
    # ------------------------------------------------------------------
    @staticmethod
    def recover_stale_events(stale_epochs: set[str] | None = None) -> int:
        """把上次 worker 异常退出留下的 recording 行标成 failed。

        worker 是 AudioEvent 的唯一写入方；出现 recording 状态的旧行只可能是
        上个进程没走完收尾。不回收的话页面会永远显示"录制中"。

        ``stale_epochs`` 限定回收范围（默认 ``{""}`` = 引入 ``worker_epoch``
        之前的历史行）：只回收"本次接管前那个租约"产出的行，其它 epoch 的行一律
        不动——即便出现另一个异常存活的实例，也不会把它的在途事件判成 failed。
        """
        epochs = {""} if stale_epochs is None else set(stale_epochs)
        qs = AudioEvent.objects.filter(
            status=AudioEvent.STATUS_RECORDING, worker_epoch__in=epochs,
        )
        n = 0
        for row in qs:
            row.status = AudioEvent.STATUS_FAILED
            row.degraded = True
            row.ended_at_ts = row.ended_at_ts or int(time.time())
            row.ended_at = row.ended_at or ts_to_dt(row.ended_at_ts)
            row.save(update_fields=[
                "status", "degraded", "ended_at_ts", "ended_at", "updated_at",
            ])
            n += 1
        if n:
            logger.warning("[audio] 回收了 %d 条上次 worker 中断遗留的 recording 事件", n)
        return n

    # ------------------------------------------------------------------
    @property
    def status(self) -> dict[str, Any]:
        return {
            "pending": self._pending is not None,
            "closing": len(self._closing),
            "recent": [t[0] for t in self._recent],
            "energy": self._energy.snapshot(),
        }

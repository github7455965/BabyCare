"""自适应静音检测（spec §6.3，Phase 3）。

为什么不用固定 dBFS 阈值
-----------------------
家用摄像头 mic 普遍开 AGC（自动增益），会把绝对电平拉平——``-42 dBFS`` 这类
固定值在不同设备、不同 AGC 状态下不可比，现场校准也救不回来。

改为**相对自适应阈值**：维护最近 N 秒（默认 30s）的窗口能量分布，取低分位数
（默认 P20）作为动态静音线。它只依赖"当前环境的能量分布"，与绝对电平无关。

两段职责
--------
1. :class:`EnergyTracker`：每 tick 喂推理窗音频 → 维护分布 → 出动态阈值；
2. :func:`find_silence_ranges`：事件结束时扫描整段录音 → 静音区间列表
   （相对**录音起点**的秒偏移；换算成事件内偏移要减
   :attr:`~apps.audio_detect.models.AudioEvent.recording_lead_sec`）。

纯 numpy 计算，无模型、无 DB，便于单测。
"""

from __future__ import annotations

import math
from collections import deque

_EPS = 1e-10


def rms_db(audio) -> float:
    """整段音频的 RMS 电平（dBFS，float32 [-1,1] 输入）。"""
    n = len(audio)
    if n == 0:
        return -120.0
    acc = 0.0
    # 分块累加，避免超长数组的精度损失
    step = 65536
    for i in range(0, n, step):
        chunk = audio[i:i + step]
        acc += float((chunk * chunk).sum())
    return 20.0 * math.log10(math.sqrt(acc / n) + _EPS)


class EnergyTracker:
    """最近 N 秒的窗口能量分布 → 动态静音线（低分位数）。

    :param window_sec: 分布统计窗口（默认 30s，spec §6.3）
    :param percentile: 低分位数（默认 20 → P20）
    :param hop_sec: 每 tick 一个样本（与推理 hop 一致）
    """

    def __init__(
        self,
        window_sec: float = 30.0,
        percentile: float = 20.0,
        hop_sec: float = 1.0,
    ):
        self._percentile = float(percentile)
        capacity = max(int(window_sec / max(hop_sec, 1e-6)), 1)
        self._samples: deque[float] = deque(maxlen=capacity)
        self._min_samples = max(min(capacity, 5), 1)   # 至少几个样本才给阈值

    def push(self, audio) -> None:
        """喂一个推理窗的音频（tick 一次喂一次）。"""
        self._samples.append(rms_db(audio))

    @property
    def threshold_db(self) -> float | None:
        """当前动态静音线；样本不足时 None（调用方应跳过静音判定）。"""
        if len(self._samples) < self._min_samples:
            return None
        xs = sorted(self._samples)
        # 线性插值分位数
        idx = (self._percentile / 100.0) * (len(xs) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return xs[lo]
        return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

    def snapshot(self) -> dict:
        return {
            "n_samples": len(self._samples),
            "percentile": self._percentile,
            "threshold_db": self.threshold_db,
        }


def find_silence_ranges(
    pcm,
    sample_rate: int = 16000,
    threshold_db: float | None = None,
    min_ms: int = 500,
    frame_ms: int = 50,
) -> list[tuple[float, float]]:
    """扫描整段录音，返回低于动态线的连续区间。

    :param pcm: float32 [-1,1]，事件完整录音（含 pre/post roll）
    :param threshold_db: 动态静音线（EnergyTracker.threshold_db）
    :return: ``[(start_sec, end_sec), ...]`` 相对**录音起点**；None 时返回空
    """
    if threshold_db is None or len(pcm) == 0:
        return []

    import numpy as np

    frame_len = max(int(sample_rate * frame_ms / 1000), 1)
    n = len(pcm)
    n_frames = n // frame_len
    if n_frames == 0:
        return []

    # 分块计算帧能量：一次性 reshape 整段录音在超长事件（防御值 1h）下会
    # 多占几百 MB 临时内存；按 ~50s 一块扫描，内存恒定。
    block_frames = 1000
    quiet = np.empty(n_frames, dtype=bool)
    for start_f in range(0, n_frames, block_frames):
        end_f = min(start_f + block_frames, n_frames)
        frames = np.asarray(
            pcm[start_f * frame_len: end_f * frame_len], dtype="float32",
        ).reshape(end_f - start_f, frame_len)
        # 与 rms_db 用同一公式 sqrt(mean)+EPS —— 能量分布（P20）与帧判定
        # 必须用同一把尺，否则全静默时 P20=-200dB 而帧=-194dB，永远判不出静音。
        rms = np.sqrt((frames * frames).mean(axis=1))
        db = 20.0 * np.log10(rms + _EPS)
        # <= 而非 <：恒定电平录音（如全静默误报事件）时 P20 = 当前电平，
        # 严格小于会让"整段都是静音"判不出来（退化场景）。
        quiet[start_f:end_f] = db <= threshold_db

    min_frames = max(int(min_ms / max(frame_ms, 1)), 1)
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= min_frames:
                ranges.append((start * frame_ms / 1000.0, i * frame_ms / 1000.0))
            start = None
    if start is not None and n_frames - start >= min_frames:
        ranges.append((start * frame_ms / 1000.0, n_frames * frame_ms / 1000.0))
    return ranges

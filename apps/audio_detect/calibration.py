"""Phase 8：阈值标定的**统计核心**（spec §12 P0-5 / P0-6 推迟项的落地工具）。

背景
----
P0-3 / P0-5 / P0-6 当时推迟：没有哭声标注集，无法离线标定。改为**上线后用生产
``SoundDetectionLog``（含近阈值窗）+ 人工试听**来标定 —— 本模块就是把"统计"这半边
自动化，"试听与拍板"留给人工（spec §12「推迟项的处理方式」）。

能回答的三个问题
----------------
1. **分数长什么样**（:func:`score_distribution`）：per (模型 × 业务标签) 的分位数，
   用来圈定候选阈值区间；
2. **换阈值会发生什么**（:func:`threshold_scan`）：对每个候选阈值 t，统计落库窗里
   ``and`` / ``or`` / 单侧阳性的数量 —— 阈值越低"判阳窗"越多，人工按可接受的通知
   频率反推阈值；
3. **GAP 参数合不合理**（:func:`event_report`）：事件时长分布 + 相邻事件间隔分布。
   间隔在 ``EVENT_GAP_SEC`` 附近聚集 = 有些本应合并的事件被拆开了。

必须写在脸上的偏差
------------------
落库是**稀疏**的（positive / near_threshold / state_change 才落库，spec §5.1），
所以本报表的分数分布**不代表全体窗口** —— 它偏向"有动静"的窗口。拿它定相对阈值
（分位点）没问题，别拿它当全局误报率。
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Iterable, Optional

from apps.audio_detect.label_map import (
    BUSINESS_CRY,
    BUSINESS_SPEECH,
    MODEL_PANNS,
    MODEL_YAMNET,
)
from apps.audio_detect.models import AudioEvent, SoundDetectionLog

#: 报表覆盖的业务标签（spec §4.2 当前只有这两个）
LABELS = (BUSINESS_CRY, BUSINESS_SPEECH)

#: 分位数位点（线性插值）
QUANTILE_POINTS = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


# ---------------------------------------------------------------------------
# 取数
# ---------------------------------------------------------------------------
def load_logs(days: Optional[int] = None):
    """取参与统计的落库窗（``-window_end_ts`` 倒序；稀疏落库，量级可控）。"""
    qs = SoundDetectionLog.objects.all()
    if days:
        cutoff = int(time.time()) - int(days) * 86400
        qs = qs.filter(window_end_ts__gte=cutoff)
    return list(qs.order_by("-window_end_ts"))


def _model_scores(log: SoundDetectionLog, model: str) -> dict[str, float]:
    """从 payload 取业务标签分数（``InferenceService._model_payload`` 的 ``scores``）。"""
    payload = (log.yamnet_payload if model == MODEL_YAMNET else log.panns_payload) or {}
    scores = payload.get("scores") or {}
    return scores if isinstance(scores, dict) else {}


def iter_label_scores(logs: Iterable[SoundDetectionLog]):
    """→ ``(label, y, p)`` 三元组流；``y`` / ``p`` 缺失为 ``None``。"""
    for log in logs:
        y = _model_scores(log, MODEL_YAMNET)
        p = _model_scores(log, MODEL_PANNS)
        for label in LABELS:
            yield label, y.get(label), p.get(label)


# ---------------------------------------------------------------------------
# 统计原语
# ---------------------------------------------------------------------------
def quantile(values: list[float], q: float) -> Optional[float]:
    """线性插值分位数；空列表返回 ``None``。纯 Python 实现（不依赖 numpy）。"""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def score_distribution(logs: list[SoundDetectionLog]) -> dict:
    """per (模型 × 标签) 的分位数 + 样本数。

    Returns:
        ``{(model, label): {"n": int, "min": f, "q25": f, ..., "max": f}}``
        没有样本的 (model, label) 不出现在结果里。
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for log in logs:
        for model in (MODEL_YAMNET, MODEL_PANNS):
            for label, score in _model_scores(log, model).items():
                if score is None:
                    continue
                buckets.setdefault((model, label), []).append(float(score))

    out: dict[tuple[str, str], dict] = {}
    for key, values in buckets.items():
        out[key] = {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            **{f"q{int(q * 100)}": quantile(values, q) for q in QUANTILE_POINTS},
        }
    return out


def threshold_scan(
    logs: list[SoundDetectionLog],
    thresholds: Iterable[float],
) -> list[dict]:
    """候选阈值扫描（spec Phase 8 动作 1 + P0-6）。

    对每个 (label, t) 统计**落库窗**里：

    - ``and_n``：两模型都 ≥ t（``BABYCARE_AUDIO_CONSENSUS=and`` 会判阳）；
    - ``or_n``：任一模型 ≥ t（``or`` 会判阳）；
    - ``single_n``：or 阳但 and 不阳（单侧阳性，即 and/or 的差异部分）。

    双侧都缺分数（标签 disabled / 模型异常窗）的窗不计数。
    """
    thresholds = [round(float(t), 4) for t in thresholds]
    counts: dict[str, dict[float, dict[str, int]]] = {
        label: {t: {"and": 0, "or": 0, "single": 0} for t in thresholds}
        for label in LABELS
    }
    for label, y, p in iter_label_scores(logs):
        for t in thresholds:
            y_pos = y is not None and y >= t
            p_pos = p is not None and p >= t
            if not (y_pos or p_pos):
                continue
            row = counts[label][t]
            row["or"] += 1
            if y_pos and p_pos:
                row["and"] += 1
            else:
                row["single"] += 1
    return [
        {
            "label": label,
            "threshold": t,
            "and_n": row["and"],
            "or_n": row["or"],
            "single_n": row["single"],
        }
        for label in LABELS
        for t, row in counts[label].items()
    ]


def single_side_samples(
    logs: list[SoundDetectionLog],
    threshold: float | None = None,
    limit: int = 30,
) -> list[dict]:
    """单侧阳性窗明细（人工去页面复核误报 / 漏报用）。

    :param threshold: 缺省用当前生产阈值快照（``consensus_payload.thresholds``）；
        传入则按候选阈值重判。
    """
    if threshold is None:
        threshold = _current_threshold(logs)
    samples: list[dict] = []
    for log in logs:
        if len(samples) >= limit:
            break
        for label, y, p in iter_label_scores([log]):
            y_pos = y is not None and y >= threshold
            p_pos = p is not None and p >= threshold
            if (y_pos or p_pos) and not (y_pos and p_pos):
                samples.append({
                    "log_id": log.id,
                    "window_end_ts": log.window_end_ts,
                    "camera_id": log.camera_id,
                    "label": label,
                    "yamnet": y,
                    "panns": p,
                    "positive_model": MODEL_YAMNET if y_pos else MODEL_PANNS,
                })
                break  # 每窗取一个标签即可（明细用途）
    return samples


def _current_threshold(logs: list[SoundDetectionLog]) -> float:
    """从最近的落库窗读当前生产阈值快照；读不到退回 0.3（settings 占位值）。"""
    for log in logs:
        payload = log.consensus_payload if isinstance(log.consensus_payload, dict) else {}
        thresholds = payload.get("thresholds") or {}
        cry = thresholds.get(MODEL_YAMNET) or {}
        v = cry.get(BUSINESS_CRY)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.3


# ---------------------------------------------------------------------------
# 事件 / 参数复核（spec Phase 8 动作 3）
# ---------------------------------------------------------------------------
def event_report(days: Optional[int] = None, gap_sec: float = 3.0) -> dict:
    """事件时长 / 相邻间隔分布 + 描述状态，用于复核 ``EVENT_GAP_SEC`` 等参数。

    间隔判读（写在命令输出里的建议）：
    - 大量间隔 **略大于** ``GAP_SEC`` → 有些本应合并的事件被拆开了（GAP 偏小）；
    - 间隔都远大于 GAP → GAP 没有过度合并的迹象（偏保守是安全的）。
    """
    qs = AudioEvent.objects.exclude(status=AudioEvent.STATUS_RECORDING)
    if days:
        cutoff = int(time.time()) - int(days) * 86400
        qs = qs.filter(started_at_ts__gte=cutoff)
    events = list(qs.order_by("camera_id", "started_at_ts"))

    durations = [
        float(ev.ended_at_ts - ev.started_at_ts)
        for ev in events
        if ev.ended_at_ts and ev.ended_at_ts >= ev.started_at_ts
    ]
    gaps_by_cam: dict[int, list[float]] = {}
    prev: dict[int, int] = {}
    for ev in events:
        cam = ev.camera_id
        if cam is None:
            continue
        if cam in prev:
            gaps_by_cam.setdefault(cam, []).append(float(ev.started_at_ts - prev[cam]))
        prev[cam] = ev.started_at_ts

    all_gaps = [g for gaps in gaps_by_cam.values() for g in gaps]
    near_gap = sum(1 for g in all_gaps if g <= gap_sec * 3)

    return {
        "event_count": len(events),
        "recording_count": AudioEvent.objects.filter(
            status=AudioEvent.STATUS_RECORDING,
        ).count(),
        "duration": {
            "n": len(durations),
            "q50": quantile(durations, 0.5),
            "q90": quantile(durations, 0.9),
            "max": max(durations) if durations else None,
        },
        "gap": {
            "n": len(all_gaps),
            "min": min(all_gaps) if all_gaps else None,
            "q50": quantile(all_gaps, 0.5),
            "max": max(all_gaps) if all_gaps else None,
            "near_gap_count": near_gap,
        },
        "degraded_count": sum(1 for ev in events if ev.degraded),
        "desc_status": dict(
            Counter(ev.description_status for ev in events),
        ),
    }


def reason_counts(logs: list[SoundDetectionLog]) -> dict:
    """落库原因 / 决策 / 失败原因计数（样本概况）。"""
    return {
        "window_total": len(logs),
        "log_reason": dict(Counter(l.log_reason or "-" for l in logs)),
        "decision": dict(Counter(l.decision for l in logs)),
        "failure_reason": dict(
            Counter(l.failure_reason for l in logs if l.failure_reason),
        ),
    }


__all__ = [
    "LABELS",
    "event_report",
    "iter_label_scores",
    "load_logs",
    "quantile",
    "reason_counts",
    "score_distribution",
    "single_side_samples",
    "threshold_scan",
]

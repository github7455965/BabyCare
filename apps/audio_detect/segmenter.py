"""描述分段规划（Phase 4，spec §6.4）。

规则（与 spec 逐条对应）
------------------------
1. 目标片段长度 10–60 秒；
2. **优先在静音点处切分**——切点取静音区间中点，保留事件真实时间边界；
3. 事件 <10s 但 >1s 正常发送（单段）；**<1s 丢弃**；
4. 找不到静音点时：片段超过 90s 按 60s **强制切分**；
5. 任何 4B 请求都**不超过 90 秒**（硬上限）；
6. 事件 >90s 但有静音点 → 按静音点拆；无静音点 → 连续 60s 拆。

切点选择（贪心）
---------------
从 ``pos`` 出发、剩余长度超过 60s 时，在 ``(pos+10, pos+90]`` 内找静音中点：

- 有 ≤ pos+60 的 → 取**最晚**一个（片段尽量接近目标上限，减少段数）；
- 否则取 60~90 区间内**最早**一个（超目标后尽快落刀，离 90 硬上限越远越安全）；
- 都没有 → 在 pos+60 强制切（规则 4）。

尾巴处理：最后一段 <1s 时，若并入前段后仍 ≤90s 则并入（不丢音频），否则丢弃
（规则 3 的"丢弃"只发生在这种极端角落）。

本模块是纯函数，不碰 DB / 文件 / 网络，分段边界（10/60/90s）由单测覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmenterConfig:
    """分段参数（全部来自 settings，允许测试覆写）。"""

    min_sec: float = 10.0        # 目标下限：切点不早于 pos+min_sec（避免碎段）
    max_sec: float = 60.0        # 目标上限：超过就要找切点
    hard_max_sec: float = 90.0   # 硬上限：任何 4B 请求不超过此时长
    discard_sec: float = 1.0     # 短于此时长的片段丢弃

    @classmethod
    def from_settings(cls) -> "SegmenterConfig":
        from django.conf import settings

        return cls(
            min_sec=float(settings.BABYCARE_AUDIO_SEGMENT_MIN_SEC),
            max_sec=float(settings.BABYCARE_AUDIO_SEGMENT_MAX_SEC),
            hard_max_sec=float(settings.BABYCARE_AUDIO_SEGMENT_HARD_MAX_SEC),
            discard_sec=float(settings.BABYCARE_AUDIO_SEGMENT_DISCARD_SEC),
        )


@dataclass(frozen=True)
class SegmentPlan:
    """一个待描述的片段（偏移均相对**事件真实起点** started_at_ts）。"""

    start_offset: float
    end_offset: float
    #: 本段是否是在静音点之后开始的（spec §5.3 has_silence_before）
    has_silence_before: bool = False
    #: 本段终点是否为"无静音点时的 60s 强制切"（审计用）
    forced: bool = False

    @property
    def duration(self) -> float:
        return self.end_offset - self.start_offset


def silence_midpoints(silence_ranges: list, duration_sec: float) -> list[float]:
    """静音区间 ``[[s, e], ...]`` → 排序后的中点列表（只保留事件内部切点）。"""
    mids = []
    for item in silence_ranges or []:
        try:
            s, e = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if e <= s:
            continue
        mid = (s + e) / 2.0
        if 0.0 < mid < duration_sec:
            mids.append(mid)
    return sorted(mids)


def plan_segments(
    duration_sec: float,
    silence_ranges: list | None = None,
    config: SegmenterConfig | None = None,
) -> list[SegmentPlan]:
    """按 spec §6.4 规划分段；返回空列表 = 事件太短、整体丢弃。"""
    cfg = config or SegmenterConfig()
    dur = float(duration_sec)
    if dur < cfg.discard_sec:
        return []
    if dur <= cfg.max_sec:
        return [SegmentPlan(0.0, round(dur, 3))]

    cuts = silence_midpoints(silence_ranges or [], dur)
    segs: list[SegmentPlan] = []
    pos = 0.0
    next_has_silence = False          # 第一段之前没有"静音点切分"
    while dur - pos > cfg.max_sec:
        lo = pos + cfg.min_sec
        hi = pos + cfg.hard_max_sec
        cands = [c for c in cuts if lo < c <= hi]
        in_target = [c for c in cands if c <= pos + cfg.max_sec]
        if in_target:
            cut, is_silence = max(in_target), True
        elif cands:
            cut, is_silence = min(cands), True
        else:
            cut, is_silence = pos + cfg.max_sec, False
        segs.append(SegmentPlan(
            round(pos, 3), round(cut, 3),
            has_silence_before=next_has_silence, forced=not is_silence,
        ))
        pos = cut
        next_has_silence = is_silence

    tail = dur - pos
    if tail >= cfg.discard_sec:
        segs.append(SegmentPlan(
            round(pos, 3), round(dur, 3), has_silence_before=next_has_silence,
        ))
    elif segs:
        prev = segs[-1]
        if prev.duration + tail <= cfg.hard_max_sec:
            # 尾巴不足 1s：并入前段（不超 90s 硬上限就不丢音频）
            segs[-1] = SegmentPlan(
                prev.start_offset, round(dur, 3),
                has_silence_before=prev.has_silence_before, forced=prev.forced,
            )
        # 否则只能丢弃这 <1s 的尾巴（spec §6.4）
    return segs


__all__ = ["SegmentPlan", "SegmenterConfig", "plan_segments", "silence_midpoints"]

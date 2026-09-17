"""
FrameSelector（v6 选 3 帧）

设计要点
--------
- 窗口已由 PromptCursor.try_advance 找好（保证 window_sec 个连续 ts、target_classes 对应
  has_<name> 字段都 ≠ None）
- 本模块只负责"在窗口 ts_list 内选 3 帧"
- 时间顺序硬约束：3 张图必须按 ts 升序（VLM 理解"开始-过程-结尾"）

v2 多类别
---------
- target_classes 是 VLMPromptConfig.target_classes 解析出的列表（如 ["baby"] / ["person"] / ["cat"] / 多选）
- 一帧视为"有目标"当且仅当 target_classes 中任一对应 has_<name> 为 True
- 其它未选中的 class 不参与判定（多类别 OR 语义）

算法（V2：最大化 True + 保证 ts 升序）
------------------------------------
1. 收集窗口里所有 True ts 和 False ts（True ts 按 ts 升序）
2. 按 True 数量分支：
   - True >= 3：严格三段 → [true[0], true[len//2], true[-1]]（ts 天然升序）
   - True == 2：true_ts 全部用，再选 1 张"位置最靠中"的 False → 排序
   - True == 1：1 张 True + 2 张"位置最靠中"的 False → 排序
   - True == 0：返回 None（全窗口无目标）

V2 vs V1 的关键改动
-------------------
- V1（已删）：基于"默认候选位置 [首, 中, 尾]" + True/False 替换 → 边界 case 行为不一致
- V2：直接按"窗口里 True 总数"决策 → 行为统一、可预测
- v2.1：True 判定从硬编码 has_baby 改为按 target_classes 动态 getattr
"""

from __future__ import annotations

from typing import List, Optional

from apps.yolo_detect.frame_queue import FrameItem


# 已知 target_classes → FrameItem 属性名映射。FrameItem 当前硬编码支持 baby/person/cat。
def _has_target(item: FrameItem, target_classes: List[str]) -> bool:
    """任一 target_classes 对应 attr 为 True → True。

    未定义的 attr（None 或没该字段）按 False 处理。
    """
    for c in target_classes:
        v = getattr(item, f"has_{c}", None)
        if v is True:
            return True
    return False


def _has_target_attr(item: FrameItem, target_classes: List[str]) -> Optional[bool]:
    """所有 target_classes 对应 attr 都 ≠ None 且任一 True → True。

    用于 try_advance 判断"该帧所有 target 都已推理完"。
    """
    any_known = False
    for c in target_classes:
        v = getattr(item, f"has_{c}", None)
        if v is None:
            return None
        any_known = True
    if not any_known:
        return None
    return _has_target(item, target_classes)


def select_three_frames(
    frames: List[FrameItem],
    ts_list: List[int],
    target_classes: Optional[List[str]] = None,
) -> Optional[List[FrameItem]]:
    """从窗口的帧池中挑 3 帧（最大化 True + 保证 ts 升序）。

    Args:
        frames: FrameQueue 中 ts 在 ts_list 内的 FrameItem（按 ts 升序）
        ts_list: 窗口内连续的 ts 列表（长度 = window_sec，按 ts 升序）
        target_classes: 目标类别列表（默认 ["baby"] 兼容 v1）

    Returns:
        3 个 FrameItem（按 ts 升序）或 None（全 False / 数据缺失）
    """
    target_classes = target_classes or ["baby"]
    if not ts_list or not frames:
        return None

    # ts → FrameItem 索引
    by_ts: dict = {f.ts: f for f in frames}

    # 验证 ts_list 里的每个 ts 都能在 frames 查到（try_advance 保证了，但保险一下）
    for ts in ts_list:
        if ts not in by_ts:
            return None

    # 分类 True / False（按 ts 升序）
    true_ts: List[int] = [ts for ts in ts_list if _has_target(by_ts[ts], target_classes)]
    false_ts: List[int] = [ts for ts in ts_list if not _has_target(by_ts[ts], target_classes)]

    n_true = len(true_ts)

    # ---- True >= 3：严格三段 ----
    if n_true >= 3:
        mid = true_ts[len(true_ts) // 2]  # 偶数偏右（与 len//2 一致）
        picked_ts = [true_ts[0], mid, true_ts[-1]]
        return [by_ts[ts] for ts in picked_ts]

    # ---- True == 2：用 2 张 True + 1 张"位置最靠中"的 False ----
    if n_true == 2:
        if not false_ts:
            # 理论上不可能：True=2 + False=0 = 窗口长 2，不满足 try_advance window_sec>=5
            return None
        false_mid = false_ts[len(false_ts) // 2]
        picked_ts = sorted([true_ts[0], true_ts[-1], false_mid])
        return [by_ts[ts] for ts in picked_ts]

    # ---- True == 1：1 张 True + 2 张"位置最靠中"的 False ----
    if n_true == 1:
        if len(false_ts) < 2:
            # 理论上不可能（窗口长至少 5）
            return None
        # 选 2 张"位置最靠中"的 False → 取 false_ts 里最靠中间的两个
        # （先取中间偏右一个，再取中间偏左一个 → 排序后得到居中的 2 张）
        mid_right = false_ts[len(false_ts) // 2]
        # 中间偏左：在 false_ts 里"位置最靠中但比 mid_right 早"的 ts
        mid_left = false_ts[(len(false_ts) - 1) // 2]  # 偶数偏左
        picked_ts = sorted([true_ts[0], mid_left, mid_right])
        return [by_ts[ts] for ts in picked_ts]

    # ---- True == 0：全窗口无目标 → 返回 None ----
    return None
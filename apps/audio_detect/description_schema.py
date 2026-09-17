"""音频描述的解析、校验与跨分段聚合（Phase 4，spec §6.5）。

目标结构（spec §6.5）
--------------------
.. code-block:: json

    {
      "description": "妈妈说：外面那些盘子多少人用过了，不都是二手的…",
      "background_sounds": "电视声、洗衣机声"
    }

**必填只有 ``description``**（说话内容）；``background_sounds`` 是**背景音描述**
（一句话；老数据可能是 ``list[str]``，由 :func:`_bg_text` 归一）。

其余字段（``has_cry`` / ``has_adult_speech`` / 哭声时间 / ``speech_confidence``）
是"问卷式"提示词的遗留产物，现在**既是可选的、也是三值的**：

- 模型给了 → 照原样留着（历史 MOSS 数据里的 ``true`` / ``false`` 继续有效）；
- 模型没给（缺键或显式 ``null``）→ 归一成 ``None`` = **"未判定"**。

``None`` 与 ``False`` 必须分开
-----------------------------
``False`` 是一个**断言**（"判定为没有"），``None`` 是**没做判断**。这里刻意不再拿
``False`` 冒充默认值：判断归声学侧（YAMNet/PANNs + ``detected_labels``），描述模型
不参与，硬填 ``False`` 等于往数据里灌一个我们其实没做过的结论。

因此 ``transcript`` 方言（Qwen3-ASR）下这几个字段**恒为 ``None``** —— 转写模型只吐
人声文字，既不知道有没有哭，也分不清"人说话"与"电视里的人说话"。

校验原则（spec §6.5）
--------------------
- 类型校验 +（给了时间时的）**时间范围校验**：偏移必须落在片段时长内；
- 缺必填 / 类型错 / 数值越界 → **记录失败，不把非法结果写入事件描述**；
- ``description`` 是否允许为空由 ``require_description`` 决定（见
  :func:`validate_description`）：纯结构化描述必须给内容，纯转写允许"这段没人声"；
- 校验通过时返回"清洗后"的 dict（只保留已知字段、int→float 归一），
  原始文本由调用方存 ``description_raw``；
- 描述模型的"说了什么"默认视为近似描述，不保证逐字转写。
"""

from __future__ import annotations

import json
from typing import Any

#: 时间容差：模型返回的边界允许略微超出片段时长（舍入误差）
_TIME_TOL_SEC = 0.5

#: "没有背景音"的占位写法：聚合时跳过，免得拼出"无；洗衣机声"这种句子
_BG_PLACEHOLDERS = {"无", "无。", "没有", "无其他声音", "无其他声音。", "none", "None"}


def _bg_text(value: Any) -> str:
    """``background_sounds`` 归一成字符串。

    新格式是**一句描述**（"背景音描述：…没有就写无"）；老格式是 ``list[str]``
    （问卷式提示词的产物），这里用 "、" 拼回一句，两种数据在页面/聚合里表现一致。
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "、".join(str(x).strip() for x in value if str(x).strip())
    return ""


# ---------------------------------------------------------------------------
# 解析：从模型原始输出里抠出 JSON 对象
# ---------------------------------------------------------------------------
def parse_description_json(text: str) -> tuple[dict | None, str]:
    """从模型原始文本提取 JSON 对象。

    容忍 ```json 代码围栏、前后多余文字；返回 ``(obj, error)``，
    失败时 obj 为 None。
    """
    raw = (text or "").strip()
    if not raw:
        return None, "empty response"

    # 去代码围栏
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    candidates = [raw]
    # 前后带解释文字时，截取最外层 {} 区间再试一次
    start, end = raw.find("{"), raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj, ""
        return None, f"response is JSON but not an object: {type(obj).__name__}"
    return None, f"no valid JSON in response: {raw[:200]}"


# ---------------------------------------------------------------------------
# 校验 + 清洗
# ---------------------------------------------------------------------------
def _as_number(value: Any, field: str) -> tuple[float | None, str]:
    """int/float → float；bool / str / None 一律拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{field} 不是数值: {value!r}"
    return float(value), ""


def validate_description(
    obj: Any,
    segment_duration_sec: float,
    require_description: bool = True,
) -> tuple[dict | None, str]:
    """schema + 类型 + 时间范围校验。

    **必填只有 ``description``**（一段自然语言描述）。其余字段是从早先“问卷式”
    提示词沿用下来的，现在**可选**：模型自愿给出就照旧校验（旧数据、旧输出仍然
    干净），不给就写 ``None``（= 未判定，**不是** ``False``）—— 因为强求
    ``cry_start_offset_sec`` / ``speech_confidence`` 会逼模型硬猜，反而产出脏数据；
    而默认填 ``False`` 等于替它下一个"没有哭"的结论，同样是脏数据。

    :param obj: :func:`parse_description_json` 抠出的 dict
    :param segment_duration_sec: 片段真实时长（时间范围校验的上限）
    :param require_description: ``description`` 是否必须非空。
        ``True``（json 方言）：空描述意味着模型没按要求干活 → 判失败去重试；
        ``False``（transcript 方言）：**空描述是合法结果** —— 这一段确实没有可
        转写的人声，不该因此判失败、更不该塞一句含糊的占位文本进库。
    :return: ``(clean, error)``；任何违规 → ``(None, error)``，不写脏数据
    """
    if not isinstance(obj, dict):
        return None, f"description 不是 object: {type(obj).__name__}"

    limit = float(segment_duration_sec) + _TIME_TOL_SEC

    # --- 描述：必填（是否必须非空由方言决定）---
    text = obj.get("description")
    if text is None and not require_description:
        text = ""                       # 转写模型什么都没转出来 = 这段没人声
    if not isinstance(text, str):
        return None, f"description 缺失或不是字符串: {text!r}"
    if require_description and not text.strip():
        return None, f"description 缺失或不是非空字符串: {text!r}"

    clean: dict[str, Any] = {
        "description": text.strip(),
        # 全部默认为 None = 未判定；模型给了才覆盖（见下）
        "has_cry": None,
        "cry_start_offset_sec": None,
        "cry_duration_sec": None,
        "has_adult_speech": None,
        "speech_confidence": None,
        "background_sounds": None,
    }

    # --- 以下都是"给了才校验"；**显式 null 一律视为未提供** ---
    # 注意：不能省掉 None 判断 —— `field in obj` 对 `{"has_cry": null}` 是 True，
    # 直接走 isinstance 会把它当类型错，把一次本来正常的描述判成脏数据。
    for field in ("has_cry", "has_adult_speech"):
        value = obj.get(field)
        if value is None:
            continue
        if not isinstance(value, bool):
            return None, f"{field} 不是 bool: {value!r}"
        clean[field] = value

    bg = obj.get("background_sounds")
    if bg is not None:                      # null 视为未提供
        if isinstance(bg, str):
            clean["background_sounds"] = bg.strip()
        elif isinstance(bg, list) and all(isinstance(x, str) for x in bg):
            clean["background_sounds"] = _bg_text(bg)     # 老格式 list[str] → 一句话
        else:
            return None, f"background_sounds 不是字符串或 list[str]: {bg!r}"

    # 哭声时间：两个偏移都给全才校验（只给一半按未提供处理，不判失败）
    start_raw = obj.get("cry_start_offset_sec")
    dur_raw = obj.get("cry_duration_sec")
    if start_raw is not None and dur_raw is not None:
        start, err = _as_number(start_raw, "cry_start_offset_sec")
        if err:
            return None, err
        dur, err = _as_number(dur_raw, "cry_duration_sec")
        if err:
            return None, err
        if not (0.0 <= start <= limit):
            return None, f"cry_start_offset_sec 越界: {start}（片段 {segment_duration_sec:.1f}s）"
        if dur <= 0.0:
            return None, f"cry_duration_sec 必须为正: {dur}"
        if start + dur > limit:
            return None, (
                f"cry 时间范围越界: {start}+{dur} > {segment_duration_sec:.1f}s"
            )
        clean["cry_start_offset_sec"] = round(start, 3)
        clean["cry_duration_sec"] = round(dur, 3)

    conf_raw = obj.get("speech_confidence")
    if conf_raw is not None:
        conf, err = _as_number(conf_raw, "speech_confidence")
        if err:
            return None, err
        if not (0.0 <= conf <= 1.0):
            return None, f"speech_confidence 越界: {conf}（应在 [0,1]）"
        clean["speech_confidence"] = round(conf, 4)

    return clean, ""


# ---------------------------------------------------------------------------
# 跨分段聚合（spec §5.2 description_json 的合并规则）
# ---------------------------------------------------------------------------
def _merge_claim(current: bool | None, claim: Any) -> bool | None:
    """三值"任一为真"合并：``None``（未判定）不参与投票。

    全都是未判定 → 结果仍未判定（``None``）。**不能回落成 ``False``**：那会把
    "没人做判断"写成"判定为没有"（描述模型本就不负责这类判定，spec §6.5）。
    """
    if claim is None:
        return current
    claim = bool(claim)
    return claim if current is None else (current or claim)


def aggregate_descriptions(segments: list) -> dict:
    """把各片段的清洗后描述合并成事件级描述。

    :param segments: 有 ``start_offset`` / ``end_offset`` / ``sequence`` /
        ``description_json`` 属性的对象列表（AudioEventSegment 或测试替身）

    合并规则
    --------
    - ``description``：各段描述去重后用 "；" 连接（顺带收老数据的
      ``adult_speech_summary``，升级后历史文本不丢）；空描述（转写方言下
      "这段没人声"）直接丢弃，不会在事件描述里留下空档或占位噪声；
    - ``has_cry`` / ``has_adult_speech``：**三值** —— 给了判断的段按"任一为真即真"，
      全都没给（``None``）则事件级也是 ``None``（未判定）；
    - 哭声时间换算成**事件内偏移**：start = min(段偏移 + 段内起点)，
      duration = max(段偏移 + 段内终点) − start —— **两个偏移都给全才换算**：
      新提示词不再索要时间，缺了只记"有哭"，绝不能因此崩（旧实现直接
      ``desc["cry_start_offset_sec"]`` 取键，缺键即 KeyError）；
    - ``speech_confidence``：取各段最大值（最确定的一次）；
    - ``background_sounds``：各段背景音描述去重后拼接（"无"这类占位写法丢弃）；
      **一段都没有 → ``None``**（未判定，不写空串冒充"没有背景音"）；
    - 另附 ``segments`` 审计列表（段号 + 偏移）。
    """
    has_cry: bool | None = None
    cry_start: float | None = None
    cry_end: float | None = None
    has_speech: bool | None = None
    texts: list[str] = []
    confidence: float | None = None
    backgrounds: list[str] = []
    seg_meta: list[dict] = []

    for seg in segments:
        desc = seg.description_json or {}
        seg_meta.append({
            "sequence": seg.sequence,
            "start_offset": seg.start_offset,
            "end_offset": seg.end_offset,
        })
        # ``str()`` 是防历史脏数据（老行里 description 可能是 list/dict）：这里
        # 崩会连累整个事件的聚合，而页面侧对同一份 JSON 早就做了同样的防御。
        text = str(
            desc.get("description") or desc.get("adult_speech_summary") or "",
        ).strip()
        if text and text not in texts:
            texts.append(text)
        has_cry = _merge_claim(has_cry, desc.get("has_cry"))
        if desc.get("has_cry"):
            start, _ = _as_number(desc.get("cry_start_offset_sec"), "cry_start_offset_sec")
            dur, _ = _as_number(desc.get("cry_duration_sec"), "cry_duration_sec")
            if start is not None and dur is not None:
                s = float(seg.start_offset) + start
                e = s + dur
                cry_start = s if cry_start is None else min(cry_start, s)
                cry_end = e if cry_end is None else max(cry_end, e)
        has_speech = _merge_claim(has_speech, desc.get("has_adult_speech"))
        conf = desc.get("speech_confidence")
        if conf is not None:
            confidence = conf if confidence is None else max(confidence, conf)
        bg = _bg_text(desc.get("background_sounds"))
        if bg and bg not in _BG_PLACEHOLDERS and bg not in backgrounds:
            backgrounds.append(bg)

    return {
        "description": "；".join(texts),
        "has_cry": has_cry,
        "cry_start_offset_sec": round(cry_start, 3) if cry_start is not None else None,
        "cry_duration_sec": (
            round(cry_end - cry_start, 3) if cry_start is not None else None
        ),
        "has_adult_speech": has_speech,
        "speech_confidence": confidence,
        # 一段背景音描述都没有（含全部为 null）= 未判定，不留空串
        "background_sounds": "；".join(backgrounds) if backgrounds else None,
        "segments": seg_meta,
    }


__all__ = [
    "aggregate_descriptions",
    "parse_description_json",
    "validate_description",
]

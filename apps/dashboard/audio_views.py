"""Phase 7：音频线页面（spec §8）。

四个页面
--------
- ``GET  /sound-detection/logs/``            声音检测日志（§8.1）—— **按窗口**
- ``GET  /sound-detection/events/``          音频事件列表 —— **按事件**
- ``GET  /sound-events/<id>/``               音频事件详情（§8.2）+ FLAC 播放
- ``GET/POST /sound-detection/cleanup/``     手动清理（§8.3）

两种展示粒度
------------
同一份音频数据有两种自然粒度，排查时用途不同，所以做成顶部可切换的两个视图
（不是同一个列表的两种排序）：

- **按窗口**（``SoundDetectionLog``）：一行 = 一个推理窗。回答"模型当时怎么判的"、
  调阈值、查单模型阳性；
- **按事件**（``AudioEvent``）：一行 = 一次完整声音。回答"到底响了几次、各多久、
  4B 怎么描述、录到没有"。

两者靠「摄像头 + 时间相交」互相跳转（事件行 → 该时段的窗口；窗口行 → 已有事件编号），
切换视图时保留摄像头与时间范围。

为什么单独成模块
----------------
``views.py`` 已经承载 VLM 事件列表 / 详情 / 控制页；音频线页面自带两套筛选
与媒体文件服务，塞进同一个文件会让它难以阅读。这里只复用 ``views.py`` 的两个
纯 helper（``_parse_dt_param`` / ``_human_ts``），不反向依赖，所以没有循环 import。

设计要点
--------
1. **两个模型、两套筛选**（spec §8.1）：日志表 ``SoundDetectionLog`` 是稀疏落库的
   推理窗口；事件表 ``AudioEvent`` 是完整声音事件。列表页显示日志，事件详情页显示
   事件，二者用「摄像头 + 时间相交」关联（事件表没有指向日志的 FK）。
2. **"无声音"只代表被采样落库的窗口**（spec §8.1 注）：稀疏落库决定了本页看不到
   全部窗口，模板里必须写出这条免责说明，否则会误导排查。
3. **媒体文件走专门 view 而不是 ``/media/``**（spec §8.3）：``settings.DEBUG`` 才
   挂 ``/media/``，生产不暴露。音频播放用 ``django.views.static.serve``（它自带
   Range 支持，``<audio>`` 拖进度条不会整段重下）。
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.views.static import serve as static_serve

from apps.audio_detect.models import (
    AudioEvent,
    AudioEventSegment,
    AudioRuntimeState,
    SoundDetectionLog,
)
from apps.dashboard.views import _human_ts, _parse_dt_param
from apps.streaming.models import Camera
from apps.vlm.models import (
    PromptNotificationDelivery,
    VLMCheckStateAudioEvent,
)

if TYPE_CHECKING:
    from django.http import HttpRequest

LOG_PAGE_SIZE = 25

#: 「按事件」列表每页条数（事件比窗口少得多，但一行信息更大，保持一致）
EVENT_PAGE_SIZE = 25

#: 业务标签 → 中文（spec §4.2 只有这两个业务标签）
LABEL_DISPLAY = {"cry": "哭声", "speech": "说话声"}

#: 「事件状态」「描述状态」下拉的合法取值（非法值静默忽略，与日志页同口径）
EVENT_STATUS_VALUES = tuple(v for v, _ in AudioEvent.STATUS_CHOICES)
DESC_STATUS_VALUES = tuple(v for v, _ in AudioEvent.DESC_STATUS_CHOICES)

#: 单侧阳性但共识不成立的原因码（``ConsensusEngine._evaluate_label``）。
#: 注意：被 §4.3.5 单侧高置信度覆盖判阳的窗，其 ``reason`` 是 ``single_side_high_confidence``
#: 且 ``decision=positive``，**不属于**本筛选（它会出现在「阳性」里）。
SINGLE_SIDE_REASON = "single_side_only"

#: 「共识结果」下拉的合法取值
DECISION_VALUES = ("positive", "single_side", "negative")

#: 「落库原因」下拉的合法取值
LOG_REASON_VALUES = (
    SoundDetectionLog.LOG_POSITIVE,
    SoundDetectionLog.LOG_NEAR_THRESHOLD,
    SoundDetectionLog.LOG_STATE_CHANGE,
)

#: ``has_event`` 筛选需要先把候选行取到 Python 里判「是否已形成事件」（事件关联是
#: 「时间相交」而非 FK，没法下推成 SQL）。上限保护：超过只扫前 N 条。
HAS_EVENT_SCAN_LIMIT = 2000

#: 清理预览的事件扫描上限（真删时不受此限）
PREVIEW_SCAN_LIMIT = 2000

#: 孤儿文件扫描上限：``os.walk`` 整个 audio_events 目录，文件极多时兜底止损
ORPHAN_SCAN_LIMIT = 20000


# ---------------------------------------------------------------------------
# 筛选：解析 + 转 Q（日志列表与清理页共用，避免两边漂移）
# ---------------------------------------------------------------------------
def _extract_log_filters(request: "HttpRequest") -> dict:
    """从 GET / POST 提 8 个筛选项（已 strip）。"""
    src = request.POST if request.method == "POST" else request.GET
    return {
        "cam_id": src.get("cam_id", "").strip(),
        "label": src.get("label", "").strip(),
        "decision": src.get("decision", "").strip(),
        "log_reason": src.get("log_reason", "").strip(),
        "model_error": src.get("model_error", "").strip(),
        "has_event": src.get("has_event", "").strip(),
        "start_dt": src.get("start_dt", "").strip(),
        "end_dt": src.get("end_dt", "").strip(),
    }


def _dt_to_epoch(raw: str):
    """``datetime-local`` 字符串 → epoch 秒（与 ``window_*_ts`` 同基）。

    ``_parse_dt_param`` 解析失败返回 None（非法值静默忽略）；这里再补一层
    ``timestamp()`` 的异常保护（越界日期在某些平台会抛 OSError）。
    """
    dt = _parse_dt_param(raw)
    if dt is None:
        return None
    try:
        return int(dt.timestamp())
    except (OSError, OverflowError, ValueError):
        return None


def _dt_range(filters: dict) -> tuple[int | None, int | None]:
    """→ ``(start_ts, end_ts)``；``end < start`` 时丢弃 end（避免查空）。"""
    start_ts = _dt_to_epoch(filters.get("start_dt", ""))
    end_ts = _dt_to_epoch(filters.get("end_dt", ""))
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        end_ts = None
    return start_ts, end_ts


def _single_side_q() -> Q:
    """「单模型阳性但共识不成立」：任一业务标签带该原因码。"""
    return (
        Q(**{f"consensus_payload__labels__cry__reason": SINGLE_SIDE_REASON})
        | Q(**{f"consensus_payload__labels__speech__reason": SINGLE_SIDE_REASON})
    )


def _log_filter_q(filters: dict) -> Q:
    """把筛选项转成 ``Q``（**不含** ``has_event``，它要后处理，见视图）。"""
    q = Q()

    if str(filters.get("cam_id", "")).isdigit():
        q &= Q(camera_id=int(filters["cam_id"]))

    label = filters.get("label", "")
    if label in LABEL_DISPLAY:
        # 「按业务标签」= 该标签在本窗判阳（JSON 路径查询）
        q &= Q(**{f"consensus_payload__labels__{label}__state": "positive"})

    decision = filters.get("decision", "")
    if decision == "positive":
        q &= Q(decision=SoundDetectionLog.DECISION_POSITIVE)
    elif decision == "single_side":
        q &= _single_side_q()
    elif decision == "negative":
        # 「无声音」= 非阳性，且排掉「单侧阳性」（那条属于近阈值待看，不算无声音）
        q &= Q(decision=SoundDetectionLog.DECISION_NEGATIVE) & ~_single_side_q()

    log_reason = filters.get("log_reason", "")
    if log_reason in LOG_REASON_VALUES:
        q &= Q(log_reason=log_reason)

    if filters.get("model_error", "") == "1":
        q &= Q(failure_reason__gt="") | Q(decision=SoundDetectionLog.DECISION_DEGRADED)

    start_ts, end_ts = _dt_range(filters)
    if start_ts is not None:
        q &= Q(window_end_ts__gte=start_ts)
    if end_ts is not None:
        q &= Q(window_end_ts__lte=end_ts)
    return q


def _build_log_queryset(filters: dict):
    """日志 QuerySet（模型 Meta 已按 ``-window_end_ts, -id`` 排序）。"""
    return (
        SoundDetectionLog.objects
        .select_related("camera")
        .filter(_log_filter_q(filters))
    )


def _querystring_without_page(request: "HttpRequest") -> str:
    """除 ``page`` 外的 GET 参数编码串（分页链接复用筛选条件；空串时无 ``&``）。"""
    params = request.GET.copy()
    params.pop("page", None)
    encoded = params.urlencode()
    return f"{encoded}&" if encoded else ""


def _shared_time_qs(filters: dict) -> str:
    """视图切换时携带的公共筛选（摄像头 + 时间范围）→ ``""`` 或 ``"?cam_id=1&..."``。

    只带这三个：它们的语义在两个视图里完全一致（都是"哪路 + 哪段时间"），而
    ``label`` / ``decision`` 之类是各自粒度专有的，带了反而会让人误以为结果集对应。
    """
    params = {
        key: filters[key]
        for key in ("cam_id", "start_dt", "end_dt")
        if filters.get(key)
    }
    encoded = urlencode(params)
    return f"?{encoded}" if encoded else ""


def _epoch_to_dt_param(ts) -> str:
    """epoch 秒 → ``datetime-local`` 字符串（与 :func:`_parse_dt_param` 同一本地时间基）。"""
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


# ---------------------------------------------------------------------------
# 事件关联：日志（窗口）↔ 事件（连续时间段）
# ---------------------------------------------------------------------------
def _overlaps(log: SoundDetectionLog, ev: AudioEvent) -> bool:
    """窗口与事件是否有时间交集（进行中的事件 ``ended_at_ts`` 为空也算相交）。"""
    if log.camera_id is None or ev.camera_id != log.camera_id:
        return False
    if ev.started_at_ts > log.window_end_ts:
        return False
    return ev.ended_at_ts is None or ev.ended_at_ts >= log.window_start_ts


def _events_for_logs(logs: list[SoundDetectionLog]) -> dict[int, list[AudioEvent]]:
    """批量取候选事件，返回 ``{log.id: [event, ...]}``（避免 N+1）。"""
    out: dict[int, list[AudioEvent]] = {log.id: [] for log in logs}
    if not logs:
        return out
    cam_ids = {log.camera_id for log in logs if log.camera_id is not None}
    if not cam_ids:
        return out
    t_min = min(log.window_start_ts for log in logs)
    t_max = max(log.window_end_ts for log in logs)
    candidates = list(
        AudioEvent.objects
        .select_related("camera")
        .filter(camera_id__in=cam_ids, started_at_ts__lte=t_max)
        .filter(Q(ended_at_ts__gte=t_min) | Q(ended_at_ts__isnull=True))
        .order_by("-started_at_ts", "-id")
    )
    for log in logs:
        out[log.id] = [ev for ev in candidates if _overlaps(log, ev)]
    return out


def _apply_has_event_filter(qs, filters: dict):
    """``has_event`` 筛选：候选行取到 Python 里判「是否已形成音频事件」。

    「相交」判定没法下推成 SQL（``ended_at_ts IS NULL`` + 区间重叠），所以只能
    先扫前 :data:`HAS_EVENT_SCAN_LIMIT` 条候选。本项目稀疏落库，量级可控；
    超出上限时只按前 N 条判定（页面会提示），不会静默全表扫。
    """
    want = filters.get("has_event", "")
    if want not in ("1", "0"):
        return qs, False
    candidates = list(qs[:HAS_EVENT_SCAN_LIMIT])
    index = _events_for_logs(candidates)
    matched = [
        log.id for log in candidates
        if bool(index.get(log.id)) == (want == "1")
    ]
    truncated = qs.count() > HAS_EVENT_SCAN_LIMIT
    return qs.filter(id__in=matched), truncated


# ---------------------------------------------------------------------------
# §8.1 声音检测日志
# ---------------------------------------------------------------------------
def _label_rows(log: SoundDetectionLog) -> list[dict]:
    """本窗的业务标签判定（供列表内 ``<details>` 展开）。"""
    payload = log.consensus_payload if isinstance(log.consensus_payload, dict) else {}
    labels = payload.get("labels") or {}
    if not isinstance(labels, dict):
        return []
    rows = []
    for label, lv in labels.items():
        if not isinstance(lv, dict):
            continue
        scores = lv.get("scores") or {}
        thresholds = lv.get("thresholds") or {}
        rows.append({
            "label": label,
            "display": LABEL_DISPLAY.get(label, label),
            "state": lv.get("state", ""),
            "reason": lv.get("reason", ""),
            "yamnet": scores.get("yamnet"),
            "panns": scores.get("panns"),
            "threshold_yamnet": thresholds.get("yamnet"),
            "threshold_panns": thresholds.get("panns"),
        })
    return rows


def _runtime_rows(cam_ids: set) -> dict[int, dict]:
    """摄像头 → 当前音频健康状态（spec §8.1「当前错误和健康状态」）。"""
    out: dict[int, dict] = {}
    if not cam_ids:
        return out
    for st in AudioRuntimeState.objects.filter(camera_id__in=cam_ids):
        out[st.camera_id] = {
            "status": st.status,
            "status_display": st.get_status_display(),
            "last_error": st.last_error,
            "last_packet_at": st.last_packet_at,
        }
    return out


def _build_rows(logs: list[SoundDetectionLog]) -> list[dict]:
    index = _events_for_logs(logs)
    runtimes = _runtime_rows({log.camera_id for log in logs if log.camera_id})
    rows = []
    for log in logs:
        payload = log.consensus_payload if isinstance(log.consensus_payload, dict) else {}
        rows.append({
            "log": log,
            "window_start": _human_ts(log.window_start_ts),
            "window_end": _human_ts(log.window_end_ts),
            "labels": _label_rows(log),
            "strategy": payload.get("strategy", ""),
            "decision": log.decision,
            "decision_display": log.get_decision_display(),
            "log_reason_display": log.get_log_reason_display(),
            "failure_reason": log.failure_reason,
            "yamnet_top": (log.yamnet_payload or {}).get("top_k") or [],
            "panns_top": (log.panns_payload or {}).get("top_k") or [],
            "yamnet_ok": (log.yamnet_payload or {}).get("ok", True),
            "panns_ok": (log.panns_payload or {}).get("ok", True),
            "events": index.get(log.id, []),
            "runtime": runtimes.get(log.camera_id) if log.camera_id else None,
        })
    return rows


@require_GET
def sound_detection_logs(request):
    """§8.1 声音检测日志列表：8 项筛选 + 分页 + 每行展开判定明细。"""
    filters = _extract_log_filters(request)
    qs = _build_log_queryset(filters)
    qs, truncated = _apply_has_event_filter(qs, filters)

    paginator = Paginator(qs, LOG_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    rows = _build_rows(list(page.object_list))

    ctx = {
        "page": page,
        "rows": rows,
        "filters": filters,
        "truncated": truncated,
        "scan_limit": HAS_EVENT_SCAN_LIMIT,
        "querystring": _querystring_without_page(request),
        "cameras": Camera.objects.all().order_by("id"),
        "label_display": LABEL_DISPLAY,
        "log_reasons": SoundDetectionLog.LOG_REASON_CHOICES,
        "audio_view": "window",
        "shared_qs": _shared_time_qs(filters),
    }
    return TemplateResponse(request, "dashboard/sound_detection_logs.html", ctx)


# ---------------------------------------------------------------------------
# 音频事件列表（另一种展示粒度：按事件而不是按窗口）
# ---------------------------------------------------------------------------
def _extract_event_filters(request: "HttpRequest") -> dict:
    """从 GET 提事件筛选项（已 strip）。

    与日志页的 8 项筛选**不共用**：事件没有"落库原因"和"单模型阳性"这两个概念
    （那是窗口级的判定产物），却有状态 / 描述状态 / 是否有录音这些事件级字段。
    """
    src = request.GET
    return {
        "cam_id": src.get("cam_id", "").strip(),
        "label": src.get("label", "").strip(),
        "status": src.get("status", "").strip(),
        "desc_status": src.get("desc_status", "").strip(),
        "degraded": src.get("degraded", "").strip(),
        "has_audio": src.get("has_audio", "").strip(),
        "start_dt": src.get("start_dt", "").strip(),
        "end_dt": src.get("end_dt", "").strip(),
    }


def _event_filter_q(filters: dict) -> Q:
    """事件筛选项 → ``Q``。

    时间范围用**相交**语义：事件须在 end 前开始（``started_at_ts <= end``），且
    在 start 后结束——``ended_at_ts`` 为空（进行中）也算"尚未结束"，因此
    [start, end] 区间内的进行中事件会命中，区间之后才开始的不会（``start`` 那次
    比较已经把它排掉了）。与日志页 ``has_event`` 的判定同一口径，两边互相跳转时
    看到的行才对得上。
    """
    q = Q()

    if str(filters.get("cam_id", "")).isdigit():
        q &= Q(camera_id=int(filters["cam_id"]))

    label = filters.get("label", "")
    if label in LABEL_DISPLAY:
        # 该标签在事件里出现过（detected_labels 的键），不是"最后一次判定为阳"
        q &= Q(**{"detected_labels__has_key": label})

    status = filters.get("status", "")
    if status in EVENT_STATUS_VALUES:
        q &= Q(status=status)

    desc_status = filters.get("desc_status", "")
    if desc_status in DESC_STATUS_VALUES:
        q &= Q(description_status=desc_status)

    if filters.get("degraded", "") == "1":
        q &= Q(degraded=True)

    has_audio = filters.get("has_audio", "")
    if has_audio == "1":
        q &= ~Q(audio_path="")
    elif has_audio == "0":
        q &= Q(audio_path="")

    start_ts, end_ts = _dt_range(filters)
    if end_ts is not None:
        q &= Q(started_at_ts__lte=end_ts)
    if start_ts is not None:
        q &= (Q(ended_at_ts__gte=start_ts) | Q(ended_at_ts__isnull=True))
    return q


def _build_event_queryset(filters: dict):
    """事件 QuerySet，按 ``-started_at_ts, -id`` 排序。

    显式 ``order_by``：调用方会再 ``annotate(Count("segments"))`` 带上 GROUP BY，
    那时不该再依赖 Meta 默认排序（Paginator 也会因"看起来无序"告警）。
    """
    return (
        AudioEvent.objects
        .select_related("camera")
        .filter(_event_filter_q(filters))
        .order_by("-started_at_ts", "-id")
    )


def _status_breakdown(q: Q) -> list[dict]:
    """当前筛选下各事件状态的条数（"共 N 个：已完成 x / 待描述 y / 失败 z"）。

    **不能**在带 ``Count("segments")`` 的 QuerySet 上 ``values().annotate()``——
    那个 annotation 会变成分组字段，计数是错的。所以单独查一次。
    """
    labels = dict(AudioEvent.STATUS_CHOICES)
    rows = (
        AudioEvent.objects.filter(q)
        .values("status")
        .annotate(n=Count("id"))
        .order_by()
    )
    return [
        {"status": r["status"], "display": labels.get(r["status"], r["status"]), "n": r["n"]}
        for r in rows
    ]


def _state_link_counts(event_ids: list[int]) -> dict[int, int]:
    """``{event_id: 关联 Prompt 命中数}``（批量，避免 N+1）。"""
    if not event_ids:
        return {}
    rows = (
        VLMCheckStateAudioEvent.objects
        .filter(audio_event_id__in=event_ids)
        .values("audio_event_id")
        .annotate(n=Count("id"))
    )
    return {r["audio_event_id"]: r["n"] for r in rows}


def _event_labels(ev: AudioEvent) -> list[dict]:
    """事件里出现过的业务标签（含阳性窗数），供列表打徽章。"""
    raw = ev.detected_labels if isinstance(ev.detected_labels, dict) else {}
    rows = []
    for label, info in raw.items():
        info = info if isinstance(info, dict) else {}
        windows = info.get("positive_windows") or []
        rows.append({
            "label": label,
            "display": LABEL_DISPLAY.get(label, label),
            "windows": len(windows) if isinstance(windows, list) else 0,
        })
    rows.sort(key=lambda r: r["label"])
    return rows


def _desc_brief(ev: AudioEvent) -> dict:
    """事件级描述摘要（``aggregate_descriptions`` 的产物，spec §5.2）。

    同一份 JSON 有三代写法，都要能显示：
    - 现在：``description``（说话内容）/ ``background_sounds``（背景音描述，字符串）；
    - 早先：``adult_speech_summary``（成人说话摘要）/ ``background_sounds`` 列表；
    - 更早：``summary``（列表页测试数据里出现过）。
    太短跳过或失败的事件是 ``{"skipped": ...}``，一律按"没有"降级展示。
    """
    js = ev.description_json if isinstance(ev.description_json, dict) else {}
    summary = js.get("description") or js.get("adult_speech_summary") or js.get("summary") or ""
    background = js.get("background_sounds") or ""
    if isinstance(background, list):     # 老数据是 list[str]
        background = "、".join(str(b) for b in background)
    return {
        "has_cry": bool(js.get("has_cry")),
        "has_speech": bool(js.get("has_adult_speech")),
        "summary": str(summary)[:60],
        "background": str(background)[:120],
        "skipped": js.get("skipped", ""),
    }


def _window_link_qs(ev: AudioEvent) -> str:
    """事件 → 窗口视图的反查参数（``cam_id`` + 该事件的时间范围）。"""
    params = {}
    if ev.camera_id:
        params["cam_id"] = str(ev.camera_id)
    start = _epoch_to_dt_param(ev.started_at_ts)
    if start:
        params["start_dt"] = start
    # 进行中的事件没有结束时间 → 不传 end_dt（开区间），别把查询框死
    end = _epoch_to_dt_param(ev.ended_at_ts) if ev.ended_at_ts else ""
    if end:
        params["end_dt"] = end
    return urlencode(params)


def _build_event_rows(events: list[AudioEvent]) -> list[dict]:
    counts = _state_link_counts([ev.pk for ev in events])
    rows = []
    for ev in events:
        rows.append({
            "ev": ev,
            "started": _human_ts(ev.started_at_ts),
            "ended": _human_ts(ev.ended_at_ts) if ev.ended_at_ts else "",
            "ongoing": ev.ended_at_ts is None,
            "duration": round(ev.duration_sec, 1),
            "labels": _event_labels(ev),
            "desc": _desc_brief(ev),
            "status_display": ev.get_status_display(),
            "desc_status_display": ev.get_description_status_display(),
            "has_audio": bool(ev.audio_path),
            "seg_count": getattr(ev, "seg_count", 0),
            "state_link_count": counts.get(ev.pk, 0),
            "window_qs": _window_link_qs(ev),
        })
    return rows


@require_GET
def sound_events_list(request):
    """**按音频事件**展示：一行 = 一次完整声音（可播、可下钻到详情/相关窗口）。

    与 :func:`sound_detection_logs`（按窗口）互为两种粒度，顶部标签页切换。
    """
    filters = _extract_event_filters(request)
    q = _event_filter_q(filters)
    qs = _build_event_queryset(filters).annotate(seg_count=Count("segments"))

    paginator = Paginator(qs, EVENT_PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))

    ctx = {
        "page": page,
        "rows": _build_event_rows(list(page.object_list)),
        "filters": filters,
        "querystring": _querystring_without_page(request),
        "breakdown": _status_breakdown(q),
        "cameras": Camera.objects.all().order_by("id"),
        "label_display": LABEL_DISPLAY,
        "event_statuses": AudioEvent.STATUS_CHOICES,
        "desc_statuses": AudioEvent.DESC_STATUS_CHOICES,
        "audio_view": "event",
        "shared_qs": _shared_time_qs(filters),
    }
    return TemplateResponse(request, "dashboard/sound_events_list.html", ctx)


# ---------------------------------------------------------------------------
# §8.2 音频事件详情
# ---------------------------------------------------------------------------
def _serve_media(request: "HttpRequest", abs_path: str):
    """只服务 ``MEDIA_ROOT`` 内的文件（spec §8.3：不暴露 ``/media/``）。

    ``static.serve`` 自带 ``Content-Type`` / ``Range`` / ``Last-Modified``，
    比手写 ``FileResponse`` 更省事，且能拖 ``<audio>`` 进度条。
    """
    if not abs_path:
        raise Http404("该记录没有音频文件")
    try:
        rel = Path(abs_path).relative_to(settings.MEDIA_ROOT)
    except ValueError:
        raise Http404("文件不在媒体目录内")
    return static_serve(
        request, str(rel).replace("\\", "/"), document_root=str(settings.MEDIA_ROOT),
    )


def _label_timeline(ev: AudioEvent) -> list[dict]:
    """``detected_labels`` → 模板友好的时间线行（spec §8.2）。"""
    raw = ev.detected_labels if isinstance(ev.detected_labels, dict) else {}
    rows = []
    for label, info in raw.items():
        info = info if isinstance(info, dict) else {}
        windows = []
        for pair in (info.get("positive_windows") or []):
            try:
                start_ts, end_ts = int(pair[0]), int(pair[1])
            except (TypeError, IndexError, ValueError):
                continue
            windows.append({
                "start": _human_ts(start_ts),
                "end": _human_ts(end_ts),
                # 事件内偏移（相对 started_at_ts），便于和分段 / 静音点对齐
                "start_offset": round(start_ts - ev.started_at_ts, 1),
                "end_offset": round(end_ts - ev.started_at_ts, 1),
            })
        rows.append({
            "label": label,
            "display": LABEL_DISPLAY.get(label, label),
            "first": _human_ts(info["first_positive_ts"]) if info.get("first_positive_ts") else "",
            "last": _human_ts(info["last_positive_ts"]) if info.get("last_positive_ts") else "",
            "windows": windows,
        })
    return rows


def _silence_timeline(ev: AudioEvent) -> list[dict]:
    """静音区间（相对事件起点的秒偏移）→ 展示行。"""
    raw = ev.silence_ranges if isinstance(ev.silence_ranges, list) else []
    rows = []
    for pair in raw:
        try:
            start, end = float(pair[0]), float(pair[1])
        except (TypeError, IndexError, ValueError):
            continue
        rows.append({
            "start": round(start, 1),
            "end": round(end, 1),
            "duration": round(end - start, 1),
        })
    return rows


def _state_links(ev: AudioEvent) -> list[dict]:
    """关联的 Prompt 事件与通知条件（spec §8.2）。

    两条线：
    - ``VLMCheckStateAudioEvent``：命中时算出的音频三态快照；
    - ``PromptNotificationDelivery``：每个目标按 ``condition`` 的投递结果。
    """
    links = []
    for link in (
        VLMCheckStateAudioEvent.objects
        .filter(audio_event=ev)
        .select_related("vlm_state", "vlm_state__prompt_config")
        .order_by("-id")
    ):
        links.append({
            "vlm_state": link.vlm_state,
            "prompt": link.vlm_state.prompt_config if link.vlm_state else None,
            "snapshot": link.snapshot_json,
            "created_at": link.created_at,
            "deliveries": list(
                PromptNotificationDelivery.objects
                .filter(vlm_state_id=link.vlm_state_id, audio_event=ev)
                .select_related("notify_target")
                .order_by("id")
            ),
        })
    return links


@require_http_methods(["GET"])
def audio_event_detail(request, pk: int):
    """§8.2 音频事件详情：时间线 / FLAC 播放 / 分段 / 4B 描述 / 关联 Prompt。"""
    ev = get_object_or_404(
        AudioEvent.objects.select_related("camera"),
        pk=pk,
    )
    segments = []
    for seg in ev.segments.all().order_by("sequence"):
        # 时长模板里算不了，这里补成展示属性
        seg.duration_sec = round(seg.end_offset - seg.start_offset, 1)
        segments.append(seg)
    ctx = {
        "ev": ev,
        "started_at_human": _human_ts(ev.started_at_ts) if ev.started_at_ts else "",
        "ended_at_human": _human_ts(ev.ended_at_ts) if ev.ended_at_ts else "",
        "labels": _label_timeline(ev),
        "silences": _silence_timeline(ev),
        "segments": segments,
        "has_audio": bool(ev.audio_path),
        "state_links": _state_links(ev),
    }
    return TemplateResponse(request, "dashboard/audio_event_detail.html", ctx)


@require_GET
def audio_event_file(request, pk: int):
    """事件 FLAC 播放（``<audio>`` 源）。"""
    ev = get_object_or_404(AudioEvent.objects.only("audio_path"), pk=pk)
    return _serve_media(request, ev.audio_path)


@require_GET
def audio_segment_file(request, pk: int, seq: int):
    """分段 FLAC 播放（``<audio>`` 源）。"""
    seg = get_object_or_404(
        AudioEventSegment.objects.only("audio_path"),
        audio_event_id=pk,
        sequence=seq,
    )
    return _serve_media(request, seg.audio_path)


# ---------------------------------------------------------------------------
# §8.3 手动清理
# ---------------------------------------------------------------------------
def _dir_size(paths: list[str]) -> tuple[int, int]:
    """→ ``(存在文件数, 总字节)``；stat 失败按缺失处理。"""
    n, total = 0, 0
    for p in paths:
        try:
            total += os.path.getsize(p)
            n += 1
        except OSError:
            continue
    return n, total


def _cleanup_preview(filters: dict) -> dict:
    """清理预览（spec §8.3「预览匹配数量和占用空间」）。

    - ``log_count`` 用 ``COUNT(*)``：准，且不会把行读进内存；
    - 关联事件只扫前 :data:`PREVIEW_SCAN_LIMIT` 条日志（只取 4 个小字段，不碰三个
      JSON 大字段）——预览是给人看的估算，真删时由 ``_cleanup_by_filters`` 全量算。
    """
    log_qs = _build_log_queryset(filters)
    log_count = log_qs.count()
    logs = list(
        log_qs
        .only("id", "camera_id", "window_start_ts", "window_end_ts")[:PREVIEW_SCAN_LIMIT]
    )
    events = {}
    for evs in _events_for_logs(logs).values():
        for ev in evs:
            events[ev.pk] = ev

    event_paths = [ev.audio_path for ev in events.values() if ev.audio_path]
    seg_paths = [
        s.audio_path for s in
        AudioEventSegment.objects.filter(audio_event_id__in=list(events)).only("audio_path")
        if s.audio_path
    ]
    n_files, total_bytes = _dir_size(event_paths + seg_paths)
    return {
        "log_count": log_count,
        "event_count": len(events),
        "events": sorted(events.values(), key=lambda e: -e.started_at_ts)[:20],
        "file_count": n_files,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "truncated": log_count > PREVIEW_SCAN_LIMIT,
        "scan_limit": PREVIEW_SCAN_LIMIT,
    }


def _orphan_audio_files() -> dict:
    """扫描 ``MEDIA_ROOT/audio_events`` 下无人引用的 FLAC（spec §8.3 重试入口）。

    为什么要有这个入口：清理是「先删库行、再删文件」，删文件那步失败（占用中 /
    权限）不会让库行回来，也就没法靠"再点一次删除"补偿。孤儿扫描把"库里没有、
    盘上还在"这个状态**显式**列出来，重试按钮就是再扫一次再删一次。
    """
    root = Path(settings.MEDIA_ROOT) / "audio_events"
    empty = {
        "count": 0, "total_bytes": 0, "total_mb": 0.0, "paths": [], "truncated": False,
    }
    if not root.exists():
        return empty
    known = set()
    for p in AudioEvent.objects.exclude(audio_path="").values_list("audio_path", flat=True):
        known.add(os.path.normcase(str(p)))
    for p in (
        AudioEventSegment.objects.exclude(audio_path="")
        .values_list("audio_path", flat=True)
    ):
        known.add(os.path.normcase(str(p)))

    orphans = []
    total = 0
    seen = 0
    truncated = False
    for cur, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".flac"):
                continue
            seen += 1
            if seen > ORPHAN_SCAN_LIMIT:
                truncated = True
                break
            full = os.path.join(cur, name)
            if os.path.normcase(full) in known:
                continue
            try:
                total += os.path.getsize(full)
            except OSError:
                continue
            orphans.append(full)
        if truncated:
            break
    return {
        "count": len(orphans),
        "total_bytes": total,
        "total_mb": round(total / 1024 / 1024, 2),
        "paths": orphans,
        "truncated": truncated,
        "scan_limit": ORPHAN_SCAN_LIMIT,
    }


@require_GET
def audio_cleanup(request):
    """§8.3 手动清理页：筛选 + 预览 + 二次确认（不自动清理）。"""
    filters = _extract_log_filters(request)
    try:
        preview = _cleanup_preview(filters)
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"预览失败：{e}")
        preview = {
            "log_count": 0, "event_count": 0, "events": [],
            "file_count": 0, "total_bytes": 0, "total_mb": 0.0,
            "truncated": False, "scan_limit": PREVIEW_SCAN_LIMIT,
        }
    try:
        orphans = _orphan_audio_files()
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"孤儿文件扫描失败：{e}")
        orphans = {
            "count": 0, "total_bytes": 0, "total_mb": 0.0, "paths": [],
            "truncated": False, "scan_limit": ORPHAN_SCAN_LIMIT,
        }

    ctx = {
        "filters": filters,
        "preview": preview,
        "orphans": orphans,
        "cameras": Camera.objects.all().order_by("id"),
        "label_display": LABEL_DISPLAY,
    }
    return TemplateResponse(request, "dashboard/audio_cleanup.html", ctx)


@require_POST
def audio_cleanup_run(request):
    """执行清理：后台线程删库行 + 删文件（spec §8.3）。

    只接受 POST（二次确认在模板里做 ``confirm()``，这里是服务端兜底）。
    """
    from apps.dashboard.audio_cleanup import _run_cleanup_in_thread

    filters = _extract_log_filters(request)
    preview = _cleanup_preview(filters)
    _run_cleanup_in_thread(dict(filters))
    messages.success(
        request,
        f"已启动后台清理：{preview['log_count']} 条日志 / "
        f"{preview['event_count']} 个事件 / {preview['file_count']} 个音频文件。",
    )
    return redirect("dashboard:audio_cleanup")


@require_POST
def audio_cleanup_orphans(request):
    """删除孤儿音频文件并重扫（spec §8.3「清理失败显示失败数量并允许重试」）。"""
    from apps.dashboard.audio_cleanup import purge_orphan_files

    removed, failed = purge_orphan_files()
    if failed:
        messages.warning(request, f"孤儿文件：删除 {removed} 个，失败 {failed} 个（可重试）。")
    else:
        messages.success(request, f"孤儿文件已清理：{removed} 个。")
    return redirect("dashboard:audio_cleanup")


__all__ = [
    "audio_cleanup",
    "audio_cleanup_orphans",
    "audio_cleanup_run",
    "audio_event_detail",
    "audio_event_file",
    "audio_segment_file",
    "sound_detection_logs",
    "sound_events_list",
    "_build_event_queryset",
    "_build_log_queryset",
    "_cleanup_preview",
    "_extract_event_filters",
    "_extract_log_filters",
    "_log_filter_q",
    "_orphan_audio_files",
    "_querystring_without_page",
    "_shared_time_qs",
]

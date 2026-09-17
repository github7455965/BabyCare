"""Dashboard views: events list / detail + logs placeholder.

v2 范围（在 Step 13.5 基础上增强）
-----------------------------------
- `/events/`     列表：
  - GET 过滤：cam_id / prompt_id / hit（B7 之前已有）+ **kind（B11）** + **start_dt / end_dt（B9）**
  - 分页 25；**B8 跳转指定页**（template 实现）
- `/events/<pk>/` 详情：
  - **B10** `ts_list` 显示为 `2026年09月01日 13:45:30` 格式（SHANGHAI UTC+8）
  - **B7** dl 每个字段 `<dt>X</dt>` 后加中文同义词 `<small>（中文）</small>`
- `/logs/`                占位（"v1 暂未实装"）

设计要点
- 图字段 VLMCheckState.img1/2/3 存**绝对路径**（Step 12 决策），
  Django template 不允许 file:// → helper 转 MEDIA_URL 相对 URL
- 不做 WebSocket；列表只读；dismiss/undismiss 走 /api/（Step 10）
- ts_list 是 JSONField[int, int, int] 秒级时间戳；详情 view 预渲染成上海时区字符串
- 过滤：kind ∈ {judge, describe} 白名单；非法忽略；start_dt / end_dt 用 parse_datetime 解析；
  `end_dt < start_dt` 时忽略 end_dt（避免查空）
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect
from django.template.response import TemplateResponse
from django.utils.dateparse import parse_datetime
from django.views.decorators.http import require_GET, require_POST

from apps.streaming.models import Camera
from apps.vlm.models import VLMCheckState, VLMPromptConfig

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from django.http import HttpRequest

PAGE_SIZE = 25

# 上海时区（UTC+8）；用于 ts_list 的人类可读渲染。
# apps/vlm/notify.py 也有同名常量，这里复制一份以保持 dashboard App 独立（避免反向依赖）。
SHANGHAI = timezone(timedelta(hours=8), name="Asia/Shanghai")

# kind 过滤白名单（仅这 2 个值有效；其他值忽略）
KIND_CHOICES = ("judge", "describe")


def _extract_filters(request: "HttpRequest") -> dict:
    """从 GET 或 POST 提筛选项（dict）；POST 用于 bulk_delete。

    返回 dict 包含 6 个 key（已 strip）：cam_id / prompt_id / kind / hit / start_dt / end_dt。
    """
    src = request.POST if request.method == "POST" else request.GET
    return {
        "cam_id": src.get("cam_id", "").strip(),
        "prompt_id": src.get("prompt_id", "").strip(),
        "kind": src.get("kind", "").strip(),
        "hit": src.get("hit", "").strip(),
        "start_dt": src.get("start_dt", "").strip(),
        "end_dt": src.get("end_dt", "").strip(),
    }


def _filter_kwargs(filters: dict) -> dict:
    """把 filters dict 转 ORM filter 关键字。

    公共 helper：view 的 _build_events_queryset 和 bulk_delete worker 都用它，
    保证两边 filter 逻辑不漂移。

    过滤项非法值（cam_id 非数字 / kind 不在白名单 / 时间无法解析）静默忽略。
    `end_dt < start_dt` 时忽略 end_dt（避免查空）。
    """
    kw: dict = {}
    if filters.get("cam_id", "").isdigit():
        kw["camera_id"] = int(filters["cam_id"])
    if filters.get("prompt_id", "").isdigit():
        kw["prompt_config_id"] = int(filters["prompt_id"])
    hit = filters.get("hit", "")
    if hit == "1":
        kw["hit"] = True
    elif hit == "0":
        kw["hit"] = False
    kind = filters.get("kind", "")
    if kind in KIND_CHOICES:
        kw["prompt_config__kind"] = kind
    start_dt = _parse_dt_param(filters.get("start_dt", ""))
    end_dt = _parse_dt_param(filters.get("end_dt", ""))
    if start_dt and end_dt and end_dt < start_dt:
        end_dt = None
    if start_dt:
        kw["detected_at__gte"] = start_dt
    if end_dt:
        kw["detected_at__lte"] = end_dt
    return kw


def _build_events_queryset(filters: dict) -> "QuerySet":
    """根据 filters dict 构造 VLMCheckState QuerySet（带 select_related）。"""
    return (
        VLMCheckState.objects
        .select_related("camera", "prompt_config")
        .filter(**_filter_kwargs(filters))
        .order_by("-id")
    )


def _media_url_for(abs_path: str) -> str:
    """绝对路径 → MEDIA_URL 相对 URL（用于 <img src>）。

    路径不在 MEDIA_ROOT 下时返回空串（前端 <img> 留 broken 图标）。
    """
    if not abs_path:
        return ""
    try:
        rel = Path(abs_path).relative_to(settings.MEDIA_ROOT)
        return settings.MEDIA_URL + str(rel).replace("\\", "/")
    except ValueError:
        return ""


def _attach_image_urls(obj: VLMCheckState) -> VLMCheckState:
    """给 obj 加 img1_url/img2_url/img3_url 属性（template 用）。"""
    obj.img1_url = _media_url_for(obj.img1)
    obj.img2_url = _media_url_for(obj.img2)
    obj.img3_url = _media_url_for(obj.img3)
    return obj


def _parse_dt_param(raw: str):
    """解析 datetime-local 字符串（'YYYY-MM-DDTHH:MM' / 'YYYY-MM-DDTHH:MM:SS'）→ naive datetime。

    返回 None 表示解析失败（忽略）。C2：parse_datetime 对越界值（'2024-13-45T10:00'）
    raise ValueError，必须 catch；同样 drop tzinfo（USE_TZ=False，MySQL 不接受 aware）。
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        dt = parse_datetime(raw)
    except (ValueError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _human_ts(ts) -> str:
    """把单个 int_ts 渲染成 "2026年09月01日 13:45:30"（上海时区）；无效原样返回。"""
    try:
        ts_int = int(ts)
    except (TypeError, ValueError):
        return str(ts)
    return datetime.fromtimestamp(ts_int, tz=SHANGHAI).strftime("%Y年%m月%d日 %H:%M:%S")


def _human_ts_list(ts_list) -> list[str]:
    """把 [int_ts, ...] 渲染成 ["2026年09月01日 13:45:30", ...]（上海时区）。"""
    if not ts_list:
        return []
    return [_human_ts(ts) for ts in ts_list]


def events_list(request):
    filters = _extract_filters(request)
    qs = _build_events_queryset(filters)
    paginator = Paginator(qs, PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    for _obj in page.object_list:
        _obj.img1_url = _media_url_for(_obj.img1)

    ctx = {
        "page": page,
        "filters": filters,
        "cameras": Camera.objects.all().order_by("id"),
        "prompts": VLMPromptConfig.objects.all().order_by("id"),
    }
    return TemplateResponse(request, "dashboard/events_list.html", ctx)


def _run_bulk_delete_in_thread(filters: dict) -> None:
    """包装：起 daemon 线程跑 _run_bulk_delete，方便测试 mock。

    n_total 不传：worker 自己 count（用同一 _filter_kwargs 公共 helper），
    避免 view/worker 算两次 filter 漂移。
    """
    from apps.dashboard.bulk_delete import _run_bulk_delete
    t = threading.Thread(
        target=_run_bulk_delete,
        args=(dict(filters),),
        daemon=True,
        name="bulk-delete",
    )
    t.start()


@require_POST
def bulk_delete(request):
    """按当前筛选条件异步删 VLMCheckState + 清孤儿图片。

    POST body 传同 GET 的 filter key（cam_id / prompt_id / kind / hit / start_dt / end_dt）。
    立即 302 回 events_list（同 filter）+ flash「已启动后台删除，共 N 条事件」。

    n_total 仅用于 flash 消息（用户看的），不传给 worker；worker 自己 count。
    两边共用 _filter_kwargs 公共 helper，filter 逻辑不会漂移。
    """
    filters = _extract_filters(request)
    n_total = _build_events_queryset(filters).count()

    _run_bulk_delete_in_thread(dict(filters))

    messages.success(request, f"已启动后台删除，共 {n_total} 条事件。")
    qs_params = request.POST.urlencode()
    return redirect(f"/events/?{qs_params}")


#: 音频三态的 reason → 人话（未知的原样显示，便于排查）
_AUDIO_REASON_LABEL = {
    "no_camera": "无法定位摄像头",
    "no_rule": "未配置声音条件",
    "rule_disabled": "声音条件已停用",
    "no_service_state": "音频服务状态缺失",
    "service_state_error": "读取音频服务状态失败",
    "desired_off": "音频线被手动关闭",
    "no_lease": "音频服务未持有租约",
    "lease_timeout": "音频服务心跳超时",
    "no_runtime_state": "摄像头无音频状态",
    "no_audio_profile": "摄像头无音频轨",
    "capture_error": "音频采集异常",
    "model_error": "音频模型异常",
    "initializing": "音频初始化中",
    "stopped": "音频已停止",
    "coverage_starts_later": "覆盖率起点晚于判定窗口",
    "no_recent_packet": "无最近音频包",
    "packet_stalled": "音频包停滞",
    "packet_time_error": "音频包时间异常",
}

#: 音频三态（audio_rule.SATISFIED / NOT_SATISFIED / UNKNOWN）→ 展示徽章
_AUDIO_RESULT_BADGE = {
    "SATISFIED": ("bg-success", "声音条件满足"),
    "NOT_SATISFIED": ("bg-secondary", "不满足"),
    "UNKNOWN": ("bg-warning text-dark", "无法判定"),
}


def _audio_links(state) -> list[dict]:
    """该 VLM 窗的音频判定（spec §5.7）：三态 + 条件 + 覆盖 + 参考音频事件。

    阴性 / UNKNOWN 也有一行（``audio_event=None``），保证三态可审计；
    没配 ``audio_rule`` 目标的检查项则一行都没有。
    """
    from apps.vlm.models import VLMCheckStateAudioEvent

    links = (
        VLMCheckStateAudioEvent.objects
        .filter(vlm_state=state)
        .select_related("audio_event")
        .order_by("id")
    )
    rows = []
    for link in links:
        snap = link.snapshot_json if isinstance(link.snapshot_json, dict) else {}
        result = snap.get("result", "")
        badge, label = _AUDIO_RESULT_BADGE.get(
            result, ("bg-light text-dark border", result or "-"),
        )
        reason = snap.get("reason") or ""
        rule_enabled = bool(snap.get("rule_enabled"))
        condition = snap.get("condition") or ""
        rows.append({
            "badge": badge,
            "label": label,
            "condition": condition or "-",
            "rule_enabled": rule_enabled,
            "covered": snap.get("covered"),
            "service_available": snap.get("service_available"),
            "camera_status": snap.get("camera_status") or "",
            "event_count": snap.get("event_count"),
            "cry_count": snap.get("cry_count"),
            "speech_count": snap.get("speech_count"),
            "window_start_ts": snap.get("window_start_ts"),
            "window_end_ts": snap.get("window_end_ts"),
            "reason": reason,
            "reason_label": _AUDIO_REASON_LABEL.get(reason, reason) or "-",
            #: 没配规则 / 规则停用时，快照里其它字段全是 UNKNOWN 的默认占位值
            #: （service_available=False、covered=False…），当真实状态展示会误导人。
            "configured": rule_enabled and bool(condition),
            "audio_event": link.audio_event,
        })
    return rows


def event_detail(request, pk: int):
    obj = get_object_or_404(
        VLMCheckState.objects.select_related("camera", "prompt_config"),
        pk=pk,
    )
    _attach_image_urls(obj)
    # B10：ts_list 渲染成上海时区的中文格式
    ctx = {
        "obj": obj,
        "human_ts_list": _human_ts_list(obj.ts_list),
        "img1_human_ts": _human_ts(obj.img1_ts) if obj.img1_ts else "",
        "img2_human_ts": _human_ts(obj.img2_ts) if obj.img2_ts else "",
        "img3_human_ts": _human_ts(obj.img3_ts) if obj.img3_ts else "",
        "audio_links": _audio_links(obj),
    }
    return TemplateResponse(request, "dashboard/event_detail.html", ctx)


def logs(request):
    return TemplateResponse(request, "dashboard/logs.html", {})


@require_POST
def toggle_pause(request, pk: int):
    """B4：从事件详情页暂停/恢复本检查项（toggle VLMPromptConfig.manual_paused）。"""
    state = get_object_or_404(
        VLMCheckState.objects.select_related("prompt_config"),
        pk=pk,
    )
    prompt = state.prompt_config
    prompt.manual_paused = not prompt.manual_paused
    prompt.save(update_fields=["manual_paused"])
    if prompt.manual_paused:
        messages.success(request, f"已暂停检查项 #{prompt.pk} {prompt.name}（误报排查期）")
    else:
        messages.success(request, f"已恢复检查项 #{prompt.pk} {prompt.name}")
    return redirect("dashboard:event_detail", pk=state.pk)


# ---------------------------------------------------------------------------
# VLM 控制页（常驻模式专用）
# ---------------------------------------------------------------------------
def _vlm_breaker_snapshot() -> dict:
    """VLM 视觉服务的熔断状态（给控制页显示）。

    **优先读进程内单例**，只在"本进程还没观察过任何结果"时回落到 DB 镜像。
    取法与 ASR 侧（`worker_manager._breaker_snapshot`）相反，因为两者的
    "状态住在哪"不同：

    - ASR 状态住在 `audio_worker` 进程，控制页在 daphne → **只能**读 DB 镜像；
    - VLM 状态就住在 **daphne 自己**（`apps/vlm/apps.py` 的 `ready()` 在本进程起
      Runner / Drainer）→ 单例才是权威且实时的。

    为什么不能**只**读 DB：`record_failure` 仅在**达到阈值或状态变化**时落库
    （稳态零写的优化，见 `LLMBreaker.record_failure`），所以**阈值以下查不到**。
    VLM 阈值是 3 → 页面上只会出现 0 或 3，"已经错了 2 次"看不见 —— 恰好丢掉了
    §8.3 要的预警价值。（ASR 阈值是 1，读 DB 没这个问题。）
    """
    from apps.core.models import LLMHealthState

    try:
        from apps.core.llm_breaker import LLMBreaker

        snap = LLMBreaker.for_service(LLMHealthState.SERVICE_VLM).snapshot()
        if (
            snap["consecutive_failures"]
            or snap["last_failure_at"]
            or snap["last_success_at"]
        ):
            return snap          # 本进程观察过 → 单例权威且实时
    except Exception:  # noqa: BLE001
        logger.exception("[dashboard] 读 VLM 熔断单例失败（改用 DB 镜像）")

    # 本进程没观察过（例如控制页与 Runner 不同进程）→ 退回 DB 镜像
    try:
        row = LLMHealthState.objects.filter(
            service=LLMHealthState.SERVICE_VLM,
        ).first()
        return {
            "enabled": _env_bool("BABYCARE_LLM_BREAKER_ENABLED", True),
            "state": row.state if row else LLMHealthState.STATE_CLOSED,
            "consecutive_failures": row.consecutive_failures if row else 0,
            "last_failure_at": row.last_failure_at if row else None,
            "last_probe_at": row.last_probe_at if row else None,
            "last_success_at": row.last_success_at if row else None,
            "last_error": (row.last_error or "") if row else "",
        }
    except Exception:  # noqa: BLE001
        logger.exception("[dashboard] 读 VLM 熔断状态失败")
        return {}


def _vlm_control_ctx() -> dict:
    """拼装 vlm_control 模板上下文：llama 状态 + 队列计数 + 熔断 + env 摘要。"""
    from django.utils import timezone

    from apps.vlm.llama_manager import LlamaManager
    from apps.vlm.models import VLMQueuedTask

    llama = LlamaManager.instance()
    stats = llama.status()
    pending = VLMQueuedTask.objects.filter(status="pending").count()
    failed = VLMQueuedTask.objects.filter(status="failed").count()
    done = VLMQueuedTask.objects.filter(status="done").count()
    oldest = (
        VLMQueuedTask.objects.filter(status="pending")
        .order_by("created_at")
        .values_list("created_at", flat=True)
        .first()
    )
    return {
        "resident_mode": _env_bool("BABYCARE_LLM_RESIDENT", False),
        # 分体部署：llama-server 在另一台机器上，本页不能启停它（§5 Phase 5）
        "external": llama.is_external(),
        "llama_url": str(stats.get("url", "")),
        "llama": {
            "running": stats["running"],
            "pid": stats["pid"],
            "uptime_hours": round(stats.get("uptime_hours", 0.0), 2),
            "max_uptime_hours": stats.get("max_uptime_hours", 0),
            "forced_off": llama.is_forced_off(),
        },
        "breaker": _vlm_breaker_snapshot(),
        "queue": {
            "pending": pending,
            "failed": failed,
            "done": done,
            # 最老一条待回放的年龄：判断"数字不归零"是还在攒还是卡死
            "oldest_age_sec": (
                max(0, int((timezone.now() - oldest).total_seconds()))
                if oldest else None
            ),
        },
        "queue_max_age_sec": float(
            getattr(settings, "BABYCARE_LLM_QUEUE_MAX_AGE_SEC", 0) or 0
        ),
        "auto_restart_hours": float(
            getattr(settings, "BABYCARE_LLM_AUTO_RESTART_HOURS", 24)
        ),
        "drain_interval_sec": int(
            getattr(settings, "BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC", 5)
        ),
        "max_retries": int(
            getattr(settings, "BABYCARE_LLM_QUEUE_MAX_RETRIES", 3)
        ),
    }


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@require_GET
def vlm_control(request):
    """VLM 常驻模式控制页：状态 / 队列 / 操作按钮。

    非 resident 模式仍可访问，仅顶部 banner 提示"循环模式，本页无效"。
    """
    from apps.vlm.llama_manager import LlamaStartError

    try:
        ctx = _vlm_control_ctx()
    except Exception as e:
        messages.error(request, f"读取 VLM 状态失败：{e}")
        ctx = {
            "resident_mode": False, "external": False, "llama_url": "",
            "llama": {}, "breaker": {}, "queue": {}, "auto_restart_hours": 0,
            "queue_max_age_sec": 0,
        }
    return TemplateResponse(request, "dashboard/vlm_control.html", ctx)


@require_POST
def vlm_llama_off(request):
    """手动关闭 llama-server：force_off() 后续 Runner 入队。

    外部模式（分体部署）**直接拒绝并说明原因**：远端服务不归本进程管
    （`LlamaManager.force_off()` 已经是 no-op）。不拦的话页面会"提示成功
    但什么都没发生"，是最容易误判的一类问题。
    """
    from apps.vlm.llama_manager import LlamaManager

    mgr = LlamaManager.instance()
    if mgr.is_external():
        messages.warning(
            request,
            "外部模式：llama-server 在另一台机器上，本页不能启停它。"
            "请在推理机上用本地脚本（model/run_asr_server.bat、"
            "scripts/tmp/start_llm_audio.ps1）操作。",
        )
        return redirect("dashboard:vlm_control")
    try:
        mgr.force_off()
        messages.success(request, "已手动关闭 llama-server；新请求将入库排队。")
    except Exception as e:
        messages.error(request, f"关闭失败：{e}")
    return redirect("dashboard:vlm_control")


@require_POST
def vlm_llama_on(request):
    """手动开启 llama-server：force_on() Drainer 将逐条回放队列。

    外部模式：启不了，只探活 —— 不健康就把真实原因报给用户。
    """
    from apps.vlm.llama_manager import LlamaManager, LlamaStartError

    mgr = LlamaManager.instance()
    if mgr.is_external():
        try:
            mgr.force_on()               # 外部模式退化为探活
            messages.success(
                request, "外部模式：远端 llama-server 可用；drainer 将逐条回放队列。",
            )
        except LlamaStartError as e:
            messages.error(request, f"外部模式：远端 llama-server 不可用 —— {e}")
        return redirect("dashboard:vlm_control")
    try:
        mgr.force_on()
        messages.success(request, "已开启 llama-server；drainer 将逐条回放队列。")
    except LlamaStartError as e:
        messages.error(request, f"llama-server 启动失败：{e}")
    except Exception as e:
        messages.error(request, f"开启失败：{e}")
    return redirect("dashboard:vlm_control")


# ---------------------------------------------------------------------------
# 音频线控制页（Phase 5，spec §9.3）
# ---------------------------------------------------------------------------
def _empty_audio_status() -> dict:
    """读状态失败时的兜底（模板要的 key 一个都不能少）。"""
    return {
        "enabled": False, "desired": "off",
        "running": False, "pid": None, "lease_pid": None, "worker_epoch": "",
        "stale": False, "starting": False,
        "heartbeat_ok": False, "heartbeat_age_sec": None,
        "heartbeat_timeout_sec": 0, "startup_grace_sec": 0, "cameras": [],
        "python": "", "pid_file": "", "log_file": "",
        "uptime_sec": 0.0,
        "audio_desc_configured": False, "audio_desc_url": "",
        "audio_desc_model": "", "audio_desc_provider": "",
        "audio_desc_breaker": {
            "enabled": True, "state": "closed", "consecutive_failures": 0,
            "last_failure_at": None, "last_probe_at": None,
            "last_success_at": None, "last_error": "",
        },
        "queue_pending": 0, "queue_describing": 0, "queue_oldest_age_sec": None,
    }


@require_GET
def audio_control(request):
    """音频线控制页：总开关 / 期望状态 / PID / 心跳 / 各摄像头采集状态。

    三态（spec §9.3「PID 管启停、DB 心跳管健康」+ 启动宽限期）：
    ``启动中``（刚拉起、模型加载中，心跳尚未出现）→ 正常 → ``卡死``（过了宽限期
    仍心跳停 → 进程活着却没在干活）。
    """
    from apps.audio_detect.worker_manager import AudioWorkerManager
    try:
        manager = AudioWorkerManager.instance()
        ctx = {
            "audio": manager.status(),
            "log_tail": manager.tail_log(max_chars=2000),
        }
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"读取音频线状态失败：{e}")
        ctx = {"audio": _empty_audio_status(), "log_tail": ""}
    return TemplateResponse(request, "dashboard/audio_control.html", ctx)


@require_POST
def audio_control_on(request):
    """开启音频线：Popen ``manage.py audio_worker`` 并记录 ``desired=on``。

    ``AudioWorkerStartError`` 必须转成页面提示（spec §9.3：实例互斥导致的
    启动失败不能静默）。
    """
    from apps.audio_detect.worker_manager import (
        AudioWorkerManager,
        AudioWorkerStartError,
    )
    try:
        info = AudioWorkerManager.instance().start()
        if info.get("started"):
            messages.success(request, f"音频线已启动（pid={info.get('pid')}）。")
        else:
            messages.info(request, "音频线已在运行，无需重复启动。")
    except AudioWorkerStartError as e:
        messages.error(request, f"音频线启动失败：{e}")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"音频线启动异常：{e}")
    return redirect("dashboard:audio_control")


@require_POST
def audio_control_off(request):
    """关闭音频线：杀 worker 进程树（含 FFmpeg 孙进程）并记录 ``desired=off``。"""
    from apps.audio_detect.worker_manager import AudioWorkerManager
    try:
        stopped = AudioWorkerManager.instance().stop()
        if stopped:
            messages.success(request, "音频线已关闭（进程树已回收）。")
        else:
            messages.info(request, "音频线本来就未在运行。")
    except Exception as e:  # noqa: BLE001
        messages.error(request, f"音频线关闭失败：{e}")
    return redirect("dashboard:audio_control")
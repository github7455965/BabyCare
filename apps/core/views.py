"""
apps.core.views — Step 10 HTTP API（启停 + 调试 + dismiss）。

约定（沿用 08 events/views.py）：
- POST 用 @csrf_exempt（本地工具型 API，无认证）
- 响应统一 JsonResponse({"ok": bool, "error": str, ...})
- 异常 → 500 {"ok": false, "error": "<ClassName>: <msg>"}，同时 logger.exception
- 校验失败 → 400 {"ok": false, "error": "..."}
- 资源不存在 → 404 {"ok": false, "error": "not_found"} 或 400 {"error": "already_dismissed"} / {"error": "not_dismissed"}

端点（详见 urls.py）：
- POST /api/yolo/restart/         yolo_restart
- POST /api/vlm/restart/          vlm_restart
- GET  /api/gpu/mode/             gpu_mode_get
- POST /api/gpu/mode/             gpu_mode_post
- GET  /api/state/<cam_id>/       cam_state
- POST /api/state/<id>/dismiss/   state_dismiss
- POST /api/state/<id>/undismiss/ state_undismiss
"""

from __future__ import annotations

import json as _json
import logging
import time
from typing import Any

from django.http import Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from apps.streaming.models import Camera

logger = logging.getLogger(__name__)

# 最近 N 条 VLMCheckState
RECENT_LIMIT = 10

# has_baby_count_10s 窗口（秒）
HAS_BABY_WINDOW_SEC = 10


# ---------------------------------------------------------------------------
# 响应 helper
# ---------------------------------------------------------------------------
def _ok(**extra: Any) -> JsonResponse:
    payload: dict[str, Any] = {"ok": True}
    payload.update(extra)
    return JsonResponse(payload)


def _err(msg: str, status: int = 400, **extra: Any) -> JsonResponse:
    payload: dict[str, Any] = {"ok": False, "error": msg}
    payload.update(extra)
    return JsonResponse(payload, status=status)


def _wrap_exc(prefix: str, e: Exception) -> JsonResponse:
    logger.exception("[%s] failed", prefix)
    return _err(f"{type(e).__name__}: {e}", status=500)


# ---------------------------------------------------------------------------
# 启停
# ---------------------------------------------------------------------------
@csrf_exempt
@require_POST
def yolo_restart(request):
    """POST /api/yolo/restart/ — 卸载 best.pt + 重新加载。

    返回：{"ok": true, "model_path": str, "device": "cuda|cpu", "dt_sec": float}
    失败：500 {"ok": false, "error": "..."}
    """
    from apps.yolo_detect.detector import BabyDetector
    t0 = time.time()
    try:
        det = BabyDetector.instance()
        det.unload()
        det.ensure_loaded()
        dt = time.time() - t0
        return _ok(
            model_path=det.model_path or "",
            device=det.device or "",
            dt_sec=round(dt, 3),
        )
    except Exception as e:
        return _wrap_exc("yolo_restart", e)


@csrf_exempt
@require_POST
def vlm_restart(request):
    """POST /api/vlm/restart/ — LlamaManager.restart() = stop + ensure。

    返回：{"ok": true, "uptime_hours_before": float, "port": int, "pid": int|null}
    失败：500 {"ok": false, "error": "..."}
    """
    from apps.vlm.llama_manager import LlamaManager
    try:
        mgr = LlamaManager.instance()
        before_uptime = mgr.uptime_hours()
        was_running = mgr.is_running()
        info = mgr.restart()
        return _ok(
            uptime_hours_before=round(before_uptime, 4),
            was_running_before=was_running,
            **info,
        )
    except Exception as e:
        return _wrap_exc("vlm_restart", e)


# ---------------------------------------------------------------------------
# GPU mode（GET + POST 共用 path）
# ---------------------------------------------------------------------------
@csrf_exempt
def gpu_mode_dispatch(request):
    """GET → 当前 mode + status；POST body {"mode": ...} → set_mode。"""
    if request.method == "GET":
        return gpu_mode_get(request)
    if request.method == "POST":
        return gpu_mode_post(request)
    return _err(f"method not allowed: {request.method}", status=405)


@require_GET
def gpu_mode_get(request):
    """GET /api/gpu/mode/ — 当前 mode + status 字典。"""
    from apps.yolo_detect.gpu_manager import GpuManager
    try:
        gm = GpuManager.instance()
        st = gm.status()
        return _ok(status=st, mode=st["mode"])
    except Exception as e:
        return _wrap_exc("gpu_mode_get", e)


@csrf_exempt
@require_POST
def gpu_mode_post(request):
    """POST /api/gpu/mode/ — body {"mode": "exclusive|parallel"}；仅写 log，需重启生效。"""
    from apps.yolo_detect.gpu_manager import GpuManager
    try:
        body = _json.loads(request.body or b"{}")
    except _json.JSONDecodeError as e:
        return _err(f"JSON 解析失败: {e}")
    if not isinstance(body, dict):
        return _err("body 必须是 JSON object")
    mode = body.get("mode")
    if mode not in {"exclusive", "parallel"}:
        return _err(f"mode 必须是 exclusive|parallel，当前: {mode!r}")

    try:
        gm = GpuManager.instance()
        old_mode = gm.status()["mode"]
        gm.set_mode(mode)
        return _ok(
            old_mode=old_mode,
            mode=mode,
            requires_restart=(old_mode != mode),
        )
    except Exception as e:
        return _wrap_exc("gpu_mode_post", e)


# ---------------------------------------------------------------------------
# 摄像头状态 / dismiss
# ---------------------------------------------------------------------------
@require_GET
def cam_state(request, cam_id: int):
    """GET /api/state/<cam_id>/ — 推荐版字段。

    返回字段：
    - camera: {id, name, is_active}
    - cam_t0: int 秒（0 = 未设置）
    - latest_frame_ts: int 秒（0 = 无帧）
    - lag_sec: int|null（现在 - latest_frame_ts；0 = 未采样）
    - has_baby_count_10s: int
    - prompts: [{id, name, max_read_ts, lag_sec, fail_count}]
    - yolo_loaded: bool
    - vlm_running: bool
    - recent_states: [{id, prompt, hit, status, ts, dismissed_as_false, auto_silenced}]
    """
    from apps.streaming.cam_t0 import CamT0Manager
    from apps.vlm.llama_manager import LlamaManager
    from apps.vlm.models import VLMCheckState, VLMPromptConfig
    from apps.vlm.runner import PromptRunnerManager
    from apps.yolo_detect.detector import BabyDetector
    from apps.yolo_detect.frame_queue import FrameQueueManager

    try:
        cam = Camera._default_manager.get(pk=cam_id)
    except Camera.DoesNotExist:
        raise Http404(f"camera {cam_id} not found")
    t0 = CamT0Manager.instance().get(cam_id)
    now = int(time.time())

    # FrameQueue
    latest_frame_ts = 0
    has_baby_count_10s = 0
    fq = FrameQueueManager.instance().get(cam_id)
    if fq is not None:
        items = fq.snapshot_sorted()
        if items:
            latest_frame_ts = items[-1].ts
            window_start = now - HAS_BABY_WINDOW_SEC
            has_baby_count_10s = sum(
                1 for it in items
                if it.has_baby is True and it.ts >= window_start
            )
    lag_sec = max(0, now - latest_frame_ts) if latest_frame_ts > 0 else None

    # PromptCursor 进度（通过 PromptRunnerManager 公共接口）
    prompts_progress: list[dict[str, Any]] = []
    runner_mgr = PromptRunnerManager.instance()
    pids: list[int] = []
    try:
        pids = runner_mgr.cursor_keys_for_cam(cam_id) or []
    except Exception:
        logger.warning("[cam_state] cursor_keys_for_cam failed cam=%d", cam_id)
    for pid in pids:
        try:
            cfg = VLMPromptConfig.objects.get(pk=pid)
        except VLMPromptConfig.DoesNotExist:
            continue
        cur = runner_mgr.cursor_get(cam_id, pid, cfg.window_sec)
        if cur is None:
            continue
        prompts_progress.append({
            "id": pid,
            "name": cfg.name,
            "max_read_ts": cur.max_read_ts,
            "lag_sec": max(0, now - cur.max_read_ts) if cur.max_read_ts >= 0 else None,
            "fail_count": cur.fail_count,
        })

    # 检测器 / llama-server
    try:
        yolo_loaded = BabyDetector.instance().is_loaded
    except Exception:
        yolo_loaded = False
    try:
        vlm_running = LlamaManager.instance().is_running()
    except Exception:
        vlm_running = False

    # 最近 10 条 VLMCheckState
    recent: list[dict[str, Any]] = []
    qs = (VLMCheckState.objects
          .filter(camera_id=cam_id)
          .select_related("prompt_config")
          .order_by("-id")[:RECENT_LIMIT])
    for s in qs:
        recent.append({
            "id": s.id,
            "prompt": s.prompt_config.name,
            "hit": s.hit,
            "status": s.status or "",
            "ts": s.img1_ts or 0,
            "dismissed_as_false": s.dismissed_as_false,
            "auto_silenced": s.auto_silenced,
        })

    return _ok(
        camera={"id": cam.id, "name": cam.name, "is_active": cam.is_active},
        cam_t0=t0,
        latest_frame_ts=latest_frame_ts,
        lag_sec=lag_sec,
        has_baby_count_10s=has_baby_count_10s,
        prompts=prompts_progress,
        yolo_loaded=yolo_loaded,
        vlm_running=vlm_running,
        recent_states=recent,
    )


@csrf_exempt
@require_POST
def state_dismiss(request, state_id: int):
    """POST /api/state/<state_id>/dismiss/ — 标误报；重复 → 400 already_dismissed。

    返回：{"ok": true, "state_id", "prompt", "camera", "status"}
    """
    from apps.vlm.models import VLMCheckState
    try:
        state = (VLMCheckState.objects
                 .select_related("camera", "prompt_config")
                 .get(pk=state_id))
    except VLMCheckState.DoesNotExist:
        return _err("not_found", status=404)

    if state.dismissed_as_false:
        return _err("already_dismissed", status=400)

    state.dismissed_as_false = True
    state.save(update_fields=["dismissed_as_false"])

    logger.info("[state_dismiss] state=%d cam=%s prompt=%s status=%s",
                state.id, state.camera.name, state.prompt_config.name, state.status)

    return _ok(
        state_id=state.id,
        prompt=state.prompt_config.name,
        camera=state.camera.name,
        status=state.status or "",
    )


@csrf_exempt
@require_POST
def state_undismiss(request, state_id: int):
    """POST /api/state/<state_id>/undismiss/ — 取消误报；未标过 → 400 not_dismissed。

    返回：{"ok": true, "state_id"}
    """
    from apps.vlm.models import VLMCheckState
    try:
        state = VLMCheckState.objects.get(pk=state_id)
    except VLMCheckState.DoesNotExist:
        return _err("not_found", status=404)

    if not state.dismissed_as_false:
        return _err("not_dismissed", status=400)

    state.dismissed_as_false = False
    state.save(update_fields=["dismissed_as_false"])

    logger.info("[state_undismiss] state=%d", state.id)
    return _ok(state_id=state.id)
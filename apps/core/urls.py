"""
apps.core.urls — Step 10 URL 路由。

约定：
- 所有 API 挂在 /api/ 前缀（由 config/urls.py include 时定）
- POST view 用 @csrf_exempt（沿用 08 events/views.py 风格；本地工具型 API 无认证）
- 返回 JsonResponse({"ok": bool, "error": str, ...})（08 风格）
"""

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    # 启停
    path("yolo/restart/", views.yolo_restart, name="yolo_restart"),
    path("vlm/restart/", views.vlm_restart, name="vlm_restart"),
    # GPU mode
    path("gpu/mode/", views.gpu_mode_dispatch, name="gpu_mode"),
    # 摄像头 / 状态
    path("state/<int:cam_id>/", views.cam_state, name="cam_state"),
    path("state/<int:state_id>/dismiss/", views.state_dismiss, name="state_dismiss"),
    path("state/<int:state_id>/undismiss/", views.state_undismiss, name="state_undismiss"),
]
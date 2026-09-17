"""根 URL：home + Step 10 /api/ 路由"""
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

from home import views as home_views

urlpatterns = [
    path("", home_views.index, name="index"),
    path("healthz", home_views.healthz, name="healthz"),
    # Step 10：HTTP API（启停 + dismiss）
    path("api/", include("apps.core.urls", namespace="core")),
    # Step 11：VLM 检查项配置面板
    path("config/", include("apps.config_panel.urls", namespace="config_panel")),
    # Dashboard 顶级路由：/events/ /events/<pk>/ /logs/
    path("", include("apps.dashboard.urls", namespace="dashboard")),
]

# Step 12：开发期 serving 媒体文件（帧落盘绝对路径 = MEDIA_ROOT/frames/<date>/<cam>_<ts>.jpg）
# 仅 DEBUG：生产应走 nginx / 静态文件服务器
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
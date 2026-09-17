"""
apps.core — Step 10 HTTP API 容器（启停 + 调试 + dismiss）。

设计：
- 不引入 auto_init；v1 简单启动（views 单例通过 apps.*.instance() 取得各 Manager）
- INSTALLED_APPS 注册为 "apps.core"（配置见 config/settings.py）
- URL 挂载点：path("api/", include("apps.core.urls"))（见 config/urls.py）
"""

from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = "apps.core"
    verbose_name = "Core HTTP API"
    default_auto_field = "django.db.models.BigAutoField"
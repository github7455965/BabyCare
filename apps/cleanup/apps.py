from django.apps import AppConfig


class CleanupConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.cleanup"
    verbose_name = "清理管理命令"

"""ASGI entry（daphne 用；Step 1 不启用，仅占位）"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
application = get_asgi_application()
"""home 视图：首页 dashboard 入口 + /healthz。

- `/` 渲染控制台首页（卡片：摄像头 / VLM 检查项 / 事件 / 日志）
- `/healthz` 返回 JSON 状态（v1 基础；后续步骤可扩 vlm / gpu）
"""
from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone

from apps.streaming.models import Camera
from apps.vlm.models import VLMCheckState, VLMPromptConfig


def index(request):
    now = timezone.now()
    day_ago = now - timedelta(days=1)
    ctx = {
        "project": "09_web_vlm_manage",
        "vlm_url": settings.BABYCARE_LLAMA_SERVER_URL,
        "gpu_mode": settings.BABYCARE_GPU_MODE,
        "counts": {
            "cameras": Camera.objects.count(),
            "prompts": VLMPromptConfig.objects.count(),
            "events_today": VLMCheckState.objects.filter(created_at__gte=day_ago).count(),
            "events_hit_today": VLMCheckState.objects.filter(
                hit=True, created_at__gte=day_ago
            ).count(),
        },
    }
    return render(request, "home/index.html", ctx)


def healthz(request):
    return JsonResponse(
        {
            "status": "ok",
            "step": 1,
            "vlm_url": settings.BABYCARE_LLAMA_SERVER_URL,
            "gpu_mode": settings.BABYCARE_GPU_MODE,
        }
    )
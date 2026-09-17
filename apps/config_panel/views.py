from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from apps.vlm.models import VLMPromptConfig
from .forms import VLMPromptConfigForm

PAGE_SIZE = 25


def list_view(request):
    qs = VLMPromptConfig.objects.all().prefetch_related("camera_ids").order_by("id")
    kind = request.GET.get("kind", "").strip()
    if kind in ("judge", "describe"):
        qs = qs.filter(kind=kind)
    enabled = request.GET.get("enabled", "").strip()
    if enabled == "1":
        qs = qs.filter(enabled=True)
    elif enabled == "0":
        qs = qs.filter(enabled=False)
    paginator = Paginator(qs, PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    ctx = {
        "page": page,
        "filters": {"kind": kind, "enabled": enabled},
    }
    return render(request, "config_panel/prompts_list.html", ctx)


@require_http_methods(["GET", "POST"])
def create_view(request):
    if request.method == "POST":
        form = VLMPromptConfigForm(request.POST)
        if form.is_valid():
            obj = form.save()
            # 目标类别关键词提示：只警告，不拦保存（2026-09-14）
            for warning in form.warnings:
                messages.warning(request, warning)
            messages.success(request, f"已创建 #{obj.pk} {obj.name}")
            return redirect("config_panel:prompts_list")
    else:
        form = VLMPromptConfigForm()
    return render(request, "config_panel/prompt_detail.html", {"form": form, "obj": None})


@require_http_methods(["GET", "POST"])
def detail_view(request, pk: int):
    obj = get_object_or_404(VLMPromptConfig, pk=pk)
    if request.method == "POST":
        form = VLMPromptConfigForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            # 目标类别关键词提示：只警告，不拦保存（2026-09-14）
            for warning in form.warnings:
                messages.warning(request, warning)
            messages.success(request, f"已保存 #{obj.pk} {obj.name}")
            return redirect("config_panel:prompt_detail", pk=obj.pk)
    else:
        form = VLMPromptConfigForm(instance=obj)
    return render(request, "config_panel/prompt_detail.html", {"form": form, "obj": obj})


@require_http_methods(["POST"])
def delete_view(request, pk: int):
    obj = get_object_or_404(VLMPromptConfig, pk=pk)
    name = obj.name
    obj.delete()
    messages.success(request, f"已删除 {name}")
    return redirect("config_panel:prompts_list")
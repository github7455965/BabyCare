from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from apps.streaming.models import Camera

from .camera_forms import CameraForm

PAGE_SIZE = 25


def list_view(request):
    qs = Camera.objects.all().order_by("id")
    source_type = request.GET.get("source_type", "").strip()
    if source_type in (Camera.SOURCE_FILE, Camera.SOURCE_ONVIF):
        qs = qs.filter(source_type=source_type)
    is_active = request.GET.get("is_active", "").strip()
    if is_active == "1":
        qs = qs.filter(is_active=True)
    elif is_active == "0":
        qs = qs.filter(is_active=False)
    paginator = Paginator(qs, PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    ctx = {
        "page": page,
        "filters": {"source_type": source_type, "is_active": is_active},
    }
    return render(request, "config_panel/cameras_list.html", ctx)


@require_http_methods(["GET", "POST"])
def create_view(request):
    if request.method == "POST":
        form = CameraForm(request.POST)
        if form.is_valid():
            obj = form.save()
            messages.success(request, f"已创建 #{obj.pk} {obj.name}")
            return redirect("config_panel:camera_list")
    else:
        form = CameraForm()
    return render(request, "config_panel/camera_detail.html", {"form": form, "obj": None})


@require_http_methods(["GET", "POST"])
def detail_view(request, pk: int):
    obj = get_object_or_404(Camera, pk=pk)
    if request.method == "POST":
        form = CameraForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            messages.success(request, f"已保存 #{obj.pk} {obj.name}")
            return redirect("config_panel:camera_detail", pk=obj.pk)
    else:
        form = CameraForm(instance=obj)
    referenced_prompts = list(obj.prompt_configs.all().order_by("id"))
    return render(
        request,
        "config_panel/camera_detail.html",
        {
            "form": form,
            "obj": obj,
            "referenced_by": len(referenced_prompts),
            "referenced_prompts": referenced_prompts,
        },
    )


@require_http_methods(["POST"])
def delete_view(request, pk: int):
    obj = get_object_or_404(Camera, pk=pk)
    name = obj.name
    obj.delete()
    messages.success(request, f"已删除 {name}")
    return redirect("config_panel:camera_list")
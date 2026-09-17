from django.contrib import messages
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from apps.vlm.models import NotifyTarget

from .forms import NotifyTargetForm

PAGE_SIZE = 25


def list_view(request):
    qs = NotifyTarget.objects.all().order_by("id")
    paginator = Paginator(qs, PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "config_panel/notify_targets_list.html", {"page": page})


@require_http_methods(["GET", "POST"])
def create_view(request):
    if request.method == "POST":
        form = NotifyTargetForm(request.POST)
        if form.is_valid():
            obj = form.save()
            messages.success(request, f"已创建通知目标 #{obj.pk} {obj.name}")
            return redirect("config_panel:notify_targets_list")
    else:
        form = NotifyTargetForm()
    return render(
        request,
        "config_panel/notify_target_detail.html",
        {"form": form, "obj": None, "mode": "new"},
    )


@require_http_methods(["GET", "POST"])
def detail_view(request, pk: int):
    obj = get_object_or_404(NotifyTarget, pk=pk)
    if request.method == "POST":
        form = NotifyTargetForm(request.POST, instance=obj)
        if form.is_valid():
            form.save()
            messages.success(request, f"已保存通知目标 #{obj.pk} {obj.name}")
            return redirect("config_panel:notify_target_detail", pk=obj.pk)
    else:
        form = NotifyTargetForm(instance=obj)
    referenced_prompts = list(obj.prompts.all().order_by("id"))
    return render(
        request,
        "config_panel/notify_target_detail.html",
        {
            "form": form,
            "obj": obj,
            "mode": "edit",
            "referenced_by": len(referenced_prompts),
            "referenced_prompts": referenced_prompts,
        },
    )


@require_http_methods(["POST"])
def delete_view(request, pk: int):
    obj = get_object_or_404(NotifyTarget, pk=pk)
    name = obj.name
    obj.delete()
    messages.success(request, f"已删除通知目标 {name}")
    return redirect("config_panel:notify_targets_list")
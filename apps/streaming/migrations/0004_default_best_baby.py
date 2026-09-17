"""
Camera.yolo_models 默认挂 best-baby：保证 v1 (单 detector 跑 best.pt) 行为不变。

幂等
----
- 找不到 best-baby 模型（说明 0002 没跑）→ 跳过（migration 顺序保证一定有，但保险）
- cam 已经挂了 best-baby → add() 是幂等的（不会重复）
"""

from django.db import migrations


def attach_default(apps, schema_editor):
    YOLOModel = apps.get_model("yolo_detect", "YOLOModel")
    Camera = apps.get_model("streaming", "Camera")
    try:
        baby = YOLOModel.objects.get(name="best-baby")
    except YOLOModel.DoesNotExist:
        return
    for cam in Camera.objects.all():
        cam.yolo_models.add(baby)


def detach_default(apps, schema_editor):
    """回滚：拆掉所有 cam 的 best-baby。"""
    YOLOModel = apps.get_model("yolo_detect", "YOLOModel")
    Camera = apps.get_model("streaming", "Camera")
    try:
        baby = YOLOModel.objects.get(name="best-baby")
    except YOLOModel.DoesNotExist:
        return
    for cam in Camera.objects.all():
        cam.yolo_models.remove(baby)


class Migration(migrations.Migration):

    dependencies = [
        ("streaming", "0003_camera_yolo_models"),
        ("yolo_detect", "0002_seed_yolomodel"),
    ]

    operations = [
        migrations.RunPython(attach_default, detach_default),
    ]
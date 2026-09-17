"""
Seed YOLOModel 默认两条：best-baby（自定义 baby 模型）+ yolov8s-coco（COCO 80 类，用 0+15 拿 person/cat）。

幂等
----
已存在的同名记录会被跳过（用 get_or_create），重复 migrate 不会重复插入。
"""

from django.db import migrations


def seed(apps, schema_editor):
    YOLOModel = apps.get_model("yolo_detect", "YOLOModel")

    YOLOModel.objects.get_or_create(
        name="best-baby",
        defaults={
            "file_path": "model/best.pt",
            "class_names": ["baby"],
            "coco_class_ids": [],
            "enabled": True,
        },
    )
    YOLOModel.objects.get_or_create(
        name="yolov8s-coco",
        defaults={
            "file_path": "model/yolov8s.pt",
            "class_names": ["person", "cat"],
            "coco_class_ids": [0, 15],
            "enabled": True,
        },
    )


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("yolo_detect", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed, noop),
    ]
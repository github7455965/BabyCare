"""
yolo_detect 数据模型：YOLOModel 模型注册表。

设计
----
- 单实例 YOLOModel = 一个可加载的 .pt 文件 + 其负责识别的类别
- Camera.yolo_models (M2M) 表示"该 cam 启用哪些 model"
- 同一 cam 可挂多个 model（每个 model 负责不同 class），结果在 FrameItem 上聚合

字段
----
- name: 唯一名（admin 显示 + 日志标识）
- file_path: .pt 绝对路径或相对 BASE_DIR 路径
- class_names: 该 model 覆盖的目标类别名（FrameItem 上对应的 has_<name> 字段）
  例：["baby"] 或 ["person", "cat"]
- coco_class_ids: ultralytics predict() 的 classes 过滤参数（int 列表）
  例：best.pt 是自定义模型 → []；yolov8s.pt 取 person(0)/cat(15) → [0, 15]
- enabled: 全局开关；False 时 YoloLoop 跳过
- extra: 留给 conf / imgsz / device 等运行时调参（当前版本未消费）

历史
----
- 09 v1 阶段：单 detector 加载 best.pt；此表为 v2 multi-model 改造引入
"""

from __future__ import annotations

from django.db import models


class YOLOModel(models.Model):
    name = models.CharField("名称", max_length=64, unique=True)
    file_path = models.CharField(
        ".pt 路径",
        max_length=512,
        help_text="绝对路径；或相对项目根 09_web_vlm_manage/ 的路径",
    )
    class_names = models.JSONField(
        "覆盖类别",
        default=list,
        help_text="该 model 输出的目标类别列表（如 ['baby'] 或 ['person','cat']）；FrameItem 上对应 has_<name> 字段",
    )
    coco_class_ids = models.JSONField(
        "COCO 类别 ID",
        default=list,
        help_text="ultralytics predict(classes=...) 过滤参数；自定义模型（如 best.pt）填 []",
    )
    enabled = models.BooleanField("启用", default=True)

    extra = models.JSONField("扩展", default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "YOLO 模型"
        verbose_name_plural = "YOLO 模型"
        ordering = ["id"]
        db_table = "yolo_model"

    def __str__(self) -> str:
        return f"#{self.pk} {self.name} ({','.join(self.class_names)})"
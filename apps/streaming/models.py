"""
Camera 模型。

字段说明
--------
- name: 摄像头显示名
- source_type: file / onvif
- file_path: 视频文件绝对路径（source_type=file 时必填）
- onvif_host / onvif_port / onvif_username: ONVIF 摄像头连接信息
  （密码统一从 .env 读，不存 DB）
- is_active: True=启线程；False=停线程
- extra: JSONField，留扩展位（rtsp 直连模式、ffmpeg 参数等）
- created_at / updated_at
"""

from django.db import models


class Camera(models.Model):
    SOURCE_FILE = "file"
    SOURCE_ONVIF = "onvif"
    SOURCE_CHOICES = [
        (SOURCE_FILE, "视频文件"),
        (SOURCE_ONVIF, "ONVIF 摄像头"),
    ]

    name = models.CharField("名称", max_length=128, unique=True)
    source_type = models.CharField(
        "源类型", max_length=16, choices=SOURCE_CHOICES, default=SOURCE_FILE,
    )
    file_path = models.CharField(
        "视频文件路径（绝对路径）", max_length=512, blank=True, default="",
    )
    onvif_host = models.CharField("ONVIF 主机", max_length=128, blank=True, default="")
    onvif_port = models.PositiveIntegerField("ONVIF 端口", default=80)
    onvif_username = models.CharField("ONVIF 用户名", max_length=64, blank=True, default="")
    # 密码不放 DB；统一从 .env BABYCARE_ONVIF_PASSWORD 读
    is_active = models.BooleanField("启用", default=True)
    yolo_models = models.ManyToManyField(
        "yolo_detect.YOLOModel",
        verbose_name="YOLO 模型",
        related_name="cameras",
        blank=True,
        help_text="该摄像头启用的 YOLO 模型；空 = 不识别。改了会通过 m2m_changed signal 通知 YoloLoop 刷新缓存。",
    )
    extra = models.JSONField("扩展", default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "摄像头"
        verbose_name_plural = "摄像头"
        ordering = ["id"]
        db_table = "vlm_camera"  # 09 业务表统一加 vlm_ 前缀

    def __str__(self) -> str:
        return f"#{self.pk} {self.name} ({self.source_type})"

    # ------------------------------------------------------------------
    def resolved_file_path(self) -> str:
        """FileSource 需要的绝对路径。

        09 v1 仅支持绝对路径（不解析相对路径——避免 BASE_DIR 假设污染）。
        """
        return (self.file_path or "").strip()
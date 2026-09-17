from django import forms

from apps.streaming.models import Camera


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = [
            "name", "source_type",
            "file_path", "onvif_host", "onvif_port", "onvif_username",
            "yolo_models", "is_active",
        ]
        widgets = {
            "file_path": forms.TextInput(attrs={"placeholder": "D:\\path\\to\\video.mp4"}),
            "onvif_port": forms.NumberInput(attrs={"min": 1, "max": 65535}),
            "yolo_models": forms.CheckboxSelectMultiple,
        }
        help_texts = {
            "yolo_models": (
                "该摄像头启用的 YOLO 模型（可多选）。"
                "改了会通过 m2m_changed signal 自动通知 YoloLoop 刷新缓存，无需重启。"
            ),
        }

    def clean(self):
        cleaned = super().clean()
        st = cleaned.get("source_type")
        if st == Camera.SOURCE_FILE and not (cleaned.get("file_path") or "").strip():
            self.add_error("file_path", "file 源必须填 file_path")
        if st == Camera.SOURCE_ONVIF:
            if not (cleaned.get("onvif_host") or "").strip():
                self.add_error("onvif_host", "onvif 源必须填 onvif_host")
            if not (cleaned.get("onvif_username") or "").strip():
                self.add_error("onvif_username", "onvif 源必须填 onvif_username")
        return cleaned
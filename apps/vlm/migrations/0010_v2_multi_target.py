"""
v2 重构：引入多模型多类别支持。

字段变更
--------
1. VLMPromptConfig.target_classes 新增（CharField，逗号分隔，默认 "baby"）
2. VLMCheckState.has_baby_in_window 重命名为 has_target_in_window（保留数据）
3. VLMQueuedTask.has_baby_in_window 重命名为 has_target_in_window（保留数据）

注：用 RenameField 而非 RemoveField+AddField，保证老记录的 bool 值保留
（语义等同：旧逻辑只查 baby，新逻辑下老 prompt 默认 target_classes=["baby"]，
has_target_in_window 与老 has_baby_in_window 等价）。
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("vlm", "0009_lower_max_tokens_default"),
    ]

    operations = [
        migrations.AddField(
            model_name="vlmpromptconfig",
            name="target_classes",
            field=models.CharField(
                default="baby",
                help_text=(
                    "逗号分隔；可选项 baby / person / cat。"
                    "多选时 prompt 文案必须涵盖所有 target（如 '图中有宝宝或猫吗？'），"
                    "否则 VLM 看到的是未提及的 target，会一直 miss。"
                ),
                max_length=128,
                verbose_name="目标类别",
            ),
        ),
        migrations.RenameField(
            model_name="vlmcheckstate",
            old_name="has_baby_in_window",
            new_name="has_target_in_window",
        ),
        migrations.RenameField(
            model_name="vlmqueuedtask",
            old_name="has_baby_in_window",
            new_name="has_target_in_window",
        ),
    ]
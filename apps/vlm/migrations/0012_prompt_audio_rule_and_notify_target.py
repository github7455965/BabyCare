"""Phase 6 迁移第一步：**只建表**（spec §5.6 第 1 步）。

本步故意**不动** `VLMPromptConfig.notify_targets` 字段——它此刻仍指向旧的隐式
M2M 中间表 `vlm_prompt_config_notify_targets`。切 through 放在 0014，中间夹一个
0013 用 raw SQL 把旧表数据搬进新表，三步顺序不能合并，否则旧中间表会被直接删掉、
数据静默丢失（spec §5.6 的"迁移必须分三步"）。

新增：
- `PromptAudioRule`：Prompt 的声音条件（不持有阈值，spec §5.5）；
- `PromptNotifyTarget`：通知目标的显式 through 表（spec §5.6）；
- `PromptNotificationDelivery`：通知投递审计（spec §7.2）；
- `VLMCheckStateAudioEvent`：VLM 事件 ↔ 音频事件关联 + 判定快照（spec §5.7）。
"""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("audio_detect", "0004_audio_service_state_and_worker_epoch"),
        ("vlm", "0011_v2_target_field_meta"),
    ]

    operations = [
        migrations.CreateModel(
            name="PromptAudioRule",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("enabled", models.BooleanField(default=False, help_text="关闭 / 未配置 → 音频判为 UNKNOWN，不参与通知筛选（spec §7.1）。", verbose_name="启用声音条件")),
                ("condition", models.CharField(choices=[("any", "窗口内存在任一声音事件"), ("cry", "存在哭声"), ("speech", "存在说话声"), ("cry_or_speech", "哭声或说话声任一"), ("no_cry", "没有哭声"), ("no_speech", "没有说话声"), ("count_cry", "哭声事件数达标"), ("count_speech", "说话声事件数达标")], default="any", max_length=24, verbose_name="声音条件")),
                ("window_sec", models.PositiveIntegerField(default=60, help_text="Prompt 命中时刻往前看多少秒的 AudioEvent。", verbose_name="判定窗口(秒)")),
                ("min_event_count", models.PositiveIntegerField(default=1, help_text="仅 count_cry / count_speech 使用。", verbose_name="最少事件数")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("prompt_config", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="audio_rule", to="vlm.vlmpromptconfig", verbose_name="Prompt")),
            ],
            options={
                "verbose_name": "Prompt 声音规则",
                "verbose_name_plural": "Prompt 声音规则",
                "db_table": "vlm_prompt_audio_rule",
            },
        ),
        migrations.CreateModel(
            name="PromptNotifyTarget",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("condition", models.CharField(choices=[("always", "总是发送"), ("audio_rule", "声音条件满足时发送")], default="always", max_length=16, verbose_name="发送条件")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("notify_target", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="prompt_links", to="vlm.notifytarget", verbose_name="通知目标")),
                ("prompt_config", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notify_target_links", to="vlm.vlmpromptconfig", verbose_name="Prompt")),
            ],
            options={
                "verbose_name": "Prompt 通知目标",
                "verbose_name_plural": "Prompt 通知目标",
                "db_table": "vlm_prompt_notify_target",
                "ordering": ["prompt_config", "notify_target"],
            },
        ),
        migrations.CreateModel(
            name="VLMCheckStateAudioEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("snapshot_json", models.JSONField(blank=True, default=dict, verbose_name="判定快照")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("audio_event", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="vlm_state_links", to="audio_detect.audioevent", verbose_name="音频事件")),
                ("vlm_state", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="audio_event_links", to="vlm.vlmcheckstate", verbose_name="VLM 事件")),
            ],
            options={
                "verbose_name": "VLM 事件音频关联",
                "verbose_name_plural": "VLM 事件音频关联",
                "db_table": "vlm_check_state_audio_event",
                "ordering": ["-id"],
            },
        ),
        migrations.CreateModel(
            name="PromptNotificationDelivery",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("audio_state", models.CharField(choices=[("satisfied", "声音条件满足"), ("not_satisfied", "确认不满足"), ("unknown", "音频不可用（不筛选）")], default="unknown", max_length=16, verbose_name="音频三态")),
                ("condition", models.CharField(blank=True, default="", max_length=16, verbose_name="目标条件")),
                ("delivered", models.BooleanField(default=False, verbose_name="已发送")),
                ("error", models.TextField(blank=True, default="", verbose_name="发送错误")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="记录时间")),
                ("audio_event", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="notification_deliveries", to="audio_detect.audioevent", verbose_name="参考音频事件")),
                ("notify_target", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notification_deliveries", to="vlm.notifytarget", verbose_name="通知目标")),
                ("prompt_config", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notification_deliveries", to="vlm.vlmpromptconfig", verbose_name="Prompt")),
                ("vlm_state", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notification_deliveries", to="vlm.vlmcheckstate", verbose_name="VLM 事件")),
            ],
            options={
                "verbose_name": "通知投递记录",
                "verbose_name_plural": "通知投递记录",
                "db_table": "vlm_prompt_notification_delivery",
                "ordering": ["-id"],
                "indexes": [
                    models.Index(fields=["prompt_config", "created_at"], name="vlm_pnd_prompt_created_idx"),
                    models.Index(fields=["vlm_state"], name="vlm_pnd_state_idx"),
                ],
            },
        ),
        migrations.AddConstraint(
            model_name="promptnotifytarget",
            constraint=models.UniqueConstraint(fields=("prompt_config", "notify_target"), name="vlm_pnt_prompt_target_uq"),
        ),
    ]

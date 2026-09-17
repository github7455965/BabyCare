"""Phase 6 迁移第三步：`VLMPromptConfig.notify_targets` **切换到显式 through**（spec §5.6 第 3 步）。

为什么不能直接用 `AlterField`
------------------------------
Django **不支持**用 `AlterField` 给 M2M 加 / 去 `through=`：

```text
ValueError: Cannot alter field ... - they are not compatible types
(you cannot alter to or from M2M fields, or add or remove through= on M2M fields)
```

（`django/db/backends/base/schema.py::alter_field` 明示拒绝；autodetector 生成的
裸 `AlterField` 一 apply 就炸。）因此拆成 `SeparateDatabaseAndState`：

- **state_operations**：只改迁移状态，让 `notify_targets` 指向 `vlm.PromptNotifyTarget`；
- **database_operations**：`DROP TABLE IF EXISTS` 掉旧的隐式中间表。

为什么旧表要用 raw SQL 删（而不是 `DeleteModel`）
------------------------------------------------
隐式 M2M 的中间表是**自动生成**的，它只存在于渲染出来的 app registry 里，
**不在 `ProjectState.models` 中**，所以 `DeleteModel` 取不到它
（`from_state.apps.get_model(...)` 会 `LookupError`），只能按表名删。

旧表名 = `<宿主模型 db_table>_<字段名>` = `vlm_prompt_config_notify_targets`
（宿主 `VLMPromptConfig.db_table = "vlm_prompt_config"`）。

> ⚠️ apply 前必须 `python manage.py sqlmigrate vlm 0014` 确认：
> 删的是**旧自动表** `vlm_prompt_config_notify_targets`，
> 不是新 through 表 `vlm_prompt_notify_target`。
> 数据已由 0013 搬完，本步只改 schema 指向、清掉旧表。

不可逆性
--------
旧表里是"升级前"的数据快照，0013 正向搬走后即可丢弃。反向 migrate 时
本步用 `RunSQL.noop`（不重建旧表）——反向只用于开发回退，生产不需要。
"""

from django.db import migrations, models

OLD_TABLE = "vlm_prompt_config_notify_targets"

ALTER_NOTIFY_TARGETS_THROUGH = migrations.AlterField(
    model_name="vlmpromptconfig",
    name="notify_targets",
    field=models.ManyToManyField(
        blank=True,
        help_text=(
            "命中时要通知的目标；空 = 不通知。与 notify_on_hit AND 关系。"
            "每个目标带独立 condition（always / audio_rule），见 PromptNotifyTarget。"
        ),
        related_name="prompts",
        through="vlm.PromptNotifyTarget",
        to="vlm.notifytarget",
        verbose_name="通知目标",
    ),
)


class Migration(migrations.Migration):

    dependencies = [
        ("vlm", "0013_backfill_prompt_notify_targets"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[ALTER_NOTIFY_TARGETS_THROUGH],
            database_operations=[
                migrations.RunSQL(
                    sql=f"DROP TABLE IF EXISTS `{OLD_TABLE}`;",
                    reverse_sql=migrations.RunSQL.noop,
                ),
            ],
        ),
    ]

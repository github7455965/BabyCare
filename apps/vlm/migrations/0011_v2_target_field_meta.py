"""追平 `has_target_in_window` 的描述文字（纯 metadata，无库结构变更）。

背景
----
`0010_v2_multi_target` 用 `RenameField` 把 `has_baby_in_window` 改名成
`has_target_in_window`；`RenameField` 会**原样保留旧字段的全部属性**，其中
`verbose_name` 仍是 `"窗口内有 baby"`。之后模型侧改成了 `"窗口内有目标"`，
`VLMCheckState` 还补了 `help_text`，于是 autodetector 要求生成这个迁移。

影响
----
- 不改列名 / 类型 / 默认值，**不动数据**；
- `verbose_name` / `help_text` 都不是 DB 属性，`schema_editor.alter_field` 会
  判定无需改表并**跳过实际 SQL**，因此这是一个"只改迁移状态"的迁移。
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vlm', '0010_v2_multi_target'),
    ]

    operations = [
        migrations.AlterField(
            model_name='vlmcheckstate',
            name='has_target_in_window',
            field=models.BooleanField(default=False, help_text='窗口内任一 target_classes 命中为 True；旧名 has_baby_in_window（v2 rename）', verbose_name='窗口内有目标'),
        ),
        migrations.AlterField(
            model_name='vlmqueuedtask',
            name='has_target_in_window',
            field=models.BooleanField(default=False, verbose_name='窗口内有目标'),
        ),
    ]

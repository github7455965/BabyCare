"""Phase 6 迁移第二步：**回填**旧隐式 M2M 中间表 → 新 through 表（spec §5.6 第 2 步）。

为什么必须单独一步
------------------
Django 把 `ManyToManyField` 从"隐式 through"改成"显式 through"时**不会自动搬数据**：
旧中间表被删、新表是空的，`migrate` 不报错，升级后表现为"通知全停且没有任何异常"，
极难发现（spec §5.6）。因此用 raw SQL 在"建表之后、切 through（0014）之前"把旧表搬过来。

为什么用 raw SQL 而不是 ORM
---------------------------
迁移文件里的历史 model state **不认识旧中间表**（它从来不是真正的 model），ORM 读不到，
只能直接 `SELECT` 旧表名。

旧表名
------
隐式 M2M 的表名 = ``<宿主模型 db_table>_<字段名>``，即
``vlm_prompt_config_notify_targets``（宿主 `VLMPromptConfig.db_table = "vlm_prompt_config"`）。
额外保留一个"按 model_name 推导"的候选名做兜底，兼容不同 Django 版本的命名差异。

新表名
------
`PromptNotifyTarget.db_table = "vlm_prompt_notify_target"`（0012 已建）。

回填语义
--------
`condition` 一律写 `always`——**保持升级前的行为完全不变**：老配置"命中就发"，
不会因为引入音频维度而被静默（spec §5.6）。
"""

from django.db import migrations

#: 隐式 M2M 中间表候选名（第一个是当前项目的真实名字）
OLD_TABLE_CANDIDATES = (
    "vlm_prompt_config_notify_targets",
    "vlm_vlmpromptconfig_notify_targets",
)
NEW_MODEL = "PromptNotifyTarget"
DEFAULT_CONDITION = "always"


def _existing_old_table(connection) -> str | None:
    """返回实际存在的旧中间表名；都不存在 → None（全新部署 / 已切过 through）。"""
    with connection.cursor() as cursor:
        existing = set(connection.introspection.table_names(cursor))
    for name in OLD_TABLE_CANDIDATES:
        if name in existing:
            return name
    return None


def backfill(apps, schema_editor):
    """旧中间表 → 新 through 表；旧表不存在时 no-op。"""
    connection = schema_editor.connection
    old_table = _existing_old_table(connection)
    if old_table is None:
        return  # 全新部署 / 已经切过 through：没有可搬的数据

    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT vlmpromptconfig_id, notifytarget_id FROM `{old_table}`"
        )
        rows = cursor.fetchall()
    if not rows:
        return

    PromptNotifyTarget = apps.get_model("vlm", NEW_MODEL)
    objs = [
        PromptNotifyTarget(
            prompt_config_id=prompt_id,
            notify_target_id=target_id,
            condition=DEFAULT_CONDITION,
        )
        for prompt_id, target_id in rows
    ]
    # INSERT IGNORE：重复执行 / 已有部分行时不炸（MySQL）
    PromptNotifyTarget.objects.bulk_create(objs, ignore_conflicts=True)


class Migration(migrations.Migration):

    dependencies = [
        ("vlm", "0012_prompt_audio_rule_and_notify_target"),
    ]

    operations = [
        # reverse 用 noop：反向时 0014 会把字段切回旧隐式 M2M（重建旧表），
        # 再由 0012 的反向删掉新表；回搬没有意义（旧表是新建的、本来就是空的）。
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]

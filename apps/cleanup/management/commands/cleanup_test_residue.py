"""清理回归测试在库里留下的残留行。

背景
----
`apps/config_panel/tests.py` / `apps/cleanup/tests.py` 早期用 plain
`unittest.TestCase` 写。**当一次测试选择里没有 `django.test.TestCase` 时**，
Django 会判定"用不到 DB"并跳过 test 库创建（日志里的
`Skipping setup of unused database(s)`），测试于是直接写进**真实库**，
把摄像头 / Prompt / 通知目标越写越多（名字形如 `t_1a2b3c4d_卧房`）。

那些测试现已改为 `django.test.TestCase`（独立 test 库 + 自动回滚 + 兜底
`_purge`），不会再新增残留。本命令用来清理**历史上已经写进真实库**的那批行。

匹配规则（刻意收紧）
------------------
只匹配 `t_` + 8 位小写十六进制 + `_`，例如 `t_1f829299_cam_a`。
用户自己起的名字不会长这样 → 不会误删真实数据。

用法
----
    python manage.py cleanup_test_residue           # dry-run（默认，只报告）
    python manage.py cleanup_test_residue --yes     # 真删

注意
----
`--yes` 会真删真实库数据且**不可恢复**；先跑一次 dry-run 看清单再决定。
"""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

# 测试残留命名约定：t_<8 位小写 hex>_
RESIDUE_REGEX = r"^t_[0-9a-f]{8}_"
SAMPLE_LIMIT = 10


class Command(BaseCommand):
    help = "清理回归测试在真实库里留下的残留行（t_<8hex>_ 前缀）"

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes", action="store_true",
            help="真删（默认只做 dry-run 报告）。",
        )

    def handle(self, *args, **opts):
        # 函数内 import：避免 command 在相关 app 不可用时 import 失败
        from apps.streaming.models import Camera
        from apps.vlm.models import (
            NotifyTarget,
            VLMPromptConfig,
            VLMCheckState,
            VLMQueuedTask,
        )

        cams = Camera.objects.filter(name__regex=RESIDUE_REGEX)
        prompts = VLMPromptConfig.objects.filter(name__regex=RESIDUE_REGEX)
        targets = NotifyTarget.objects.filter(name__regex=RESIDUE_REGEX)

        self.stdout.write("=" * 72)
        self.stdout.write(f"测试残留匹配规则：{RESIDUE_REGEX}")
        self.stdout.write("=" * 72)
        for label, qs in (
            ("Camera", cams),
            ("VLMPromptConfig", prompts),
            ("NotifyTarget", targets),
        ):
            ids = list(qs.order_by("id").values_list("id", flat=True)[:SAMPLE_LIMIT])
            self.stdout.write(f"  [{qs.count():>4}] {label:<17} ids={ids}")

        # 报告会被连带清掉的行
        cascaded_states = VLMCheckState.objects.filter(prompt_config__in=prompts).count()
        cascaded_queued = VLMQueuedTask.objects.filter(prompt_config__in=prompts).count()
        self.stdout.write(
            f"  连带清除（CASCADE）：VLMCheckState≈{cascaded_states} "
            f"VLMQueuedTask≈{cascaded_queued}"
        )

        total = cams.count() + prompts.count() + targets.count()
        if total == 0:
            self.stdout.write(self.style.SUCCESS("没有残留，无需清理。"))
            return

        if not opts["yes"]:
            self.stdout.write("")
            self.stdout.write(
                "DRY-RUN：未删任何数据。确认上面的清单后加 --yes 真删。"
            )
            return

        # 按模型分别统计（delete() 的第一个返回值是"含 CASCADE"的总数，不能直接当行数用）
        deleted: dict[str, int] = {}

        def _accumulate(detail: dict) -> None:
            for label, n in detail.items():
                deleted[label] = deleted.get(label, 0) + n

        with transaction.atomic():
            # 1) prompt：CASCADE 清 VLMCheckState / VLMQueuedTask
            _accumulate(prompts.delete()[1])
            # 2) cam 关联的 state/queued（prompt 先删后，仍可能有 camera 指向的行）
            _accumulate(VLMCheckState.objects.filter(camera__in=cams).delete()[1])
            _accumulate(VLMQueuedTask.objects.filter(camera__in=cams).delete()[1])
            # 3) NotifyTarget
            _accumulate(targets.delete()[1])
            # 4) Camera：CASCADE 清 vlm_audio_runtime_state
            _accumulate(cams.delete()[1])

        summary = "  ".join(f"{label}={n}" for label, n in sorted(deleted.items()))
        self.stdout.write(self.style.SUCCESS(f"已清理：{summary}"))

"""
清理过期的 VLMCheckState 记录。

用法
----
python manage.py cleanup_vlm_states [--days N] [--keep-hits] [--dry-run]

参数
----
--days N      保留最近 N 天（默认 30）。早于 cutoff 的会被删。
--keep-hits   同时保留 hit=True 的记录（不论时间）。
--dry-run     只统计 / 打印，不实际删除。

行为
----
- cutoff = timezone.now() - timedelta(days=opts["days"])
- qs = VLMCheckState.objects.filter(created_at__lt=cutoff)
- 若 --keep-hits → qs = qs.filter(hit=False)（只删 hit=False 记录）
- 用 qs.count() 给 dry-run 一个精确计数；真实执行用 qs.delete()

注意
----
- VLMCheckState 用 USE_TZ=False（settings.py），但 timezone.now() 返回的是 tz-aware
  naive → aware 比较时会自动转 local（settings.TIME_ZONE）。
- 不联级删除 Camera / VLMPromptConfig（FK=CASCADE 走 ORM 自动行为；本命令不删模型）。
"""
from __future__ import annotations

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "清理过期的 VLMCheckState 记录"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days", type=int, default=30,
            help="保留最近 N 天（默认 30）。早于 cutoff 的会被删。",
        )
        parser.add_argument(
            "--keep-hits", action="store_true",
            help="同时保留 hit=True 的记录（不论时间）。",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="只统计 / 打印，不实际删除。",
        )

    def handle(self, *args, **opts):
        # 函数内 import：避免 command 在 `apps.vlm` 不可用时 import 失败
        from apps.vlm.models import VLMCheckState

        cutoff = timezone.now() - timedelta(days=opts["days"])
        qs = VLMCheckState.objects.filter(created_at__lt=cutoff)
        if opts["keep_hits"]:
            qs = qs.filter(hit=False)

        count = qs.count()
        if opts["dry_run"]:
            self.stdout.write(
                f"[dry-run] would delete {count} VLMCheckState records "
                f"(cutoff={cutoff.isoformat(timespec='seconds')}, "
                f"keep_hits={opts['keep_hits']}, days={opts['days']})"
            )
            return

        # qs.delete() 返回 (总行数, {表名: 行数})；取总数 n
        n, _ = qs.delete()
        self.stdout.write(
            f"deleted {n} VLMCheckState records "
            f"(cutoff={cutoff.isoformat(timespec='seconds')}, "
            f"keep_hits={opts['keep_hits']}, days={opts['days']})"
        )

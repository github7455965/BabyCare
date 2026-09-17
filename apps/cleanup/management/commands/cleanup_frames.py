"""
清理过期的采样帧图片目录。

用法
----
python manage.py cleanup_frames [--days N] [--dry-run]

参数
----
--days N     删除 N 天前的日期目录（默认 7）。
             目录命名格式 YYYY-MM-DD（YYYY-MM-DD < cutoff.date() 的删）。
--dry-run    只打印，不实际删除。

行为
----
- frames_root = MEDIA_ROOT / "frames"
- 不存在 → 直接返回 "no frames dir"，不报错
- 遍历 frames_root.iterdir()：
  - 跳过非日期格式目录（如 random.txt、临时文件）；不抛错
  - dir_date >= cutoff.date() → 保留
  - 否则：dry-run 打印 / 真删 shutil.rmtree(date_dir)

注意
----
- cleanup_vlm_states 删的是 DB 记录，img 字段存的是绝对路径；删 DB 不删文件。
  本命令只管磁盘文件。
- 同一天目录下的 jpg 是按 cam_id + ts_int 命名的；删除整目录不影响其他日期。
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "清理过期的采样帧图片目录（MEDIA_ROOT/frames/<YYYY-MM-DD>/）"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days", type=int, default=7,
            help="删除 N 天前的日期目录（默认 7）。",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="只打印，不实际删除。",
        )

    def handle(self, *args, **opts):
        cutoff = timezone.now() - timedelta(days=opts["days"])
        frames_root = Path(settings.MEDIA_ROOT) / "frames"
        if not frames_root.exists():
            self.stdout.write("no frames dir")
            return

        deleted = 0
        skipped_nondate = 0
        kept = 0
        for date_dir in frames_root.iterdir():
            # 非目录（如有人偷放 random.txt）跳过
            if not date_dir.is_dir():
                skipped_nondate += 1
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                # 非日期目录名 → 跳过，不抛错
                skipped_nondate += 1
                continue
            if dir_date >= cutoff.date():
                kept += 1
                continue
            if opts["dry_run"]:
                self.stdout.write(f"[dry-run] would delete {date_dir}")
                continue
            shutil.rmtree(date_dir)
            deleted += 1

        if opts["dry_run"]:
            self.stdout.write(
                f"[dry-run] would delete {deleted} date dirs "
                f"(cutoff={cutoff.date().isoformat()}, days={opts['days']}, "
                f"kept={kept}, skipped_nondate={skipped_nondate})"
            )
        else:
            self.stdout.write(
                f"deleted {deleted} date dirs "
                f"(cutoff={cutoff.date().isoformat()}, days={opts['days']}, "
                f"kept={kept}, skipped_nondate={skipped_nondate})"
            )

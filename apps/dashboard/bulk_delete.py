"""daemon 线程批量删除 VLMCheckState + 清孤儿图片。

入口
----
- `apps.dashboard.views._run_bulk_delete_in_thread(filters)` 起 daemon 线程
  调 `_run_bulk_delete`；mock 包装函数避免测试时真起线程。

阶段
----
1. `_delete_states(filters)` — 自己算 n_total + 分批（1000/批）按 filter 删 state，
   返回累计待清图片路径 set。
2. 算剩余 VLMCheckState 引用的图片路径 set（孤儿判定）。
3. `paths_to_potentially_clean - remaining` = 孤儿，逐个 unlink（missing_ok=True）。

边界
----
- LARGE DELETE warning: n_total > 10000 时 logger.warning
- ORM connection finally close（防死循环/卡死时 fd 漏）
- unlink 失败仅 logger.warning，不阻断

filter 逻辑复用 views._filter_kwargs，与 view 算 n_total 的逻辑同一源，避免漂移。

不重试；异常 logger.exception 留 traceback，UI 不感知。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set

from django.db import close_old_connections, transaction

from apps.dashboard.views import _filter_kwargs
from apps.vlm.models import VLMCheckState

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000
LARGE_DELETE_THRESHOLD = 10000


def _delete_states(filters: Dict) -> Set[str]:
    """阶段 A：分批删 state，返回累计待清图片路径 set。

    `transaction.atomic()` 每批一个事务，避免长事务；chunked iterator 不一次加载全表。

    自己算 n_total + 应用 LARGE DELETE warning（不依赖外部传入的 n_total，
    避免 view/worker 算两次 filter 漂移风险）。
    """
    qs = VLMCheckState.objects.filter(**_filter_kwargs(filters)).order_by("-id")
    n_total = qs.count()
    if n_total > LARGE_DELETE_THRESHOLD:
        logger.warning("[bulk-delete] LARGE DELETE: %d states", n_total)
    paths: Set[str] = set()
    batch_ids: list[int] = []
    deleted = 0
    for item in qs.iterator(chunk_size=BATCH_SIZE):
        batch_ids.append(item.id)
        for p in (item.img1, item.img2, item.img3):
            if p:
                paths.add(p)
        if len(batch_ids) >= BATCH_SIZE:
            with transaction.atomic():
                n, _ = VLMCheckState.objects.filter(pk__in=batch_ids).delete()
            deleted += n
            batch_ids.clear()
            logger.info("[bulk-delete] progress: %d / %d", deleted, n_total)
    if batch_ids:
        with transaction.atomic():
            n, _ = VLMCheckState.objects.filter(pk__in=batch_ids).delete()
        deleted += n
    logger.info("[bulk-delete] state deleted: %d (expected %d)", deleted, n_total)
    return paths


def _release_db_connection() -> None:
    """包装 close_old_connections，便于测试 mock（Django TestCase 在事务里跑，
    直接关闭连接会让后续 query 失败）。
    """
    close_old_connections()


def _run_bulk_delete(filters: Dict) -> None:
    """daemon 线程入口：删 state → 算剩余 → 清孤儿图片。"""
    try:
        paths_to_potentially_clean = _delete_states(filters)
        if not paths_to_potentially_clean:
            logger.info("[bulk-delete] no images to check")
            return
        # 算当前剩余 state 引用的图片
        remaining: Set[str] = set()
        for s in (
            VLMCheckState.objects
            .only("img1", "img2", "img3")
            .iterator(chunk_size=BATCH_SIZE)
        ):
            for p in (s.img1, s.img2, s.img3):
                if p:
                    remaining.add(p)
        orphans = paths_to_potentially_clean - remaining
        removed = 0
        for p in orphans:
            try:
                Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError:
                logger.warning("[bulk-delete] unlink failed: %s", p)
        logger.info(
            "[bulk-delete] orphan images removed: %d / %d candidates",
            removed,
            len(paths_to_potentially_clean),
        )
    except Exception:
        logger.exception("[bulk-delete] unhandled error")
    finally:
        _release_db_connection()

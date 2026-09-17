"""Phase 7 §8.3：音频线手动清理的**后台执行**部分。

两件事
------
1. :func:`_cleanup_by_filters` —— 按页面的筛选条件删日志 + 事件 + 音频文件；
2. :func:`purge_orphan_files` —— 删「库里没人引用、盘上还在」的 FLAC
   （spec §8.3 的「失败可重试」入口）。

执行顺序（对应 spec §8.3 的「先标记删除，再删除录音、片段和数据库记录」）
--------------------------------------------------------------------
=====================  ====================================================
步骤                    说明
=====================  ====================================================
1. 定格待删集合         先用筛选条件把「日志 id 集合 + 事件 id 集合 + 文件路径
                        集合」算出来 —— 这一步就是 spec 说的「标记」：后面所有
                        删除动作都以这份内存清单为准，不再重新查询。
2. 删数据库记录         日志 → 事件（``AudioEvent`` 的 ``segments`` 随之
                        CASCADE 删除；``VLMCheckStateAudioEvent`` /
                        ``PromptNotificationDelivery`` 上的 FK 是 ``SET_NULL``，
                        审计行保留但不再指向已删事件）。
3. 删媒体文件           事件 FLAC + 分段 FLAC，``missing_ok=True``。
=====================  ====================================================

为什么不反过来先删文件：文件先删、库删失败 → 事件变成「有记录没录音」，
页面播放直接坏掉，而用户看不到任何可重试的入口。先删库行，剩下的就是纯孤儿
文件，由 :func:`purge_orphan_files` 兜底清扫。

线程模型与 ``bulk_delete`` 一致：view 起 daemon 线程，日志留痕，异常不冒泡。
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict

from django.db import close_old_connections, transaction

from apps.audio_detect.models import AudioEvent, AudioEventSegment, SoundDetectionLog
from apps.dashboard.audio_views import (
    _build_log_queryset,
    _events_for_logs,
    _orphan_audio_files,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def _delete_logs(log_ids: list[int]) -> int:
    """分批删日志，返回删除行数。"""
    deleted = 0
    for i in range(0, len(log_ids), BATCH_SIZE):
        chunk = log_ids[i:i + BATCH_SIZE]
        with transaction.atomic():
            n, _ = SoundDetectionLog.objects.filter(pk__in=chunk).delete()
        deleted += n
    return deleted


def _delete_events(event_ids: list[int]) -> int:
    """分批删事件（含 CASCADE 分段），返回删除行数（含级联行）。"""
    deleted = 0
    for i in range(0, len(event_ids), BATCH_SIZE):
        chunk = event_ids[i:i + BATCH_SIZE]
        with transaction.atomic():
            n, _ = AudioEvent.objects.filter(pk__in=chunk).delete()
        deleted += n
    return deleted


def _cleanup_by_filters(filters: Dict) -> Dict:
    """按筛选条件定格 → 删库 → 删文件。返回统计（含失败数，供页面/日志）。"""
    result = {
        "logs_deleted": 0, "events_deleted": 0,
        "files_deleted": 0, "files_failed": 0,
    }

    # ---- 1. 定格待删集合（不重新查询，避免边删边漂移）----
    # 只取 4 个小字段：三个 JSON 大字段（top-K / 分数 / 阈值）这里完全用不到，
    # 加载它们会让"清几万条日志"变成几百 MB 内存。
    logs = list(
        _build_log_queryset(filters)
        .only("id", "camera_id", "window_start_ts", "window_end_ts")
    )
    if not logs:
        logger.info("[audio-cleanup] 筛选无匹配日志，无需清理")
        return result

    events = {}
    for evs in _events_for_logs(logs).values():
        for ev in evs:
            events[ev.pk] = ev
    event_ids = list(events)

    file_paths = [ev.audio_path for ev in events.values() if ev.audio_path]
    if event_ids:
        file_paths += [
            p for p in (
                AudioEventSegment.objects
                .filter(audio_event_id__in=event_ids)
                .exclude(audio_path="")
                .values_list("audio_path", flat=True)
            )
        ]

    # ---- 2. 删数据库记录 ----
    result["logs_deleted"] = _delete_logs([log.pk for log in logs])
    if event_ids:
        result["events_deleted"] = _delete_events(event_ids)

    # ---- 3. 删媒体文件（失败只计数，不阻断；孤儿页可重试）----
    for path in file_paths:
        try:
            os.unlink(path)
            result["files_deleted"] += 1
        except FileNotFoundError:
            continue
        except OSError as e:
            result["files_failed"] += 1
            logger.warning("[audio-cleanup] 删除文件失败 %s: %s", path, e)

    logger.info(
        "[audio-cleanup] 完成：logs=%d events=%d files=%d failed=%d",
        result["logs_deleted"], result["events_deleted"],
        result["files_deleted"], result["files_failed"],
    )
    return result


def purge_orphan_files() -> tuple[int, int]:
    """删孤儿 FLAC，返回 ``(成功数, 失败数)``（spec §8.3 重试入口）。"""
    info = _orphan_audio_files()
    removed = failed = 0
    for path in info.get("paths", []):
        try:
            os.unlink(path)
            removed += 1
        except FileNotFoundError:
            continue
        except OSError as e:
            failed += 1
            logger.warning("[audio-cleanup] 孤儿文件删除失败 %s: %s", path, e)
    logger.info("[audio-cleanup] 孤儿清理：removed=%d failed=%d", removed, failed)
    return removed, failed


def _release_db_connection() -> None:
    """包装 ``close_old_connections``，便于测试 patch（同 bulk_delete）。"""
    close_old_connections()


def _run_cleanup(filters: Dict) -> None:
    """daemon 线程入口：异常记日志，连接 finally 归还。"""
    try:
        _cleanup_by_filters(filters)
    except Exception:  # noqa: BLE001
        logger.exception("[audio-cleanup] unhandled error")
    finally:
        _release_db_connection()


def _run_cleanup_in_thread(filters: Dict) -> None:
    """起 daemon 线程跑清理；测试 patch 本函数避免真起线程。"""
    t = threading.Thread(
        target=_run_cleanup,
        args=(dict(filters),),
        daemon=True,
        name="audio-cleanup",
    )
    t.start()


__all__ = [
    "purge_orphan_files",
    "_cleanup_by_filters",
    "_run_cleanup",
    "_run_cleanup_in_thread",
]

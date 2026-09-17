"""音频 worker 实例互斥（spec §9.3，Phase 5 前置）。

音频线是独立进程，但**同一时刻只允许一个实例**：两个实例并存会互相破坏——
B 启动时执行 recover 会把 A 正在处理的事件打回 pending，导致同一事件被重复送
描述服务、状态互相覆盖（A/B 各自 save，结果随机）。

租约模型
--------
`AudioServiceState` 单行表当锁（``pk = SINGLETON_PK``）：

- **抢锁**（:func:`acquire_worker_slot`）：事务内 ``select_for_update`` 原子读改。
  租约"活跃"的判据是 **``pid`` 非空 且 ``updated_at`` 在心跳超时内**：
  满足 → 抛 :class:`WorkerLockError`；不满足 → 接管（生成新 ``worker_epoch``、
  写入自己的 pid）；
- **续租**（:func:`touch_worker_slot`）：manager 心跳时刷新 ``updated_at``；
- **释放**（:func:`release_worker_slot`）：只清自己 epoch 的 ``pid``
  （**保留 ``worker_epoch``**，供下次启动界定 recover 范围）。

判死口径（为什么以心跳为准，而不是 pid）
----------------------------------------
spec §9.3 定义"超过 ``BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC``（默认 10s）没心跳
→ 判定进程死了/卡住"。以心跳为权威有两个好处：跨平台可靠（Windows 上判断"进程
存在但已退出"不可靠），并且**卡死的 worker 也能被替换**——若以 pid 存活为准，
一个挂起但没退出的实例会让用户永远起不来。

``pid`` 的存活状态只用于日志：心跳过期但进程仍在时打 WARNING，提示可能是
"卡死"而不是"崩溃"，便于排查。

`worker_epoch` 的回收语义
-------------------------
接管成功后得到 :attr:`WorkerSlot.stale_epochs` = ``{接管前的 epoch, ""}``
（``""`` 覆盖加这个字段之前产生的历史行）。``EventAssembler`` /
``DescribeService`` 的 recover **只回收这些 epoch 的遗留行**；其它 epoch 的行
一律不动——即便出现"另一个异常存活的实例"，也不会被误回收。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from uuid import uuid4

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import AudioServiceState

logger = logging.getLogger(__name__)


class WorkerLockError(RuntimeError):
    """已有活跃的音频 worker 实例，拒绝启动。"""


@dataclass(frozen=True)
class WorkerSlot:
    """抢到的 worker 租约。"""

    #: 本实例的 epoch（写进事件/分段行）
    epoch: str
    #: 抢锁前那个租约的 epoch（``""`` = 没有 / 字段存在之前的历史数据）
    previous_epoch: str

    @property
    def stale_epochs(self) -> set[str]:
        """recover 可以安全回收的 epoch 集合。"""
        return {"", self.previous_epoch}


def _heartbeat_timeout_sec() -> float:
    return float(
        getattr(settings, "BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC", 10),
    )


def _pid_alive(pid: int | None) -> bool:
    """跨平台判断进程是否存活。

    用途：抢锁时的接管日志、web 侧 ``AudioWorkerManager.is_running()`` 的启停判断、
    ``stop()`` 的优雅等待。**不参与抢锁判据**——租约是否活跃一律以心跳为准
    （见模块文档"判死口径"）。

    Windows 实现有坑，见 :func:`_pid_alive_windows` 里的说明。
    """
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True             # 存在但无权访问
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes

    SYNCHRONIZE = 0x00100000
    ERROR_ACCESS_DENIED = 5
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # 64 位下 HANDLE 是指针；默认 restype 会被截断，必须显式声明
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if not handle:
        # 打不开但报"拒绝访问" → 进程存在（只是权限不够）
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        # ⚠️ 打开成功 ≠ 进程还活着：进程退出后，只要还有**任何**句柄指向它，内核对象
        # 就不会销毁，OpenProcess 依旧成功——最典型的就是本进程 Popen 持有的那个句柄。
        # 必须再问一次"这个对象是否已 signaled（= 进程已终止）"。
        # 现场踩到（2026-09-12）：worker 已优雅退出 1s，这里仍返回 True → web 白等
        # 10s 判超时、误报"仍然存活"、PID 文件不清理。
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def acquire_worker_slot(
    reason: str = AudioServiceState.REASON_AUTOSTART,
) -> WorkerSlot:
    """抢 worker 租约；已有活跃实例时抛 :class:`WorkerLockError`。

    Raises:
        WorkerLockError: 租约活跃（``pid`` 非空且心跳新鲜）。
    """
    timeout = _heartbeat_timeout_sec()
    now = timezone.now()
    with transaction.atomic():
        # 先确保单行存在，再对它加行锁：get_or_create 和 select_for_update 分开写，
        # 避免把 create 也裹进锁语义里（首次并发启动时最多一方失败，属可接受）。
        AudioServiceState.objects.get_or_create(pk=AudioServiceState.SINGLETON_PK)
        row = AudioServiceState.objects.select_for_update().get(
            pk=AudioServiceState.SINGLETON_PK,
        )
        if row.pid is not None:
            age = (now - row.updated_at).total_seconds()
            if age < timeout:
                raise WorkerLockError(
                    f"已有音频 worker 在运行（pid={row.pid}, "
                    f"epoch={(row.worker_epoch or '-')[:8]}, 心跳 {age:.1f}s 前）"
                )
            # 心跳过期 → 旧实例已崩/卡死，允许接管（spec §9.3 的判死口径）
            logger.warning(
                "[audio] 接管过期租约：旧 pid=%s epoch=%s 距今 %.1fs"
                "（超时 %.0fs，进程%s存活）",
                row.pid, (row.worker_epoch or "-")[:8], age, timeout,
                "" if _pid_alive(row.pid) else "不",
            )

        slot = WorkerSlot(epoch=uuid4().hex, previous_epoch=row.worker_epoch or "")
        row.pid = os.getpid()
        row.worker_epoch = slot.epoch
        row.reason = reason
        row.updated_at = now
        row.save(update_fields=["pid", "worker_epoch", "reason", "updated_at"])
        logger.info(
            "[audio] worker 租约已获取：pid=%s epoch=%s（上一个 epoch=%s）",
            row.pid, slot.epoch[:8], slot.previous_epoch[:8] or "-",
        )
        return slot


def touch_worker_slot(epoch: str) -> bool:
    """续租（心跳）。只刷新自己 epoch 的行；返回是否仍是持有者。"""
    if not epoch:
        return False
    return bool(
        AudioServiceState.objects.filter(
            pk=AudioServiceState.SINGLETON_PK, worker_epoch=epoch,
        ).update(updated_at=timezone.now())
    )


def release_worker_slot(epoch: str) -> bool:
    """释放租约：清 ``pid``、保留 ``worker_epoch``（供下次 recover 界定范围）。

    只动自己 epoch 的行，避免误清已经接管的新实例。
    """
    if not epoch:
        return False
    released = bool(
        AudioServiceState.objects.filter(
            pk=AudioServiceState.SINGLETON_PK, worker_epoch=epoch,
        ).update(pid=None, updated_at=timezone.now())
    )
    if released:
        logger.info("[audio] worker 租约已释放：epoch=%s", epoch[:8])
    return released


__all__ = [
    "WorkerLockError",
    "WorkerSlot",
    "acquire_worker_slot",
    "release_worker_slot",
    "touch_worker_slot",
]

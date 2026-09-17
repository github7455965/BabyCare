"""
SamplerWorker（per-cam 1Hz 采样线程）。

设计要点
--------
- 1Hz 严格整秒对齐（v6 设计：ts 全部 int 秒）
  next_ts = max(int(time.time()) + 1, prev_ts + 1)
  → 漂移自校正，长期不累积
- 从 FrameBus.get_snapshot() 拉最新帧 → FrameQueue.push(ts, ndarray)
- **不写盘**（Step 5+ Storage 按需落盘）
- **不调 YOLO**（YoloLoop 异步从 FrameQueue 队头 detect）
- frame=None 时 skip（不消耗一个 ts 位）

线程模型
--------
- 每路 active Camera 一个 worker（daemon thread）
- 启停由 VideoStreamManager 风格同构的 SamplerManager 维护
"""

from __future__ import annotations

import logging
import threading
import time

from apps.streaming.frame_bus import FrameBus
from .frame_queue import FrameQueue, FrameQueueManager


logger = logging.getLogger(__name__)


class _SamplerWorker:
    """单路摄像头 1Hz 采样线程。"""

    def __init__(self, cam_id: int):
        self.cam_id = cam_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_ts: int = 0  # 下一个目标秒

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # 第一个目标 ts：当前秒 +1（避免从过去帧开始）
        self._next_ts = int(time.time()) + 1
        self._thread = threading.Thread(
            target=self._run, name=f"sample-{self.cam_id}", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        bus = FrameBus.instance()
        fq_mgr = FrameQueueManager.instance()
        fq = fq_mgr.get_or_create(self.cam_id)

        while not self._stop.is_set():
            # 对齐到 next_ts
            now = time.time()
            sleep_left = self._next_ts - now
            if sleep_left > 0:
                # 用 Event.wait 支持中途 stop
                if self._stop.wait(sleep_left):
                    return

            ts_int = self._next_ts
            # 拿当前最新帧
            _, frame, _ = bus.get_snapshot(self.cam_id)
            if frame is not None:
                fq.push(ts_int, frame, has_baby=None)

            # 推进 next_ts（自校正：如果落后 ≥2s 就跳到当前下一秒，不追历史）
            now_after = time.time()
            self._next_ts = max(int(now_after) + 1, ts_int + 1)


# ---------------------------------------------------------------------------
class SamplerManager:
    """全局单例：维护 per-cam SamplerWorker。

    设计
    ----
    - reload_from_db() 改完内部状态后，会调注册的 on_cams_changed 回调
      让 YoloLoop / VideoStreamManager 等同步知道 active cam 集合变了
    - 注册方（YoloLoop / VideoStreamManager）在自己 ready() 里 register_callback()
    - **不在 signal handler 里直接调 YoloLoop**——避免跨模块耦合 + 锁复杂
    """

    _instance: "SamplerManager | None" = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._workers: dict[int, _SamplerWorker] = {}
        self._lock = threading.Lock()
        self._callbacks: list = []  # [(name, fn(active_ids: list[int])) ...]

    @classmethod
    def instance(cls) -> "SamplerManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def register_callback(self, name: str, fn) -> None:
        """其他模块注册：reload_from_db 完后会被调 fn(active_ids)。"""
        with self._lock:
            self._callbacks.append((name, fn))

    # ------------------------------------------------------------------
    def reload_from_db(self) -> None:
        """
        按 Camera 表当前 is_active=True 拉起/关闭 sampler。

        - 新增 / is_active False→True → 启 worker
        - 删除 / is_active True→False → 停 worker
        - 其余保持
        - 末尾：广播给所有注册回调（YoloLoop / VideoStreamManager 等）
        """
        from apps.streaming.models import Camera

        with self._lock:
            try:
                cams = list(Camera.objects.filter(is_active=True))
            except Exception as e:
                logger.warning("[sampler] reload_from_db: query failed: %s", e)
                return
            active_ids = {c.id for c in cams}

            # 停掉不再 active 的
            for cam_id in list(self._workers.keys()):
                if cam_id not in active_ids:
                    logger.info("[sampler] stopping worker for cam=%d", cam_id)
                    self._workers.pop(cam_id).stop()
                    FrameQueueManager.instance().remove(cam_id)

            # 启 active 但没在跑的
            for cam in cams:
                if cam.id in self._workers:
                    continue
                w = _SamplerWorker(cam.id)
                self._workers[cam.id] = w
                w.start()
                logger.info("[sampler] started worker for cam=%d", cam.id)

            active_ids_list = sorted(active_ids)
            callbacks = list(self._callbacks)

        # 锁外调回调：避免回调里反向调 SamplerManager 时死锁
        for name, fn in callbacks:
            try:
                fn(active_ids_list)
            except Exception as e:
                logger.warning("[sampler] callback %s failed: %s", name, e)

    def stop_all(self) -> None:
        with self._lock:
            for cam_id in list(self._workers.keys()):
                self._workers.pop(cam_id).stop()

    def list_active(self) -> list[int]:
        with self._lock:
            return list(self._workers.keys())
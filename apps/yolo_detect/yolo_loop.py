"""
YoloLoop（全局单线程）：从所有 cam 的 FrameQueue 取最早未推理帧，逐个 detect。

设计要点（v6）
--------------
- 帧**永远 CPU 采集 + 入 FrameQueue**（不依赖 YOLO）
- YOLO 不必立刻启动；未 detect 的帧 has_baby=None
- YoloLoop 一启动：从 FrameQueue 队头（最早的）未推理帧开始，逐帧 detect → mark_detected
- 与 SamplerWorker 完全解耦：SamplerWorker 永不阻塞
- 轮询各 cam 的 FrameQueue（保证不饿死任何 cam）
- 所有 FrameQueue 都空时 sleep 0.5s
- Step 3 阶段 gpu_manager 是 stub（no-op）；Step 4 才实装真实状态机

v2 多模型
---------
- 每个 cam 有自己的 enabled YOLOModel 列表（由 on_cam_models_changed 维护）
- detect 路径走 YoloRegistry.instance().detect(frame, models) → Dict[class_name, bool]
- 结果一次性 mark_detected(ts, **det_dict) 写回 FrameItem 的 has_<name> 字段
- cam 未配 model → 该 cam 跳过 detect（FrameItem 永远 None；Cursor/Selector 视为"未推理"）
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

from .frame_queue import FrameItem, FrameQueueManager
from .gpu_manager import GpuManager
from .yolo_registry import YoloRegistry


logger = logging.getLogger(__name__)


#: cam → models 缓存的存活时间（秒）。过期后下次取用会重读 DB。
#:
#: 为什么需要 TTL：缓存刷新依赖 ``Camera.yolo_models`` 的 ``m2m_changed`` signal，
#: 而 signal **只在触发它的那个进程里生效** —— 用 shell / 脚本 / 另一个进程改绑定，
#: 跑着的 daphne 收不到通知，缓存会永久停在旧列表（症状：某类别一直是 None，
#: 依赖它的 prompt 永远凑不出窗口）。TTL 让它在 30s 内自愈。
_CAM_MODELS_TTL_SEC = 30.0


# 在循环外延迟 import，避免启动期 django.db 没就绪时炸
def _get_yolo_models_for_cam(cam_id: int) -> list:
    """从 DB 读 cam 当前 enabled models 列表（每次外部 cache miss 时调用）。"""
    from apps.yolo_detect.models import YOLOModel
    from apps.streaming.models import Camera

    try:
        cam = Camera.objects.get(pk=cam_id)
    except Camera.DoesNotExist:
        return []
    return list(cam.yolo_models.filter(enabled=True))


class _YoloLoop:
    """全局单线程：扫描所有 cam FrameQueue，detect 最早未推理帧。

    cam 列表怎么维护
    ----------------
    - 由 SamplerManager.reload_from_db() 末尾通过 on_cams_changed() 显式通知
    - 不用每轮自扫（之前有 _refresh_cam_order，已删；改显式注册）

    cam → models 列表怎么维护
    ------------------------
    - 由 Camera.yolo_models m2m_changed signal 通过 on_cam_models_changed() 显式通知
    - YoloLoop 端缓存 _cam_models: Dict[cam_id, List[YOLOModel]]
    - 缓存未命中（cam_id 不在 dict）→ 重新从 DB 读（cam 刚加载，缓存还是空）
    """

    def __init__(self):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # cam 轮询顺序（保证不饿死任何 cam）
        self._cam_order: List[int] = []
        self._cursor_idx: int = 0
        # cam_id → 该 cam 启用的 YOLOModel 列表（v2）
        self._cam_models: dict = {}
        # cam_id → 该缓存写入时刻（time.monotonic），用于 TTL 过期重读
        self._cam_models_at: dict = {}
        # 保护 _cam_order / _cursor_idx / _cam_models 的写
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="yolo-loop", daemon=True,
        )
        self._thread.start()
        logger.info("[yolo-loop] started")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None
        logger.info("[yolo-loop] stopped")

    # ------------------------------------------------------------------
    def on_cams_changed(self, cam_ids: List[int]) -> None:
        """由 SamplerManager.reload_from_db() 末尾回调。

        - 新增 cam：放到队尾（不抢 cursor）
        - 删除 cam：从 cam_order 中移除
        - cursor 越界时重置为 0
        """
        with self._lock:
            new_set = set(cam_ids)
            old_set = set(self._cam_order)
            added = sorted(new_set - old_set)
            removed = sorted(old_set - new_set)

            if removed:
                self._cam_order = [c for c in self._cam_order if c not in removed]
                # 清掉被删 cam 的 models 缓存
                for c in removed:
                    self._cam_models.pop(c, None)
                    self._cam_models_at.pop(c, None)
                logger.info("[yolo-loop] cam removed: %s; cam_order=%s",
                            removed, self._cam_order)
            if added:
                self._cam_order = self._cam_order + added
                # 新 cam 不预拉 models，第一次 detect 时 _get_cam_models 兜底
                logger.info("[yolo-loop] cam added: %s; cam_order=%s",
                            added, self._cam_order)

            if self._cam_order and self._cursor_idx >= len(self._cam_order):
                self._cursor_idx = 0

    def on_cam_models_changed(self, cam_id: int) -> None:
        """由 Camera.yolo_models m2m_changed signal 回调：刷新该 cam 的 models 缓存。"""
        models = _get_yolo_models_for_cam(cam_id)
        with self._lock:
            self._cam_models[cam_id] = models
            self._cam_models_at[cam_id] = time.monotonic()
        logger.info(
            "[yolo-loop] cam=%d models refreshed: %s",
            cam_id, [m.name for m in models],
        )

    def _get_cam_order_snapshot(self) -> List[int]:
        """线程安全读：给 _take_next / 日志用。"""
        with self._lock:
            return list(self._cam_order)

    def _get_cam_models(self, cam_id: int) -> list:
        """线程安全读 cam 的 models 缓存；miss 或 TTL 过期时重读 DB。

        TTL 是"signal 丢失 / 跨进程改绑定"的自愈手段，见 ``_CAM_MODELS_TTL_SEC``。
        """
        now = time.monotonic()
        with self._lock:
            models = self._cam_models.get(cam_id)
            loaded_at = self._cam_models_at.get(cam_id, 0.0)
        if models is not None and (now - loaded_at) <= _CAM_MODELS_TTL_SEC:
            return models
        models = _get_yolo_models_for_cam(cam_id)
        with self._lock:
            self._cam_models[cam_id] = models
            self._cam_models_at[cam_id] = now
        return models

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        registry = YoloRegistry.instance()
        gpu = GpuManager.instance()
        fq_mgr = FrameQueueManager.instance()

        # 重置 cam 轮询（启动期 SamplerManager 已 reload，on_cams_changed 已被调）
        cam_order = self._get_cam_order_snapshot()
        logger.info("[yolo-loop] initial cam_order=%s", cam_order)

        while not self._stop.is_set():
            try:
                self._loop_iteration(fq_mgr, registry, gpu)
            except Exception:
                # 防御性兜底：任何未预料异常不能让 YoloLoop 线程死掉
                logger.exception("[yolo-loop] unhandled error in iteration; backoff 1s")
                self._stop.wait(1.0)

    def _loop_iteration(self, fq_mgr, registry, gpu) -> None:
        """单次循环体（拆出来便于外层 try/except 兜底）。"""
        # 1) 轮询各 cam，取最早未推理帧
        cam_order = self._get_cam_order_snapshot()
        item, cam_id = self._take_next(fq_mgr, cam_order)
        if item is None:
            # 都空 → release_yolo_if_idle + 睡
            gpu.release_yolo_if_idle()
            self._stop.wait(0.5)
            return

        # 2) acquire_yolo（Step 3 是 stub，Step 4 会真正等 VLM 让位）
        gpu.acquire_yolo()
        try:
            # 3) detect（v2 多 model）
            try:
                models = self._get_cam_models(cam_id)
            except Exception:
                # DB error 等瞬时故障：sleep 后重试，不让循环死掉
                logger.exception(
                    "[yolo-loop] get_cam_models failed cam=%d; backoff 1s",
                    cam_id,
                )
                gpu.release_yolo_if_idle()
                self._stop.wait(1.0)
                return
            if not models:
                # cam 未配 model → 该 cam 不 detect；为避免无限循环睡 0.5s
                logger.debug("[yolo-loop] cam=%d no yolo models configured",
                             cam_id)
                gpu.release_yolo_if_idle()
                self._stop.wait(0.5)
                return
            t0 = time.monotonic()
            try:
                det_dict = registry.detect(item.ndarray, models)
            except Exception:
                logger.exception("[yolo-loop] detect failed cam=%d ts=%d",
                                 cam_id, item.ts)
                # 异常路径也要 release（让对面有机会抢 GPU）；sleep 避免
                # 同帧立即重试（busy spin）。FrameItem 保持 None → cursor 等下一帧
                gpu.release_yolo_if_idle()
                self._stop.wait(0.5)
                return
            dt = time.monotonic() - t0

            # 4) 回填 FrameQueue（v2 一次性写入多 attr）
            fq_mgr.get_or_create(cam_id).mark_detected(item.ts, **det_dict)
            logger.debug(
                "[yolo-loop] cam=%d ts=%d detections=%s (%.3fs, models=%s)",
                cam_id, item.ts, det_dict, dt,
                [m.name for m in models],
            )
        finally:
            # 关键钩子（Step 4 实装时）：detect 完调 release_yolo_if_idle()
            # - 当前若还有 pending 帧：本轮不释放，循环再进 acquire（no-op 也 OK）
            # - 当前若 FrameQueue 全空：释放，让 VLM 有机会抢 GPU
            # Step 3 阶段 gpu_manager 是 stub，此调用 no-op；不阻塞 detect 节奏
            gpu.release_yolo_if_idle()

    def _take_next(
        self, fq_mgr: FrameQueueManager, cams: List[int],
    ) -> Tuple[Optional[FrameItem], int]:
        """轮询各 cam FrameQueue，取最早未推理帧。返回 (item, cam_id)。

        v2：未推理判定改为"任意 target 字段 == None"。但 FrameItem 默认 has_baby/has_person/has_cat
        全是 None，逻辑上等价于 has_baby=None。
        """
        if not cams:
            return None, -1
        n = len(cams)
        # 从 self._cursor_idx 起轮一圈
        with self._lock:
            start = self._cursor_idx
        for i in range(n):
            cam_id = cams[(start + i) % n]
            fq = fq_mgr.get(cam_id)
            if fq is None:
                continue
            item = fq.peek_oldest()
            if item is not None:
                # 推进 cursor（保证下次优先查下一 cam；轮询公平）
                with self._lock:
                    self._cursor_idx = (start + i + 1) % n
                return item, cam_id
        return None, -1


# ---------------------------------------------------------------------------
class YoloLoopManager:
    """全局单例：启停 _YoloLoop。"""

    _instance: "YoloLoopManager | None" = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._loop = _YoloLoop()

    @classmethod
    def instance(cls) -> "YoloLoopManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        self._loop.start()

    def stop(self) -> None:
        self._loop.stop()

    def on_cams_changed(self, cam_ids: List[int]) -> None:
        """外部调用：active cam 列表变了 → 同步 _cam_order。"""
        self._loop.on_cams_changed(cam_ids)

    def on_cam_models_changed(self, cam_id: int) -> None:
        """外部调用：cam 的 yolo_models 改了 → 刷新缓存。"""
        self._loop.on_cam_models_changed(cam_id)
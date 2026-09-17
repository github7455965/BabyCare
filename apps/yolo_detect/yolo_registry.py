"""
YoloRegistry（v2）：多 YOLO 模型注册表 + 缓存 + 顺序 predict。

设计要点
--------
- 单例全局，缓存已加载的 ultralytics.YOLO 实例（_models_cache: Dict[path, YOLO]）
- 首次加载用 _load_lock 串行化（避免并发首次加载同一 model）；二次进入直接走 cache
- predict 全程持 _predict_lock（与 v1 BabyDetector 共享同一思路，ultralytics
  predictor 状态串号）；多 model 顺序跑同一帧（GPU 单 stream 并发不可控，顺序简单可靠）
- detect() 返回 Dict[class_name, bool]，caller（YoloLoop）自行写回 FrameItem

缓存失效
--------
- 当前没有 hot-unload（模型一旦加载常驻显存）；如需下掉某个 model，外部手动调 unload(path)
- enabled=False 的 model YoloLoop 端跳过（不进 detect 调用），缓存仍保留（不卸载）

性能
----
- 单帧 2 model（best + yolov8s）顺序 predict ≈ 40ms（GPU 推断时间）
- 首次加载各 ~1-2s；一次性
- 显存：2 个 model + 中间特征 ≈ 640MB 峰值（顺序跑时实际只有 320MB）
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class YoloRegistry:
    """全局单例：管理多个 YOLO 模型的加载 + 缓存 + predict。"""

    _instance: Optional["YoloRegistry"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        # path(str) -> (YOLO, device, ref_count_used_for_warmup)
        self._models_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()    # 保护 _models_cache 读写
        self._load_lock = threading.Lock()     # 串行化"首次加载"（防并发 load 同一 model）
        self._predict_lock = threading.Lock()  # 串行化 predict（与 v1 BabyDetector 同源）
        self._device: str = ""

    @classmethod
    def instance(cls) -> "YoloRegistry":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        return self._device

    @property
    def loaded_paths(self) -> List[str]:
        with self._cache_lock:
            return list(self._models_cache.keys())

    # ------------------------------------------------------------------
    def _resolve_path(self, file_path: str) -> str:
        """相对 BASE_DIR 解析为绝对路径；不存在仍返回原值（让 ultralytics 自己抛错）。"""
        p = Path(file_path)
        if p.is_absolute():
            return str(p)
        # 09_web_vlm_manage 是 BASE_DIR（detector.py 用过的同套路）
        base = Path(__file__).resolve().parent.parent.parent
        return str(base / file_path)

    def _load(self, file_path: str) -> Any:
        """懒加载并缓存。线程安全（双检锁）。"""
        resolved = self._resolve_path(file_path)
        with self._cache_lock:
            cached = self._models_cache.get(resolved)
        if cached is not None:
            return cached
        with self._load_lock:
            # 双检
            with self._cache_lock:
                cached = self._models_cache.get(resolved)
            if cached is not None:
                return cached
            if not Path(resolved).exists():
                raise FileNotFoundError(f"model not found: {resolved}")
            t0 = time.time()
            from ultralytics import YOLO
            yolo = YOLO(resolved)
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                yolo.to(self._device)
            except Exception:
                self._device = "cpu"
            with self._cache_lock:
                self._models_cache[resolved] = yolo
            dt = time.time() - t0
            logger.info("[yolo-registry] loaded %.2fs path=%s device=%s",
                        dt, resolved, self._device)
        return yolo

    def unload(self, file_path: Optional[str] = None) -> None:
        """卸载缓存。file_path=None → 全部。"""
        with self._load_lock:  # 与 _load 互斥
            with self._cache_lock:
                if file_path is None:
                    paths = list(self._models_cache.keys())
                else:
                    paths = [self._resolve_path(file_path)]
            for p in paths:
                with self._cache_lock:
                    m = self._models_cache.pop(p, None)
                if m is not None:
                    try:
                        del m
                    except Exception:
                        pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("[yolo-registry] unloaded %d model(s)", len(paths))

    # ------------------------------------------------------------------
    def detect(
        self,
        frame,
        models: List,  # List[YOLOModel]；避免循环 import
        conf: float = 0.25,
        imgsz: int = 640,
        iou: float = 0.45,
    ) -> Dict[str, bool]:
        """对同一帧顺序跑所有 model，返回 {class_name: bool}。

        - 仅返回 models 中出现的 class_name（其它 FrameItem 已有字段保持原状）
        - coco_class_ids 空 → 不过滤（全类）；非空 → ultralytics predict(classes=...) 过滤
        - 任一 model 抛异常 → 该 model 贡献的 class 全 False（不阻断其它 model）
        """
        out: Dict[str, bool] = {}
        if not models:
            return out
        with self._predict_lock:
            for m in models:
                if not m.enabled:
                    for cn in m.class_names:
                        out[cn] = False
                    continue
                try:
                    yolo = self._load(m.file_path)
                    classes_arg = m.coco_class_ids if m.coco_class_ids else None
                    results = yolo.predict(
                        source=frame, conf=conf, iou=iou, imgsz=imgsz,
                        classes=classes_arg, verbose=False,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("[yolo-registry] predict failed model=%s err=%s",
                                   m.name, e)
                    for cn in m.class_names:
                        out[cn] = False
                    continue
                # 提取该 model 检测到的 class_name 集合
                hits: set = set()
                if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                    try:
                        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                    except Exception:
                        cls_ids = []
                    for i in cls_ids:
                        if 0 <= i < len(m.class_names):
                            hits.add(m.class_names[i])
                for cn in m.class_names:
                    out[cn] = out.get(cn, False) or (cn in hits)
        return out
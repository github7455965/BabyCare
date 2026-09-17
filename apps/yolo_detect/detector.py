"""
BabyDetector（兼容层）：09 v1 单 best.pt 接口保留作 deprecated wrapper。

v2 改造
-------
- 新增 YoloRegistry（apps/yolo_detect/yolo_registry.py）：多 model + 缓存 + detect(frame, models)
- BabyDetector 保留旧的 detect_has_baby() 走单 model 路径，给 gpu_manager 等老调用方
  或单 model 测试场景使用
- _predict_lock 仍是单实例属性，多线程调用 detect_has_baby 安全

新代码请用 YoloRegistry；本类不再扩展。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional


logger = logging.getLogger(__name__)


# 项目根下的 model/ 目录
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL = _BASE_DIR / "model" / "best.pt"


def baby_model_path() -> str:
    """BABYCARE_BABY_MODEL 覆盖 → 否则 model/best.pt。"""
    p = os.environ.get("BABYCARE_BABY_MODEL", "").strip()
    if p:
        return p
    return str(_DEFAULT_MODEL)


# ---------------------------------------------------------------------------
class BabyDetector:
    """单例 + 懒加载 best.pt（v1 兼容层，deprecated）。

    v2 阶段保留作 wrapper：
    - ensure_loaded() / unload() / is_loaded / device / model_path 给 gpu_manager 用
    - detect_has_baby() 给老测试用
    - detect() 内部委托给 YoloRegistry.instance()（单一 predict 路径）

    新代码请直接用 YoloRegistry。
    """

    _instance: Optional["BabyDetector"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._model: Any = None
        self._model_path: Optional[str] = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._device: str = ""

    @classmethod
    def instance(cls) -> "BabyDetector":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    def ensure_loaded(self, model_path: Optional[str] = None) -> Any:
        """懒加载 best.pt（v1 兼容 API）。新代码用 YoloRegistry.ensure_loaded。"""
        mp = model_path or baby_model_path()
        if self._model is not None and self._model_path == mp:
            return self._model
        with self._load_lock:
            if self._model is not None and self._model_path == mp:
                return self._model
            if not Path(mp).exists():
                raise FileNotFoundError(
                    f"model not found: {mp}（请把 best.pt 放到 09_web_vlm_manage/model/ 下，"
                    f"或用环境变量 BABYCARE_BABY_MODEL 覆盖路径）"
                )
            t0 = time.time()
            from ultralytics import YOLO
            m = YOLO(mp)
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                m.to(self._device)
            except Exception:
                self._device = "cpu"
            self._model = m
            self._model_path = mp
            dt = time.time() - t0
            logger.info("[baby_detector] loaded in %.2fs path=%s device=%s",
                        dt, mp, self._device)
        return self._model

    def unload(self) -> None:
        """释放模型 + 显存（Step 4 GpuManager 调用）。"""
        with self._load_lock:
            if self._model is None:
                return
            try:
                del self._model
            except Exception:
                pass
            self._model = None
            self._model_path = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[baby_detector] unloaded (cuda cache cleared)")

    # ------------------------------------------------------------------
    def detect_has_baby(
        self, frame,
        conf: float = 0.25, imgsz: int = 640, iou: float = 0.45,
    ) -> bool:
        """v1 接口：返回画面里有没有 baby。

        内部用 ultralytics 全帧 predict + 任意 box 即 True。
        新代码请直接用 YoloRegistry.detect()。
        """
        model = self.ensure_loaded()
        with self._predict_lock:
            results = model.predict(
                source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
            )
        if not results:
            return False
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return False
        return True

    @staticmethod
    def candidates(result) -> List[dict]:
        """调试/Step 9+ 可能用到：从 ultralytics Results 提取所有候选。"""
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        out = []
        for i in range(len(xyxy)):
            out.append({
                "xyxy": [float(xyxy[i, 0]), float(xyxy[i, 1]),
                         float(xyxy[i, 2]), float(xyxy[i, 3])],
                "conf": float(conf[i]),
            })
        out.sort(key=lambda x: -x["conf"])
        return out
"""双模型推理封装：YAMNet(TF) + PANNs(PyTorch)（spec §4.1/§4.2/§4.4）。

三类职责
--------
1. **模型常驻**：两模型在进程内各只加载一份，多路摄像头共享（spec §4.4）。
   每路一份会成倍吃内存/显存。
2. **线程限制**：P0-4 实测，不限制时 TF 线程池空闲自旋，2 路常驻进程平均吃
   ~296% CPU（≈3 核），而真实推理只用 ~100ms/窗。限制成单线程后降到 38%
   （PANNs 走 GPU 时 6%），延迟 104→192ms 仍远低于 1s hop。**必须配**。
3. **两级聚合**：先跨帧（YAMNet 2s 窗 → 4 个 patch 聚成一个），再跨标签
   （在业务标签支持的原始类里取 max）。只做第二步会把 4 个 patch 当成
   4 个独立结果，与"每 2 秒一个决策窗"的语义不符。

推理异常**不抛**给调用方：返回 ``ok=False`` 的输出，由共识引擎判成 ``abstain``。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .label_map import MODEL_PANNS, MODEL_YAMNET, AudioLabelMap

logger = logging.getLogger(__name__)

FRAME_AGG_MAX = "max"
FRAME_AGG_MEAN = "mean"
FRAME_AGG_MODES = (FRAME_AGG_MAX, FRAME_AGG_MEAN)

# 模型连续失败后，隔多久再尝试重新加载
RELOAD_BACKOFF_SEC = 60.0

#: PANNs Cnn14 要求 32kHz；对外统一 16kHz，推理前在内存重采样（spec §4.1）
PANNS_SAMPLE_RATE = 32000


# ---------------------------------------------------------------------------
# 跨帧聚合（第一级，spec §4.1）
# ---------------------------------------------------------------------------
def aggregate_frames(frame_matrix: Any, mode: str = FRAME_AGG_MAX) -> Any:
    """把 ``(frames, classes)`` 的逐帧分数聚成 ``(classes,)``。

    用 numpy 的 ``max``/``mean``（axis=0）。**不排序、不截断**：返回完整类别向量，
    下游按业务标签的类索引取值。
    """
    import numpy as np

    arr = np.asarray(frame_matrix, dtype="float32")
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"预期 (frames, classes) 二维分数矩阵，实得 shape={arr.shape}")
    if mode == FRAME_AGG_MAX:
        return arr.max(axis=0)
    if mode == FRAME_AGG_MEAN:
        return arr.mean(axis=0)
    raise ValueError(f"未知跨帧聚合方式: {mode!r}（可选 {FRAME_AGG_MODES}）")


def top_k_names(scores: Sequence[float], class_names: Sequence[str], k: int = 5):
    """仅用于日志展示的 top-K（**不参与判定**，spec §4.2）。"""
    import numpy as np

    arr = np.asarray(scores, dtype="float32")
    k = max(1, min(int(k), arr.shape[0]))
    idx = np.argpartition(-arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return [(class_names[i], round(float(arr[i]), 4)) for i in idx]


# ---------------------------------------------------------------------------
# 模型输出
# ---------------------------------------------------------------------------
@dataclass
class ModelOutput:
    """单个模型在一个窗口上的输出。"""

    model: str
    ok: bool
    #: 业务标签置信度
    scores: dict[str, float] = field(default_factory=dict)
    #: 每个业务标签命中的原始类分数（审计用）
    raw_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    version: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    top_k: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 单模型
# ---------------------------------------------------------------------------
class YamnetModel:
    """YAMNet（TensorFlow）。

    本地 SavedModel（spec §4.2 要求本地化，不能依赖在线 tfhub）：
    实测 2 秒 16kHz → ``scores`` shape ``(4, 521)``，约 0.48s/patch。
    """

    name = MODEL_YAMNET

    def __init__(self, model_dir: str, class_map: str, agg_mode: str = FRAME_AGG_MAX):
        self.model_dir = str(model_dir)
        self.class_map = str(class_map)
        self.agg_mode = agg_mode
        self._model = None
        self._version = ""

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def version(self) -> str:
        return self._version

    def load(self) -> None:
        import hashlib

        import tensorflow as tf

        t0 = time.perf_counter()
        self._model = tf.saved_model.load(self.model_dir)
        pb = Path(self.model_dir) / "saved_model.pb"
        try:
            digest = hashlib.sha256(pb.read_bytes()).hexdigest()[:12]
        except OSError:
            digest = "unknown"
        self._version = f"yamnet-savedmodel-{digest}"
        logger.info(
            "[audio] YAMNet loaded in %.1fs: %s (%s)",
            time.perf_counter() - t0, self.model_dir, self._version,
        )

    def warmup(self, seconds: float = 2.0, sample_rate: int = 16000) -> None:
        import numpy as np

        n = int(seconds * sample_rate)
        self._model(np.zeros(n, dtype="float32"))

    def infer(self, audio16: Any, label_map: AudioLabelMap) -> ModelOutput:
        import numpy as np

        if self._model is None:
            return ModelOutput(self.name, False, error="模型未加载", version=self._version)

        t0 = time.perf_counter()
        try:
            wave = np.ascontiguousarray(audio16, dtype="float32")
            scores, _emb, _spec = self._model(wave)
            frame_scores = np.asarray(scores, dtype="float32")   # (frames, 521)
        except Exception as e:  # noqa: BLE001
            return ModelOutput(
                self.name, False,
                error=f"{type(e).__name__}: {e}", version=self._version,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        # 第一级：跨帧（4 个 patch → 1 个向量）
        vector = aggregate_frames(frame_scores, self.agg_mode)
        return self._build(vector, label_map, t0)

    def _build(self, vector: Any, label_map: AudioLabelMap, t0: float) -> ModelOutput:
        names = label_map.class_names.get(MODEL_YAMNET) or []
        return ModelOutput(
            model=self.name,
            ok=True,
            scores=label_map.score_business(vector, MODEL_YAMNET),
            raw_scores=label_map.raw_score_table(vector, MODEL_YAMNET),
            version=self._version,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            top_k=top_k_names(vector, names),
        )


class PannsModel:
    """PANNs Cnn14 32kHz（PyTorch）。

    ``panns-inference`` 只有 32kHz 的 Cnn14，没有 16k 变体。因此推理前在内存
    用 ``resample_poly``（多相滤波）重采样到 32kHz——**不能用线性插值**，
    镜像频率会污染高频特征，而哭声能量恰好集中在高频段（spec §4.1）。
    """

    name = MODEL_PANNS

    def __init__(self, checkpoint: str, agg_mode: str = FRAME_AGG_MAX, use_gpu: bool = False):
        self.checkpoint = str(checkpoint)
        self.agg_mode = agg_mode
        self.use_gpu = bool(use_gpu)
        self._device = "cpu"
        self._tagging = None
        self._version = ""

    @property
    def loaded(self) -> bool:
        return self._tagging is not None

    @property
    def version(self) -> str:
        return self._version

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        import torch
        from panns_inference import AudioTagging

        t0 = time.perf_counter()
        device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        if self.use_gpu and device == "cpu":
            logger.warning("[audio] 请求 GPU 但 cuda 不可用 → PANNs 回退 CPU")
        self._tagging = AudioTagging(checkpoint_path=self.checkpoint, device=device)
        self._device = device
        ckpt = Path(self.checkpoint)
        self._version = f"panns-cnn14-32k-{ckpt.stem}"
        logger.info(
            "[audio] PANNs loaded in %.1fs: %s (device=%s, %s)",
            time.perf_counter() - t0, ckpt.name, device, self._version,
        )

    def warmup(self, seconds: float = 2.0, sample_rate: int = 16000) -> None:
        import numpy as np

        n = int(seconds * sample_rate)
        wave32 = self._to_32k(np.zeros(n, dtype="float32"))
        self._tagging.inference(np.stack([wave32]))

    def _to_32k(self, audio16: Any) -> Any:
        import numpy as np
        from scipy.signal import resample_poly

        ratio = PANNS_SAMPLE_RATE // 16000
        return np.ascontiguousarray(resample_poly(audio16, ratio, 1), dtype="float32")

    def infer(self, audio16: Any, label_map: AudioLabelMap) -> ModelOutput:
        import numpy as np

        if self._tagging is None:
            return ModelOutput(self.name, False, error="模型未加载", version=self._version)

        t0 = time.perf_counter()
        try:
            audio32 = self._to_32k(audio16)
            clipwise, _embedding = self._tagging.inference(np.stack([audio32]))
            vector = np.asarray(clipwise, dtype="float32").reshape(-1)   # (527,)
        except Exception as e:  # noqa: BLE001
            return ModelOutput(
                self.name, False,
                error=f"{type(e).__name__}: {e}", version=self._version,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

        names = label_map.class_names.get(MODEL_PANNS) or []
        return ModelOutput(
            model=self.name,
            ok=True,
            scores=label_map.score_business(vector, MODEL_PANNS),
            raw_scores=label_map.raw_score_table(vector, MODEL_PANNS),
            version=self._version,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            top_k=top_k_names(vector, names),
        )


# ---------------------------------------------------------------------------
# 双模型
# ---------------------------------------------------------------------------
class DualModelDetector:
    """两模型常驻 + 线程限制 + 预热 + 失败后带退避重载。

    模型只在本类里加载一次；多路摄像头共享同一个实例（spec §4.4）。
    """

    def __init__(
        self,
        label_map: AudioLabelMap,
        yamnet: YamnetModel,
        panns: PannsModel,
        tf_threads: int = 1,
        torch_threads: int = 1,
    ):
        self.label_map = label_map
        self.yamnet = yamnet
        self.panns = panns
        self.tf_threads = max(1, int(tf_threads))
        self.torch_threads = max(1, int(torch_threads))
        self._threads_configured = False
        self._failures = 0
        self._fail_counts = {MODEL_YAMNET: 0, MODEL_PANNS: 0}
        self._last_reload = 0.0

    @classmethod
    def from_settings(cls, label_map: AudioLabelMap) -> "DualModelDetector":
        from django.conf import settings

        return cls(
            label_map=label_map,
            yamnet=YamnetModel(
                model_dir=settings.BABYCARE_AUDIO_YAMNET_MODEL_DIR,
                class_map=settings.BABYCARE_AUDIO_YAMNET_CLASS_MAP,
                agg_mode=settings.BABYCARE_AUDIO_FRAME_AGG,
            ),
            panns=PannsModel(
                checkpoint=settings.BABYCARE_AUDIO_PANNS_CHECKPOINT,
                agg_mode=settings.BABYCARE_AUDIO_FRAME_AGG,
                use_gpu=settings.BABYCARE_AUDIO_USE_GPU,
            ),
            tf_threads=settings.BABYCARE_AUDIO_TF_THREADS,
            torch_threads=settings.BABYCARE_AUDIO_TORCH_THREADS,
        )

    # ------------------------------------------------------------------
    def _configure_threads(self) -> None:
        """必须在加载/推理**之前**设置（spec §4.4 / P0-4 结论）。"""
        if self._threads_configured:
            return
        import tensorflow as tf
        import torch

        tf.config.threading.set_intra_op_parallelism_threads(self.tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(self.tf_threads)
        torch.set_num_threads(self.torch_threads)
        self._threads_configured = True
        logger.info(
            "[audio] 线程限制: TF=%s torch=%s（P0-4：不限制会白烧约 3 核）",
            self.tf_threads, self.torch_threads,
        )

    def load(self) -> dict[str, str]:
        """加载两模型。单个模型失败不抛（记为不可用 → 共识判 abstain）。

        **映射表校验失败会抛**（由 :meth:`AudioLabelMap.from_settings` 完成）——
        那是配置错误，必须让 worker 起不来。
        """
        self._validate_sample_rate()
        self._configure_threads()
        errors: dict[str, str] = {}
        for model in (self.yamnet, self.panns):
            try:
                model.load()
            except Exception as e:  # noqa: BLE001
                errors[model.name] = f"{type(e).__name__}: {e}"
                logger.exception("[audio] %s 加载失败", model.name)
        return errors

    def _validate_sample_rate(self) -> None:
        """推理链路按 16kHz 硬编码假设（YAMNet 输入规格、PANNs 重采样比例）。

        配了别的采样率不会崩，只会**静默给出错误分数**（重采样比例错、频谱
        整体平移）——比启动失败难排查得多，所以必须 fail fast。
        """
        from django.conf import settings

        sr = int(settings.BABYCARE_AUDIO_SAMPLE_RATE)
        if sr != 16000:
            raise RuntimeError(
                f"BABYCARE_AUDIO_SAMPLE_RATE={sr}：YAMNet 要求 16kHz 输入、"
                "PANNs 的 32kHz 重采样比例也按 16k 写死，不支持其它采样率"
            )

    def warmup(self, seconds: float = 2.0) -> None:
        for model in (self.yamnet, self.panns):
            if not model.loaded:
                continue
            try:
                model.warmup(seconds)
            except Exception:  # noqa: BLE001
                logger.exception("[audio] %s 预热失败", model.name)

    # ------------------------------------------------------------------
    def infer(self, audio16: Any) -> tuple[ModelOutput, ModelOutput]:
        """返回 ``(yamnet_output, panns_output)``；失败以 ``ok=False`` 表达。"""
        out_y = self._infer_one(self.yamnet, audio16)
        out_p = self._infer_one(self.panns, audio16)
        self._maybe_reload()
        return out_y, out_p

    def _infer_one(self, model: YamnetModel | PannsModel, audio16: Any) -> ModelOutput:
        if not model.loaded:
            self._fail_counts[model.name] += 1
            return ModelOutput(model.name, False, error="模型未加载", version=model.version)
        out = model.infer(audio16, self.label_map)
        if out.ok:
            self._fail_counts[model.name] = 0
        else:
            self._fail_counts[model.name] += 1
            self._failures += 1
            logger.warning("[audio] %s 推理失败: %s", model.name, out.error)
        return out

    def _maybe_reload(self) -> None:
        """某模型连续失败后，隔 :data:`RELOAD_BACKOFF_SEC` 重试加载一次。"""
        broken = [m for m in (self.yamnet, self.panns) if self._fail_counts[m.name] >= 3]
        if not broken:
            return
        now = time.monotonic()
        if now - self._last_reload < RELOAD_BACKOFF_SEC:
            return
        self._last_reload = now
        for model in broken:
            logger.warning("[audio] 尝试重新加载 %s", model.name)
            try:
                model.load()
                self._fail_counts[model.name] = 0
                logger.info("[audio] %s 重新加载成功", model.name)
            except Exception:  # noqa: BLE001
                logger.exception("[audio] %s 重新加载仍失败", model.name)

    # ------------------------------------------------------------------
    @property
    def status(self) -> dict[str, Any]:
        return {
            "yamnet": {
                "loaded": self.yamnet.loaded,
                "version": self.yamnet.version,
                "fail_count": self._fail_counts[MODEL_YAMNET],
            },
            "panns": {
                "loaded": self.panns.loaded,
                "version": self.panns.version,
                "device": self.panns.device,
                "fail_count": self._fail_counts[MODEL_PANNS],
            },
        }

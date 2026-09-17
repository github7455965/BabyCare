"""
Step 12 帧落盘工具：3 帧被选中 → 真落盘到 MEDIA_ROOT/frames/<date>/<cam>_<ts>.jpg

设计要点
--------
- 落盘**按需**：只在 PromptRunner._resolve_img_paths / _save_failure 里被调；
  v1-simplify 阶段采样器 (apps/yolo_detect/sampler.py) 不写盘（按需来）。
- **去重**：同 (cam_id, ts) 已存在 → 跳过 imwrite（同一帧被多 prompt 复用）。
- 目录不存在自动建（mkdir -p）。
- JPEG quality=70（cv2.IMWRITE_JPEG_QUALITY=70）。
- 路径全部**绝对路径**（VLMCheckState.img1/2/3 存绝对路径，便于后续 dev serving）。
- 落盘失败 → 返回 Path("")（或抛出被上层 catch → 留空串）。Runner 端决定如何降级。

- **送给模型的 bytes 必须缩图**：落盘留的是全分辨率（1080p/2K/4K，给页面看证据），
  而 VLM 只吃长边 ≤1024。实时路径（``PromptRunner._frame_to_bytes``）与回放路径
  （``Drainer`` → :func:`load_frame_bytes`）**共用** :func:`encode_vlm_jpeg`，不许各写一份
  ——2026-09-14 就是因为回放这边直接 ``read_bytes()`` 原图，把成本放大了 5 倍
  （实测 5986 tokens / 4.0s vs 实时 1225 tokens / 0.8s；几百条回放就把单 slot 的
  llama 打满，实时窗延迟从 1.5s 涨到 90s）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import cv2  # type: ignore
import numpy as np

from apps.core.imaging import fit_long_side


logger = logging.getLogger(__name__)


# JPEG 压缩质量（与设计/计划文档一致：1080p 单帧约 100-150KB）
_JPEG_QUALITY = 70

# 送给 VLM 的图长边上限（与 Runner._frame_to_bytes 的原实现一致：qwen-vl 视觉 token
# 随像素数增长，1080p ≈1500 token/张，4K/2K 必须缩）
_VLM_MAX_LONG_SIDE = 1024

# VLM 请求内 JPEG 编码质量（cv2.imencode 默认值，显式写出来免得以后改了落盘质量
# 连带改变请求内容）
_VLM_JPEG_QUALITY = 95

# frames 子目录名（与 media/frames/<YYYY-MM-DD>/<cam>_<ts>.jpg 约定）
_FRAMES_SUBDIR = "frames"


def frames_root(media_root: Union[str, Path]) -> Path:
    """返回 MEDIA_ROOT / frames 的绝对路径。"""
    return Path(media_root) / _FRAMES_SUBDIR


def frame_path(media_root: Union[str, Path], cam_id: int, ts: int) -> Path:
    """返回某帧的绝对路径（不实际写盘）。

    路径格式：<media_root>/frames/<YYYY-MM-DD>/<cam_id>_<ts>.jpg

    日期用本地时间（与 settings.TIME_ZONE 隐式对齐；USE_TZ=False 时 datetime
    无 tz 信息，标准 fromtimestamp(ts) 即返回 local time）。
    早期代码用 django.utils.timezone.localtime() 转 aware，结果 USE_TZ=False
    下抛 "localtime() cannot be applied to a naive datetime"。
    """
    import datetime as _dt

    # naive = 本地时间（settings.TIME_ZONE 不会改变 fromtimestamp 的字面意义）
    #      但操作系统会按当前 TZ 转换（env 配 Asia/Shanghai 时 → CST）
    dt = _dt.datetime.fromtimestamp(ts)
    date_dir = frames_root(media_root) / dt.strftime("%Y-%m-%d")
    return date_dir / f"{cam_id}_{ts}.jpg"


def save_frame(
    frame_ndarray: np.ndarray,
    cam_id: int,
    ts: int,
    media_root: Union[str, Path],
) -> Optional[Path]:
    """落盘单帧 JPEG（去重）。目录不存在自动建。

    Args:
        frame_ndarray: BGR ndarray（cv2 默认格式）。
        cam_id: 摄像头 ID。
        ts: int 秒时间戳。
        media_root: Django MEDIA_ROOT（绝对路径或 django settings 引用）。

    Returns:
        绝对路径对象；若文件已存在 → 直接返回（不重写）；
        若帧 ndarray 非法 / imwrite 失败 → 返回 None（由 caller 决定记日志 / 留空）。
    """
    if frame_ndarray is None or not isinstance(frame_ndarray, np.ndarray) or frame_ndarray.size == 0:
        logger.warning("frame_storage: skip invalid ndarray cam=%d ts=%d", cam_id, ts)
        return None

    path = frame_path(media_root, cam_id, ts)
    if path.exists():
        # 去重：同 cam+ts 已落盘 → 跳过 imwrite
        return path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("frame_storage: mkdir failed %s: %s", path.parent, e)
        return None

    try:
        ok = cv2.imwrite(
            str(path), frame_ndarray,
            [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("frame_storage: imwrite raised %s: %s", path, e)
        return None

    if not ok:
        logger.warning("frame_storage: imwrite returned False for %s", path)
        return None

    return path


def resize_long_side(
    frame_ndarray: np.ndarray, max_long_side: int = _VLM_MAX_LONG_SIDE,
) -> np.ndarray:
    """长边超过 ``max_long_side`` 时按比例缩小（**只缩不放**，保持长宽比）。

    未超限时原样返回（不复制），所以小图（测试用的 1x1 / 640x480）不会被改动。

    实现在 :func:`apps.core.imaging.fit_long_side`（与 FrameQueue 入队缩图共用同一份
    逻辑，避免两处各写一套比例计算）。
    """
    return fit_long_side(frame_ndarray, max_long_side)


def encode_vlm_jpeg(
    frame_ndarray: np.ndarray, max_long_side: int = _VLM_MAX_LONG_SIDE,
) -> bytes:
    """ndarray → 送给 VLM 的 JPEG bytes（长边限到 ``max_long_side``）。

    **实时与回放共用这一个函数**：Runner 拿内存里的帧调它，Drainer 读回落盘的图
    解码后也调它。历史上这两条路径各写一份，回放那份漏了 resize，代价差 5 倍。
    """
    arr = resize_long_side(frame_ndarray, max_long_side)
    ok, buf = cv2.imencode(
        ".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, _VLM_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError(f"cv2.imencode failed (shape={arr.shape})")
    return bytes(buf)


def load_frame_bytes(rel_or_abs_path: str, media_root: Union[str, Path]) -> bytes:
    """读已落盘的 JPEG 帧 → **送给 VLM 的 bytes**（Drainer 回放 VLMQueuedTask 用）。

    入参允许：
    - 绝对路径（VLMCheckState.img1/2/3 / VLMQueuedTask.img1/2/3 当前都存绝对路径）
    - MEDIA_ROOT 下的相对路径

    **读出来必须回炉缩图**（:func:`encode_vlm_jpeg`）：落盘的帧是全分辨率，直接
    ``read_bytes()`` 发给模型会让回放单次视觉 token 变成实时的 ~5 倍（实测
    5986 tokens/4.0s vs 1225 tokens/0.8s）——几百条回放就把单 slot llama 打满，
    实时窗跟着排队（2026-09-14）。

    异常 → 让上层 raise（drainer 转 retry/failed）。**解码失败不抛**：文件本身
    坏了，缩不缩都一样，退回原始 bytes 交给 llama 判失败，别把整个任务判成
    "帧不存在"。
    """
    if not rel_or_abs_path:
        raise FileNotFoundError("empty path")
    p = Path(rel_or_abs_path)
    if not p.is_absolute():
        p = Path(media_root) / rel_or_abs_path
    if not p.exists():
        raise FileNotFoundError(f"frame not found: {p}")

    raw = p.read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("frame_storage: imdecode failed, send raw bytes: %s", p)
        return raw
    return encode_vlm_jpeg(img)


"""帧图像工具（纯函数，跨 app 共用）。

为什么单独抽出这个模块
----------------------
``fit_long_side`` 原本只实现在 ``apps/vlm/frame_storage.py``（送 VLM 的图限长边）。
2026-09-15 起 ``apps/yolo_detect/frame_queue.py`` **入队时也要限长边**（队列存 1024
长边，把帧缓冲从 ~1.95GB 降到 ~0.4GB），但两个模块分属 ``vlm`` / ``yolo_detect``，
互相 import 会造成方向混乱（现有方向是 vlm → yolo_detect 的 GpuManager），所以把
纯函数上提到 ``apps/core``，两侧薄封装调用。

为什么惰性 import cv2
---------------------
``frame_queue`` 顶层刻意不依赖 numpy（``FrameItem.ndarray`` 声明为 ``object``，
见该模块 docstring），所以本模块也不在顶层 import cv2 —— 只有真的需要缩图时才加载。
"""

from __future__ import annotations


def fit_long_side(frame, max_long_side: int):
    """长边超过 ``max_long_side`` 时等比缩小（**只缩不放**），否则原样返回。

    - 未超限：直接返回入参（**不复制**），所以小图不会被改动，调用方也可用
      ``result is frame`` 判断"没缩"。
    - 超限：``INTER_AREA`` 重采样（缩小场景的标准选择，比 LINEAR 少摩尔纹）。

    Args:
        frame: BGR ndarray（cv2 默认格式），或任何有 ``shape`` 的 ndarray。
        max_long_side: 长边上限（像素）。

    Returns:
        缩小后的新 ndarray，或原对象（未超限时）。
    """
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return frame

    import cv2  # 惰性：见模块 docstring

    scale = max_long_side / long_side
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

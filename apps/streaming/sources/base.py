"""
所有摄像头源的抽象基类。

09 精简点（相对 08）：
- 不支持 pause/resume/speed/seek/step（v6 设计无回放交互；视频文件源只跑循环）
- 不存 snapshot_state（worker 重建时不需要保留播放位置）

- read()      : 拉一帧；返回 None 表示暂时没有帧（不要把它当错误）
- release()   : 释放底层硬件
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SourceInfo:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    source_type: str = ""     # "file" / "onvif"


class CameraSource(ABC):
    is_file: bool = False

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._info = SourceInfo()

    @property
    def info(self) -> SourceInfo:
        return self._info

    @abstractmethod
    def open(self) -> bool:
        """打开源；返回 True=成功。"""

    @abstractmethod
    def read(self):
        """拉一帧 BGR ndarray；失败/暂时无帧返回 None。"""

    @abstractmethod
    def release(self) -> None:
        """释放。"""
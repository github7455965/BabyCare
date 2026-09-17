"""
视频文件源（精简版）：cv2.VideoCapture 循环读取。

09 相对 08 的精简：
- 去掉 pause / speed / seek / step / snapshot_state / restore_state
  → 09 v1 视频源只用来"循环跑测试视频"，不提供回放控制
- 默认按源 fps 节流（wall clock 补偿）；无控制命令

行为：
- 读完末尾 → cap.set(POS_FRAMES, 0) → 形成循环
- read() 失败 → 返回 None；manager 走 retry 路径
- t0（视频循环时刻）由 manager 在首次成功 read 后调 cam_t0_manager.set(...)；
  本类不直接持有 t0。
"""

import logging
import threading
import time

import cv2

from .base import CameraSource, SourceInfo


logger = logging.getLogger(__name__)


class FileSource(CameraSource):
    is_file = True

    def __init__(self, source_url: str):
        # source_url 必须是绝对路径（manager 已解析过）
        super().__init__(source_url)
        self._cap: cv2.VideoCapture | None = None
        self._fps_native = 25.0
        # 帧率节流基准时刻（wall clock 补偿，避免 cv2.read 全速跑导致 fps 虚高）
        self._next_frame_t: float = 0.0
        self._lock = threading.Lock()  # open/release 与 read 互斥

    # ------------------------------------------------------------------
    def open(self) -> bool:
        url = self.source_url
        # cv2 在 Windows 上能直接打开中文路径吗？答：早期不行。保险起见用 imread 风格：先短路径？
        # 我们直接传原路径，新版 cv2 (4.x) 已支持 UTF-8。如果失败再回退。
        with self._lock:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                logger.warning("[file_source] open failed: %s", url)
                return False
            self._cap = cap
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            self._fps_native = fps
            self._info = SourceInfo(width=w, height=h, fps=fps, source_type="file")
            self._next_frame_t = 0.0
            logger.info("[file_source] opened %s (%dx%d @ %.1f fps)",
                        url, w, h, fps)
            return True

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    # ------------------------------------------------------------------
    def read(self):
        with self._lock:
            cap = self._cap
            if cap is None:
                return None

            ok, frame = cap.read()
            if not ok:
                # 末尾：循环回去
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    return None

        # 按源 fps 节流（wall clock 补偿，避免 cv2.read 全速跑导致 fps 虚高）
        fps_effective = max(self._fps_native, 1.0)
        frame_interval = 1.0 / fps_effective
        now_t = time.monotonic()
        if self._next_frame_t == 0.0:
            # 首帧：以"现在"为基准
            self._next_frame_t = now_t
        self._next_frame_t += frame_interval
        sleep_left = self._next_frame_t - now_t
        if sleep_left > 0:
            time.sleep(sleep_left)
        else:
            # 落后了（cv2.read 比预期慢）：重置基准避免追赶风暴
            self._next_frame_t = now_t
        return frame if ok else None
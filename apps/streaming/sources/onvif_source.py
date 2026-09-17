"""
ONVIF 摄像头源（精简版）。

协议分层
--------
- ONVIF 仅用于发现 + 拿 RTSP URL（GetProfiles + GetStreamUri）
- 真正的拉流走 RTSP（cv2.VideoCapture）：标准、稳定、onvif-zeep 之外的依赖最少

凭证
----
- 用户名 / host / port 来自 Camera 表（onvif_username / onvif_host / onvif_port）
- 密码统一从 settings.ONVIF_PASSWORD（= .env BABYCARE_ONVIF_PASSWORD）读
  设计理由：所有摄像头同一密码，集中管理；DB 只存非敏感信息

wsdl
----
onvif-zeep 0.2.12 pip 包内不带 wsdl 文件：
- 优先用 settings.ONVIF_WSDL_DIR (= BABYCARE_ONVIF_WSDL_DIR)
- 缺省用 onvif-zeep 包内默认路径（通常能工作；本地失败再手动指定）

回放控制
--------
is_file=False；本类无回放控制方法（精简自 08）。

重连
----
- open() 失败 → manager 走"连续 5 次失败退 worker"路径，由看门狗 30s 接管
- read() 失败 → 走 worker 内 read exception → release() + reopen
- 不在 OnvifSource 内做退避（保持单一职责）
"""

from __future__ import annotations

import logging
import threading
from urllib.parse import quote

import cv2

from django.conf import settings

from .base import CameraSource, SourceInfo


logger = logging.getLogger(__name__)


class OnvifSource(CameraSource):
    """ONVIF 摄像头源：ONVIF 拿 RTSP URL + cv2.VideoCapture 拉流。"""

    is_file = False

    def __init__(self, host: str, port: int, username: str):
        """
        :param host: 摄像头 IP
        :param port: ONVIF 服务端口（默认 80）
        :param username: 用户名
        密码不传——从 settings.ONVIF_PASSWORD 读（全局默认）
        """
        # source_url 用 host:port 形式存，方便日志/调试
        super().__init__(source_url=f"{host}:{port}")
        self._host = host
        self._port = int(port)
        self._username = username
        self._password = settings.ONVIF_PASSWORD  # 全局默认
        self._wsdl_dir = settings.ONVIF_WSDL_DIR or None

        self._onvif_cam = None     # onvif.ONVIFCamera 实例
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()  # open/release 与 read 互斥

    # ------------------------------------------------------------------
    def open(self) -> bool:
        """
        1. ONVIFCamera 连接 + GetStreamUri
        2. 把用户名:密码拼到 URL（cv2 不支持 ONVIF 鉴权）
        3. cv2.VideoCapture(rtsp_url) 打开 RTSP
        失败返回 False（不抛）；worker 会走"连续失败退 worker"路径。
        """
        with self._lock:
            if not self._host:
                logger.warning("[onvif] empty host, skip")
                return False
            if not self._password:
                logger.warning(
                    "[onvif %s:%s] BABYCARE_ONVIF_PASSWORD 未配置，跳过该摄像头。"
                    "所有 ONVIF 摄像头用同一全局密码。",
                    self._host, self._port,
                )
                return False

            # 1) ONVIF 连接
            try:
                from onvif import ONVIFCamera  # 延迟导入，未配置 ONVIF 摄像头时不浪费
            except ImportError:
                logger.error(
                    "[onvif %s:%s] 缺 onvif-zeep 包："
                    "pip install onvif-zeep",
                    self._host, self._port,
                )
                return False

            try:
                kwargs = {}
                if self._wsdl_dir:
                    kwargs["wsdl_dir"] = self._wsdl_dir
                cam = ONVIFCamera(
                    self._host, self._port, self._username, self._password,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(
                    "[onvif %s:%s] ONVIF connect failed: %s",
                    self._host, self._port, e,
                )
                return False

            # 2) GetStreamUri（profiles[0]）
            try:
                media = cam.create_media_service()
                profiles = media.GetProfiles()
                if not profiles:
                    logger.warning("[onvif %s:%s] no media profiles", self._host, self._port)
                    return False
                profile_token = profiles[0].token

                stream_setup = media.create_type("GetStreamUri")
                stream_setup.ProfileToken = profile_token
                stream_setup.StreamSetup = {
                    "Stream": "RTP-Unicast",
                    "Transport": {"Protocol": "RTSP"},
                }
                uri_obj = media.GetStreamUri(stream_setup)
                uri = uri_obj.Uri if hasattr(uri_obj, "Uri") else uri_obj.get("Uri")
                if not uri:
                    logger.warning("[onvif %s:%s] GetStreamUri returned empty", self._host, self._port)
                    return False
            except Exception as e:
                logger.warning(
                    "[onvif %s:%s] GetStreamUri failed: %s",
                    self._host, self._port, e,
                )
                return False

            # 3) 把用户名:密码嵌入 RTSP URL（cv2 不支持 ONVIF 鉴权）
            rtsp_url = self._inject_credentials(uri)

            # 4) cv2 打开 RTSP
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logger.warning(
                    "[onvif %s:%s] cv2.VideoCapture open failed (rtsp=%s)",
                    self._host, self._port,
                    self._redact_url(rtsp_url),
                )
                return False
            # 缓冲调小，降低推流延迟
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            if fps <= 0:
                fps = 25.0  # 兜底

            self._onvif_cam = cam
            self._cap = cap
            self._info = SourceInfo(
                width=w, height=h, fps=fps, source_type="onvif",
            )
            logger.info(
                "[onvif %s:%s] opened rtsp=%s %dx%d @ %.1f fps",
                self._host, self._port,
                self._redact_url(rtsp_url), w, h, fps,
            )
            return True

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._onvif_cam = None

    # ------------------------------------------------------------------
    def read(self):
        """读一帧 BGR ndarray；拉不到返回 None（不要当错误）。"""
        with self._lock:
            cap = self._cap
            if cap is None:
                return None
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            return frame

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _inject_credentials(self, url: str) -> str:
        """
        把 user:pass 拼到 rtsp://user:pass@host:port/path。
        凭据含特殊字符（@/:）走 urllib.parse.quote。
        """
        user = quote(self._username or "", safe="")
        pw = quote(self._password or "", safe="")
        if url.startswith("rtsp://"):
            rest = url[len("rtsp://"):]
            # 如果已经有 user@，不重复注入
            if "@" not in rest.split("/", 1)[0]:
                return f"rtsp://{user}:{pw}@{rest}"
            return url
        # 非 rtsp://：原样返回（理论上不会发生；ONVIF GetStreamUri 默认 rtsp）
        return url

    @staticmethod
    def _redact_url(url: str) -> str:
        """日志用：把 rtsp://user:pass@ 中的密码段涂掉。"""
        try:
            if "://" in url and "@" in url:
                scheme = url.split("://", 1)[0]
                rest = url.split("://", 1)[1]
                if "@" in rest:
                    creds, hostpart = rest.split("@", 1)
                    if ":" in creds:
                        user = creds.split(":", 1)[0]
                        return f"{scheme}://{user}:***@{hostpart}"
            return url
        except Exception:
            return "<redacted>"
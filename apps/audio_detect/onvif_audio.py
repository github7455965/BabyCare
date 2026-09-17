"""ONVIF 音频源探测（Phase 1，spec §3.1）。

职责
----
给一路摄像头找到"带音频"的 RTSP URI（已注入凭据），并把结果分成三种：

| 返回 | 含义 | 对应 AudioRuntimeState.status |
|---|---|---|
| `ok=True` | 找到可用音频 URI | `initializing` → `ready` |
| `no_audio=True` | 确认该摄像头没有音轨 | `no_audio_profile` |
| `error != ""` | 连接/查询/GetStreamUri 失败 | `capture_error` |

为什么由音频 worker 自己探测
---------------------------
worker 启动时、以及每次断流重连时都要重新拿 URI；若依赖 web 进程往 DB 写 URI，
两边状态容易不一致（spec §9.2）。所以音频 venv 里也装了 `onvif-zeep`。

现场实测（P0-1，spec §3.1）
--------------------------
- 两台摄像头（客厅 / 过道）的**每个 profile 都声明了** `AudioEncoderConfiguration`
  → 选 profile 规则可以很简单：优先 `profiles[0]`，它没有音频再往后找；
- `GetAudioSources()` / `GetAudioEncoderConfigurations()` 都返回 `ONVIFError`
  → **设备级能力查询不可用**，只能看 profile 级 + `ffprobe`；
- profile 上报的 `Bitrate`（65536kbps）/ `Channels`（缺失）**不可信**
  （G.711 实际 64kbps），真实格式一律以 `ffprobe` 为准；
- `GetAudioEncoderConfigurationOptions` 只返回 `G711` → 8kHz，有效带宽 4kHz，
  这是**设备硬限制**，改配置解决不了（阈值必须在真实 8kHz 音频上标定）。

不修改 `apps.streaming` 的 `CameraSource` 协议：音频线独立于视频线，只复用 Camera 表。
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class AudioStreamProbe:
    """一次音频 URI 探测的结果。"""

    ok: bool = False
    uri: str = ""                     # 已注入 user:pass 的 RTSP URI
    profile_token: str = ""
    # profile 声明的音频参数（现场实测不可信，仅作诊断）
    audio_encoding: str = ""
    audio_sample_rate: str = ""
    audio_channels: str = ""
    video_encoding: str = ""
    used_ffprobe_fallback: bool = False
    no_audio: bool = False
    error: str = ""
    profiles: list = field(default_factory=list)   # 诊断：每个 profile 的摘要

    def summary(self) -> str:
        if self.error:
            return f"probe error: {self.error}"
        if self.no_audio:
            return "no audio stream"
        tail = " (ffprobe fallback)" if self.used_ffprobe_fallback else ""
        return (
            f"profile={self.profile_token} audio={self.audio_encoding or '?'}"
            f"{tail}"
        )


# ---------------------------------------------------------------------------
# URL 工具
# ---------------------------------------------------------------------------
def inject_credentials(url: str, user: str, password: str) -> str:
    """把 user:pass 拼进 rtsp://user:pass@host/...（特殊字符走 quote）。

    与 `apps/streaming/sources/onvif_source.py::_inject_credentials` 行为一致；
    这里单独实现是为了**不改动视频线代码**（spec §12：不直接修改 CameraSource 协议）。
    """
    u = quote(user or "", safe="")
    p = quote(password or "", safe="")
    if url.startswith("rtsp://"):
        rest = url[len("rtsp://"):]
        # 已经有 user@ 时不重复注入
        if "@" not in rest.split("/", 1)[0]:
            return f"rtsp://{u}:{p}@{rest}"
    return url


#: 日志脱敏：``//user:pass@`` → ``//user:***@``。
#: 用正则而不是字符串切分——要处理的往往是 **ffmpeg 的原样 stderr**，一行里可能
#: 带多个 URL、URL 后面还跟着别的文字。
_CREDENTIALS_RE = re.compile(r"//([^:/@\s]+):[^@\s]+@")


def redact_text(text: str) -> str:
    """把文本中所有 URL 的密码替换成 ``***``。

    ffmpeg 连接失败时会把**完整 RTSP URL（含密码）**打在自己的 stderr 里，而这些
    文本会被写进日志和 ``AudioRuntimeState.last_error``（控制页可见）——
    必须在写入前脱敏，否则凭据就长期躺在日志文件和数据库里。
    """
    if not text:
        return text
    try:
        return _CREDENTIALS_RE.sub(r"//\1:***@", text)
    except Exception:  # noqa: BLE001
        return "<redacted>"


def redact_url(url: str) -> str:
    """日志用：rtsp://user:***@host/..."""
    return redact_text(url)


# ---------------------------------------------------------------------------
# ffprobe 兜底
# ---------------------------------------------------------------------------
def ffprobe_has_audio(
    uri: str, *, ffprobe_bin: str = "ffprobe", timeout: int = 20,
) -> bool | None:
    """探测 URI 里是否有音轨。

    :return: True=有音频；False=确认无音频；None=探测本身失败（无法下结论）
    """
    cmd = [
        ffprobe_bin, "-rtsp_transport", "tcp", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,sample_rate,channels",
        "-of", "json", "-i", uri,
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        logger.warning("[audio] 找不到 %s（用 BABYCARE_FFPROBE_BIN 指定）", ffprobe_bin)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("[audio] ffprobe 超时 %ss", timeout)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("[audio] ffprobe 执行异常: %s: %s", type(e).__name__, e)
        return None

    if cp.returncode != 0:
        tail = (cp.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        logger.warning("[audio] ffprobe rc=%s: %s", cp.returncode, tail[-1] if tail else "?")
        return None
    try:
        data = json.loads(cp.stdout.decode("utf-8", "replace"))
    except Exception as e:  # noqa: BLE001
        logger.warning("[audio] ffprobe 输出解析失败: %s", e)
        return None

    streams = (data or {}).get("streams") or []
    for s in streams:
        if s.get("codec_type") == "audio":
            logger.info(
                "[audio] ffprobe 发现音轨: %s %sHz %sch",
                s.get("codec_name"), s.get("sample_rate"), s.get("channels"),
            )
            return True
    return False


# ---------------------------------------------------------------------------
# profile 选择
# ---------------------------------------------------------------------------
def _resolution_text(profile) -> str:
    """诊断用：'2304x1296' / '纯音频' / ''。"""
    vec = getattr(profile, "VideoEncoderConfiguration", None)
    if vec is None:
        return "纯音频"
    res = getattr(vec, "Resolution", None)
    if res is None:
        return ""
    return f"{getattr(res, 'Width', '?')}x{getattr(res, 'Height', '?')}"


def _video_pixels(profile) -> int:
    """该 profile 的视频像素数（选择排序键）。越小越省。

    - 无 ``VideoEncoderConfiguration``（纯音频 profile）→ ``-1``：最优先
    - 有配置但读不出分辨率 → ``0``：次优先（比任何具体分辨率省）
    """
    vec = getattr(profile, "VideoEncoderConfiguration", None)
    if vec is None:
        return -1
    res = getattr(vec, "Resolution", None)
    if res is None:
        return 0
    try:
        return int(getattr(res, "Width", 0)) * int(getattr(res, "Height", 0))
    except (TypeError, ValueError):
        return 0


def _pick_audio_profile(profiles):
    """带音频的 profile 里选视频像素最少的；一个都没有 → 回退 ``profiles[0]``。

    为什么不是"第一个带音频的"：现场两台摄像头**每个 profile 都带音频**，旧规则等价于
    永远选 ``profiles[0]``（主码流），音频侧 `-vn` 丢掉视频轨，等于把同一路主码流白拉
    一份（带宽 2048 vs 256kbps、demux 负担差 8 倍）。音轨本身与 profile 无关。

    ``min`` 稳定：像素数并列时取靠前的 profile（与旧行为的顺序偏好一致）。
    """
    with_audio = [
        p for p in profiles
        if getattr(p, "AudioEncoderConfiguration", None) is not None
    ]
    if not with_audio:
        return profiles[0]
    return min(with_audio, key=_video_pixels)


# ---------------------------------------------------------------------------
# 主探测
# ---------------------------------------------------------------------------
def probe_audio_uri(
    host: str,
    port: int,
    username: str,
    *,
    password: str,
    wsdl_dir: str = "",
    ffprobe_bin: str = "ffprobe",
    timeout: int = 20,
) -> AudioStreamProbe:
    """探测一路摄像头的音频 RTSP URI。

    profile 选择规则（spec §3.1；2026-09-15 修订）：
    1. 在所有声明了 `AudioEncoderConfiguration` 的 profile 中，取**视频像素最少**的那个
       —— 音轨数据与 profile 无关（本机设备只有 G711，硬限制），但小码流的网络与
       demux 开销小得多：实测 mainStream 2304×1296/2048kbps → minorStream
       640×480/256kbps，省 87%。没有 `VideoEncoderConfiguration` 的纯音频
       profile 优先级最高（零视频负担）；并列时取靠前者（`min` 稳定）。
    2. 全都没有声明音频 → 用 `profiles[0]` 的 URI 做 `ffprobe` 兜底（有的设备把音频
       塞在同一 profile 里但 ONVIF 没声明）；
    3. ffprobe 也确认没有 → `no_audio=True`（`no_audio_profile`）。

    旧规则（2026-09-12 ~ 09-15）是"优先 `profiles[0]`"，即默认拉了主码流 ——
    音频侧只是 `-vn` 丢掉视频轨，等于把同一路主码流白拉一份（见 spec §3.4 备注）。
    """
    res = AudioStreamProbe()
    if not host:
        res.error = "onvif_host 为空"
        return res
    if not password:
        res.error = "BABYCARE_ONVIF_PASSWORD 未配置（所有 ONVIF 摄像头用同一全局密码）"
        return res

    try:
        from onvif import ONVIFCamera  # 延迟导入：未启用音频线时不必付这个成本
    except ImportError as e:
        res.error = f"缺 onvif-zeep: {e}"
        return res

    kwargs = {}
    if wsdl_dir:
        kwargs["wsdl_dir"] = wsdl_dir

    try:
        cam = ONVIFCamera(host, int(port or 80), username or "", password, **kwargs)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
    except Exception as e:  # noqa: BLE001
        res.error = f"ONVIF 连接/GetProfiles 失败: {type(e).__name__}: {e}"
        return res

    if not profiles:
        res.error = "GetProfiles 返回空"
        return res

    # 记录所有 profile（诊断用）
    for p in profiles:
        aec = getattr(p, "AudioEncoderConfiguration", None)
        vec = getattr(p, "VideoEncoderConfiguration", None)
        res.profiles.append({
            "token": p.token,
            "video": getattr(vec, "Encoding", "") if vec else "",
            "audio": getattr(aec, "Encoding", "") if aec else "",
            "has_audio": aec is not None,
            "resolution": _resolution_text(p),
        })

    # 1) 选 profile：带音频的里面取"视频最省"的（见 docstring 规则 1）
    chosen = _pick_audio_profile(profiles)

    aec = getattr(chosen, "AudioEncoderConfiguration", None)
    vec = getattr(chosen, "VideoEncoderConfiguration", None)
    res.profile_token = chosen.token
    if aec is not None:
        res.audio_encoding = str(getattr(aec, "Encoding", "") or "")
        res.audio_sample_rate = str(getattr(aec, "SampleRate", "") or "")
        res.audio_channels = str(getattr(aec, "Channels", "") or "")
    if vec is not None:
        res.video_encoding = str(getattr(vec, "Encoding", "") or "")

    # 诊断：把全部候选打出来（含分辨率），便于事后核对"为什么选它"
    logger.info(
        "[audio %s:%s] profile 选择 %s；候选=%s",
        host, port, _resolution_text(chosen),
        "; ".join(
            "%s %s%s" % (p["token"], p["resolution"] or "-",
                         " [audio]" if p["has_audio"] else "")
            for p in res.profiles
        ),
    )

    # 2) GetStreamUri
    try:
        req = media.create_type("GetStreamUri")
        req.ProfileToken = chosen.token
        req.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"},
        }
        uri_obj = media.GetStreamUri(req)
        uri = uri_obj.Uri if hasattr(uri_obj, "Uri") else uri_obj.get("Uri")
    except Exception as e:  # noqa: BLE001
        res.error = f"GetStreamUri 失败: {type(e).__name__}: {e}"
        return res

    if not uri:
        res.error = "GetStreamUri 返回空 URI"
        return res

    res.uri = inject_credentials(uri, username or "", password)

    # 3) profile 声明了音频 → 直接可用
    if aec is not None:
        res.ok = True
        return res

    # 4) 没有 profile 声明音频 → ffprobe 兜底
    logger.info(
        "[audio %s:%s] 没有 profile 声明 AudioEncoderConfiguration，"
        "用 ffprobe 探测 profile=%s 是否含音轨",
        host, port, chosen.token,
    )
    has_audio = ffprobe_has_audio(
        res.uri, ffprobe_bin=ffprobe_bin, timeout=timeout,
    )
    if has_audio is None:
        # 探测失败 ≠ 没有音频；按错误处理，下轮重试（不能误标 no_audio_profile）
        res.error = "无 profile 声明音频，且 ffprobe 探测失败（无法确认）"
        return res
    if has_audio:
        res.used_ffprobe_fallback = True
        res.ok = True
        return res

    res.no_audio = True
    return res


def default_ffprobe_bin() -> str:
    return os.environ.get("BABYCARE_FFPROBE_BIN", "ffprobe")

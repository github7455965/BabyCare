"""每路摄像头一个 FFmpeg 子进程，持续产出 16kHz 单声道 PCM（Phase 1，spec §3.2/§3.3）。

关键设计
--------
1. **读取线程与推理解耦**：FFmpeg stdout 管道有缓冲，如果读取端被推理阻塞，
   FFmpeg 会写不进去 → 反压 → 卡死整条采集。所以读取线程只做"搬进环形缓冲"，
   不碰模型、不碰 DB。
2. **这里不写 DB**：读线程是热路径。`AudioCapture` 只维护内存状态，
   由 `AudioCaptureManager` 的心跳线程定期快照落库（spec §9.3）。
3. **coverage_ok_from_ts**：任一"断过"的事件都把它重置；**停机期间还会持续前推**
   （否则掉线时段会被误判成"覆盖完整"，spec §3.3）。
4. **进程树回收**：Windows 下必须 `taskkill /F /T /PID`（spec §3.2）。

状态机（spec §10）
-----------------
    initializing ──首包──> ready ──断流/退出/卡死──> capture_error ──退避──> initializing
    探测无音频 ──> no_audio_profile ──长退避──> 重试
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from django.conf import settings

from .models import AudioRuntimeState
from .onvif_audio import probe_audio_uri, redact_text, redact_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 环形缓冲
# ---------------------------------------------------------------------------
class PcmRingBuffer:
    """定长字节环形缓冲（存放 s16le 原始字节）。

    只用字节长度做容量，不关心采样率——调用方按
    `sample_rate * 2 * seconds` 算容量。
    """

    def __init__(self, capacity_bytes: int):
        self._cap = max(int(capacity_bytes), 0)
        self._buf = bytearray()
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._buf.extend(data)
            overflow = len(self._buf) - self._cap
            if overflow > 0:
                del self._buf[:overflow]

    def snapshot(self) -> bytes:
        """取当前缓冲内容（最旧的在前）。Phase 3 拼事件录音时用。"""
        with self._lock:
            return bytes(self._buf)

    def clear(self) -> None:
        """清空缓冲。

        **断流重连时必须调用**：ring 只按容量滑窗，断流 10s 后重连，里面还
        混着断流前的旧 PCM——推理窗"最近 2s"会拿旧音频去喂模型，事件 pre_roll
        也会复制错时序的上下文（code review 发现的真实 bug）。
        """
        with self._lock:
            self._buf.clear()

    def tail(self, nbytes: int) -> bytes:
        """取最近 ``nbytes`` 字节（推理窗用：只要窗口那一段）。"""
        if nbytes <= 0:
            return b""
        with self._lock:
            return bytes(self._buf[-nbytes:])

    @property
    def capacity(self) -> int:
        return self._cap

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ---------------------------------------------------------------------------
# 子进程回收
# ---------------------------------------------------------------------------
def terminate_process_tree(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    """结束 ffmpeg 进程（含子进程）。

    Windows 必须用 `taskkill /F /T`：ffmpeg 是音频 worker 的孙进程，
    只 kill worker 会留下孤儿 ffmpeg 继续占着 RTSP 连接（spec §3.2）。
    """
    if proc is None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("[audio] taskkill pid=%s failed: %s", proc.pid, e)
    else:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                    proc.wait(timeout=timeout)
                except Exception:  # noqa: BLE001
                    pass

    try:
        proc.wait(timeout=timeout)
    except Exception:  # noqa: BLE001
        pass

    for pipe in (proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class CaptureConfig:
    """采集参数（全部来自 settings，允许测试覆写）。"""

    sample_rate: int = 16000
    pre_roll_sec: int = 5
    read_chunk_ms: int = 200
    start_timeout_sec: int = 15
    stall_sec: float = 5.0
    reconnect_min_sec: float = 2.0
    reconnect_max_sec: float = 30.0
    no_audio_retry_sec: float = 60.0
    onvif_timeout_sec: int = 20
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    @classmethod
    def from_settings(cls) -> "CaptureConfig":
        s = settings
        return cls(
            sample_rate=int(s.BABYCARE_AUDIO_SAMPLE_RATE),
            pre_roll_sec=int(s.BABYCARE_AUDIO_PRE_ROLL_SEC),
            read_chunk_ms=int(s.BABYCARE_AUDIO_READ_CHUNK_MS),
            start_timeout_sec=int(s.BABYCARE_AUDIO_START_TIMEOUT_SEC),
            stall_sec=float(s.BABYCARE_AUDIO_STALL_SEC),
            reconnect_min_sec=float(s.BABYCARE_AUDIO_RECONNECT_MIN_SEC),
            reconnect_max_sec=float(s.BABYCARE_AUDIO_RECONNECT_MAX_SEC),
            no_audio_retry_sec=float(s.BABYCARE_AUDIO_NO_AUDIO_RETRY_SEC),
            onvif_timeout_sec=int(s.BABYCARE_AUDIO_ONVIF_TIMEOUT_SEC),
            ffmpeg_bin=str(s.BABYCARE_FFMPEG_BIN),
            ffprobe_bin=str(s.BABYCARE_FFPROBE_BIN),
        )


@dataclass
class CaptureSnapshot:
    """内存状态快照（心跳线程据此落库）。"""

    status: str
    last_packet_at: datetime | None
    coverage_ok_from_ts: int
    bytes_total: int
    bytes_per_sec: float
    profile_token: str
    audio_encoding: str
    error: str

    def as_log(self) -> str:
        rate = f"{self.bytes_per_sec / 1024:.1f}KB/s"
        return (
            f"status={self.status} rate={rate} total={self.bytes_total / 1024:.0f}KB "
            f"cov_from={self.coverage_ok_from_ts} "
            f"last_packet={self.last_packet_at} err={self.error or '-'}"
        )


# ---------------------------------------------------------------------------
# 单路采集
# ---------------------------------------------------------------------------
class AudioCapture:
    """一路摄像头的 FFmpeg 采集 + 断流重连（spec §3.2）。

    只维护内存状态；`snapshot()` 给心跳线程取用。
    """

    def __init__(self, camera, config: CaptureConfig | None = None):
        self.camera_id: int = camera.id
        self.camera_name: str = camera.name
        self._host: str = camera.onvif_host or ""
        self._port: int = int(camera.onvif_port or 80)
        self._username: str = camera.onvif_username or ""
        self._password: str = settings.ONVIF_PASSWORD or ""
        self._wsdl_dir: str = settings.ONVIF_WSDL_DIR or ""
        self._cfg: CaptureConfig = config or CaptureConfig.from_settings()

        self._ring = PcmRingBuffer(
            self._cfg.sample_rate * 2 * self._cfg.pre_roll_sec,
        )
        # ---- taps（事件录音缓冲，Phase 3）----
        # 读线程每收一段 PCM 就同步 append 到所有 tap；tap 是纯内存字节缓冲，
        # 不碰模型/DB，不违背"读线程只搬数据"的约定（spec §3.2）。
        self._taps: dict[str, "PcmRingBuffer"] = {}
        self._tap_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # ---- 内存状态（_lock 保护）----
        self._status = AudioRuntimeState.STATUS_DISABLED
        self._last_packet_at: datetime | None = None
        self._last_packet_mono: float = 0.0
        self._coverage_ok_from_ts: int = 0
        self._bytes_total: int = 0
        self._bytes_win_start: float = 0.0
        self._bytes_win_base: int = 0
        self._bytes_per_sec: float = 0.0
        self._error: str = ""
        self._profile_token: str = ""
        self._audio_encoding: str = ""

        self._proc: subprocess.Popen | None = None
        self._stderr_lines: deque[str] = deque(maxlen=5)
        self._supervisor: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        with self._lock:
            self._status = AudioRuntimeState.STATUS_INITIALIZING
            self._coverage_ok_from_ts = int(time.time())
        t = threading.Thread(
            target=self._supervise, name=f"audio-cap-{self.camera_id}", daemon=True,
        )
        self._supervisor = t
        t.start()
        logger.info("[audio cam#%s %s] capture thread started", self.camera_id, self.camera_name)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            terminate_process_tree(proc, timeout=3.0)
        t = self._supervisor
        if t is not None:
            t.join(timeout=timeout)
        self._supervisor = None
        logger.info("[audio cam#%s] capture stopped", self.camera_id)

    # ------------------------------------------------------------------
    # 状态查询（给心跳线程 / 后续 Phase）
    # ------------------------------------------------------------------
    def snapshot(self) -> CaptureSnapshot:
        with self._lock:
            return CaptureSnapshot(
                status=self._status,
                last_packet_at=self._last_packet_at,
                coverage_ok_from_ts=self._coverage_ok_from_ts,
                bytes_total=self._bytes_total,
                bytes_per_sec=self._bytes_per_sec,
                profile_token=self._profile_token,
                audio_encoding=self._audio_encoding,
                error=self._error,
            )

    def is_covered_for(self, window_start_ts: int) -> bool:
        """窗口 `[window_start_ts, ...]` 是否被音频采集完整覆盖（spec §3.3）。

        只有**阴性结论**需要它：覆盖不完整时不能断言"没听到"。
        阳性结论不要求全覆盖（"听到了"本身就是硬证据）。
        """
        with self._lock:
            if self._status != AudioRuntimeState.STATUS_READY:
                return False
            last_mono = self._last_packet_mono
            cov = self._coverage_ok_from_ts
        # 兜底：进程活着但流已经死了（假活）
        if last_mono and (time.monotonic() - last_mono) > self._cfg.stall_sec:
            return False
        return cov <= int(window_start_ts)

    def pre_roll_snapshot(self) -> bytes:
        return self._ring.snapshot()

    @property
    def sample_rate(self) -> int:
        """采集采样率（当前恒为 16kHz，见 `ffmpeg -ar`）。"""
        return int(self._cfg.sample_rate)

    def last_samples(self, seconds: float):
        """最近 ``seconds`` 秒的 PCM，转成 float32 [-1, 1]（推理输入，spec §4.1）。

        窗口比环形缓冲短时（2s < pre_roll 5s）拿到的就是完整窗口；缓冲还没喂满
        则返回不足长度的数组，由调用方按 ``min_audio_ratio`` 决定跳过本窗——
        **不能拿半截窗口当阴性结论**。
        """
        import numpy as np

        need = int(self._cfg.sample_rate * 2 * float(seconds))
        data = self._ring.tail(need)
        if not data:
            return np.zeros(0, dtype="float32")
        return np.frombuffer(data, dtype="<i2").astype("float32") / 32768.0

    # ------------------------------------------------------------------
    # 事件录音 tap（Phase 3，spec §6.2）
    # ------------------------------------------------------------------
    def add_tap(self, key: str, buf: "PcmRingBuffer") -> None:
        """注册一个事件录音缓冲；读线程会同步把 PCM 追加进去。"""
        with self._tap_lock:
            self._taps[key] = buf

    def remove_tap(self, key: str) -> "PcmRingBuffer | None":
        """摘除并返回该缓冲（事件结束后由 assembler 取走落盘）。"""
        with self._tap_lock:
            return self._taps.pop(key, None)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _supervise(self) -> None:
        backoff = self._cfg.reconnect_min_sec
        try:
            while not self._stop.is_set():
                probe = self._probe()
                if self._stop.is_set():
                    break

                if not probe.ok:
                    if probe.no_audio:
                        self._set_error(
                            AudioRuntimeState.STATUS_NO_AUDIO_PROFILE,
                            probe.summary(),
                        )
                        self._sleep(self._cfg.no_audio_retry_sec)
                    else:
                        self._set_error(
                            AudioRuntimeState.STATUS_CAPTURE_ERROR, probe.summary(),
                        )
                        self._sleep(backoff)
                        backoff = min(backoff * 2, self._cfg.reconnect_max_sec)
                    continue

                backoff = self._cfg.reconnect_min_sec
                self._set_status(AudioRuntimeState.STATUS_INITIALIZING)
                logger.info(
                    "[audio cam#%s] opening ffmpeg: %s",
                    self.camera_id, probe.summary(),
                )
                self._run_ffmpeg(probe)
                if self._stop.is_set():
                    break
                self._sleep(backoff)
        except Exception:  # noqa: BLE001
            logger.exception("[audio cam#%s] supervisor crashed", self.camera_id)
        finally:
            self._mark_coverage_now()
            self._set_status(AudioRuntimeState.STATUS_STOPPED)

    def _probe(self):
        return probe_audio_uri(
            self._host, self._port, self._username,
            password=self._password,
            wsdl_dir=self._wsdl_dir,
            ffprobe_bin=self._cfg.ffprobe_bin,
            timeout=self._cfg.onvif_timeout_sec,
        )

    def _run_ffmpeg(self, probe) -> None:
        """起 ffmpeg → 等首包 → 监督到断流/退出。阻塞直到该段流结束。"""
        cmd = [
            self._cfg.ffmpeg_bin, "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", probe.uri,
            "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", str(self._cfg.sample_rate),
            "-f", "s16le", "-",
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0,
                creationflags=creationflags,
            )
        except FileNotFoundError:
            self._set_error(
                AudioRuntimeState.STATUS_CAPTURE_ERROR,
                f"找不到 ffmpeg（{self._cfg.ffmpeg_bin}）；用 BABYCARE_FFMPEG_BIN 指定绝对路径",
            )
            return
        except Exception as e:  # noqa: BLE001
            self._set_error(
                AudioRuntimeState.STATUS_CAPTURE_ERROR,
                f"ffmpeg 启动失败: {type(e).__name__}: {e}",
            )
            return

        self._proc = proc
        self._stderr_lines.clear()
        # 新一段流开始：清掉上一段流的残留 PCM（否则断流前的旧音频会混进
        # 推理窗和事件 pre_roll；窗口音频不足时按 starved 跳过更安全）
        self._ring.clear()
        with self._lock:
            self._profile_token = probe.profile_token
            self._audio_encoding = probe.audio_encoding
            self._last_packet_mono = 0.0
            self._bytes_win_start = 0.0
            self._bytes_win_base = self._bytes_total
            self._bytes_per_sec = 0.0

        reader = threading.Thread(
            target=self._read_stdout, args=(proc,),
            name=f"audio-read-{self.camera_id}", daemon=True,
        )
        err_reader = threading.Thread(
            target=self._read_stderr, args=(proc,),
            name=f"audio-err-{self.camera_id}", daemon=True,
        )
        reader.start()
        err_reader.start()

        # --- 等首包：initializing → ready；超时/退出 → capture_error ---
        t0 = time.monotonic()
        ready = False
        while not self._stop.is_set():
            with self._lock:
                got_packet = self._last_packet_mono > 0.0
            if got_packet:
                ready = True
                break
            if proc.poll() is not None:
                break
            if (time.monotonic() - t0) > self._cfg.start_timeout_sec:
                break
            self._touch_coverage_while_down()
            time.sleep(0.1)

        if ready:
            self._mark_ready()
            logger.info(
                "[audio cam#%s] ready: profile=%s audio=%s (%s)",
                self.camera_id, probe.profile_token,
                probe.audio_encoding or "?", redact_url(probe.uri),
            )

        # --- 监督循环 ---
        while not self._stop.is_set():
            if proc.poll() is not None:
                tail = self._stderr_tail()
                msg = f"ffmpeg 退出 rc={proc.returncode}"
                if tail:
                    msg += f"：{tail}"
                self._set_error(AudioRuntimeState.STATUS_CAPTURE_ERROR, msg)
                break
            with self._lock:
                last_mono = self._last_packet_mono
            if last_mono and (time.monotonic() - last_mono) > self._cfg.stall_sec:
                self._set_error(
                    AudioRuntimeState.STATUS_CAPTURE_ERROR,
                    f"超过 {self._cfg.stall_sec:.0f}s 没有收到音频（疑似断流）",
                )
                break
            if not ready:
                # 首包一直没来（超时/提前退出），错误在上面已写
                if proc.poll() is not None:
                    break
                if (time.monotonic() - t0) > self._cfg.start_timeout_sec:
                    tail = self._stderr_tail()
                    self._set_error(
                        AudioRuntimeState.STATUS_CAPTURE_ERROR,
                        f"{self._cfg.start_timeout_sec}s 内没有收到 PCM"
                        + (f"：{tail}" if tail else ""),
                    )
                    break
            time.sleep(0.25)

        terminate_process_tree(proc, timeout=3.0)
        reader.join(timeout=2.0)
        err_reader.join(timeout=2.0)
        self._proc = None
        # 不管哪种原因断开，覆盖连续性都从这里断掉
        self._mark_coverage_now()
        with self._lock:
            self._last_packet_mono = 0.0

    # ------------------------------------------------------------------
    # 读取线程（热路径：不碰 DB、不碰模型）
    # ------------------------------------------------------------------
    def _read_stdout(self, proc: subprocess.Popen) -> None:
        stdout = proc.stdout
        if stdout is None:
            return
        chunk = max(
            int(self._cfg.sample_rate * 2 * self._cfg.read_chunk_ms / 1000), 1,
        )
        try:
            while not self._stop.is_set():
                data = stdout.read(chunk)
                if not data:
                    break   # EOF：ffmpeg 退出或管道被关
                now_mono = time.monotonic()
                # 先入环形缓冲（有独立锁），再更新标量状态
                self._ring.append(data)
                # 事件录音 tap：活跃事件期间同步追加（纯内存搬运，不碰模型/DB）
                if self._taps:
                    with self._tap_lock:
                        taps = tuple(self._taps.values())
                    for tap in taps:
                        tap.append(data)
                with self._lock:
                    self._bytes_total += len(data)
                    self._last_packet_mono = now_mono
                    self._last_packet_at = datetime.now()
                    if self._bytes_win_start == 0.0:
                        self._bytes_win_start = now_mono
                        self._bytes_win_base = self._bytes_total
                    elif now_mono - self._bytes_win_start >= 1.0:
                        dt = now_mono - self._bytes_win_start
                        self._bytes_per_sec = (self._bytes_total - self._bytes_win_base) / dt
                        self._bytes_win_start = now_mono
                        self._bytes_win_base = self._bytes_total
        except Exception as e:  # noqa: BLE001
            logger.debug("[audio cam#%s] stdout read ended: %s", self.camera_id, e)
        finally:
            try:
                stdout.close()
            except Exception:  # noqa: BLE001
                pass

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        stderr = proc.stderr
        if stderr is None:
            return
        try:
            for raw in iter(stderr.readline, b""):
                if self._stop.is_set():
                    break
                line = (raw or b"").decode("utf-8", "replace").strip()
                if not line:
                    continue
                # 脱敏后再存/再打：ffmpeg 的报错里带完整 RTSP URL（含密码），
                # 这段文本还会经 _stderr_tail → last_error 进 DB 和错误日志。
                line = redact_text(line)
                self._stderr_lines.append(line)
                logger.debug("[audio cam#%s] ffmpeg: %s", self.camera_id, line)
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                stderr.close()
            except Exception:  # noqa: BLE001
                pass

    def _stderr_tail(self) -> str:
        # 双保险：即便某条路径绕过了上面的脱敏，这里也再过滤一次
        return redact_text(" | ".join(self._stderr_lines))

    # ------------------------------------------------------------------
    # 状态工具
    # ------------------------------------------------------------------
    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            if error is not None:
                self._error = error

    def _set_error(self, status: str, error: str) -> None:
        with self._lock:
            self._status = status
            self._error = error or ""
        if status != AudioRuntimeState.STATUS_READY:
            self._touch_coverage_while_down()
        logger.warning(
            "[audio cam#%s %s] %s", self.camera_id, self.camera_name, error or status,
        )

    def _mark_ready(self) -> None:
        with self._lock:
            self._status = AudioRuntimeState.STATUS_READY
            self._error = ""
            self._coverage_ok_from_ts = int(time.time())

    def _mark_coverage_now(self) -> None:
        """连续性断点：从此刻重新开始算覆盖。"""
        with self._lock:
            self._coverage_ok_from_ts = int(time.time())

    def _touch_coverage_while_down(self) -> None:
        """非 ready 期间持续前推 coverage_ok_from_ts。

        不加这个，掉线时段的窗口会因为 `cov <= window_start` 被判成"覆盖完整"，
        阴性结论（"没有哭声"）就会变得不可信却看不出来（spec §3.3）。
        """
        with self._lock:
            if self._status == AudioRuntimeState.STATUS_READY:
                return
            self._coverage_ok_from_ts = int(time.time())

    def _sleep(self, seconds: float) -> None:
        """可被 stop() 打断的退避等待；期间保持覆盖状态前推。"""
        deadline = time.monotonic() + max(float(seconds), 0.0)
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= deadline:
                return
            self._touch_coverage_while_down()
            time.sleep(min(0.5, deadline - now))

r"""
LlamaManager（Step 4）：llama-server 子进程管理。

职责
----
- 启动 llama-server 子进程（subprocess.Popen），复用你脚本里的启动参数
- health 探测（轮询 /health 端点）
- 优雅停服（SIGTERM 5s → SIGKILL）
- 按需 unload（调 /unload 端点；不支持则 stop_server 兜底）
- parallel 模式下的 48h 重启（防止 llama.cpp 长时间运行显存膨胀）

启动参数
--------
全部由下面这些 settings 拼出（见 `_build_cmd`），与 model/run_vlm_server.bat 是同一套
（改一处要同步另一处）：
    llama-server.exe ^
      -m MODELS\qwen35-4b-Q4_K_M.gguf ^
      --mmproj MODELS\mmproj-Qwen3.5-4B-F16.gguf ^
      --ctx-size 8192 ^
      -ngl 99 ^
      -ctk q8_0 -ctv q8_0 ^
      --cache-ram 0 ^
      -np 1 ^
      --host 127.0.0.1 ^
      --port 8082

⚠ `--cache-ram 0` 与 `-np 1` 不是可省的调优项：
    --cache-ram 0  关掉只增不减的 prompt cache。漏配时实测宿主内存 WS 9.4GB /
                   PM 12.5GB（权重才 ~2.2GB），内存只增不减。
    -np 1          4 slot × ctx_size 的 KV cache 会白占大量显存；Runner 是单线程
                   顺序处理，1 slot 足够。
⚠ 默认 `BABYCARE_LLAMA_HOST=127.0.0.1`（只给本机）。**不要**随手改成 0.0.0.0 ——
  那等于把无鉴权的推理端点交给整个局域网（llama-server 默认 CORS 全开且没有 key，
  而 LlamaClient 目前不支持 api_key）。只有分体部署（推理机在另一台机器）才需要它。

配置（settings 读 .env）
------------------------
- BABYCARE_LLAMA_SERVER_BIN     llama-server 可执行路径（默认 llama-server.exe，走 PATH）
- BABYCARE_LLAMA_MODEL_DIR      模型所在目录（默认 D:\\AI\\LLAMA）
- BABYCARE_LLAMA_MODEL_NAME     gguf 文件名（默认 qwen35-4b-Q4_K_M.gguf）
- BABYCARE_LLAMA_MMPROJ_NAME    mmproj 文件名（默认 mmproj-Qwen3.5-4B-F16.gguf）
- BABYCARE_LLAMA_CTX_SIZE       ctx-size（默认 8192）
- BABYCARE_LLAMA_NGL            ngl（默认 99）
- BABYCARE_LLAMA_CTK            K cache q8_0（默认 q8_0）
- BABYCARE_LLAMA_CTV            V cache q8_0（默认 q8_0）
- BABYCARE_LLAMA_HOST           --host（默认 127.0.0.1；分体部署才改 0.0.0.0）
- BABYCARE_LLAMA_PORT           --port（默认 8082）
- BABYCARE_LLAMA_PID_FILE       PID 文件路径（默认 data/llama_server.pid）
- BABYCARE_LLAMA_MAX_UPTIME_HOURS parallel 模式最长常驻小时（默认 48）
- BABYCARE_LLAMA_START_TIMEOUT_SEC health 超时秒（默认 60）
- BABYCARE_LLAMA_START_RETRIES   启动失败重试次数（默认 1）

线程模型
--------
- Popen + 守护子进程；父进程退出 → 子进程自动回收（Windows 下 subprocess 行为）
- 不起专门 watchdog 线程；每次 ensure_running 时检查 uptime
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class LlamaStartError(Exception):
    """llama-server 启动失败。"""


class LlamaTimeoutError(LlamaStartError):
    """llama-server 在 timeout_sec 内未通过 /health 探测。"""


class LlamaExternalError(LlamaStartError):
    """外部模式下调了"只有本机托管才允许"的操作（启停 / unload / 重启）。

    单独一个类型是为了让调用方（和单测）能明确区分"启不动"与"压根不该启"。
    """


#: 外部模式下 `/health` 结果的 TTL 缓存（秒）。
#: is_running() 在热路径上（每个 VLM 窗 / 每条回放任务都会问），
#: 不能每次都发一次 HTTP；2s 足够新鲜，又把请求量压到可忽略。
_EXTERNAL_HEALTH_TTL_SEC = 2.0
#: 外部模式探测 `/health` 的单次超时（秒）。必须短 —— 它是"探活"，不是业务请求。
_EXTERNAL_HEALTH_TIMEOUT_SEC = 3.0


class LlamaManager:
    """全局单例：管理 llama-server 子进程的生命周期。"""

    _instance: Optional["LlamaManager"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        # 子进程引用
        self._proc: Optional[subprocess.Popen] = None
        self._started_at: float = 0.0  # time.time()；uptime 检查用
        # 保护 _proc / _started_at 的写
        self._lock = threading.Lock()
        # 序列化整个启动流程（ensure_running 入口；与 _lock 解耦——_lock 是细粒度读写锁，
        # 长时间持 _lock 会卡 is_running() / uptime_hours()；_start_lock 是粗粒度 mutex）
        self._start_lock = threading.Lock()
        # 常驻模式状态（apps.ready() 调 start_auto_restart_scheduler 启动后台线程）
        self._user_forced_off: bool = False
        self._auto_restart_enabled: bool = False
        self._auto_restart_hours: float = 24.0
        self._auto_restart_thread: Optional[threading.Thread] = None
        self._auto_restart_stop: threading.Event = threading.Event()
        # 外部模式（分体部署）的远端 health TTL 缓存
        self._ext_lock = threading.Lock()
        self._ext_healthy: bool = False
        self._ext_checked_at: float = 0.0

    @classmethod
    def instance(cls) -> "LlamaManager":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------
    def _config(self) -> dict:
        from django.conf import settings

        def _env_or_setting(env_key: str, setting_attr: str, default: Any) -> Any:
            v = os.environ.get(env_key)
            if v is not None and v != "":
                return v
            return getattr(settings, setting_attr, default)

        # int 字段单独处理
        def _env_int(env_key: str, setting_attr: str, default: int) -> int:
            raw = os.environ.get(env_key)
            if raw is not None and raw != "":
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    pass
            return int(getattr(settings, setting_attr, default))

        model_dir = _env_or_setting("BABYCARE_LLAMA_MODEL_DIR", "BABYCARE_LLAMA_MODEL_DIR", r"D:\AI\LLAMA")
        pid_file = _env_or_setting("BABYCARE_LLAMA_PID_FILE", "BABYCARE_LLAMA_PID_FILE", "")

        if not pid_file:
            # 缺省放到 data/llama_server.pid
            try:
                from django.conf import settings as _s
                data_dir = Path(getattr(_s, "BASE_DIR", Path.cwd())) / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                pid_file = str(data_dir / "llama_server.pid")
            except Exception:
                pid_file = str(Path.cwd() / "data" / "llama_server.pid")

        return {
            "bin": _env_or_setting("BABYCARE_LLAMA_SERVER_BIN", "BABYCARE_LLAMA_SERVER_BIN", "llama-server.exe"),
            "model_dir": model_dir,
            "model_name": _env_or_setting("BABYCARE_LLAMA_MODEL_NAME", "BABYCARE_LLAMA_MODEL_NAME",
                                          "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
            "mmproj_name": _env_or_setting("BABYCARE_LLAMA_MMPROJ_NAME", "BABYCARE_LLAMA_MMPROJ_NAME",
                                           "mmproj-Qwen3VL-4B-Instruct-F16.gguf"),
            "ctx_size": _env_int("BABYCARE_LLAMA_CTX_SIZE", "BABYCARE_LLAMA_CTX_SIZE", 4096),
            "ngl": _env_int("BABYCARE_LLAMA_NGL", "BABYCARE_LLAMA_NGL", 99),
            "ctk": _env_or_setting("BABYCARE_LLAMA_CTK", "BABYCARE_LLAMA_CTK", "q8_0"),
            "ctv": _env_or_setting("BABYCARE_LLAMA_CTV", "BABYCARE_LLAMA_CTV", "q8_0"),
            "host": _env_or_setting("BABYCARE_LLAMA_HOST", "BABYCARE_LLAMA_HOST", "0.0.0.0"),
            "port": _env_int("BABYCARE_LLAMA_PORT", "BABYCARE_LLAMA_PORT", 8082),
            "pid_file": pid_file,
            "max_uptime_hours": _env_int("BABYCARE_LLAMA_MAX_UPTIME_HOURS", "BABYCARE_LLAMA_MAX_UPTIME_HOURS", 48),
            "start_timeout_sec": _env_int("BABYCARE_LLAMA_START_TIMEOUT_SEC", "BABYCARE_LLAMA_START_TIMEOUT_SEC", 60),
            "start_retries": _env_int("BABYCARE_LLAMA_START_RETRIES", "BABYCARE_LLAMA_START_RETRIES", 1),
        }

    def _build_cmd(self, cfg: dict) -> list[str]:
        """构造命令行参数列表。"""
        model_path = Path(cfg["model_dir"]) / cfg["model_name"]
        mmproj_path = Path(cfg["model_dir"]) / cfg["mmproj_name"]
        return [
            cfg["bin"],
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "--ctx-size", str(cfg["ctx_size"]),
            "-ngl", str(cfg["ngl"]),
            "-ctk", cfg["ctk"],
            "-ctv", cfg["ctv"],
            "--host", cfg["host"],
            "--port", str(cfg["port"]),
            "-np", "1",  # n_slots=1：4 slot × ctx_size KV cache 占大量 VRAM/RAM；Runner 是单线程顺序处理，1 slot 足够
            # prompt cache 默认开启 + cache-idle-slots 默认 enabled → 新请求来时把当前 KV
            # 自动存到 prompt cache，下次匹配前缀时复用。Runner 每次都是新对话，cache 永远
            # 用不上但持续累积，导致内存只增不减。彻底关掉：
            "--cache-ram", "0",  # 0 = disable prompt cache
            "-rea", "off",  # 关 Qwen3 thinking 模式（否则会消耗上千 token 推理，单请求 3-4s）
        ]

    # ------------------------------------------------------------------
    # 外部模式（分体部署）
    # ------------------------------------------------------------------
    def is_external(self) -> bool:
        """llama-server 是否由**别的机器**托管（`BABYCARE_LLAMA_EXTERNAL`）。

        为 True 时本类**只探活、绝不启停**：不 spawn、不 kill、不 unload、
        不清理 pidfile、不自动重启。原因是分体后服务机既**管不到**对面，
        也**不该**管 —— 对面是共享资源，`release_vlm_if_idle()` 一发
        `/unload` 就会把别人正在用的服务关掉。

        每次调用都重读 env/settings（与 `_config()` 同款约定），
        这样单测可以靠 patch 环境变量切换，不需要清单例。
        """
        raw = os.environ.get("BABYCARE_LLAMA_EXTERNAL")
        if raw is not None and raw != "":
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
        from django.conf import settings

        return bool(getattr(settings, "BABYCARE_LLAMA_EXTERNAL", False))

    def _server_url(self) -> str:
        """远端 llama-server 根地址（外部模式的唯一地址来源）。

        **不能**复用 `_config()` 里的 host/port：那两个是"本机启动参数"，
        分体时它们描述的是推理机的监听设置，而真正要连的是
        `BABYCARE_LLAMA_SERVER_URL`（服务机侧配置）。
        """
        raw = os.environ.get("BABYCARE_LLAMA_SERVER_URL")
        if raw is None or raw == "":
            from django.conf import settings

            raw = getattr(settings, "BABYCARE_LLAMA_SERVER_URL", "")
        return str(raw or "http://127.0.0.1:8082").rstrip("/")

    def _health_url(self, cfg: dict) -> str:
        # 外部模式：探的是**服务机侧配的那个地址**（分体部署的关键，见 §2 #2）
        if self.is_external():
            return f"{self._server_url()}/health"
        # 本机托管：/health 端口 = 启动端口；host 配 0.0.0.0 时探测用 127.0.0.1
        host = "127.0.0.1" if cfg["host"] in ("0.0.0.0", "::") else cfg["host"]
        return f"http://{host}:{cfg['port']}/health"

    def _external_health(self, force: bool = False) -> bool:
        """远端 `/health` 是否 200（带 TTL 缓存；任何异常一律算不健康）。"""
        now = time.monotonic()
        if not force:
            with self._ext_lock:
                if now - self._ext_checked_at < _EXTERNAL_HEALTH_TTL_SEC:
                    return self._ext_healthy
        ok = False
        try:
            import requests  # 延迟 import（与本模块其它地方一致）

            r = requests.get(
                self._health_url(self._config()),
                timeout=_EXTERNAL_HEALTH_TIMEOUT_SEC,
            )
            ok = r.status_code == 200
        except Exception as e:  # noqa: BLE001
            logger.debug("[llama] external health failed: %s", e)
            ok = False
        with self._ext_lock:
            self._ext_healthy = ok
            self._ext_checked_at = time.monotonic()
        return ok

    def remote_status(self) -> dict:
        """外部模式下给控制页看的远端信息（含真实 URL，便于确认"发去哪台机器了"）。"""
        return {
            "external": True,
            "url": self._server_url(),
            "health_url": f"{self._server_url()}/health",
            "healthy": self._external_health(force=True),
        }

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        """**服务当前可用吗** —— 语义按模式切换（这是分体改造的核心之一）。

        - 本机托管：子进程活着（`self._proc.poll() is None`）
        - 外部模式：远端 `/health` 返回 200（带 TTL 缓存）

        为什么外部模式不能仍看 `self._proc.poll()`：分体后本进程永远没有子进程，
        它会**恒为 False** —— 后果是 Drainer 的回放门槛永不通过（回放永远不触发）
        且控制页永远显示"未运行"（checklist §2 #1）。
        """
        if self.is_external():
            return self._external_health()
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def uptime_hours(self) -> float:
        """距上次启动的小时数；未启动返回 0。"""
        with self._lock:
            if self._started_at <= 0:
                return 0.0
            return (time.time() - self._started_at) / 3600.0

    def needs_restart(self) -> bool:
        """判断是否需要重启（uptime 超过 `BABYCARE_LLAMA_MAX_UPTIME_HOURS`）。

        外部模式**恒 False**：远端服务不归我们管，重启它是别人的事
        （而且 `ensure_running()` 在外部模式也不会 spawn，判 True 只会造成
        "说要重启但什么也没做"的假象）。
        """
        if self.is_external():
            return False
        if not self.is_running():
            return False
        cfg = self._config()
        return self.uptime_hours() >= cfg["max_uptime_hours"]

    # ------------------------------------------------------------------
    # 启停
    # ------------------------------------------------------------------
    def _ensure_external(self) -> None:
        """外部模式下的 "ensure"：**只探活，绝不 spawn**。

        健康 → 静默返回；不健康 → raise :class:`LlamaExternalError`，
        让调用方（Runner / Drainer）据此走"入队等回放"，而不是白等一个超时。
        """
        if self._external_health(force=True):
            return
        raise LlamaExternalError(
            f"外部模式：远端 llama-server 不可用（{self._server_url()}）；"
            "本进程不会尝试启动它"
        )

    def ensure_running(self, timeout_sec: Optional[int] = None, max_retries: Optional[int] = None) -> None:
        """保证 llama-server 跑着：

        1. 检查是否在跑（in_running=True 直接返回）
        2. 检查是否超过 max_uptime_hours；到时 → stop + 重启
        3. 否则启动 + 等 health
        失败重试 max_retries 次（默认 1）。

        入口持 _start_lock：序列化整个启动流程，防止并发 ensure_running 撞 pidfile
        误杀对方（_cleanup_stale_pidfile 会杀活 PID）+ 双 spawn 抢端口。

        常驻模式短路：若 _user_forced_off=True（手动关闭 / 自动重启窗口期）→ raise，
        防止 Runner / Drainer 把被用户关掉的 llama 自动拉起。

        **外部模式**：整段跳过，改走 :meth:`_ensure_external`（只探活）。
        """
        if self.is_external():
            self._ensure_external()
            return
        with self._start_lock:
            if self._user_forced_off:
                raise LlamaStartError("llama-server is user-forced-off")
            cfg = self._config()
            timeout = timeout_sec if timeout_sec is not None else cfg["start_timeout_sec"]
            retries = max_retries if max_retries is not None else cfg["start_retries"]

            # 已跑但到时 → 重启
            if self.is_running() and self.needs_restart():
                logger.info("[llama] uptime=%.1fh >= %dh, restarting", self.uptime_hours(), cfg["max_uptime_hours"])
                self._stop_blocking()
                # 走 _start_one

            # 已跑且不用重启 → 直接返回
            if self.is_running():
                return

            # 启动（含重试）
            last_err: Optional[Exception] = None
            for attempt in range(retries + 1):
                try:
                    self._start_one(cfg, timeout)
                    return
                except Exception as e:
                    last_err = e
                    logger.warning("[llama] start attempt %d/%d failed: %s",
                                   attempt + 1, retries + 1, e)
                    # 清掉残留子进程
                    self._cleanup_dead()
                    if attempt < retries:
                        time.sleep(1.0)

            # 全部失败
            raise LlamaStartError(f"llama-server 启动失败（重试 {retries} 次）：{last_err}")

    def _start_one(self, cfg: dict, timeout_sec: int) -> None:
        """一次启动尝试 + health 探测。

        外部模式在此**再挡一次**：正常路径上 `ensure_running()` 已经提前返回，
        这里是防御 —— 挡住将来有人直接调 `_start_one` 造成的"误 spawn + 误杀
        远端同名进程"（`_cleanup_stale_pidfile` 会按进程名杀 llama-server，
        见 checklist §2 #6）。
        """
        if self.is_external():
            raise LlamaExternalError("外部模式：拒绝 spawn llama-server")
        cmd = self._build_cmd(cfg)
        logger.info("[llama] spawning: %s", " ".join(cmd))

        # 先清理上一轮残留：PID 文件 + 还活着的同 PID 进程（防止孤儿占显存 / 占端口）
        # own_pid 守卫：避免误杀本实例 self._proc 还登记的活跃子进程
        own_pid = self._proc.pid if self._proc else None
        self._cleanup_stale_pidfile(cfg["pid_file"], own_pid=own_pid)

        # stderr 重定向到日志文件（失败时能查到原因）
        log_path = Path(cfg["pid_file"]).with_suffix(".log")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_fp = open(log_path, "ab", buffering=0)
        except Exception as e:
            logger.warning("[llama] open stderr log failed: %s; fallback DEVNULL", e)
            stderr_fp = subprocess.DEVNULL

        # spawn
        try:
            # Windows: CREATE_NEW_PROCESS_GROUP 让 SIGTERM 走 terminate() 路径
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=stderr_fp,
                creationflags=creationflags,
            )
        except FileNotFoundError as e:
            raise LlamaStartError(f"找不到可执行文件: {cfg['bin']}（请设 BABYCARE_LLAMA_SERVER_BIN）") from e
        except Exception as e:
            raise LlamaStartError(f"Popen 失败: {e}") from e

        # 写 PID
        try:
            Path(cfg["pid_file"]).write_text(str(proc.pid), encoding="utf-8")
        except Exception as e:
            logger.warning("[llama] write pid file failed: %s", e)

        # health 探测
        url = self._health_url(cfg)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if proc.poll() is not None:
                # 子进程提前退出 → tail 一下 stderr 日志
                err_tail = self._tail_log(log_path, lines=20)
                raise LlamaStartError(
                    f"llama-server 进程已退出，rc={proc.returncode}；stderr (tail): {err_tail}"
                )
            try:
                import requests  # 延迟 import
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    with self._lock:
                        self._proc = proc
                        self._started_at = time.time()
                    logger.info("[llama] up after %.1fs pid=%d url=%s; stderr→%s",
                                timeout_sec - (deadline - time.time()), proc.pid, url, log_path)
                    return
            except Exception:
                pass
            time.sleep(1.0)

        # 超时 → 杀进程 + tail 日志
        self._terminate(proc)
        err_tail = self._tail_log(log_path, lines=10)
        raise LlamaTimeoutError(f"llama-server /health 超时 {timeout_sec}s: {url}; stderr (tail): {err_tail}")

    @staticmethod
    def _tail_log(path: Path, lines: int = 20) -> str:
        try:
            if not path.exists():
                return "<no log file>"
            data = path.read_bytes()
            # 取最后 N 行
            all_lines = data.splitlines()
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return b"\n".join(tail).decode("utf-8", errors="replace")
        except Exception as e:
            return f"<read failed: {e}>"

    def stop_server(self) -> None:
        """优雅停止 llama-server（SIGTERM 5s → SIGKILL）。

        外部模式**什么都不做**：远端服务不归本进程管，关它等于搞掉别人的服务。
        """
        if self.is_external():
            logger.warning("[llama] 外部模式：忽略 stop_server（不关远端服务）")
            return
        self._stop_blocking()

    # ------------------------------------------------------------------
    # 常驻模式：手动启停 + 自动重启调度
    # ------------------------------------------------------------------
    def is_forced_off(self) -> bool:
        """手动关闭 / 自动重启窗口期 → True。

        Runner 在调 VLM 前查此值；为 True 时把窗口入 VLMQueuedTask，不调 llama。
        """
        with self._lock:
            return self._user_forced_off

    def force_off(self) -> None:
        """手动关闭 llama-server：设旗标 + 停子进程。

        调用方：dashboard view POST /vlm-control/llama/off/。
        副作用：后续 Runner / ensure_running 看到旗标会走队列；自动重启 scheduler 仍按周期
        触发 force_on（用户可再 force_off 关闭）。

        外部模式：**直接返回，连旗标都不设** —— 设了会让 Runner 永远入队，
        但"谁去把远端开起来"根本不在本进程；可用性一律交给 health + 熔断器判。
        """
        if self.is_external():
            logger.warning("[llama] 外部模式：忽略 force_off（远端服务不受本进程控制）")
            return
        with self._lock:
            self._user_forced_off = True
        try:
            self._stop_blocking()
        except Exception as e:
            logger.warning("[llama] force_off stop_blocking failed: %s", e)
        logger.info("[llama] force_off done")

    def force_on(self) -> None:
        """手动开启 llama-server：清旗标 + ensure_running。

        调用方：dashboard view POST /vlm-control/llama/on/。
        失败 raise（LlamaStartError）；调用方负责捕获并提示用户。

        外部模式：只探活（:meth:`_ensure_external`）—— 不能启，不健康就 raise
        把真实原因报给页面。
        """
        if self.is_external():
            self._ensure_external()
            logger.info("[llama] 外部模式：force_on 退化为探活，远端可用")
            return
        with self._lock:
            self._user_forced_off = False
        self.ensure_running()
        logger.info("[llama] force_on done; pid=%s",
                    self._proc.pid if (self._proc and self._proc.poll() is None) else None)

    def start_auto_restart_scheduler(self, hours: float) -> None:
        """常驻模式后台线程：每 hours 小时触发一次 force_off + force_on。

        apps.ready() 在 BABYCARE_LLM_RESIDENT=true 时调。
        hours <= 0 → 不启线程（仅用户手动启停）。

        外部模式：**不启**（周期 force_off/force_on 会不断尝试动远端服务）。
        """
        if self.is_external():
            logger.info("[llama] 外部模式：不启自动重启调度（远端服务不受本进程控制）")
            return
        if hours <= 0:
            logger.info("[llama] auto_restart_hours<=0; scheduler disabled")
            return
        self._auto_restart_enabled = True
        self._auto_restart_hours = float(hours)
        self._auto_restart_stop.clear()
        with self._lock:
            if self._auto_restart_thread is not None and self._auto_restart_thread.is_alive():
                logger.info("[llama] auto_restart scheduler already running")
                return
        t = threading.Thread(
            target=self._auto_restart_loop,
            name="llama-auto-restart",
            daemon=True,
        )
        with self._lock:
            self._auto_restart_thread = t
        t.start()
        logger.info("[llama] auto_restart scheduler started (every %.1fh)", hours)

    def stop_auto_restart_scheduler(self) -> None:
        """停后台线程；signal handler / atexit 调。"""
        self._auto_restart_stop.set()
        with self._lock:
            t = self._auto_restart_thread
        if t is not None:
            t.join(timeout=2.0)

    def _auto_restart_loop(self) -> None:
        """后台线程主循环：wait(hours*3600) → force_off → sleep 2s → force_on。"""
        while not self._auto_restart_stop.wait(timeout=self._auto_restart_hours * 3600.0):
            try:
                logger.info("[llama] auto_restart cycle begin")
                self.force_off()
                time.sleep(2.0)
                self.force_on()
                logger.info("[llama] auto_restart cycle done")
            except Exception:
                logger.exception("[llama] auto_restart cycle error")

    def restart(self) -> dict:
        """手动重启 llama-server：stop_server() + ensure_running()。

        调用方通常为 /api/vlm/restart/（Step 10）。返回字典供 view 拼 JSON：
        - port: int（.env 配置的端口）
        - pid: int | None（启动后子进程 PID；失败保持 None）

        Raises:
            LlamaStartError / LlamaTimeoutError: ensure_running 失败
            LlamaExternalError: 外部模式（远端服务不归本进程重启）
        """
        if self.is_external():
            raise LlamaExternalError(
                "外部模式：不能重启远端 llama-server（请在推理机上用本地脚本操作）"
            )
        # stop（即便之前没在跑也安全）
        self.stop_server()
        # 重启（处理 ensure_running 内部 max_retries / uptime 判定）
        self.ensure_running()
        cfg = self._config()
        with self._lock:
            pid = self._proc.pid if (self._proc is not None and self._proc.poll() is None) else None
        return {
            "port": cfg["port"],
            "pid": pid,
        }

    def _stop_blocking(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
            self._started_at = 0.0
        if proc is None:
            return
        self._terminate(proc)

    def _terminate(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            logger.info("[llama] terminating pid=%d", proc.pid)
            if os.name == "nt":
                # Windows: 用 CTRL_BREAK_EVENT（CREATE_NEW_PROCESS_GROUP 才能用）或 terminate()
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                except Exception:
                    proc.terminate()
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
                logger.info("[llama] terminated cleanly pid=%d", proc.pid)
                return
            except subprocess.TimeoutExpired:
                pass
            # 兜底：kill
            logger.warning("[llama] kill -9 pid=%d", proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except Exception as e:
            logger.warning("[llama] terminate failed: %s", e)

    def _cleanup_dead(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is not None:
            with self._lock:
                self._proc = None
                self._started_at = 0.0

    @staticmethod
    def _cleanup_stale_pidfile(pid_file: str, own_pid: Optional[int] = None) -> None:
        """启动新子进程前清孤儿：
        - PID 文件存在但指向的进程已死 → 清文件
        - PID 文件指向进程仍活着但不是 llama-server.exe（或本实例启的）→ kill
        - own_pid 不为 None 且与文件 PID 相同 → 跳过（我们自己刚启的活跃进程；不杀）

        解决：之前 runserver Ctrl+C 后 LlamaManager.stop_server() 没人调 → 残留孤儿
        占显存；下次重启 LlamaManager 看到 PID 文件 + is_running()=False（self._proc=None）
        会启新进程，但旧孤儿仍占显存 + 端口。

        own_pid 守卫：异常路径（health probe 中途被打断 → self._proc 已被前一次 _start_one
        写入但实际死掉 + 紧接着的 _start_one 看到的就是自己刚启的活跃进程）下避免自杀。
        """
        p = Path(pid_file)
        if not p.exists():
            return
        try:
            raw = p.read_text(encoding="utf-8").strip()
            old_pid = int(raw)
        except (ValueError, OSError):
            # 文件损毁或非数字 → 清掉
            try:
                p.unlink()
                logger.info("[llama] removed unparseable pid file: %s", pid_file)
            except OSError:
                pass
            return

        if old_pid == os.getpid():
            # 不可能，但防呆
            return

        if own_pid is not None and old_pid == own_pid:
            # 我们自己刚启的活跃进程；不杀，只清文件让新一轮 spawn 写新 pid
            logger.info("[llama] pidfile points to our own live pid=%d; skipping kill", own_pid)
            try:
                p.unlink()
            except OSError:
                pass
            return

        try:
            import psutil  # type: ignore
        except ImportError:
            # 无 psutil 时退化为只查 PID 存在性
            psutil = None

        alive = False
        try:
            proc = psutil.Process(old_pid) if psutil else None
            if proc and proc.is_running():
                # 同名进程（llama-server.exe）才算；避免误杀
                if proc.name().lower().startswith("llama-server"):
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    logger.info("[llama] killed stale llama-server pid=%d", old_pid)
                alive = proc.is_running()
            else:
                alive = False
        except (psutil.NoSuchProcess if psutil else Exception):
            alive = False
        except Exception as e:
            logger.debug("[llama] psutil check pid=%d failed: %s", old_pid, e)
            alive = False

        # 不管活不活都删文件，下次 ensure_running 重新写干净的
        try:
            p.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # unload（设计 §5.3：先调 /unload 端点；不支持则 stop_server）
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """释放 VLM 显存。

        优先调 /unload 端点（如果 llama-server 支持）；失败/不支持 → stop_server。
        exclusive 模式：GpuManager.release_vlm_if_idle 调 → 完全停服释放显存。
        parallel 模式：不会调到这里（parallel 不 release）。

        外部模式**禁止**：它会给远端发 `/unload`（把对面正在用的服务关掉），
        而且 `/unload` 不被支持时下面的 fallback 会走到**本机** `stop_server()`
        —— 两条路都是错的（checklist §2 #4）。
        """
        if self.is_external():
            logger.warning("[llama] 外部模式：忽略 unload（不 unload 远端服务）")
            return
        if not self.is_running():
            return
        url = self._health_url(self._config()).replace("/health", "/unload")
        try:
            import requests
            r = requests.post(url, timeout=5)
            if r.status_code == 200:
                logger.info("[llama] /unload ok pid=%d", self._proc.pid if self._proc else -1)
                return
            logger.warning("[llama] /unload returned %d; falling back to stop_server", r.status_code)
        except Exception as e:
            logger.info("[llama] /unload not supported (%s); stop_server", e)
        self.stop_server()

    # ------------------------------------------------------------------
    # 监控
    # ------------------------------------------------------------------
    def status(self) -> dict:
        """给控制页/接口的状态 dict。

        外部模式没有本地子进程，`pid` / `uptime` 一律为 None/0，`running` 取远端
        health；额外带 `external` 与 `url`，让页面能直接显示"发去哪台机器了"。
        """
        cfg = self._config()
        if self.is_external():
            healthy = self._external_health()
            return {
                "running": healthy,
                "pid": None,
                "started_at": 0.0,
                "uptime_hours": 0.0,
                "max_uptime_hours": cfg["max_uptime_hours"],
                "needs_restart": False,
                "forced_off": False,
                "url": self._server_url(),
                "external": True,
                "remote": self.remote_status() if healthy else {
                    "external": True,
                    "url": self._server_url(),
                    "health_url": f"{self._server_url()}/health",
                    "healthy": False,
                },
                "step": 4,
            }
        with self._lock:
            proc = self._proc
            started_at = self._started_at
        running = proc is not None and proc.poll() is None
        return {
            "running": running,
            "pid": proc.pid if (proc is not None and running) else None,
            "started_at": started_at,
            "uptime_hours": (time.time() - started_at) / 3600.0 if (running and started_at > 0) else 0.0,
            "max_uptime_hours": cfg["max_uptime_hours"],
            "needs_restart": running and self.uptime_hours() >= cfg["max_uptime_hours"],
            "forced_off": self.is_forced_off(),
            "url": self._health_url(cfg),
            "external": False,
            "step": 4,
        }
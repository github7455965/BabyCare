"""Django settings for 09_web_vlm_manage

Step 2：apps.streaming 上线，FrameBus / FileSource / OnvifSource / Camera 模型。
后续步骤会按需扩展 INSTALLED_APPS / Channels 配置。
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / ".env")
except ImportError:  # 兜底：没装 dotenv 也能跑（用环境变量即可）
    BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 核心
# ---------------------------------------------------------------------------
SECRET_KEY = env("DJANGO_SECRET_KEY", "dev-insecure-secret-key-do-not-use-in-prod")
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [h.strip() for h in env("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()]

# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    "daphne",  # Channels 文档惯例：放第一个
    "channels",
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # 本项目
    "home",
    "apps.streaming",  # Step 2：摄像头 + FrameBus
    "apps.yolo_detect",  # Step 3：1Hz 采样 + FrameQueue + YoloLoop
    "apps.vlm",  # Step 4：LlamaManager（llama-server 进程管理）
    "apps.core",  # Step 10：HTTP API（启停 + dismiss）
    "apps.config_panel",  # Step 11：VLM 检查项配置面板
    "apps.cleanup",  # Step 12：cleanup 管理命令（cleanup_vlm_states / cleanup_frames）
    "apps.dashboard",  # 首页改造 + events 列表/详情 + 日志占位
    "apps.audio_detect",  # Step 15：音频线（独立进程 audio_worker，见 §9.3）
]

MIDDLEWARE = [
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# 数据库：MySQL / SQLite 二选一（DB_ENGINE 切换）
# ---------------------------------------------------------------------------
# DB_ENGINE=mysql   （默认，= 现状，零影响）外置 MySQL；09 独立 database `babycare_vlm`，与 08 隔离。
# DB_ENGINE=sqlite  「没有 MySQL 服务器」的本机部署：单文件 SQLite，零安装、零驱动
#                   （Python 自带 sqlite3，不需要 mysqlclient）。
#                   ⚠ 只适合**单机**：daphne / 音频 worker / 管理命令共用一个文件，
#                     SQLite 靠文件锁串行化写入；放到网络盘或共享目录上会损坏数据。
#
# SQLite 那三个 OPTIONS 不是可选项，逐条理由（改之前先读）：
#   timeout=20
#       并发写抢不到锁时最多等 20s，而不是立刻抛 "database is locked"。
#   transaction_mode=IMMEDIATE
#       Django 默认的 BEGIN 是 DEFERRED：锁在"第一次写"时才升级，
#       于是"先 SELECT 再 UPDATE"的读-改-写存在窗口，两个进程可能都读到
#       "租约空闲"然后各写各的。IMMEDIATE 让事务一开始就持有写锁，把整段串行化。
#       ⚠ 这是 MySQL → SQLite 唯一的**语义差异**：`select_for_update()` 在 SQLite 上
#         被 Django **静默忽略**（backend features.has_select_for_update = False），
#         apps/audio_detect/worker_lock.py 的 worker 互斥完全靠这个 IMMEDIATE 兜住。
#         删了它 = 两个 audio worker 可能同时抢到租约（互相覆盖事件状态）。
#   init_command=PRAGMA...
#       WAL 让"多读 + 一写"并行（默认 rollback journal 是读写互相阻塞）；
#       synchronous=NORMAL 省掉每次提交的 fsync；busy_timeout 在 Python 侧 timeout
#       之外再兜一层（毫秒）。Django 5.1+ 的 SQLite backend 支持 init_command，
#       会按 ";" 拆开逐条在建连后执行。
#
# 变量名故意用**小写**：Django 的 Settings 会把模块里 `isupper()` 为真的属性收进
# settings 命名空间，而 `"_DB_ENGINE".isupper()` 恰好是 True（下划线不参与判断）
# —— 用全大写下划线风格会让这两个中间变量以 `settings._DB_ENGINE` 的形式泄漏出去。
# 与本文件既有的 `_audio_scripts_dir` / `_audio_py_exe` 保持同一约定。
_db_engine = env("DB_ENGINE", "mysql").strip().lower()

if _db_engine in ("sqlite", "sqlite3"):
    _sqlite_path = Path(env("DB_sqlite_path", "") or "data/db.sqlite3")
    if not _sqlite_path.is_absolute():
        _sqlite_path = BASE_DIR / _sqlite_path
    # sqlite3 不会自动创建父目录，缺目录会直接 "unable to open database file"
    _sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(_sqlite_path),
            "OPTIONS": {
                "timeout": env_int("DB_SQLITE_TIMEOUT_SEC", 20),
                "transaction_mode": "IMMEDIATE",
                "init_command": (
                    "PRAGMA journal_mode=WAL;"
                    "PRAGMA synchronous=NORMAL;"
                    "PRAGMA busy_timeout=20000"
                ),
            },
        }
    }
elif _db_engine == "mysql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": env("DB_NAME", "babycare_vlm"),
            "USER": env("DB_USER", "root"),
            "PASSWORD": env("DB_PASSWORD", ""),
            "HOST": env("DB_HOST", "127.0.0.1"),
            "PORT": env("DB_PORT", "3306"),
            "OPTIONS": {
                "charset": "utf8mb4",
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
else:
    raise ValueError(f"DB_ENGINE 必须是 mysql|sqlite，当前: {_db_engine}")

# ---------------------------------------------------------------------------
# i18n（09 内部用 int 秒，不用 TZ-aware DateTimeField；USE_TZ 保持 False）
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "zh-hans"
TIME_ZONE = env("BABYCARE_TIMEZONE", "Asia/Shanghai")
USE_I18N = True
USE_TZ = False

# ---------------------------------------------------------------------------
# 静态文件
# ---------------------------------------------------------------------------
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "data" / "static"

# ---------------------------------------------------------------------------
# 媒体文件（Step 12：帧落盘 + 命令清理）
#   - MEDIA_ROOT: 帧图片根目录（MEDIA_ROOT/frames/<YYYY-MM-DD>/<cam>_<ts>.jpg）
#   - MEDIA_URL:  浏览器访问前缀（仅 DEBUG 下由 config.urls.static() serving）
# ---------------------------------------------------------------------------
MEDIA_ROOT = BASE_DIR / "media"
MEDIA_URL = "/media/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Channels（Step 1 用内存版）
# ---------------------------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"},
}

# ---------------------------------------------------------------------------
# 09 业务配置
# ---------------------------------------------------------------------------
# 外部 llama-server
BABYCARE_LLAMA_SERVER_URL = env("BABYCARE_LLAMA_SERVER_URL", "http://127.0.0.1:8082")

# 外部模式（分体部署）：llama-server 跑在另一台机器上，本进程**只探活、绝不启停**。
#   0（默认）= 单机：LlamaManager 照旧 spawn/kill/pidfile/health（现状，零影响）
#   1        = 分体：不 spawn、不 kill、不 unload、不清理 pidfile、不自动重启；
#              可用性一律由「远端 health + 熔断器」判定，URL 从 BABYCARE_LLAMA_SERVER_URL 推导。
# 注意：开着它时 `BABYCARE_LLAMA_SERVER_BIN` 等启动参数只用于"本机方案"，不再被消费。
# 详见 docs/superpowers/specs/2026-09-16-split-deploy-plan.md Phase 2。
BABYCARE_LLAMA_EXTERNAL = env_bool("BABYCARE_LLAMA_EXTERNAL", False)

# 显存模式：exclusive=6G（YOLO/VLM 互斥）；parallel=22G（并行常驻）；off=不做本机仲裁
# （分体部署用 off：两卡在不同机器上，没有显存竞争，状态机是纯负担）
BABYCARE_GPU_MODE = env("BABYCARE_GPU_MODE", "exclusive")
if BABYCARE_GPU_MODE not in {"exclusive", "parallel", "off"}:
    raise ValueError(
        f"BABYCARE_GPU_MODE 必须是 exclusive|parallel|off，当前: {BABYCARE_GPU_MODE}"
    )

# ---------------------------------------------------------------------------
# llama-server 进程管理（Step 4：LlamaManager）
# ---------------------------------------------------------------------------
# 所有字段都支持 .env 覆盖；缺省值与你现有启动脚本一致。
BABYCARE_LLAMA_SERVER_BIN   = env("BABYCARE_LLAMA_SERVER_BIN", "llama-server.exe")
BABYCARE_LLAMA_MODEL_DIR    = env("BABYCARE_LLAMA_MODEL_DIR", r"D:\AI\LLAMA")
BABYCARE_LLAMA_MODEL_NAME   = env("BABYCARE_LLAMA_MODEL_NAME", "qwen35-4b-Q4_K_M.gguf")
BABYCARE_LLAMA_MMPROJ_NAME  = env("BABYCARE_LLAMA_MMPROJ_NAME", "mmproj-Qwen3.5-4B-F16.gguf")
BABYCARE_LLAMA_CTX_SIZE     = env_int("BABYCARE_LLAMA_CTX_SIZE", 8192)
BABYCARE_LLAMA_NGL          = env_int("BABYCARE_LLAMA_NGL", 99)
BABYCARE_LLAMA_CTK          = env("BABYCARE_LLAMA_CTK", "q8_0")
BABYCARE_LLAMA_CTV          = env("BABYCARE_LLAMA_CTV", "q8_0")
# 绑本机：llama-server 默认 CORS 全开且**没有** api key，而 LlamaClient 目前不支持
# api_key（给它加 --api-key 会让每条 VLM 请求 401，整条视频线哑掉）。
# 因此默认值必须是 127.0.0.1 —— 只有分体部署（推理机在另一台机器）才改成 0.0.0.0。
BABYCARE_LLAMA_HOST         = env("BABYCARE_LLAMA_HOST", "127.0.0.1")
BABYCARE_LLAMA_PORT         = env_int("BABYCARE_LLAMA_PORT", 8082)
BABYCARE_LLAMA_PID_FILE     = env("BABYCARE_LLAMA_PID_FILE", "")
BABYCARE_LLAMA_MAX_UPTIME_HOURS    = env_int("BABYCARE_LLAMA_MAX_UPTIME_HOURS", 48)
BABYCARE_LLAMA_START_TIMEOUT_SEC   = env_int("BABYCARE_LLAMA_START_TIMEOUT_SEC", 60)
BABYCARE_LLAMA_START_RETRIES       = env_int("BABYCARE_LLAMA_START_RETRIES", 1)

# ---------------------------------------------------------------------------
# 常驻模式（Step 14）：YOLO + LLM 启动后常驻；LLM 周期重启 + 手动控制
# ---------------------------------------------------------------------------
# BABYCARE_LLM_RESIDENT            是否开启常驻模式（true / false）
#                                 false → 维持原 exclusive 循环行为（每次 VLM 启/停 llama）
#                                 true  → YOLO/LLM 启动后不卸载；LLM 按 AUTO_RESTART_HOURS 自动重启；
#                                         关闭期间（手动或重启窗口）新请求入 VLMQueuedTask，
#                                         DrainerThread 后台回放。
# BABYCARE_LLM_AUTO_RESTART_HOURS   常驻模式下 llama-server 自动重启周期（小时；0 = 仅手动）
# BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC  Drainer **空转**时的轮询间隔（秒）。
#                                 只在"队列空 / llama 不可用 / 处理失败重试"时生效；
#                                 成功消费一条后立刻抢下一条，让位 live 流量时用固定短睡
#                                 0.5s 重探（见 apps/vlm/drainer.py 模块 docstring）。
# BABYCARE_LLM_QUEUE_MAX_RETRIES    单条 queued task 调 llama 失败重试上限
BABYCARE_LLM_RESIDENT                 = env_bool("BABYCARE_LLM_RESIDENT", False)
BABYCARE_LLM_AUTO_RESTART_HOURS       = env_float("BABYCARE_LLM_AUTO_RESTART_HOURS", 24.0)
BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC = env_int("BABYCARE_LLM_QUEUE_DRAIN_INTERVAL_SEC", 5)
BABYCARE_LLM_QUEUE_MAX_RETRIES        = env_int("BABYCARE_LLM_QUEUE_MAX_RETRIES", 3)

# ---------------------------------------------------------------------------
# 熔断器 + 队列时效（分体部署；见 apps/core/llm_breaker.py）
# ---------------------------------------------------------------------------
# 推理机非 7×24 → 两条 llama 线会长时间不可达。熔断器让调用方别再白试，
# 请求留在各自的 DB 队列里攒着（VLM: VLMQueuedTask；ASR: pending_description），
# 恢复后回放。**关掉它 = 回到旧行为**（VLM 丢窗口 / ASR 8 分钟熬成永久 failed）。
# BABYCARE_LLM_BREAKER_ENABLED      总开关（0 = 完全旁路，永远放行）
# BABYCARE_LLM_FAIL_THRESHOLD       **通用默认**：连续"服务侧"失败多少次 → 熔断打开
#                                   （解析错/校验错/4xx **不计**，那是内容问题不是服务不可用）
# BABYCARE_LLM_FAIL_THRESHOLD_ASR   音频描述线（:8136）覆盖，默认 **1**
# BABYCARE_LLM_FAIL_THRESHOLD_VLM   视觉线（:8082）覆盖，默认 = 通用默认
#
# **两条线默认值不同，因为代价不对称**（用户 2026-09-16 定）：
#   - ASR **没有 live 流量**（唯一消费者就是 DescribeService）→「攒着」零成本，
#     所以 1 次即熔断：早一点停止白试，排队的工作照样回放。1 够用的前提是
#     connect 超时已独立压到 3s（见 AudioDescClient._post_once）—— 误判一次的
#     代价只有一个 3s 请求 + 最多 60s 排队，且不丢数据。
#   - VLM **有 live 窗口竞争** → 一次读超时（服务活着但慢）就熔断的话，live 窗口
#     会被压进队列最多 60s，代价是真的，所以沿用 3。
# BABYCARE_LLM_BREAKER_COOLDOWN_SEC 打开后多久放一条探测（探活 = 真实请求）
# BABYCARE_LLM_PROBE_TIMEOUT_SEC    半开态给调用方用的**短超时**：
#                                   探测必须便宜，否则自己就把链路堵死
#                                   （ASR 正常 30s×3=90s；VLM timeout_sec_first=60s 且全局串行）
# BABYCARE_LLM_QUEUE_MAX_AGE_SEC    超期的队列项标 expired 不再处理（0 = 不收割）
# BABYCARE_LLM_REPLAY_NOTIFY_MAX_DELAY_SEC
#                                   回放时事件距今超该值 → 只落库不通知（0 = 迟到多久都发）
BABYCARE_LLM_BREAKER_ENABLED           = env_bool("BABYCARE_LLM_BREAKER_ENABLED", True)
BABYCARE_LLM_FAIL_THRESHOLD            = env_int("BABYCARE_LLM_FAIL_THRESHOLD", 3)
BABYCARE_LLM_FAIL_THRESHOLD_ASR        = env_int("BABYCARE_LLM_FAIL_THRESHOLD_ASR", 1)
BABYCARE_LLM_FAIL_THRESHOLD_VLM        = env_int(
    "BABYCARE_LLM_FAIL_THRESHOLD_VLM", BABYCARE_LLM_FAIL_THRESHOLD,
)
BABYCARE_LLM_BREAKER_COOLDOWN_SEC      = env_int("BABYCARE_LLM_BREAKER_COOLDOWN_SEC", 60)
BABYCARE_LLM_PROBE_TIMEOUT_SEC         = env_int("BABYCARE_LLM_PROBE_TIMEOUT_SEC", 5)
BABYCARE_LLM_QUEUE_MAX_AGE_SEC         = env_int("BABYCARE_LLM_QUEUE_MAX_AGE_SEC", 0)
BABYCARE_LLM_REPLAY_NOTIFY_MAX_DELAY_SEC = env_int(
    "BABYCARE_LLM_REPLAY_NOTIFY_MAX_DELAY_SEC", 0
)

# ONVIF 摄像头（Step 2：所有摄像头共用一个密码，从 .env 读）
# BABYCARE_ONVIF_PASSWORD       ONVIF 全局默认密码（所有摄像头同一密码）
#                              缺省 = "" → ONVIF 摄像头 open() 直接返回 False
# BABYCARE_ONVIF_WSDL_DIR       wsdl 文件夹路径（onvif-zeep 0.2.12 不带 wsdl，需手动指定）
ONVIF_PASSWORD = env("BABYCARE_ONVIF_PASSWORD", "")
ONVIF_WSDL_DIR = env("BABYCARE_ONVIF_WSDL_DIR", "")

# ---------------------------------------------------------------------------
# 音频检测（Step 15：音频线，独立进程 + 独立 venv）
#   设计：docs/superpowers/specs/2026-09-07-audio-detection-design.md
#   本段只放 Phase 1（采集）需要的配置；模型/阈值/静音等在 Phase 2/3 追加。
# ---------------------------------------------------------------------------
# 总开关：false → 音频线永不启动（spec §9.3「用户期望状态」的总闸）
BABYCARE_AUDIO_ENABLED = env_bool("BABYCARE_AUDIO_ENABLED", False)

# 音频线解释器：独立 venv（spec §9.2）；缺省 = <BASE_DIR>/.venv-audio/Scripts/python.exe
_audio_scripts_dir = "Scripts" if os.name == "nt" else "bin"
_audio_py_exe = "python.exe" if os.name == "nt" else "python"
BABYCARE_AUDIO_PYTHON = env(
    "BABYCARE_AUDIO_PYTHON",
    str(Path(BASE_DIR) / ".venv-audio" / _audio_scripts_dir / _audio_py_exe),
)

# FFmpeg / FFprobe 可执行文件（不在 PATH 上时填绝对路径）
BABYCARE_FFMPEG_BIN = env("BABYCARE_FFMPEG_BIN", "ffmpeg")
# ffprobe：没有 profile 声明音频时的兜底探测（spec §3.1 第 4 条）
BABYCARE_FFPROBE_BIN = env("BABYCARE_FFPROBE_BIN", "ffprobe")

# 采集输出格式：单声道 16kHz s16le（spec §3.2）
BABYCARE_AUDIO_SAMPLE_RATE = env_int("BABYCARE_AUDIO_SAMPLE_RATE", 16000)
# 每次从 ffmpeg stdout 读取的时长（毫秒）：越大越省 CPU，断流发现越慢
BABYCARE_AUDIO_READ_CHUNK_MS = env_int("BABYCARE_AUDIO_READ_CHUNK_MS", 200)
# 事件前预录长度（Phase 3 用；此处决定环形缓冲容量）
BABYCARE_AUDIO_PRE_ROLL_SEC = env_int("BABYCARE_AUDIO_PRE_ROLL_SEC", 5)

# 启动后等首包超时；超过判 capture_error 并重连
BABYCARE_AUDIO_START_TIMEOUT_SEC = env_int("BABYCARE_AUDIO_START_TIMEOUT_SEC", 15)
# 超过该秒数没有新 PCM → 判"进程活着但流死了"，视为断流 + 覆盖不完整（spec §3.3）
BABYCARE_AUDIO_STALL_SEC = env_float("BABYCARE_AUDIO_STALL_SEC", 5.0)
# 断流重连退避（秒），指数增长到上限
BABYCARE_AUDIO_RECONNECT_MIN_SEC = env_float("BABYCARE_AUDIO_RECONNECT_MIN_SEC", 2.0)
BABYCARE_AUDIO_RECONNECT_MAX_SEC = env_float("BABYCARE_AUDIO_RECONNECT_MAX_SEC", 30.0)
# no_audio_profile 属于配置问题，不需要频繁重试
BABYCARE_AUDIO_NO_AUDIO_RETRY_SEC = env_float("BABYCARE_AUDIO_NO_AUDIO_RETRY_SEC", 60.0)
# ONVIF 连接 / 查询超时
BABYCARE_AUDIO_ONVIF_TIMEOUT_SEC = env_int("BABYCARE_AUDIO_ONVIF_TIMEOUT_SEC", 20)

# 心跳：worker 每 N 秒把内存状态写入 AudioRuntimeState（spec §9.3）
BABYCARE_AUDIO_HEARTBEAT_SEC = env_int("BABYCARE_AUDIO_HEARTBEAT_SEC", 2)
BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC = env_int("BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC", 10)
# worker 重读 Camera 表同步增删的间隔（spec §9.3）
BABYCARE_AUDIO_CAMERA_POLL_SEC = env_int("BABYCARE_AUDIO_CAMERA_POLL_SEC", 5)

# web 侧（daphne）启停 worker 用的文件路径；留空 = <BASE_DIR>/data/audio_worker.pid|.log|.stop
BABYCARE_AUDIO_PID_FILE = env("BABYCARE_AUDIO_PID_FILE", "")
BABYCARE_AUDIO_LOG_FILE = env("BABYCARE_AUDIO_LOG_FILE", "")
# 优雅停止哨兵：web 写、worker 读到即收尾（Windows 上跨进程发信号不可靠，见 paths.py）
BABYCARE_AUDIO_STOP_FILE = env("BABYCARE_AUDIO_STOP_FILE", "")

# web 侧拉起 worker 子进程时使用的 settings：音频 venv 没有 daphne / channels，
# 必须用 settings_audio（它只摘掉这两个 app），否则子进程 import daphne 失败秒退。
BABYCARE_AUDIO_SETTINGS_MODULE = env(
    "BABYCARE_AUDIO_SETTINGS_MODULE", "config.settings_audio",
)

# worker 启动宽限期：这段时间内"没有心跳"不算故障（模型加载 + 等首包需要 15~40s，
# 没有它的话用户点完按钮一刷新就会看到误报的"卡死"）。
BABYCARE_AUDIO_WORKER_STARTUP_GRACE_SEC = env_int(
    "BABYCARE_AUDIO_WORKER_STARTUP_GRACE_SEC", 45,
)

# 优雅停止等待上限：web 写哨兵后等 worker 自己收尾（落 stopped + 释放租约）多久，
# 超时才 taskkill /F /T 强杀。要覆盖"心跳发现(≤2s) + 逐路停 FFmpeg(每路≤3s)"。
BABYCARE_AUDIO_STOP_GRACE_SEC = env_int("BABYCARE_AUDIO_STOP_GRACE_SEC", 10)

# ---------------------------------------------------------------------------
# 音频检测：双模型推理（Step 15 Phase 2，spec §4）
# ---------------------------------------------------------------------------
# 模型资产：**必须本地化**（spec §4.2）。YAMNet 的 SavedModel 已 vendor 到
# model/yamnet/；tfhub.dev 已迁到 Kaggle Models，在线地址随时可能失效，
# 且缓存落在 %TEMP% 不适合生产。
BABYCARE_AUDIO_YAMNET_MODEL_DIR = env(
    "BABYCARE_AUDIO_YAMNET_MODEL_DIR",
    str(Path(BASE_DIR) / "model" / "yamnet"),
)
BABYCARE_AUDIO_YAMNET_CLASS_MAP = env(
    "BABYCARE_AUDIO_YAMNET_CLASS_MAP",
    str(Path(BASE_DIR) / "model" / "yamnet" / "yamnet_class_map.csv"),
)
BABYCARE_AUDIO_PANNS_CHECKPOINT = env(
    "BABYCARE_AUDIO_PANNS_CHECKPOINT",
    str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"),
)
BABYCARE_AUDIO_PANNS_CLASS_MAP = env(
    "BABYCARE_AUDIO_PANNS_CLASS_MAP",
    str(Path(BASE_DIR) / "model" / "panns" / "class_labels_indices.csv"),
)

# 统一输入规格（spec §4.1）
BABYCARE_AUDIO_DECISION_WINDOW_SEC = env_int("BABYCARE_AUDIO_DECISION_WINDOW_SEC", 2)
BABYCARE_AUDIO_DECISION_HOP_SEC = env_float("BABYCARE_AUDIO_DECISION_HOP_SEC", 1.0)
# 窗口内音频不足该比例 → 跳过本窗（不算阴性结论）
BABYCARE_AUDIO_MIN_AUDIO_RATIO = env_float("BABYCARE_AUDIO_MIN_AUDIO_RATIO", 0.9)
# 跨帧聚合方式：max | mean（YAMNet 2s 窗 = 4 个 patch）
BABYCARE_AUDIO_FRAME_AGG = env("BABYCARE_AUDIO_FRAME_AGG", "max")

# 线程数（P0-4：不限制时 TF 线程池空闲自旋，2 路常驻吃 ~296% CPU）
BABYCARE_AUDIO_TF_THREADS = env_int("BABYCARE_AUDIO_TF_THREADS", 1)
BABYCARE_AUDIO_TORCH_THREADS = env_int("BABYCARE_AUDIO_TORCH_THREADS", 1)
# 推理设备开关：**只对 PANNs 生效**（YAMNet 在 Windows 上恒为 CPU）
BABYCARE_AUDIO_USE_GPU = env_bool("BABYCARE_AUDIO_USE_GPU", False)

# 判阳阈值：**必须按模型分开**（spec §4.3.1）。两个模型分数分布不同，共用一根
# 阈值会让其中一侧实际等于"永不判阳"，and 共识永远无法成立。
# ⚠️ 当前是**保守占位值**：P0-5 因缺少哭声样本推迟，Phase 8 用生产 SoundDetectionLog
#    的近阈值窗 + 人工试听标定（spec §12）。
BABYCARE_AUDIO_CRY_THRESHOLD_YAMNET = env_float("BABYCARE_AUDIO_CRY_THRESHOLD_YAMNET", 0.3)
BABYCARE_AUDIO_CRY_THRESHOLD_PANNS = env_float("BABYCARE_AUDIO_CRY_THRESHOLD_PANNS", 0.3)
BABYCARE_AUDIO_SPEECH_THRESHOLD_YAMNET = env_float("BABYCARE_AUDIO_SPEECH_THRESHOLD_YAMNET", 0.3)
BABYCARE_AUDIO_SPEECH_THRESHOLD_PANNS = env_float("BABYCARE_AUDIO_SPEECH_THRESHOLD_PANNS", 0.3)
# 近阈值落库比例（spec §5.1）：分数 ≥ 阈值 × 该比例 → 落库，供事后调阈值
BABYCARE_AUDIO_NEAR_THRESHOLD_RATIO = env_float("BABYCARE_AUDIO_NEAR_THRESHOLD_RATIO", 0.7)
# 单模型降级时阈值自动上浮系数（spec §4.3.3）
BABYCARE_AUDIO_FALLBACK_THRESHOLD_SCALE = env_float("BABYCARE_AUDIO_FALLBACK_THRESHOLD_SCALE", 1.3)
# 单侧高置信度覆盖（spec §4.3.5）：and 共识下，任一模型分数 ≥ 该值 → 即使另一侧没过
# 阈值也判阳性。用于补偿"某模型分数分布整体偏低、and 永远不成立"。0 < x ≤ 1 生效。
BABYCARE_AUDIO_HIGH_CONFIDENCE_OVERRIDE = env_float(
    "BABYCARE_AUDIO_HIGH_CONFIDENCE_OVERRIDE", 0.8,
)

# 共识策略：and（默认，误报最低）| or（召回最高）（spec §4.3.2）
BABYCARE_AUDIO_CONSENSUS = env("BABYCARE_AUDIO_CONSENSUS", "and")
# 单模型故障降级：false → 严格模式，abstain 不产生新事件（spec §4.3.3）
BABYCARE_AUDIO_SINGLE_MODEL_FALLBACK = env_bool("BABYCARE_AUDIO_SINGLE_MODEL_FALLBACK", False)

# 每窗状态日志间隔秒；0 = 每窗都打 INFO（Phase 2 验收要"每窗可见"）
BABYCARE_AUDIO_DETECT_LOG_SEC = env_float("BABYCARE_AUDIO_DETECT_LOG_SEC", 5.0)

# ---------------------------------------------------------------------------
# 音频检测：事件组装（Step 15 Phase 3，spec §4.3.4/§6.2/§6.3）
# ---------------------------------------------------------------------------
# 事件组装开关（false → 只检测落日志，不组装事件/落盘）
BABYCARE_AUDIO_EVENTS_ENABLED = env_bool("BABYCARE_AUDIO_EVENTS_ENABLED", True)
# 启动规则：最近 N 窗至少 K 窗共识阳性（2/3 规则，P0-7 实测）
BABYCARE_AUDIO_EVENT_START_N = env_int("BABYCARE_AUDIO_EVENT_START_N", 3)
BABYCARE_AUDIO_EVENT_START_K = env_int("BABYCARE_AUDIO_EVENT_START_K", 2)
# **单参数 EVENT_GAP_SEC**（P0-7 结论）：既是"多久没阳性就结束"也是"多近算同一事件"
BABYCARE_AUDIO_EVENT_GAP_SEC = env_float("BABYCARE_AUDIO_EVENT_GAP_SEC", 3.0)
# 事件后补录（尾巴）；判定结束后还要再等这段时间才真正落盘
BABYCARE_AUDIO_POST_ROLL_SEC = env_int("BABYCARE_AUDIO_POST_ROLL_SEC", 3)

# 自适应静音（spec §6.3）：最近 30s 能量分布的 P20 作为动态静音线
# **不用固定 dBFS**——家用 mic 的 AGC 会把绝对电平拉平，固定值不可比
SILENCE_ADAPTIVE_WINDOW_SEC = env_float("SILENCE_ADAPTIVE_WINDOW_SEC", 30.0)
SILENCE_ADAPTIVE_PERCENTILE = env_float("SILENCE_ADAPTIVE_PERCENTILE", 20.0)
SILENCE_MIN_MS = env_int("SILENCE_MIN_MS", 500)

# ---------------------------------------------------------------------------
# 音频描述模型（Step 15 Phase 4，spec §6.4/§6.5）
# ---------------------------------------------------------------------------
# OpenAI 兼容服务（llama-server 等）。同机 = 127.0.0.1；分体部署 = 局域网内另一台
# 推理机的地址（http://<host>:<port>），对代码没有区别，只是 URL / 超时不同。
# URL 为空 → 描述服务不启动，事件停留在 pending_description，录音播放不受影响。
#
# 输出方言（provider）——决定用哪套提示词 + 怎么解析返回：
#   transcript：**纯转写**模型（Qwen3-ASR-1.7B 等）。模型只回一段人声文字，
#               **描述原文照放，不做任何判定**：has_cry / has_adult_speech /
#               background_sounds 等一律为 null（未判定，不是 false），
#               没人声时 description 就是空串。这些判断归声学侧两个小模型。
#   json      ：**结构化描述**模型（MOSS-Audio-4B / Qwen2.5-Omni-7B 等），
#               模型回 {"description": ..., "background_sounds": ...}。
# 两种方言最终都归一成同一份 description_json，业务代码/页面不感知具体模型。
# 非法值按 transcript 处理（见 apps/audio_detect/describer.py）。
BABYCARE_AUDIO_DESC_PROVIDER = env("BABYCARE_AUDIO_DESC_PROVIDER", "transcript")
BABYCARE_AUDIO_DESC_SERVER_URL = env("BABYCARE_AUDIO_DESC_SERVER_URL", "")
BABYCARE_AUDIO_DESC_MODEL = env("BABYCARE_AUDIO_DESC_MODEL", "")
BABYCARE_AUDIO_DESC_API_KEY = env("BABYCARE_AUDIO_DESC_API_KEY", "")
# 单次请求 **read** 超时：分体部署走局域网时留足（转写模型快，几秒即可；结构化描述模型慢得多）
BABYCARE_AUDIO_DESC_TIMEOUT_SEC = env_int("BABYCARE_AUDIO_DESC_TIMEOUT_SEC", 60)
# 单次请求 **connect** 超时（秒）：只回答"服务在不在"。
# 必须与 read 分开且给得小 —— 服务停掉后连接可能挂 SYN 等超时（不是立刻 RST），
# 共用 read 超时会让"服务不可用"要 `read × (max_retries+1)` 才能被发现（见
# apps/core/llm_breaker.py 与 AudioDescClient._post_once）。
BABYCARE_AUDIO_DESC_CONNECT_TIMEOUT_SEC = env_float(
    "BABYCARE_AUDIO_DESC_CONNECT_TIMEOUT_SEC", 3.0,
)
# 单次请求内的重试次数（timeout/网络错误；4xx/解析失败不重试）
BABYCARE_AUDIO_DESC_MAX_RETRIES = env_int("BABYCARE_AUDIO_DESC_MAX_RETRIES", 2)

# 描述服务开关与轮询
BABYCARE_AUDIO_DESCRIBE_ENABLED = env_bool("BABYCARE_AUDIO_DESCRIBE_ENABLED", True)
BABYCARE_AUDIO_DESCRIBE_POLL_SEC = env_float("BABYCARE_AUDIO_DESCRIBE_POLL_SEC", 5.0)
# 单段输出上限：转写方言下 512 可能截断 60s 长语音（≈300~500 token），故默认 1024
BABYCARE_AUDIO_DESCRIBE_MAX_TOKENS = env_int("BABYCARE_AUDIO_DESCRIBE_MAX_TOKENS", 1024)
# 分段级重试：失败后至少隔 backoff 秒再试；单段最多 max_retries 次，
# 耗尽 → 事件 failed（保留分段与错误状态，录音不删，spec §6.4/§12）
BABYCARE_AUDIO_DESCRIBE_RETRY_BACKOFF_SEC = env_float(
    "BABYCARE_AUDIO_DESCRIBE_RETRY_BACKOFF_SEC", 60.0,
)
BABYCARE_AUDIO_DESCRIBE_MAX_SEGMENT_RETRIES = env_int(
    "BABYCARE_AUDIO_DESCRIBE_MAX_SEGMENT_RETRIES", 8,
)

# 分段规则（spec §6.4）：目标 10~60s，优先静音点切；无静音点超 90s 按 60s 强制切；
# 任何 4B 请求不超 90s；<1s 丢弃
BABYCARE_AUDIO_SEGMENT_MIN_SEC = env_float("BABYCARE_AUDIO_SEGMENT_MIN_SEC", 10.0)
BABYCARE_AUDIO_SEGMENT_MAX_SEC = env_float("BABYCARE_AUDIO_SEGMENT_MAX_SEC", 60.0)
BABYCARE_AUDIO_SEGMENT_HARD_MAX_SEC = env_float("BABYCARE_AUDIO_SEGMENT_HARD_MAX_SEC", 90.0)
BABYCARE_AUDIO_SEGMENT_DISCARD_SEC = env_float("BABYCARE_AUDIO_SEGMENT_DISCARD_SEC", 1.0)

# ---------------------------------------------------------------------------
# 通知去重窗口（B6）：同 (cam, prompt, status) 在 N 秒内已有 notified=True
# 记录则本次跳过通知（事件仍写库，state.notified=False）。0 = 关闭去重。
# ---------------------------------------------------------------------------
BABYCARE_NOTIFY_DEDUP_WINDOW_SEC = env_int("BABYCARE_NOTIFY_DEDUP_WINDOW_SEC", 30)

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
MESSAGE_STORAGE = "django.contrib.messages.storage.cookie.CookieStorage"

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "[{asctime}] {levelname} {name}: {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "simple"},
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "django": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # 默认 apps.* 全部降级到 WARNING（干掉 yolo_detect detector 每帧 loaded/unloaded 噪声）
        "apps": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        # B12 白名单：仅打印用户关心的 4 类事件
        # - vlm 触发 + vlm 返回（apps.vlm.runner）
        # - 启动 yolo + 关闭 yolo（apps.yolo_detect.gpu_manager）
        # - 启动 llama + 关闭 llama（apps.vlm.llama_manager）
        "apps.vlm.runner": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "apps.vlm.llama_manager": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "apps.yolo_detect.gpu_manager": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # YoloLoop：cam 模型列表刷新 / cam 增删（低频；逐帧 detections 已降为 DEBUG）
        "apps.yolo_detect.yolo_loop": {"handlers": ["console"], "level": "INFO", "propagate": False},
        # Step 15 音频线：采集健康状态（心跳）+ ONVIF 音频 profile 探测
        "apps.audio_detect": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
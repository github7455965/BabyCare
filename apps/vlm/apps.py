"""
Vlm app config（Step 8）。

启动期行为
----------
- 实例化 LlamaManager（单例）
- 通过 GpuManager.attach_llama_manager() 注入依赖
- 启动 PromptRunnerManager（per (cam, prompt) 一个 asyncio task）
- 注册 atexit + signal handler：Ctrl+C / SIGTERM 时自动 stop llama-server
  （避免孤儿子进程持续占显存）

不在本步范围
------------
- 不启 llama-server（lazy：Runner 第一次触发 VLM 时由 GpuManager.acquire_vlm() 拉起，
  避免和 YOLO 同时占显存——6G 装不下两个模型）
"""

import atexit
import logging
import os
import signal

from django.apps import AppConfig

from apps.core.startup import should_autostart


logger = logging.getLogger(__name__)


# 不需要拉起 LlamaManager / RunnerManager 的 Django 管理命令
# 注：必须包含 "audio_worker"——音频线是独立进程，它的 ready() 不能把视频线也拉起来
#     （否则会产生第二个 llama-server，8082 端口冲突；spec §9.3）。
_MGMT_SKIP = frozenset({
    "migrate", "makemigrations", "shell", "dbshell", "check",
    "createsuperuser", "changepassword", "showmigrations",
    "sqlmigrate", "loaddata", "dumpdata", "test", "collectstatic",
    "audio_worker",
})


def _should_autostart() -> bool:
    """LlamaManager / RunnerManager 是进程级单例，启动期实例化；管理命令期不需要。

    **必须排除 runserver 的 reloader 父进程**：否则父子两进程各持一个 LlamaManager
    （单例只在进程内有效），共享同一个 PID 文件，各自 `_cleanup_stale_pidfile()`
    把对方刚拉起的 llama-server 当孤儿杀掉 → 启动期互相屠杀（见 apps/core/startup.py）。
    cleanup_* 同理不启：它们只删数据/文件，常驻模式下 `ensure_running()` 会真拉起
    llama-server，进程退出时 atexit 又杀掉（白折腾 GPU）。
    """
    if os.environ.get("VLM_AUTOSTART") == "0":
        return False
    return should_autostart(_MGMT_SKIP)


def _is_external(llama_mgr) -> bool:
    """llama-server 是否在别的机器上（分体部署）。"""
    fn = getattr(llama_mgr, "is_external", None)
    return bool(fn()) if callable(fn) else False


def _shutdown_llama(_signum=None, _frame=None) -> None:
    """Ctrl+C / SIGTERM 时调；graceful 关 llama-server 释放显存。
    重复注册安全（先 remove 再 register）；signal handler 一次触发后 try 删自己。
    """
    from .llama_manager import LlamaManager
    try:
        LlamaManager.instance().stop_server()
    except Exception:
        logger.exception("[vlm] shutdown stop_server failed")
    # 关掉自己避免 signal handler 二次调用
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except (ValueError, OSError):
        # 非主线程 / 平台不支持时忽略
        pass


def _register_shutdown_hooks(llama_mgr) -> None:
    """注册 Django 退出时的 llama-server 清理钩子。

    **外部模式不注册**：这个钩子会调 `stop_server()`，而分体后那等于
    "服务机一重启就去关推理机的服务" —— 既关不掉（远端），也不该关
    （checklist §2 #7）。推理机侧的启停是它自己的事。
    """
    is_ext = getattr(llama_mgr, "is_external", None)
    if callable(is_ext) and is_ext():
        logger.info("[vlm] 外部模式：不注册 shutdown 钩子（远端服务不受本进程控制）")
        return
    # atexit：Python 正常退出（Ctrl+C 触发的 KeyboardException、sys.exit()、main 结束）
    atexit.register(llama_mgr.stop_server)
    # signal：SIGTERM（daphne 转发）/ SIGINT（直接 Ctrl+C）；Windows 下 SIGBREAK 来自 Ctrl+Break
    for sig in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGBREAK", None)):
        if sig is None:
            continue
        try:
            # 保留已有 handler（daphne/runserver 自己的）；我们套外面
            prev = signal.getsignal(sig)
            if prev is _shutdown_llama:
                continue

            def _wrapper(s, f, _prev=prev):
                _shutdown_llama(s, f)
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    try:
                        _prev(s, f)
                    except Exception:
                        logger.exception("[vlm] previous signal handler raised")

            signal.signal(sig, _wrapper)
        except (ValueError, OSError) as e:
            # 非主线程或 Windows 平台不支持时跳过
            logger.debug("[vlm] register signal %s failed: %s", sig, e)


class VlmConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.vlm"
    verbose_name = "VLM（Step 8：RunnerManager）"

    def ready(self):
        if not _should_autostart():
            return

        try:
            from .llama_manager import LlamaManager
            from apps.yolo_detect.gpu_manager import GpuManager

            llama_mgr = LlamaManager.instance()
            GpuManager.instance().attach_llama_manager(llama_mgr)
            logger.info("[vlm] ready(): LlamaManager attached to GpuManager")

            # Step：注册 atexit + signal handler，Ctrl+C / SIGTERM 时清理 llama-server
            # 防止 runserver 关停后 llama 仍占显存
            _register_shutdown_hooks(llama_mgr)

            # Step 14 常驻模式：根据 .env 切 GPU + 启 LLM + 自动重启 + Drainer
            resident = _env_bool("BABYCARE_LLM_RESIDENT", False)
            if resident:
                GpuManager.instance().set_resident(True)
                # YOLO 加载一次（resident 模式下不卸载）
                try:
                    GpuManager.instance().acquire_yolo()
                except Exception:
                    logger.exception("[vlm] resident: acquire_yolo failed")
                # LLM 预热（外部模式退化为一次探活：远端没开是**正常情况**，
                # 不该打 traceback —— 分体下"推理机不在"就是日常）
                try:
                    llama_mgr.ensure_running()
                except Exception as e:
                    if _is_external(llama_mgr):
                        logger.warning(
                            "[vlm] 外部模式：远端 llama-server 当前不可用（%s）；"
                            "新窗口将入 VLMQueuedTask 等回放", e,
                        )
                    else:
                        logger.exception("[vlm] resident: llama ensure_running failed")
                # 24h 自动重启调度
                restart_h = _env_float(
                    "BABYCARE_LLM_AUTO_RESTART_HOURS", 24.0,
                )
                llama_mgr.start_auto_restart_scheduler(restart_h)
                # Drainer（后台回放 VLMQueuedTask）
                from .drainer import DrainerThread
                DrainerThread.instance().start()
                logger.info(
                    "[vlm] ready(): resident mode ON; "
                    "auto_restart=%.1fh drainer started",
                    restart_h,
                )

            # Step 8：注册 signal + 启动 RunnerManager
            from . import signals  # noqa: F401  注册 post_save/post_delete 监听
            from .runner import PromptRunnerManager

            PromptRunnerManager.instance().start()
            logger.info("[vlm] ready(): PromptRunnerManager started")
        except Exception as e:
            logger.error("[vlm] ready() failed: %s", e)


def _env_bool(key: str, default: bool) -> bool:
    """读 .env bool 值（"true"/"1"/"yes" → True，其它非空字符串 → False）。"""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    """读 .env float 值；解析失败 → default。"""
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
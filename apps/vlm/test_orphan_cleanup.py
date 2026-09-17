"""
LlamaManager 孤儿 PID 清理 + Ctrl+C 退出钩子测试。

修复背景：Django runserver Ctrl+C 后 LlamaManager.stop_server() 不被调 → 子进程变孤儿
占显存 + 占端口。下次启动会读到旧 PID 文件但 self._proc=None → 启新进程 + 旧孤儿残留。
修复：
- apps.py:ready() 注册 atexit + signal 钩子 → 优雅关闭
- LlamaManager._start_one 先调 _cleanup_stale_pidfile() 杀同名残留
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class LlamaManagerOrphanCleanupTest(unittest.TestCase):
    """LlamaManager._cleanup_stale_pidfile 行为（不真启 llama-server）。"""

    def test_missing_pidfile_noop(self):
        from apps.vlm.llama_manager import LlamaManager
        with tempfile.TemporaryDirectory() as d:
            pid_file = str(Path(d) / "x.pid")
            LlamaManager._cleanup_stale_pidfile(pid_file)
            self.assertFalse(os.path.exists(pid_file))

    def test_unparseable_pidfile_removed(self):
        from apps.vlm.llama_manager import LlamaManager
        with tempfile.TemporaryDirectory() as d:
            pid_file = str(Path(d) / "x.pid")
            Path(pid_file).write_text("not-a-number", encoding="utf-8")
            LlamaManager._cleanup_stale_pidfile(pid_file)
            self.assertFalse(os.path.exists(pid_file))

    def test_dead_pidfile_removed(self):
        """PID 指向已死进程（mock psutil）→ 文件被删 + 不 kill。"""
        from apps.vlm.llama_manager import LlamaManager
        with tempfile.TemporaryDirectory() as d:
            pid_file = str(Path(d) / "x.pid")
            Path(pid_file).write_text("99999999", encoding="utf-8")

            fake_psutil = MagicMock()
            fake_psutil.NoSuchProcess = Exception
            fake_psutil.TimeoutExpired = Exception
            fake_proc = MagicMock()
            fake_proc.is_running.return_value = False
            fake_psutil.Process.return_value = fake_proc

            with patch.dict("sys.modules", {"psutil": fake_psutil}):
                LlamaManager._cleanup_stale_pidfile(pid_file)
            self.assertFalse(os.path.exists(pid_file))
            fake_proc.terminate.assert_not_called()

    def test_alive_llama_pidfile_killed(self):
        """PID 指向活着的 llama-server.exe → terminate + 删文件。"""
        from apps.vlm.llama_manager import LlamaManager
        with tempfile.TemporaryDirectory() as d:
            pid_file = str(Path(d) / "x.pid")
            Path(pid_file).write_text("12345", encoding="utf-8")

            fake_psutil = MagicMock()
            fake_psutil.NoSuchProcess = Exception
            fake_psutil.TimeoutExpired = Exception
            fake_proc = MagicMock()
            fake_proc.is_running.side_effect = [True, False]  # 第一次查 alive, 第二次查已死
            fake_proc.name.return_value = "llama-server.exe"
            fake_psutil.Process.return_value = fake_proc

            with patch.dict("sys.modules", {"psutil": fake_psutil}):
                LlamaManager._cleanup_stale_pidfile(pid_file)
            fake_proc.terminate.assert_called_once()
            self.assertFalse(os.path.exists(pid_file))

    def test_alive_non_llama_pidfile_preserved(self):
        """PID 指向活着但非 llama 的进程（避免误杀）→ 只删文件不 kill。"""
        from apps.vlm.llama_manager import LlamaManager
        with tempfile.TemporaryDirectory() as d:
            pid_file = str(Path(d) / "x.pid")
            Path(pid_file).write_text("12345", encoding="utf-8")

            fake_psutil = MagicMock()
            fake_psutil.NoSuchProcess = Exception
            fake_psutil.TimeoutExpired = Exception
            fake_proc = MagicMock()
            fake_proc.is_running.return_value = True
            fake_proc.name.return_value = "python.exe"  # 不是 llama-server
            fake_psutil.Process.return_value = fake_proc

            with patch.dict("sys.modules", {"psutil": fake_psutil}):
                LlamaManager._cleanup_stale_pidfile(pid_file)
            fake_proc.terminate.assert_not_called()
            # 文件仍然被删（防止下次误认）
            self.assertFalse(os.path.exists(pid_file))


class ShutdownHooksTest(unittest.TestCase):
    """apps.py:ready() 注册 atexit + signal handler 调 LlamaManager.stop_server。"""

    def setUp(self):
        from apps.vlm import apps as apps_mod
        self.apps_mod = apps_mod

    def test_shutdown_llama_calls_stop_server(self):
        """_shutdown_llama() → LlamaManager.instance().stop_server()。"""
        with patch("apps.vlm.llama_manager.LlamaManager.instance") as m_inst:
            self.apps_mod._shutdown_llama()
            m_inst.return_value.stop_server.assert_called_once()

    def test_register_shutdown_hooks_registers_atexit_and_signals(self):
        """_register_shutdown_hooks 应注册 atexit + 改 signal handler。"""
        import atexit
        llama_mgr = MagicMock()
        llama_mgr.stop_server = MagicMock()
        # 单机模式：必须显式声明，否则 MagicMock 的 is_external() 是**真值**
        # → 钩子会被当成外部模式跳过（这正是本用例之前失败的原因）
        llama_mgr.is_external.return_value = False
        # atexit + signal 计数 baseline
        with patch("atexit.register") as m_atexit, \
             patch("signal.signal") as m_signal:
            self.apps_mod._register_shutdown_hooks(llama_mgr)
            # 至少注册了 1 次 atexit
            self.assertGreaterEqual(m_atexit.call_count, 1)
            # 至少改 2 个 signal（SIGTERM + SIGINT）
            self.assertGreaterEqual(m_signal.call_count, 2)
            # 调的是 llama_mgr.stop_server
            m_atexit.assert_called_with(llama_mgr.stop_server)

    def test_register_shutdown_hooks_skipped_in_external_mode(self):
        """外部模式**不注册** —— 否则服务机一重启就去关推理机的服务（checklist §2 #7）。"""
        llama_mgr = MagicMock()
        llama_mgr.is_external.return_value = True
        with patch("atexit.register") as m_atexit, \
             patch("signal.signal") as m_signal:
            self.apps_mod._register_shutdown_hooks(llama_mgr)
            m_atexit.assert_not_called()
            m_signal.assert_not_called()
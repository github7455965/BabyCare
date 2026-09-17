"""外部模式（分体部署）单测：`LlamaManager` / `GpuManager`。

为什么这组用例必须存在
----------------------
分体的核心风险不是"功能没做"，而是**做过头**：外部模式下任何一次误操作
（`unload()` / `stop_server()` / `_cleanup_stale_pidfile()`）都可能把
**另一台机器上别人正在用的服务**关掉或杀掉。所以每个"禁止"都必须有断言。

同时另一半同样重要：**单机行为一字不能变**。所以每条禁用在 `test_local_*`
里都有对应的"照旧"断言。

跑法
----
    .venv\\Scripts\\python.exe manage.py test apps.vlm.tests_external_mode -v 2
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402

if not _django_apps.ready:
    django.setup()

from apps.vlm.llama_manager import (  # noqa: E402
    LlamaExternalError,
    LlamaManager,
    LlamaStartError,
)


def _ext_env(url: str = "http://192.168.1.50:8082"):
    """patch 出"外部模式 + 指定远端地址"的环境。"""
    return patch.dict(os.environ, {
        "BABYCARE_LLAMA_EXTERNAL": "1",
        "BABYCARE_LLAMA_SERVER_URL": url,
    })


def _local_env():
    return patch.dict(os.environ, {
        "BABYCARE_LLAMA_EXTERNAL": "0",
        # 显式清掉，避免继承开发者本机 .env 的远端地址
        "BABYCARE_LLAMA_SERVER_URL": "http://127.0.0.1:8082",
    })


def _manager() -> LlamaManager:
    """**不用 instance()**：单例会让用例之间互相污染状态。"""
    return LlamaManager()


def _ok_get(status: int = 200):
    m = MagicMock()
    m.status_code = status
    return m


# ===========================================================================
# is_external / 地址推导
# ===========================================================================
class IsExternalTest(unittest.TestCase):
    def test_env_truthy_values(self):
        for raw in ("1", "true", "TRUE", "yes", "on"):
            with patch.dict(os.environ, {"BABYCARE_LLAMA_EXTERNAL": raw}):
                self.assertTrue(_manager().is_external(), raw)

    def test_env_falsy_values(self):
        for raw in ("0", "false", "no", "off", ""):
            with patch.dict(os.environ, {"BABYCARE_LLAMA_EXTERNAL": raw}):
                self.assertFalse(_manager().is_external(), raw)

    def test_falls_back_to_settings(self):
        """env 没设 → 读 settings（.env 通过 dotenv 落进 settings，不一定落 env）。"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BABYCARE_LLAMA_EXTERNAL", None)
            with patch("django.conf.settings.BABYCARE_LLAMA_EXTERNAL", True):
                self.assertTrue(_manager().is_external())


class HealthUrlTest(unittest.TestCase):
    """checklist §2 #2：外部模式必须从 SERVER_URL 推导，不能复用启动参数。"""

    def test_external_uses_server_url_not_launch_args(self):
        with _ext_env("http://192.168.1.50:9999"):
            mgr = _manager()
            # 启动参数是**本机**的 0.0.0.0:8082；沿用它会永远探不到远端
            cfg = {"host": "0.0.0.0", "port": 8082}
            self.assertEqual(
                mgr._health_url(cfg), "http://192.168.1.50:9999/health",
            )

    def test_external_strips_trailing_slash(self):
        with _ext_env("http://192.168.1.50:8082/"):
            mgr = _manager()
            self.assertEqual(
                mgr._health_url({}), "http://192.168.1.50:8082/health",
            )

    def test_local_unchanged(self):
        """单机回归：0.0.0.0 → 127.0.0.1，端口取启动参数。"""
        with _local_env():
            mgr = _manager()
            self.assertEqual(
                mgr._health_url({"host": "0.0.0.0", "port": 8082}),
                "http://127.0.0.1:8082/health",
            )
            self.assertEqual(
                mgr._health_url({"host": "127.0.0.1", "port": 9090}),
                "http://127.0.0.1:9090/health",
            )


# ===========================================================================
# is_running 语义
# ===========================================================================
class IsRunningTest(unittest.TestCase):
    def test_external_uses_remote_health(self):
        """checklist §2 #1：外部模式必须看远端 health，否则恒 False。"""
        with _ext_env():
            mgr = _manager()
            with patch("requests.get", return_value=_ok_get(200)) as m_get:
                self.assertTrue(mgr.is_running())
                self.assertEqual(m_get.call_count, 1)
            with patch("requests.get", return_value=_ok_get(503)):
                mgr2 = _manager()
                self.assertFalse(mgr2.is_running())

    def test_external_health_failure_is_not_healthy(self):
        with _ext_env():
            mgr = _manager()
            with patch("requests.get", side_effect=OSError("no route")):
                self.assertFalse(mgr.is_running())

    def test_external_health_is_ttl_cached(self):
        """热路径（每窗都问）不能每次都发 HTTP。"""
        with _ext_env():
            mgr = _manager()
            with patch("requests.get", return_value=_ok_get(200)) as m_get:
                for _ in range(5):
                    mgr.is_running()
                self.assertEqual(m_get.call_count, 1)

    def test_local_still_uses_proc(self):
        """单机回归：本地模式看子进程，不看网络。"""
        with _local_env():
            mgr = _manager()
            with patch("requests.get") as m_get:
                self.assertFalse(mgr.is_running())            # 没有子进程

                proc = MagicMock()
                proc.poll.return_value = None                 # 活着
                mgr._proc = proc
                self.assertTrue(mgr.is_running())

                proc.poll.return_value = 1                    # 死了
                self.assertFalse(mgr.is_running())
                m_get.assert_not_called()


# ===========================================================================
# 绝不起停
# ===========================================================================
class NeverSpawnTest(unittest.TestCase):
    def test_ensure_running_ok_when_healthy(self):
        with _ext_env():
            with patch("requests.get", return_value=_ok_get(200)):
                _manager().ensure_running()               # 不抛

    def test_ensure_running_raises_when_unhealthy(self):
        with _ext_env():
            with patch("requests.get", side_effect=OSError("down")):
                with self.assertRaises(LlamaExternalError):
                    _manager().ensure_running()

    def test_ensure_running_never_spawns(self):
        with _ext_env():
            with patch("requests.get", side_effect=OSError("down")), \
                 patch.object(LlamaManager, "_start_one") as m_start:
                with self.assertRaises(LlamaExternalError):
                    _manager().ensure_running()
                m_start.assert_not_called()

    def test_start_one_refuses_in_external_mode(self):
        """防将来有人绕过 ensure_running 直接调 _start_one。"""
        with _ext_env():
            with self.assertRaises(LlamaExternalError):
                _manager()._start_one({}, 1)

    def test_restart_raises(self):
        with _ext_env():
            with self.assertRaises(LlamaExternalError):
                _manager().restart()


class NoStopTest(unittest.TestCase):
    def test_stop_server_is_noop(self):
        with _ext_env():
            mgr = _manager()
            with patch.object(mgr, "_stop_blocking") as m_stop:
                mgr.stop_server()
                m_stop.assert_not_called()

    def test_unload_does_not_touch_remote_or_local(self):
        """checklist §2 #4：既不发远端 /unload，也不 fallback 到本机 stop_server。"""
        with _ext_env():
            mgr = _manager()
            with patch("requests.post") as m_post, \
                 patch.object(mgr, "stop_server") as m_stop:
                mgr.unload()
                m_post.assert_not_called()
                m_stop.assert_not_called()

    def test_unload_still_works_locally(self):
        """单机回归：本地 unload 仍会发 /unload（或 fallback 停服）。"""
        with _local_env():
            mgr = _manager()
            proc = MagicMock()
            proc.poll.return_value = None
            mgr._proc = proc
            with patch("requests.post", return_value=_ok_get(200)) as m_post:
                mgr.unload()
                m_post.assert_called_once()

    def test_force_off_does_not_set_flag(self):
        """设了旗标会让 Runner 永远入队，而没人能把远端开起来。"""
        with _ext_env():
            mgr = _manager()
            with patch.object(mgr, "_stop_blocking") as m_stop:
                mgr.force_off()
                m_stop.assert_not_called()
            self.assertFalse(mgr.is_forced_off())

    def test_auto_restart_scheduler_not_started(self):
        with _ext_env():
            mgr = _manager()
            with patch("threading.Thread") as m_thread:
                mgr.start_auto_restart_scheduler(1.0)
                m_thread.assert_not_called()

    def test_needs_restart_always_false(self):
        with _ext_env():
            mgr = _manager()
            with patch("requests.get", return_value=_ok_get(200)):
                self.assertFalse(mgr.needs_restart())


class StatusTest(unittest.TestCase):
    def test_external_status_shape(self):
        with _ext_env("http://10.0.0.9:8082"):
            mgr = _manager()
            with patch("requests.get", return_value=_ok_get(200)):
                st = mgr.status()
            self.assertTrue(st["external"])
            self.assertIsNone(st["pid"])
            self.assertTrue(st["running"])
            self.assertFalse(st["needs_restart"])
            self.assertEqual(st["url"], "http://10.0.0.9:8082")

    def test_local_status_shape_unchanged(self):
        with _local_env():
            st = _manager().status()
            self.assertFalse(st["external"])
            self.assertIsNone(st["pid"])
            self.assertFalse(st["running"])


# ===========================================================================
# GpuManager：不再仲裁 / 不去卸远端
# ===========================================================================
class GpuManagerModeTest(unittest.TestCase):
    def _gpu(self, mode: str, external: bool):
        from apps.yolo_detect.gpu_manager import GpuManager

        gpu = GpuManager()
        gpu.set_mode(mode)
        lm = MagicMock()
        lm.is_external.return_value = external
        gpu.attach_llama_manager(lm)
        return gpu, lm

    def test_set_mode_accepts_off(self):
        from apps.yolo_detect.gpu_manager import GpuManager

        gpu = GpuManager()
        gpu.set_mode("off")
        self.assertEqual(gpu.status()["mode"], "off")

    def test_set_mode_rejects_unknown(self):
        from apps.yolo_detect.gpu_manager import GpuManager

        with self.assertRaises(ValueError):
            GpuManager().set_mode("nope")

    def test_arbitrate_matrix(self):
        for mode, expected in (
            ("off", False), ("parallel", False), ("exclusive", True),
        ):
            gpu, _ = self._gpu(mode, external=False)
            self.assertEqual(gpu._arbitrate(), expected, mode)

    def test_resident_disables_arbitration(self):
        gpu, _ = self._gpu("exclusive", external=False)
        gpu.set_resident(True)
        self.assertFalse(gpu._arbitrate())

    def test_external_acquire_vlm_does_not_ensure_running(self):
        """checklist §2 #3：不再无条件尝试启本地 exe。"""
        gpu, lm = self._gpu("off", external=True)
        with patch.object(gpu, "_ensure_llama_running") as m_ensure:
            gpu.acquire_vlm()
            m_ensure.assert_not_called()
        self.assertTrue(gpu.status()["vlm_request_pending"])

    def test_off_mode_local_still_ensures(self):
        """`off` 与 `external` 正交：同机 + 不仲裁时，子进程该启还得启。"""
        gpu, lm = self._gpu("off", external=False)
        with patch.object(gpu, "_ensure_llama_running") as m_ensure:
            gpu.acquire_vlm()
            m_ensure.assert_called_once()

    def test_external_release_does_not_unload(self):
        """这是最危险的一条：release 会 POST 远端 /unload。"""
        gpu, lm = self._gpu("off", external=True)
        gpu.acquire_vlm()
        gpu.release_vlm_if_idle()
        lm.unload.assert_not_called()
        self.assertFalse(gpu.status()["vlm_request_pending"])

    def test_unload_llama_double_guard(self):
        """即使有人直接调 `_unload_llama()`，外部模式也必须拒绝。"""
        gpu, lm = self._gpu("exclusive", external=True)
        gpu._unload_llama()
        lm.unload.assert_not_called()

    def test_exclusive_release_still_unloads_locally(self):
        """单机回归：exclusive 下 release 仍会卸 VLM。

        **必须先把 YOLO 的 pending 清掉**：exclusive 的 `acquire_vlm()`
        会 `while state == YOLO_RUNNING: if not _yolo_request_pending: break; wait(5s)`
        —— `acquire_yolo()` 留下的 pending 不清，这个循环**永远出不来**
        （第一版本用例就是这么把整个套件挂住 5 分钟的）。
        """
        gpu, lm = self._gpu("exclusive", external=False)
        with patch.object(gpu, "_unload_yolo"):
            gpu.acquire_yolo()              # → YOLO_RUNNING
            gpu.release_yolo_if_idle()      # 清 pending + 切 IDLE
        with patch.object(gpu, "_ensure_llama_running"):
            gpu.acquire_vlm()               # → VLM_RUNNING
        with patch.object(gpu, "_unload_llama") as m_unload_llama, \
             patch.object(gpu, "_unload_yolo"):
            gpu.release_vlm_if_idle()
            m_unload_llama.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

"""
Step 10 HTTP API 单元测试。

范围
----
- 覆盖 apps.core.views 6 个 view（yolo_restart / vlm_restart / gpu_mode_get /
  gpu_mode_post / cam_state / state_dismiss / state_undismiss）
- 不连真实 DB / 不真启 llama-server / 不真 load best.pt
- 用 unittest + django.test.Client + MagicMock patch 内部 Manager

跑法
----
.venv\\Scripts\\python.exe -m unittest apps.core.tests -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.http import Http404, JsonResponse  # noqa: E402
from django.test import Client  # noqa: E402


# ---------------------------------------------------------------------------
# YOLO restart
# ---------------------------------------------------------------------------
class YoloRestartTest(unittest.TestCase):
    def setUp(self):
        # SERVER_NAME=localhost 走 settings.ALLOWED_HOSTS（默认白名单不含 testserver）
        self.client = Client(SERVER_NAME="localhost")

    def test_ok(self):
        mock_det = MagicMock()
        type(mock_det).model_path = PropertyMock(return_value="model/best.pt")
        type(mock_det).device = PropertyMock(return_value="cuda")
        with patch("apps.yolo_detect.detector.BabyDetector.instance", return_value=mock_det):
            resp = self.client.post("/api/yolo/restart/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["model_path"], "model/best.pt")
        self.assertEqual(data["device"], "cuda")
        self.assertIn("dt_sec", data)
        mock_det.unload.assert_called_once()
        mock_det.ensure_loaded.assert_called_once()

    def test_failure_returns_500(self):
        mock_det = MagicMock()
        mock_det.unload.side_effect = FileNotFoundError("model not found: x.pt")
        with patch("apps.yolo_detect.detector.BabyDetector.instance", return_value=mock_det):
            resp = self.client.post("/api/yolo/restart/")
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertIn("FileNotFoundError", data["error"])
        self.assertIn("model not found", data["error"])

    def test_get_not_allowed(self):
        resp = self.client.get("/api/yolo/restart/")
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# VLM restart
# ---------------------------------------------------------------------------
class VlmRestartTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def test_ok_was_running(self):
        mock_mgr = MagicMock()
        mock_mgr.uptime_hours.return_value = 12.3456
        mock_mgr.is_running.return_value = True
        mock_mgr.restart.return_value = {"port": 8082, "pid": 9999}
        with patch("apps.vlm.llama_manager.LlamaManager.instance", return_value=mock_mgr):
            resp = self.client.post("/api/vlm/restart/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["uptime_hours_before"], 12.3456)
        self.assertTrue(data["was_running_before"])
        self.assertEqual(data["port"], 8082)
        self.assertEqual(data["pid"], 9999)
        mock_mgr.restart.assert_called_once()

    def test_ok_was_not_running(self):
        mock_mgr = MagicMock()
        mock_mgr.uptime_hours.return_value = 0.0
        mock_mgr.is_running.return_value = False
        mock_mgr.restart.return_value = {"port": 8082, "pid": 8888}
        with patch("apps.vlm.llama_manager.LlamaManager.instance", return_value=mock_mgr):
            resp = self.client.post("/api/vlm/restart/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["was_running_before"])
        self.assertEqual(data["pid"], 8888)

    def test_failure_returns_500(self):
        mock_mgr = MagicMock()
        mock_mgr.uptime_hours.return_value = 0.0
        mock_mgr.is_running.return_value = False
        mock_mgr.restart.side_effect = RuntimeError("spawn failed")
        with patch("apps.vlm.llama_manager.LlamaManager.instance", return_value=mock_mgr):
            resp = self.client.post("/api/vlm/restart/")
        self.assertEqual(resp.status_code, 500)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertIn("RuntimeError", data["error"])

    def test_get_not_allowed(self):
        resp = self.client.get("/api/vlm/restart/")
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# GPU mode GET / POST
# ---------------------------------------------------------------------------
class GpuModeGetTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def test_ok(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"state": "IDLE", "gpu_loaded": False, "mode": "exclusive"}
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self.client.get("/api/gpu/mode/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "exclusive")
        self.assertEqual(data["status"]["state"], "IDLE")

    def test_failure_returns_500(self):
        with patch(
            "apps.yolo_detect.gpu_manager.GpuManager.instance",
            side_effect=RuntimeError("not initialized"),
        ):
            resp = self.client.get("/api/gpu/mode/")
        self.assertEqual(resp.status_code, 500)
        self.assertFalse(resp.json()["ok"])


class GpuModePostTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def _post(self, body_dict):
        return self.client.post(
            "/api/gpu/mode/",
            data=json.dumps(body_dict) if body_dict is not None else "",
            content_type="application/json",
        )

    def test_ok_no_change(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"mode": "exclusive"}
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post({"mode": "exclusive"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["mode"], "exclusive")
        self.assertFalse(data["requires_restart"])
        mock_gm.set_mode.assert_called_once_with("exclusive")

    def test_ok_changed(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"mode": "exclusive"}
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post({"mode": "parallel"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["old_mode"], "exclusive")
        self.assertEqual(data["mode"], "parallel")
        self.assertTrue(data["requires_restart"])

    def test_invalid_mode(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"mode": "exclusive"}
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post({"mode": "turbo"})
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertIn("exclusive|parallel", data["error"])
        mock_gm.set_mode.assert_not_called()

    def test_missing_mode(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"mode": "exclusive"}
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post({})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["ok"])

    def test_bad_json(self):
        mock_gm = MagicMock()
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self.client.post(
                "/api/gpu/mode/",
                data="not json",
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("JSON", resp.json()["error"])

    def test_body_not_object(self):
        mock_gm = MagicMock()
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post(["bad"])
        self.assertEqual(resp.status_code, 400)
        self.assertIn("object", resp.json()["error"])

    def test_set_mode_raises(self):
        mock_gm = MagicMock()
        mock_gm.status.return_value = {"mode": "exclusive"}
        mock_gm.set_mode.side_effect = ValueError("bad")
        with patch("apps.yolo_detect.gpu_manager.GpuManager.instance", return_value=mock_gm):
            resp = self._post({"mode": "parallel"})
        self.assertEqual(resp.status_code, 500)
        self.assertFalse(resp.json()["ok"])


# ---------------------------------------------------------------------------
# cam_state
# ---------------------------------------------------------------------------
class CamStateTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def _make_cam(self, cam_id=1, name="bedroom", is_active=True):
        cam = MagicMock()
        cam.id = cam_id
        cam.name = name
        cam.is_active = is_active
        return cam

    def _make_state(self, sid=100, prompt="face_occlusion", hit=True,
                    dismissed=False, auto_silenced=False, status="face_occlusion"):
        st = MagicMock()
        st.id = sid
        st.hit = hit
        st.status = status
        st.img1_ts = 1756700050
        st.dismissed_as_false = dismissed
        st.auto_silenced = auto_silenced
        st.prompt_config.name = prompt
        return st

    def _ctx(self, cam, states=None, prompts_keys=None,
             yolo_loaded=True, vlm_running=False,
             cur_max_read_ts=1756700100, fail_count=0,
             latest_frame_ts=0, has_baby_count_10s=0,
             cam_t0=1756700000):
        """构造所有 patch context manager。返回 list[patch]。"""
        from apps.yolo_detect.frame_queue import FrameItem

        items = []
        if latest_frame_ts > 0:
            # 生成 10 帧，让最新 ts = latest_frame_ts（倒序填）
            for i in range(10):
                ts = latest_frame_ts - (9 - i)
                items.append(FrameItem(ts=ts, ndarray=None,
                                       has_baby=(i < has_baby_count_10s)))

        mock_t0 = MagicMock()
        mock_t0.get.return_value = cam_t0

        mock_fq = MagicMock()
        mock_fq.snapshot_sorted.return_value = items

        mock_fq_inst = MagicMock()
        mock_fq_inst.get.return_value = mock_fq

        mock_cur_mgr = MagicMock()
        keys = prompts_keys if prompts_keys is not None else []
        mock_cur_mgr.stats.return_value = {"keys": keys}
        mock_cur = MagicMock()
        mock_cur.max_read_ts = cur_max_read_ts
        mock_cur.fail_count = fail_count
        mock_cur_mgr.get.return_value = mock_cur

        mock_prompt_cfg = MagicMock()
        mock_prompt_cfg.name = "face_occlusion"
        mock_prompt_cfg.window_sec = 10
        mock_prompt_qs = MagicMock()
        mock_prompt_qs.get.return_value = mock_prompt_cfg

        mock_det = MagicMock()
        type(mock_det).is_loaded = PropertyMock(return_value=yolo_loaded)

        mock_vlm = MagicMock()
        mock_vlm.is_running.return_value = vlm_running

        def _cam_get(pk):
            if pk == cam.id:
                return cam
            raise Http404
        mock_cam_default_mgr = MagicMock()
        mock_cam_default_mgr.get.side_effect = _cam_get
        # Camera.DoesNotExist 用于 raise Http404 路径
        cam_class_mock = MagicMock()
        cam_class_mock._default_manager.get = mock_cam_default_mgr.get
        cam_class_mock.DoesNotExist = Http404  # 用于 try/except 触发 raise

        # VLMCheckState.objects.filter(...).select_related(...).order_by(...)[:10]
        mock_state_qs = MagicMock()
        filtered = MagicMock()
        mock_state_qs.filter.return_value = filtered
        filtered.select_related.return_value = filtered
        filtered.order_by.return_value = filtered
        filtered.__getitem__.return_value = states or []

        mock_prm = MagicMock()
        # 公共方法（Step 10 新增）
        mock_prm.cursor_keys_for_cam.return_value = [pid for (_, pid) in keys]
        mock_prm.cursor_get.return_value = mock_cur

        return [
            patch("apps.streaming.cam_t0.CamT0Manager.instance", return_value=mock_t0),
            patch("apps.yolo_detect.frame_queue.FrameQueueManager.instance", return_value=mock_fq_inst),
            patch("apps.vlm.runner.PromptRunnerManager.instance", return_value=mock_prm),
            patch("apps.yolo_detect.detector.BabyDetector.instance", return_value=mock_det),
            patch("apps.vlm.llama_manager.LlamaManager.instance", return_value=mock_vlm),
            patch("apps.core.views.Camera", cam_class_mock),
            patch("apps.vlm.models.VLMPromptConfig.objects", mock_prompt_qs),
            patch("apps.vlm.models.VLMCheckState.objects", mock_state_qs),
        ]

    def test_ok_no_frames_no_prompts(self):
        cam = self._make_cam()
        ctx = self._ctx(cam)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]:
            resp = self.client.get("/api/state/1/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["camera"]["id"], 1)
        self.assertEqual(data["camera"]["name"], "bedroom")
        self.assertEqual(data["cam_t0"], 1756700000)
        self.assertEqual(data["latest_frame_ts"], 0)
        self.assertIsNone(data["lag_sec"])
        self.assertEqual(data["has_baby_count_10s"], 0)
        self.assertEqual(data["prompts"], [])
        self.assertTrue(data["yolo_loaded"])
        self.assertFalse(data["vlm_running"])
        self.assertEqual(data["recent_states"], [])

    def test_ok_with_frames_and_prompts(self):
        cam = self._make_cam()
        states = [self._make_state(sid=100), self._make_state(sid=101, hit=False)]
        # 用 fake now 锚定时间，避免真实 time.time() 漂移
        fake_now = 1785542400  # 2026-08-31
        latest_ts = fake_now  # 最新帧就在 now（让 10 帧全在 10s 窗口内）
        ctx = self._ctx(
            cam, states=states, prompts_keys=[(1, 5)],
            yolo_loaded=True, vlm_running=True,
            latest_frame_ts=latest_ts, has_baby_count_10s=8,
        )
        with patch("apps.core.views.time.time", return_value=float(fake_now)), \
             ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]:
            resp = self.client.get("/api/state/1/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["latest_frame_ts"], latest_ts)
        self.assertEqual(data["lag_sec"], 0)
        self.assertEqual(data["has_baby_count_10s"], 8)
        self.assertEqual(len(data["prompts"]), 1)
        self.assertEqual(data["prompts"][0]["name"], "face_occlusion")
        self.assertEqual(data["prompts"][0]["id"], 5)
        self.assertTrue(data["yolo_loaded"])
        self.assertTrue(data["vlm_running"])
        self.assertEqual(len(data["recent_states"]), 2)
        self.assertEqual(data["recent_states"][0]["prompt"], "face_occlusion")
        self.assertTrue(data["recent_states"][0]["hit"])

    def test_yolo_not_loaded(self):
        cam = self._make_cam()
        ctx = self._ctx(cam, yolo_loaded=False)
        with ctx[0], ctx[1], ctx[2], ctx[3], ctx[4], ctx[5], ctx[6], ctx[7]:
            resp = self.client.get("/api/state/1/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["yolo_loaded"])

    def test_404_camera_not_found(self):
        cam_class_mock = MagicMock()
        cam_class_mock._default_manager.get.side_effect = Http404
        cam_class_mock.DoesNotExist = Http404
        with patch("apps.core.views.Camera", cam_class_mock):
            resp = self.client.get("/api/state/999/")
        self.assertEqual(resp.status_code, 404)

    def test_post_not_allowed(self):
        resp = self.client.post("/api/state/1/")
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# state dismiss / undismiss
# ---------------------------------------------------------------------------
class StateDismissTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def _state(self, sid=42, dismissed=False, status="face_occlusion"):
        st = MagicMock()
        st.id = sid
        st.dismissed_as_false = dismissed
        st.status = status
        st.prompt_config.name = "face_occlusion"
        st.camera.name = "bedroom"
        st.save = MagicMock()
        return st

    def test_ok(self):
        st = self._state(dismissed=False)
        qs_mock = MagicMock()
        qs_mock.select_related.return_value.get.return_value = st
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/dismiss/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["state_id"], 42)
        self.assertEqual(data["prompt"], "face_occlusion")
        self.assertEqual(data["camera"], "bedroom")
        self.assertTrue(st.dismissed_as_false)
        st.save.assert_called_once()

    def test_already_dismissed(self):
        st = self._state(dismissed=True)
        qs_mock = MagicMock()
        qs_mock.select_related.return_value.get.return_value = st
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/dismiss/")
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "already_dismissed")
        st.save.assert_not_called()

    def test_not_found(self):
        from apps.vlm.models import VLMCheckState
        qs_mock = MagicMock()
        qs_mock.select_related.return_value.get.side_effect = VLMCheckState.DoesNotExist
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/dismiss/")
        self.assertEqual(resp.status_code, 404)
        self.assertFalse(resp.json()["ok"])

    def test_get_not_allowed(self):
        resp = self.client.get("/api/state/42/dismiss/")
        self.assertEqual(resp.status_code, 405)


class StateUnDismissTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def _state(self, sid=42, dismissed=True):
        st = MagicMock()
        st.id = sid
        st.dismissed_as_false = dismissed
        st.save = MagicMock()
        return st

    def test_ok(self):
        st = self._state(dismissed=True)
        qs_mock = MagicMock()
        qs_mock.get.return_value = st
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/undismiss/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["state_id"], 42)
        self.assertFalse(st.dismissed_as_false)
        st.save.assert_called_once()

    def test_not_dismissed(self):
        st = self._state(dismissed=False)
        qs_mock = MagicMock()
        qs_mock.get.return_value = st
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/undismiss/")
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"], "not_dismissed")
        st.save.assert_not_called()

    def test_not_found(self):
        from apps.vlm.models import VLMCheckState
        qs_mock = MagicMock()
        qs_mock.get.side_effect = VLMCheckState.DoesNotExist
        with patch("apps.vlm.models.VLMCheckState.objects", qs_mock):
            resp = self.client.post("/api/state/42/undismiss/")
        self.assertEqual(resp.status_code, 404)

    def test_get_not_allowed(self):
        resp = self.client.get("/api/state/42/undismiss/")
        self.assertEqual(resp.status_code, 405)


# ---------------------------------------------------------------------------
# 共性：路由没破坏
# ---------------------------------------------------------------------------
class RootRoutingTest(unittest.TestCase):
    def setUp(self):
        self.client = Client(SERVER_NAME="localhost")

    def test_healthz_still_works(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)

    def test_index_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# 启动期 autostart 判据（apps/core/startup.py）
# ---------------------------------------------------------------------------
class AutostartGuardTests(unittest.TestCase):
    """runserver 的 autoreload：**只允许子进程**启动后台服务。

    回归背景（现场踩到）：`manage.py runserver` 开 autoreload → 父子两进程都执行
    `apps.ready()`，而各 Manager 的单例只在**进程内**有效：

    - 两个 `LlamaManager` 共享同一个 `data/llama_server.pid`，各自把对方刚拉起的
      llama-server 判成"Ctrl+C 残留的孤儿"并 kill → 启动期互相屠杀（日志里
      `killed stale llama-server pid=…` 紧跟对方的 `spawning`）；
    - 两个 `VideoStreamManager` 各拉一份同一路 RTSP，再叠加音频 worker → 三方争抢，
      设备并发受限时整条视频线 `too many open failures` 退出。
    """

    SKIP = frozenset({"migrate", "test", "audio_worker"})

    def _guard(self, argv, run_main="__unset__"):
        from apps.core.startup import should_autostart

        env = {} if run_main == "__unset__" else {"RUN_MAIN": run_main}
        with patch.object(sys, "argv", list(argv)), patch.dict(os.environ, env):
            if run_main == "__unset__":
                os.environ.pop("RUN_MAIN", None)     # 显式模拟"父进程"
            return should_autostart(self.SKIP)

    def test_daphne_entry_starts(self):
        self.assertTrue(self._guard(["daphne", "config.asgi:application"]))

    def test_runserver_child_starts(self):
        self.assertTrue(self._guard(["manage.py", "runserver"], run_main="true"))

    def test_runserver_reloader_parent_skips(self):
        self.assertFalse(self._guard(["manage.py", "runserver"]))

    def test_runserver_noreload_starts(self):
        """`--noreload` 只有一个进程（父进程即服务进程），必须正常启动。"""
        self.assertTrue(self._guard(["manage.py", "runserver", "--noreload"]))

    def test_skip_commands_and_cleanup_prefix(self):
        self.assertFalse(self._guard(["manage.py", "migrate"]))
        self.assertFalse(self._guard(["manage.py", "test"]))
        self.assertFalse(self._guard(["manage.py", "audio_worker"]))
        self.assertFalse(self._guard(["manage.py", "cleanup_expired_states"]))

    def test_bare_manage_py_starts(self):
        self.assertTrue(self._guard(["manage.py"]))

    def test_all_apps_share_the_same_guard(self):
        """4 个 app 的 ready() 守卫都必须排除 runserver 的 reloader 父进程。"""
        from apps.audio_detect import apps as audio_apps
        from apps.streaming.apps import StreamingConfig
        from apps.vlm import apps as vlm_apps
        from apps.yolo_detect import apps as yolo_apps

        guards = {
            "streaming": StreamingConfig._should_autostart,
            "yolo_detect": yolo_apps._should_autostart,
            "vlm": vlm_apps._should_autostart,
            "audio_detect": audio_apps._should_autostart,
        }
        for run_main, expected in (("__unset__", False), ("true", True)):
            env = {} if run_main == "__unset__" else {"RUN_MAIN": run_main}
            with patch.object(sys, "argv", ["manage.py", "runserver"]), \
                    patch.dict(os.environ, env):
                if run_main == "__unset__":
                    os.environ.pop("RUN_MAIN", None)
                for name, guard in guards.items():
                    with self.subTest(app=name, run_main=run_main):
                        self.assertIs(guard(), expected, f"{name} run_main={run_main}")


# ---------------------------------------------------------------------------
# apps.core.imaging（帧限长边，跨 app 共用）
# ---------------------------------------------------------------------------
class FitLongSideTest(unittest.TestCase):
    """``fit_long_side``：只缩不放、等比、惰性 import cv2。

    消费方：``apps.vlm.frame_storage.resize_long_side``（送 VLM 的图）与
    ``apps.yolo_detect.frame_queue.FrameQueue.push``（入队存储，2026-09-15）。
    """

    def _fn(self):
        from apps.core.imaging import fit_long_side

        return fit_long_side

    def test_scales_down_keeping_aspect(self):
        import numpy as np

        out = self._fn()(np.zeros((1152, 2048, 3), dtype="uint8"), 1024)
        self.assertEqual(out.shape[:2], (576, 1024))

    def test_never_upscales(self):
        """小图原样返回（且是**同一对象**，不复制）。"""
        import numpy as np

        small = np.zeros((480, 640, 3), dtype="uint8")
        self.assertIs(self._fn()(small, 1024), small)

    def test_exact_boundary_untouched(self):
        """长边恰好等于上限 → 不缩。"""
        import numpy as np

        exact = np.zeros((576, 1024, 3), dtype="uint8")
        self.assertIs(self._fn()(exact, 1024), exact)

    def test_no_toplevel_cv2_import(self):
        """模块顶层不 import cv2（frame_queue 刻意零重依赖，靠惰性加载维持）。

        只看**代码行**：docstring 里为解释这个设计提到过 "import cv2" 字样。
        """
        import apps.core.imaging as imaging

        src = Path(imaging.__file__).read_text(encoding="utf-8")
        head = src.split("def fit_long_side", 1)[0]
        code_lines = [ln.strip() for ln in head.splitlines()]
        self.assertNotIn("import cv2", code_lines)


if __name__ == "__main__":
    unittest.main()
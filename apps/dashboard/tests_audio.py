"""Phase 7 音频线页面测试（spec §8）。

覆盖面
------
- §8.1 声音检测日志：渲染、8 项筛选、分页保留筛选、稀疏落库免责说明；
- §8.2 音频事件详情：时间线 / 静音点 / 分段 / 4B 描述 / 关联 Prompt 与投递条件；
- 音频文件服务：只服务 MEDIA_ROOT 内文件（Range 由 static.serve 提供）；
- §8.3 手动清理：预览、二次确认（POST-only）、后台线程、真删（库 + 文件）、
  孤儿文件扫描与重试清理。

约定
----
- 用 ``override_settings(MEDIA_ROOT=<tmp>)`` 隔离媒体目录，避免污染真实 media/；
- ``SoundDetectionLog`` 的 JSON 筛选走 Django 的 JSON 路径查询
  （``consensus_payload__labels__cry__state``），SQLite / MySQL 都支持。
"""
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.conf import settings  # noqa: E402
from django.test import Client, TestCase, override_settings  # noqa: E402

from apps.audio_detect.models import (  # noqa: E402
    AudioEvent,
    AudioEventSegment,
    AudioRuntimeState,
    SoundDetectionLog,
)
from apps.dashboard.audio_cleanup import (  # noqa: E402
    _cleanup_by_filters,
    purge_orphan_files,
)
from apps.streaming.models import Camera  # noqa: E402
from apps.vlm.models import (  # noqa: E402
    NotifyTarget,
    PromptNotificationDelivery,
    VLMCheckState,
    VLMCheckStateAudioEvent,
    VLMPromptConfig,
)

T0 = 1_700_000_000  # 固定时间基，避免依赖 now()


def _name(base: str) -> str:
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


class _Base(TestCase):
    """与 dashboard/tests.py 一致：TestCase 自动回滚 + 屏蔽 signals 拉进程。"""

    def setUp(self):
        self._orig_allowed = settings.ALLOWED_HOSTS
        settings.ALLOWED_HOSTS = list(self._orig_allowed) + ["testserver"]
        self.addCleanup(setattr, settings, "ALLOWED_HOSTS", self._orig_allowed)

        self.client = Client()
        patcher = patch("apps.vlm.signals._get_manager", lambda: None)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.cam = Camera.objects.create(
            name=_name("cam1"), source_type=Camera.SOURCE_FILE, is_active=False,
        )
        self.cam2 = Camera.objects.create(
            name=_name("cam2"), source_type=Camera.SOURCE_FILE, is_active=False,
        )

    # ---- 构造器 ----
    def _log(self, cam=None, start_ts=T0, end_ts=None, **overrides):
        defaults = {
            "camera": cam or self.cam,
            "window_start_ts": start_ts,
            "window_end_ts": end_ts if end_ts is not None else start_ts + 2,
            "yamnet_payload": {"ok": True, "top_k": [["Crying, sobbing", 0.51]]},
            "panns_payload": {"ok": True, "top_k": [["Baby cry, infant cry", 0.44]]},
            "consensus_payload": _consensus(
                cry={"state": "positive", "yamnet": 0.51, "panns": 0.44},
                speech={"state": "negative", "yamnet": 0.02, "panns": 0.03},
            ),
            "decision": SoundDetectionLog.DECISION_POSITIVE,
            "log_reason": SoundDetectionLog.LOG_POSITIVE,
        }
        defaults.update(overrides)
        return SoundDetectionLog.objects.create(**defaults)

    def _event(self, cam=None, start_ts=T0, ended_ts=None, **overrides):
        defaults = {
            "camera": cam or self.cam,
            "status": AudioEvent.STATUS_COMPLETED,
            "started_at_ts": start_ts,
            "ended_at_ts": ended_ts if ended_ts is not None else start_ts + 8,
            "duration_sec": 11.0,
            "detected_labels": {
                "cry": {
                    "first_positive_ts": start_ts,
                    "last_positive_ts": start_ts + 2,
                    "positive_windows": [[start_ts, start_ts + 2]],
                },
            },
            "silence_ranges": [[2.0, 4.0]],
            "description_status": AudioEvent.DESC_COMPLETED,
            "description_json": {"summary": "婴儿哭声"},
            "description_raw": "raw-desc-output",
            "moss_model": "Qwen3-ASR-1.7B",
        }
        defaults.update(overrides)
        return AudioEvent.objects.create(**defaults)


def _consensus(cry=None, speech=None, strategy="and"):
    """构造 ``consensus_payload``（结构见 ConsensusEngine.consensus_payload）。"""
    labels = {}
    for label, spec in (("cry", cry or {}), ("speech", speech or {})):
        labels[label] = {
            "state": spec.get("state", "negative"),
            "reason": spec.get("reason", ""),
            "scores": {"yamnet": spec.get("yamnet"), "panns": spec.get("panns")},
            "positives": {},
            "thresholds": {"yamnet": 0.3, "panns": 0.3},
        }
    return {"strategy": strategy, "labels": labels}


# ---------------------------------------------------------------------------
# §8.1 声音检测日志
# ---------------------------------------------------------------------------
class SoundDetectionLogsViewTest(_Base):
    URL = "/sound-detection/logs/"

    def test_empty(self):
        resp = self.client.get(self.URL)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("暂无日志".encode(), resp.content)

    def test_sparse_log_disclaimer_rendered(self):
        """稀疏落库免责说明必须出现（spec §8.1 注），否则会误导排查。"""
        resp = self.client.get(self.URL)
        self.assertIn("稀疏落库".encode(), resp.content)
        self.assertIn("不代表".encode(), resp.content)

    def test_row_shows_scores_event_and_runtime(self):
        self._log()
        ev = self._event()
        AudioRuntimeState.objects.create(
            camera=self.cam, status=AudioRuntimeState.STATUS_READY,
        )
        resp = self.client.get(self.URL)
        content = resp.content.decode("utf-8")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("0.510", content)          # YAMNet 业务标签分
        self.assertIn("0.440", content)          # PANNs
        self.assertIn(f"/sound-events/{ev.pk}/", content)   # 事件编号链接
        self.assertIn("采集中", content)          # 音频健康

    def test_row_shows_runtime_last_error(self):
        self._log()
        AudioRuntimeState.objects.create(
            camera=self.cam, status=AudioRuntimeState.STATUS_CAPTURE_ERROR,
            last_error="ffmpeg exited",
        )
        resp = self.client.get(self.URL)
        self.assertIn("ffmpeg exited".encode(), resp.content)

    def test_filter_cam_id(self):
        self._log(cam=self.cam)
        self._log(cam=self.cam2)
        resp = self.client.get(f"{self.URL}?cam_id={self.cam2.id}")
        rows = resp.context_data["rows"]
        self.assertEqual([r["log"].camera_id for r in rows], [self.cam2.id])

    def test_filter_decision_positive(self):
        self._log()  # positive
        self._log(start_ts=T0 + 10, decision=SoundDetectionLog.DECISION_NEGATIVE,
                  log_reason=SoundDetectionLog.LOG_NEAR_THRESHOLD)
        resp = self.client.get(f"{self.URL}?decision=positive")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "positive")

    def test_filter_decision_negative_excludes_single_side(self):
        """「无声音」必须排掉「单模型阳性」（后者属于近阈值待看）。"""
        self._log(
            start_ts=T0 + 100,
            decision=SoundDetectionLog.DECISION_NEGATIVE,
            consensus_payload=_consensus(
                cry={"state": "negative", "reason": "single_side_only"},
            ),
        )
        self._log(
            start_ts=T0 + 200,
            decision=SoundDetectionLog.DECISION_NEGATIVE,
            consensus_payload=_consensus(cry={"state": "negative"}),
        )
        resp = self.client.get(f"{self.URL}?decision=negative")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].window_start_ts, T0 + 200)

    def test_filter_decision_single_side(self):
        self._log(
            decision=SoundDetectionLog.DECISION_NEGATIVE,
            consensus_payload=_consensus(
                speech={"state": "negative", "reason": "single_side_only"},
            ),
        )
        self._log(start_ts=T0 + 100, consensus_payload=_consensus(cry={"state": "negative"}))
        resp = self.client.get(f"{self.URL}?decision=single_side")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)

    def test_filter_log_reason(self):
        self._log(log_reason=SoundDetectionLog.LOG_POSITIVE)
        self._log(start_ts=T0 + 10, log_reason=SoundDetectionLog.LOG_STATE_CHANGE)
        resp = self.client.get(f"{self.URL}?log_reason=state_change")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].log_reason, "state_change")

    def test_invalid_log_reason_ignored(self):
        self._log()
        resp = self.client.get(f"{self.URL}?log_reason=bogus")
        self.assertEqual(len(resp.context_data["rows"]), 1)

    def test_filter_label_positive(self):
        self._log()  # cry positive
        self._log(
            start_ts=T0 + 100,
            decision=SoundDetectionLog.DECISION_NEGATIVE,
            consensus_payload=_consensus(cry={"state": "negative"}),
        )
        resp = self.client.get(f"{self.URL}?label=cry")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].window_start_ts, T0)

    def test_filter_model_error(self):
        self._log(decision=SoundDetectionLog.DECISION_DEGRADED, failure_reason="all_abstain")
        self._log(start_ts=T0 + 100)
        resp = self.client.get(f"{self.URL}?model_error=1")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].decision, "degraded")

    def test_filter_time_range(self):
        self._log(start_ts=T0)
        self._log(start_ts=T0 + 3600)
        start = _dt_param(T0 + 100)
        resp = self.client.get(f"{self.URL}?start_dt={start}")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].window_start_ts, T0 + 3600)

    def test_filter_has_event_true(self):
        self._log(start_ts=T0)
        self._log(start_ts=T0 + 10_000)
        self._event(start_ts=T0)          # 只与第一条相交
        resp = self.client.get(f"{self.URL}?has_event=1")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].window_start_ts, T0)

    def test_filter_has_event_false(self):
        self._log(start_ts=T0)
        self._log(start_ts=T0 + 10_000)
        self._event(start_ts=T0)
        resp = self.client.get(f"{self.URL}?has_event=0")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["log"].window_start_ts, T0 + 10_000)

    def test_event_overlap_ignores_other_camera(self):
        """跨摄像头不算相交（事件按 camera 隔离）。"""
        self._log()
        self._event(cam=self.cam2, start_ts=T0)
        resp = self.client.get(f"{self.URL}?has_event=1")
        self.assertEqual(len(resp.context_data["rows"]), 0)

    def test_ongoing_event_counts_as_overlap(self):
        self._log(start_ts=T0)
        self._event(start_ts=T0 - 5, ended_ts=None)   # 进行中
        resp = self.client.get(f"{self.URL}?has_event=1")
        self.assertEqual(len(resp.context_data["rows"]), 1)

    def test_pagination_keeps_filters(self):
        for i in range(30):
            self._log(start_ts=T0 + i, cam=self.cam2)
        resp = self.client.get(f"{self.URL}?cam_id={self.cam2.id}")
        self.assertEqual(resp.context_data["page"].paginator.count, 30)
        # 分页链接带着筛选条件（模板里 HTML 转义成 &amp;，直接看 context 更稳）
        self.assertEqual(resp.context_data["querystring"], f"cam_id={self.cam2.id}&")
        self.assertIn(b"page=2", resp.content)


def _dt_param(ts: int) -> str:
    """epoch 秒 → ``datetime-local`` 字符串（与 ``_dt_to_epoch`` 同一本地时间基）。"""
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# 音频事件列表（按事件展示，与按窗口互为两种粒度）
# ---------------------------------------------------------------------------
class SoundEventsListTest(_Base):
    URL = "/sound-detection/events/"

    def test_empty(self):
        resp = self.client.get(self.URL)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("暂无音频事件".encode(), resp.content)

    def test_tabs_switch_between_two_views(self):
        """两种粒度入口必须互相可达（切错视图是排查时最容易踩的坑）。"""
        resp = self.client.get("/sound-detection/logs/")
        self.assertEqual(resp.context_data["audio_view"], "window")
        self.assertIn(b"/sound-detection/events/", resp.content)

        resp = self.client.get(self.URL)
        self.assertEqual(resp.context_data["audio_view"], "event")
        self.assertIn(b"/sound-detection/logs/", resp.content)

    def test_tabs_carry_camera_and_time_only(self):
        """切换时只带公共筛选（摄像头 + 时间），粒度专有的不带过去。"""
        start = _dt_param(T0)
        resp = self.client.get(
            f"/sound-detection/logs/?cam_id={self.cam2.id}&start_dt={start}&label=cry",
        )
        shared = resp.context_data["shared_qs"]
        self.assertIn(f"cam_id={self.cam2.id}", shared)
        self.assertIn("start_dt=", shared)
        self.assertNotIn("label=", shared)

    def test_row_shows_player_detail_and_window_link(self):
        ev = self._event(audio_path="C:/media/audio_events/1_100.flac")
        AudioEventSegment.objects.create(
            audio_event=ev, sequence=1, start_offset=0.0, end_offset=8.0,
        )
        resp = self.client.get(self.URL)
        content = resp.content.decode("utf-8")
        self.assertIn(f"/sound-events/{ev.pk}/audio/", content)   # 行内播放器
        self.assertIn(f"/sound-events/{ev.pk}/", content)         # 详情链接
        self.assertIn(f"cam_id={self.cam.id}", content)           # 相关窗口反查
        self.assertIn("哭声", content)
        self.assertIn("1 窗", content)
        self.assertIn("11.0s", content)
        rows = resp.context_data["rows"]
        self.assertEqual(rows[0]["seg_count"], 1)
        self.assertEqual(rows[0]["labels"][0]["windows"], 1)
        self.assertTrue(rows[0]["has_audio"])

    def test_row_shows_description_flags_and_background(self):
        self._event(description_json={
            "has_cry": True, "has_adult_speech": True,
            "adult_speech_summary": "妈妈说不要哭",
            "background_sounds": ["water"],
        })
        resp = self.client.get(self.URL)
        content = resp.content.decode("utf-8")
        self.assertIn("妈妈说不要哭", content)
        self.assertIn("water", content)
        self.assertIn("哭声", content)
        self.assertIn("人声", content)

    def test_row_handles_null_judgement_fields(self):
        """transcript 方言的 description_json（判定字段全 null）既不能崩，也不该亮徽章。

        描述模型不做判定 → 描述列没有「哭声」「人声」徽章（标签列那个「哭声」来自
        声学侧 ``detected_labels``，与描述无关），但转写文字照常展示。
        """
        self._event(description_json={
            "description": "妈妈说该吃饭了",
            "has_cry": None, "has_adult_speech": None, "background_sounds": None,
            "cry_start_offset_sec": None, "cry_duration_sec": None,
            "speech_confidence": None,
        })
        resp = self.client.get(self.URL)
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode("utf-8")
        self.assertIn("妈妈说该吃饭了", content)
        desc = resp.context_data["rows"][0]["desc"]
        self.assertFalse(desc["has_cry"])
        self.assertFalse(desc["has_speech"])
        self.assertEqual(desc["background"], "")

    def test_ongoing_event_marked(self):
        # ``_event`` 的 ``ended_ts=None`` 会被默认值补成 start+8，用 overrides 显式置空
        self._event(status=AudioEvent.STATUS_RECORDING, ended_at_ts=None)
        resp = self.client.get(self.URL)
        self.assertIn("进行中".encode(), resp.content)
        self.assertTrue(resp.context_data["rows"][0]["ongoing"])

    def test_filter_cam_id(self):
        self._event(cam=self.cam)
        self._event(cam=self.cam2)
        resp = self.client.get(f"{self.URL}?cam_id={self.cam2.id}")
        rows = resp.context_data["rows"]
        self.assertEqual([r["ev"].camera_id for r in rows], [self.cam2.id])

    def test_filter_label(self):
        """「出现过该标签」——不是"最后一次判定为阳"（那是窗口页的语义）。"""
        self._event(start_ts=T0)                       # cry
        self._event(
            start_ts=T0 + 100,
            detected_labels={"speech": {
                "first_positive_ts": T0 + 100,
                "last_positive_ts": T0 + 101,
                "positive_windows": [[T0 + 100, T0 + 101]],
            }},
        )
        resp = self.client.get(f"{self.URL}?label=speech")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ev"].started_at_ts, T0 + 100)

    def test_filter_status(self):
        self._event(status=AudioEvent.STATUS_COMPLETED)
        self._event(start_ts=T0 + 100, status=AudioEvent.STATUS_FAILED)
        resp = self.client.get(f"{self.URL}?status=failed")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ev"].status, "failed")

    def test_filter_desc_status(self):
        self._event(description_status=AudioEvent.DESC_COMPLETED)
        self._event(start_ts=T0 + 100, description_status=AudioEvent.DESC_PENDING)
        resp = self.client.get(f"{self.URL}?desc_status=pending")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ev"].description_status, "pending")

    def test_filter_has_audio(self):
        self._event(audio_path="C:/media/audio_events/1.flac")
        self._event(start_ts=T0 + 100, audio_path="")
        resp = self.client.get(f"{self.URL}?has_audio=1")
        self.assertEqual(len(resp.context_data["rows"]), 1)
        resp = self.client.get(f"{self.URL}?has_audio=0")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ev"].started_at_ts, T0 + 100)

    def test_filter_degraded(self):
        self._event(degraded=True)
        self._event(start_ts=T0 + 100, degraded=False)
        resp = self.client.get(f"{self.URL}?degraded=1")
        rows = resp.context_data["rows"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["ev"].degraded)

    def test_invalid_status_ignored(self):
        self._event()
        resp = self.client.get(f"{self.URL}?status=bogus")
        self.assertEqual(len(resp.context_data["rows"]), 1)

    def test_time_range_uses_overlap_semantics(self):
        """相交口径：跨越边界算命中；区间内已开始未结束的进行中也算（与窗口页同口径）。"""
        crossed = self._event(start_ts=T0, ended_ts=T0 + 20)          # 跨越区间左边界
        self._event(start_ts=T0 + 1000, ended_ts=T0 + 1010)           # 不相交
        ongoing = self._event(start_ts=T0 - 100, ended_at_ts=None)    # 进行中，已进入区间
        later = self._event(start_ts=T0 + 5000, ended_at_ts=None)     # 进行中，但区间后才开始
        resp = self.client.get(
            f"{self.URL}?start_dt={_dt_param(T0 + 15)}&end_dt={_dt_param(T0 + 18)}",
        )
        got = {r["ev"].pk for r in resp.context_data["rows"]}
        self.assertEqual(got, {crossed.pk, ongoing.pk})
        self.assertNotIn(later.pk, got)

    def test_breakdown_counts(self):
        self._event(status=AudioEvent.STATUS_COMPLETED)
        self._event(start_ts=T0 + 100, status=AudioEvent.STATUS_FAILED)
        resp = self.client.get(self.URL)
        got = {b["status"]: b["n"] for b in resp.context_data["breakdown"]}
        self.assertEqual(got["completed"], 1)
        self.assertEqual(got["failed"], 1)

    def test_pagination_keeps_filters(self):
        for i in range(30):
            self._event(start_ts=T0 + i, cam=self.cam2)
        resp = self.client.get(f"{self.URL}?cam_id={self.cam2.id}")
        self.assertEqual(resp.context_data["page"].paginator.count, 30)
        self.assertEqual(resp.context_data["querystring"], f"cam_id={self.cam2.id}&")
        self.assertIn(b"page=2", resp.content)


# ---------------------------------------------------------------------------
# §8.2 音频事件详情
# ---------------------------------------------------------------------------
class AudioEventDetailTest(_Base):
    def test_404(self):
        resp = self.client.get("/sound-events/999999/")
        self.assertEqual(resp.status_code, 404)

    def test_renders_timeline_segments_and_description(self):
        ev = self._event()
        AudioEventSegment.objects.create(
            audio_event=ev, sequence=1, start_offset=0.0, end_offset=8.0,
            audio_path="", description_status=AudioEvent.DESC_COMPLETED,
            description_json={"summary": "seg1"},
        )
        resp = self.client.get(f"/sound-events/{ev.pk}/")
        content = resp.content.decode("utf-8")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("哭声", content)                # 标签中文
        self.assertIn("静音点时间轴", content)
        self.assertIn("seg1", content)                # 分段描述
        self.assertIn("婴儿哭声", content)             # 结构化描述
        self.assertIn("raw-desc-output", content)     # 描述模型原始返回
        self.assertIn("Qwen3-ASR-1.7B", content)      # 产出描述的模型名

    def test_no_audio_path_hides_player(self):
        ev = self._event(audio_path="")
        resp = self.client.get(f"/sound-events/{ev.pk}/")
        self.assertIn("该事件没有可用录音文件".encode(), resp.content)
        self.assertNotIn(b"<audio", resp.content)

    def test_state_links_and_delivery_rows(self):
        ev = self._event()
        prompt = VLMPromptConfig.objects.create(
            name=_name("p"), prompt="p", positive_keyword="是",
        )
        state = VLMCheckState.objects.create(
            camera=self.cam, prompt_config=prompt, ts_list=[T0], window_sec=3, hit=True,
        )
        VLMCheckStateAudioEvent.objects.create(
            vlm_state=state, audio_event=ev,
            snapshot_json={"result": "SATISFIED", "condition": "cry", "event_ids": [ev.pk]},
        )
        target = NotifyTarget.objects.create(
            name=_name("tg"), kind="mobile_app", target_id="notify.mobile_app_x",
        )
        PromptNotificationDelivery.objects.create(
            prompt_config=prompt, vlm_state=state, audio_event=ev,
            notify_target=target, audio_state="satisfied",
            condition="audio_rule", delivered=True,
        )
        resp = self.client.get(f"/sound-events/{ev.pk}/")
        content = resp.content.decode("utf-8")
        self.assertIn(f"/events/{state.pk}/", content)
        self.assertIn("audio_rule", content)
        self.assertIn("satisfied", content)
        self.assertIn(target.name, content)

    def test_no_links_shows_hint(self):
        ev = self._event()
        resp = self.client.get(f"/sound-events/{ev.pk}/")
        self.assertIn("没有关联的 Prompt 命中".encode(), resp.content)


# ---------------------------------------------------------------------------
# 音频文件服务（只服务 MEDIA_ROOT 内）
# ---------------------------------------------------------------------------
class AudioFileServingTest(_Base):
    def setUp(self):
        super().setUp()
        self.media = Path(tempfile.mkdtemp(prefix="bc_audio_test_"))
        self.addCleanup(shutil.rmtree, str(self.media), True)
        self._media_patcher = override_settings(MEDIA_ROOT=str(self.media))
        self._media_patcher.enable()
        self.addCleanup(self._media_patcher.disable)

    def _write(self, name: str, data: bytes = b"fLaC-fake") -> str:
        path = self.media / "audio_events" / "2026-01-01" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def test_event_file_served(self):
        ev = self._event(audio_path=self._write("1_100.flac"))
        resp = self.client.get(f"/sound-events/{ev.pk}/audio/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(resp.streaming_content), b"fLaC-fake")

    def test_event_file_missing_on_disk_404(self):
        ev = self._event(audio_path=str(self.media / "audio_events" / "nope.flac"))
        resp = self.client.get(f"/sound-events/{ev.pk}/audio/")
        self.assertEqual(resp.status_code, 404)

    def test_event_file_empty_path_404(self):
        ev = self._event(audio_path="")
        resp = self.client.get(f"/sound-events/{ev.pk}/audio/")
        self.assertEqual(resp.status_code, 404)

    def test_event_file_outside_media_root_404(self):
        outside = Path(tempfile.mkdtemp(prefix="bc_audio_out_"))
        self.addCleanup(shutil.rmtree, str(outside), True)
        f = outside / "x.flac"
        f.write_bytes(b"nope")
        ev = self._event(audio_path=str(f))
        resp = self.client.get(f"/sound-events/{ev.pk}/audio/")
        self.assertEqual(resp.status_code, 404)

    def test_segment_file_served(self):
        ev = self._event()
        AudioEventSegment.objects.create(
            audio_event=ev, sequence=2, start_offset=0, end_offset=5,
            audio_path=self._write("1_100_seg2.flac", b"seg2"),
        )
        resp = self.client.get(f"/sound-events/{ev.pk}/segments/2/audio/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(b"".join(resp.streaming_content), b"seg2")

    def test_segment_file_wrong_sequence_404(self):
        ev = self._event()
        resp = self.client.get(f"/sound-events/{ev.pk}/segments/9/audio/")
        self.assertEqual(resp.status_code, 404)


# ---------------------------------------------------------------------------
# §8.3 手动清理
# ---------------------------------------------------------------------------
class AudioCleanupViewTest(_Base):
    URL = "/sound-detection/cleanup/"

    def test_get_preview_counts(self):
        log = self._log()
        ev = self._event()
        self._log(cam=self.cam2, start_ts=T0 + 10_000)
        resp = self.client.get(f"{self.URL}?cam_id={self.cam.id}")
        preview = resp.context_data["preview"]
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(preview["log_count"], 1)
        self.assertEqual(preview["event_count"], 1)
        self.assertEqual(preview["events"][0].pk, ev.pk)
        self.assertIsNotNone(log.pk)

    def test_run_requires_post(self):
        resp = self.client.get("/sound-detection/cleanup/run/")
        self.assertEqual(resp.status_code, 405)

    def test_run_posts_thread_and_redirects(self):
        self._log()
        self._event()
        with patch("apps.dashboard.audio_cleanup._run_cleanup_in_thread") as mock_thread:
            resp = self.client.post("/sound-detection/cleanup/run/", {"cam_id": self.cam.id})
        self.assertEqual(resp.status_code, 302)
        mock_thread.assert_called_once()
        args = mock_thread.call_args[0][0]
        self.assertEqual(args["cam_id"], str(self.cam.id))

    def test_orphans_requires_post(self):
        resp = self.client.get("/sound-detection/cleanup/orphans/")
        self.assertEqual(resp.status_code, 405)

    def test_orphans_post_reports_count(self):
        with patch(
            "apps.dashboard.audio_cleanup.purge_orphan_files", return_value=(3, 1),
        ):
            resp = self.client.post("/sound-detection/cleanup/orphans/", follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("失败 1 个".encode(), resp.content)


class AudioCleanupLogicTest(_Base):
    """真删路径（不 patch 线程）：库行 + 文件。"""

    def setUp(self):
        super().setUp()
        self.media = Path(tempfile.mkdtemp(prefix="bc_audio_clean_"))
        self.addCleanup(shutil.rmtree, str(self.media), True)
        self._media_patcher = override_settings(MEDIA_ROOT=str(self.media))
        self._media_patcher.enable()
        self.addCleanup(self._media_patcher.disable)

    def _flac(self, name: str) -> str:
        path = self.media / "audio_events" / "2026-01-01" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 10)
        return str(path)

    def test_cleanup_deletes_logs_events_and_files(self):
        ev_path = self._flac("1_100.flac")
        seg_path = self._flac("1_100_seg1.flac")
        keep_path = self._flac("2_999.flac")

        self._log(start_ts=T0)
        ev = self._event(start_ts=T0, audio_path=ev_path)
        AudioEventSegment.objects.create(
            audio_event=ev, sequence=1, start_offset=0, end_offset=5, audio_path=seg_path,
        )
        # 另一路：不在筛选范围内，必须存活
        self._log(cam=self.cam2, start_ts=T0 + 10_000)
        keep_ev = self._event(cam=self.cam2, start_ts=T0 + 10_000, audio_path=keep_path)

        result = _cleanup_by_filters({"cam_id": str(self.cam.id)})

        self.assertEqual(result["logs_deleted"], 1)
        self.assertGreaterEqual(result["events_deleted"], 1)
        self.assertEqual(result["files_deleted"], 2)
        self.assertEqual(result["files_failed"], 0)
        self.assertFalse(Path(ev_path).exists())
        self.assertFalse(Path(seg_path).exists())
        self.assertTrue(Path(keep_path).exists())
        self.assertTrue(AudioEvent.objects.filter(pk=keep_ev.pk).exists())
        self.assertTrue(SoundDetectionLog.objects.filter(camera=self.cam2).exists())

    def test_cleanup_missing_file_is_not_failure(self):
        self._log(start_ts=T0)
        self._event(start_ts=T0, audio_path=str(self.media / "audio_events" / "gone.flac"))
        result = _cleanup_by_filters({"cam_id": str(self.cam.id)})
        self.assertEqual(result["files_deleted"], 0)
        self.assertEqual(result["files_failed"], 0)

    def test_cleanup_no_match_is_noop(self):
        result = _cleanup_by_filters({"cam_id": "999999"})
        self.assertEqual(result["logs_deleted"], 0)
        self.assertEqual(result["events_deleted"], 0)

    def test_purge_orphans_removes_unreferenced_only(self):
        referenced = self._flac("1_100.flac")
        orphan = self._flac("orphan.flac")
        self._event(start_ts=T0, audio_path=referenced)

        removed, failed = purge_orphan_files()

        self.assertEqual(removed, 1)
        self.assertEqual(failed, 0)
        self.assertTrue(Path(referenced).exists())
        self.assertFalse(Path(orphan).exists())

    def test_purge_orphans_keeps_segment_files(self):
        seg_path = self._flac("1_100_seg1.flac")
        ev = self._event(start_ts=T0)
        AudioEventSegment.objects.create(
            audio_event=ev, sequence=1, start_offset=0, end_offset=5, audio_path=seg_path,
        )
        removed, _failed = purge_orphan_files()
        self.assertEqual(removed, 0)
        self.assertTrue(Path(seg_path).exists())

    def test_cleanup_view_shows_orphans(self):
        self._flac("1_100.flac")
        resp = self.client.get("/sound-detection/cleanup/")
        self.assertEqual(resp.context_data["orphans"]["count"], 1)


# ===========================================================================
# 事件详情页「窗口期声音判定」卡片（2026-09-14 新增）
# ===========================================================================
class EventDetailAudioLinksTest(TestCase):
    """``dashboard.views._audio_links``：三态映射 + 参考音频事件 + 阴性留痕。"""

    def _state(self):
        cam = Camera.objects.create(
            name=_name("cam"), source_type="onvif", is_active=False,
        )
        prompt = VLMPromptConfig.objects.create(
            name=_name("p"), prompt="x", positive_keyword="是",
        )
        state = VLMCheckState.objects.create(
            camera=cam, prompt_config=prompt, ts_list=[T0], window_sec=1,
            hit=True, status="在床",
        )
        return cam, state

    def test_positive_link_references_audio_event(self):
        from apps.dashboard.views import _audio_links

        cam, state = self._state()
        ev = AudioEvent.objects.create(camera=cam, started_at_ts=T0, duration_sec=9.5)
        VLMCheckStateAudioEvent.objects.create(
            vlm_state=state, audio_event=ev,
            snapshot_json={
                "result": "SATISFIED", "condition": "cry", "rule_enabled": True,
                "covered": True, "service_available": True, "camera_status": "ready",
                "event_count": 2, "cry_count": 2, "speech_count": 0,
                "window_start_ts": T0 - 600, "window_end_ts": T0,
            },
        )

        rows = _audio_links(state)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "声音条件满足")
        self.assertEqual(rows[0]["condition"], "cry")
        self.assertTrue(rows[0]["covered"])
        self.assertEqual(rows[0]["audio_event"].pk, ev.pk)

    def test_no_rule_row_is_marked_unconfigured(self):
        """``no_rule`` 行不能把 UNKNOWN 的默认占位值当真实状态展示（否则显示"音频服务未起"）。"""
        from apps.dashboard.views import _audio_links

        _cam, state = self._state()
        VLMCheckStateAudioEvent.objects.create(
            vlm_state=state, audio_event=None,
            snapshot_json={
                "result": "UNKNOWN", "reason": "no_rule",
                "rule_enabled": False, "condition": "",
                "covered": False, "service_available": False,
            },
        )

        row = _audio_links(state)[0]
        self.assertFalse(row["configured"])
        self.assertEqual(row["reason_label"], "未配置声音条件")

    def test_negative_link_has_no_reference_event(self):
        from apps.dashboard.views import _audio_links

        _cam, state = self._state()
        VLMCheckStateAudioEvent.objects.create(
            vlm_state=state, audio_event=None,
            snapshot_json={
                "result": "UNKNOWN", "covered": False,
                "reason": "coverage_incomplete",
            },
        )

        rows = _audio_links(state)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "无法判定")
        self.assertFalse(rows[0]["covered"])
        self.assertIsNone(rows[0]["audio_event"])
        self.assertEqual(rows[0]["reason"], "coverage_incomplete")

    def test_detail_view_exposes_audio_links(self):
        _cam, state = self._state()
        VLMCheckStateAudioEvent.objects.create(
            vlm_state=state, audio_event=None,
            snapshot_json={"result": "NOT_SATISFIED"},
        )

        resp = self.client.get(f"/events/{state.pk}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.context_data["audio_links"]), 1)

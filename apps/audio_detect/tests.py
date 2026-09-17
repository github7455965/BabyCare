"""音频线单元测试。

Phase 1：环形缓冲 / 覆盖完整性 / ONVIF 辅助函数。
Phase 2：标签映射 / 两级聚合 / 共识三态 / 稀疏落库策略。
Phase 4：分段边界 / 描述解析与校验（transcript + json 两种方言）/ 跨分段聚合 /
         重试与脏数据隔离。

刻意**不依赖 ffmpeg、不依赖摄像头、不加载 TF/torch/soundfile**——采集与模型的
端到端验证走 `manage.py audio_worker`（见 工程计划.md 步骤 15 的验证清单）。
模型侧用 :class:`_FakeDetector` 替身、描述侧用 :class:`_FakeAudioDescClient` + 内存
FLAC 替身，保证单测能在主 venv（无 tensorflow/soundfile）里跑。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from apps.audio_detect.assembler import AssemblerConfig, EventAssembler
from apps.audio_detect.capture import (
    AudioCapture,
    CaptureConfig,
    CaptureSnapshot,
    PcmRingBuffer,
    terminate_process_tree,
)
from apps.audio_detect.consensus import (
    DECISION_DEGRADED,
    DECISION_NEGATIVE,
    DECISION_POSITIVE,
    LOG_NEAR_THRESHOLD,
    LOG_POSITIVE,
    LOG_STATE_CHANGE,
    REASON_ALL_ABSTAIN,
    REASON_FALLBACK,
    REASON_HIGH_CONFIDENCE,
    REASON_SINGLE_SIDE_ONLY,
    STATE_ABSTAIN,
    STATE_NEGATIVE,
    STATE_POSITIVE,
    STRATEGY_AND,
    STRATEGY_OR,
    ConsensusEngine,
    LabelVerdict,
    Thresholds,
    WindowVerdict,
)
from apps.audio_detect.description_schema import (
    aggregate_descriptions,
    parse_description_json,
    validate_description,
)
from apps.audio_detect.describer import (
    DESC_PROVIDER_JSON,
    DESC_PROVIDER_TRANSCRIPT,
    NO_SPEECH_TEXT,
    DescriberConfig,
    DescribeService,
    _RawStore,          # 私有容器：单测直接注入小上限验证淘汰/截断
    build_prompt,
    build_prompt_for,
    build_transcript_prompt,
    normalize_provider,
    normalize_transcript,
    segment_audio_path,
    strip_asr_wrapper,
)
from apps.audio_detect.detector import (
    FRAME_AGG_MAX,
    FRAME_AGG_MEAN,
    ModelOutput,
    YamnetModel,
    aggregate_frames,
    top_k_names,
)
from apps.audio_detect.inference import InferenceConfig, InferenceService
from apps.audio_detect.label_map import (
    DEFAULT_RAW_LABELS,
    MODEL_PANNS,
    MODEL_YAMNET,
    MODELS,
    AudioLabelMap,
    LabelMapError,
    read_class_map,
)
from apps.audio_detect.manager import AudioCaptureManager
from apps.audio_detect.models import (
    AudioEvent,
    AudioEventSegment,
    AudioRuntimeState,
    AudioServiceState,
    SoundDetectionLog,
)
from apps.audio_detect.audio_desc_client import (
    AudioDescClient,
    AudioDescNetworkError,
    AudioDescParseError,
    AudioDescTimeoutError,
    AudioDescUnreachableError,
)
from apps.audio_detect.segmenter import SegmenterConfig, plan_segments
from apps.audio_detect.worker_lock import (
    WorkerLockError,
    _pid_alive,        # 私有：单测直接验证跨平台存活判断
    acquire_worker_slot,
    release_worker_slot,
    touch_worker_slot,
)
from apps.audio_detect.silence import EnergyTracker, find_silence_ranges, rms_db
from apps.audio_detect.onvif_audio import (
    ffprobe_has_audio,
    inject_credentials,
    probe_audio_uri,
    redact_text,
    redact_url,
)


def _fake_camera(cam_id: int = 1):
    """AudioCapture 只用到这几个属性，不需要真的落库。"""
    return SimpleNamespace(
        id=cam_id,
        name=f"cam{cam_id}",
        onvif_host="127.0.0.1",
        onvif_port=80,
        onvif_username="admin",
    )


# ---------------------------------------------------------------------------
# 环形缓冲
# ---------------------------------------------------------------------------
class PcmRingBufferTests(SimpleTestCase):
    def test_append_and_snapshot(self):
        buf = PcmRingBuffer(10)
        buf.append(b"abcde")
        buf.append(b"fghij")
        self.assertEqual(buf.snapshot(), b"abcdefghij")
        self.assertEqual(len(buf), 10)

    def test_overflow_drops_oldest(self):
        buf = PcmRingBuffer(5)
        buf.append(b"abc")
        buf.append(b"defg")
        self.assertEqual(buf.snapshot(), b"cdefg")

    def test_empty_append_is_noop(self):
        buf = PcmRingBuffer(4)
        buf.append(b"")
        self.assertEqual(len(buf), 0)

    def test_clear_drops_everything(self):
        """断流重连要清空缓冲：不能让上一段流的旧 PCM 混进新流。"""
        buf = PcmRingBuffer(10)
        buf.append(b"abcde")
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.snapshot(), b"")
        buf.append(b"xy")
        self.assertEqual(buf.snapshot(), b"xy")

    def test_capacity_for_pre_roll(self):
        """5s × 16000Hz × 2B = 160000 字节（spec §3.3）。"""
        cfg = CaptureConfig(sample_rate=16000, pre_roll_sec=5)
        buf = PcmRingBuffer(cfg.sample_rate * 2 * cfg.pre_roll_sec)
        self.assertEqual(buf.capacity, 160000)


# ---------------------------------------------------------------------------
# 采集参数
# ---------------------------------------------------------------------------
class CaptureConfigTests(SimpleTestCase):
    def test_defaults_match_spec(self):
        cfg = CaptureConfig()
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.pre_roll_sec, 5)

    def test_from_settings(self):
        cfg = CaptureConfig.from_settings()
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertGreater(cfg.stall_sec, 0)
        self.assertGreaterEqual(cfg.reconnect_max_sec, cfg.reconnect_min_sec)


# ---------------------------------------------------------------------------
# 覆盖完整性（spec §3.3 / §7.1）
# ---------------------------------------------------------------------------
class CoverageTests(SimpleTestCase):
    def setUp(self):
        self.cap = AudioCapture(_fake_camera(), config=CaptureConfig())

    def test_not_ready_is_not_covered(self):
        now = int(time.time())
        self.assertFalse(self.cap.is_covered_for(now - 10))
        self.assertFalse(self.cap.is_covered_for(now + 10))

    def test_ready_only_covers_windows_after_coverage_start(self):
        # _mark_ready 把 coverage_ok_from_ts 设为 now：之前的窗口不算全覆盖
        self.cap._mark_ready()
        now = int(time.time())
        self.assertFalse(self.cap.is_covered_for(now - 10))
        self.assertTrue(self.cap.is_covered_for(now + 1))

    def test_down_period_advances_coverage(self):
        """掉线时段必须持续前推，否则会被误判成"覆盖完整"。"""
        with self.cap._lock:
            self.cap._coverage_ok_from_ts = int(time.time()) - 3600
        self.cap._touch_coverage_while_down()
        with self.cap._lock:
            cov = self.cap._coverage_ok_from_ts
        self.assertGreaterEqual(cov, int(time.time()) - 1)

    def test_touch_is_noop_when_ready(self):
        self.cap._mark_ready()
        with self.cap._lock:
            before = self.cap._coverage_ok_from_ts
        self.cap._touch_coverage_while_down()
        with self.cap._lock:
            self.assertEqual(self.cap._coverage_ok_from_ts, before)

    def test_stale_packet_breaks_coverage(self):
        """进程活着但流死了（假活）→ 覆盖不完整。"""
        self.cap._mark_ready()
        with self.cap._lock:
            self.cap._coverage_ok_from_ts = 0
            self.cap._last_packet_mono = time.monotonic() - 999
        self.assertFalse(self.cap.is_covered_for(int(time.time()) + 1))

    def test_pre_roll_snapshot(self):
        self.cap._ring.append(b"\x00\x01\x02")
        self.assertEqual(self.cap.pre_roll_snapshot(), b"\x00\x01\x02")

    def test_snapshot_defaults(self):
        snap = self.cap.snapshot()
        self.assertEqual(snap.bytes_total, 0)
        self.assertIsNone(snap.last_packet_at)
        self.assertIn("status=", snap.as_log())


# ---------------------------------------------------------------------------
# 进程回收
# ---------------------------------------------------------------------------
class ProcessTreeTests(SimpleTestCase):
    def test_none_is_noop(self):
        terminate_process_tree(None)   # 不抛即通过


# ---------------------------------------------------------------------------
# ONVIF 辅助函数
# ---------------------------------------------------------------------------
class OnvifAudioHelperTests(SimpleTestCase):
    def test_inject_credentials_quotes_special_chars(self):
        self.assertEqual(
            inject_credentials("rtsp://1.2.3.4/stream", "admin", "p@ss"),
            "rtsp://admin:p%40ss@1.2.3.4/stream",
        )

    def test_inject_credentials_keeps_existing_userinfo(self):
        url = "rtsp://admin:x@1.2.3.4/stream"
        self.assertEqual(inject_credentials(url, "admin", "y"), url)

    def test_inject_credentials_non_rtsp_untouched(self):
        self.assertEqual(inject_credentials("http://x/y", "a", "b"), "http://x/y")

    def test_redact_url(self):
        self.assertEqual(
            redact_url("rtsp://admin:secret@1.2.3.4/s"),
            "rtsp://admin:***@1.2.3.4/s",
        )
        self.assertEqual(redact_url("rtsp://1.2.3.4/s"), "rtsp://1.2.3.4/s")

    def test_redact_text_scrubs_ffmpeg_stderr(self):
        """ffmpeg stderr 里的明文密码必须脱敏（会进日志 + last_error + 控制页）。

        回归：现场 `Error opening input file rtsp://admin:<password>@...` 被原样写进
        `data/audio_worker.log` 和 `AudioRuntimeState.last_error`。
        """
        raw = (
            "Error opening input file rtsp://admin:Secret123@192.168.7.66:554/stream1. "
            "| [in#0] rtsp://user:pw@10.0.0.1/live failed"
        )
        out = redact_text(raw)
        self.assertNotIn("Secret123", out)
        self.assertNotIn("user:pw@", out)
        self.assertIn("rtsp://admin:***@192.168.7.66:554/stream1", out)
        self.assertIn("rtsp://user:***@10.0.0.1/live", out)

    def test_redact_text_noop_without_credentials(self):
        self.assertEqual(redact_text(""), "")
        self.assertEqual(redact_text("plain text"), "plain text")
        self.assertEqual(redact_text("rtsp://1.2.3.4/s"), "rtsp://1.2.3.4/s")

    def test_ffprobe_missing_binary_returns_none(self):
        """探测失败 ≠ 没有音频：必须返回 None 而不是 False。"""
        self.assertIsNone(
            ffprobe_has_audio("rtsp://x/y", ffprobe_bin="no-such-binary-xyz"),
        )


# ---------------------------------------------------------------------------
# ONVIF profile 选择（2026-09-15：音频线改走子码流）
# ---------------------------------------------------------------------------
def _prof(token, w, h, audio=True):
    """ONVIF profile 替身；``w=None`` → 纯音频 profile（无 VideoEncoderConfiguration）。"""
    vec = None if w is None else SimpleNamespace(
        Encoding="H264", Resolution=SimpleNamespace(Width=w, Height=h),
    )
    aec = SimpleNamespace(Encoding="G711") if audio else None
    return SimpleNamespace(
        token=token, AudioEncoderConfiguration=aec, VideoEncoderConfiguration=vec,
    )


class _FakeMedia:
    def __init__(self, profiles, uri="rtsp://1.2.3.4:554/stream2"):
        self._profiles = profiles
        self._uri = uri
        self.asked_token = None

    def GetProfiles(self):
        return self._profiles

    def create_type(self, _name):
        return SimpleNamespace()

    def GetStreamUri(self, req):
        self.asked_token = req.ProfileToken
        return SimpleNamespace(Uri=self._uri)


class OnvifProfileSelectionTests(SimpleTestCase):
    """带音频的 profile 里取"视频最省"的 —— 旧行为是取 ``profiles[0]``（主码流）。

    动因：两台摄像头每个 profile 都带音频，旧规则等价于永远拉主码流
    （2304×1296/2048kbps），而音频侧 `-vn` 把视频轨丢掉 —— 白拉一份主码流。
    """

    def _probe(self, profiles, ffprobe=None):
        media = _FakeMedia(profiles)
        patcher = patch(
            "onvif.ONVIFCamera",
            lambda *a, **kw: SimpleNamespace(create_media_service=lambda: media),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        if ffprobe is not None:
            p2 = patch(
                "apps.audio_detect.onvif_audio.ffprobe_has_audio",
                return_value=ffprobe,
            )
            p2.start()
            self.addCleanup(p2.stop)

        res = probe_audio_uri("1.2.3.4", 80, "admin", password="pw")
        return res, media

    def test_picks_smallest_video_among_audio_profiles(self):
        """两个都带音频 → 选 640x480（子码流），而不是 profiles[0] 的 2304x1296。"""
        res, media = self._probe([
            _prof("profile_1", 2304, 1296),
            _prof("profile_2", 640, 480),
        ])
        self.assertTrue(res.ok)
        self.assertEqual(res.profile_token, "profile_2")
        self.assertEqual(media.asked_token, "profile_2")     # GetStreamUri 用的是它
        self.assertEqual(res.audio_encoding, "G711")          # 音轨不受影响

    def test_smaller_wins_regardless_of_order(self):
        """像素更小的胜出，与出现在列表里的先后无关。"""
        res, _ = self._probe([
            _prof("profile_2", 640, 480),
            _prof("profile_1", 1920, 1080),
        ])
        self.assertEqual(res.profile_token, "profile_2")

    def test_profiles0_without_audio_falls_back_to_audio_profile(self):
        """profiles[0] 没声明音频 → 取另一个带音频的（即便它视频更大）。"""
        res, _ = self._probe([
            _prof("profile_1", 2304, 1296, audio=False),
            _prof("profile_2", 1280, 720),
        ])
        self.assertEqual(res.profile_token, "profile_2")

    def test_pure_audio_profile_wins(self):
        """纯音频 profile（无视频配置）优先于任何带视频的 —— 零视频负担。"""
        res, _ = self._probe([
            _prof("profile_1", 640, 480),
            _prof("audio_only", None, None),
        ])
        self.assertEqual(res.profile_token, "audio_only")

    def test_tie_keeps_first(self):
        """像素并列 → 取靠前者（``min`` 稳定）。"""
        res, _ = self._probe([
            _prof("profile_1", 640, 480),
            _prof("profile_2", 640, 480),
        ])
        self.assertEqual(res.profile_token, "profile_1")

    def test_no_audio_anywhere_uses_profiles0_and_ffprobe(self):
        """全都没有音频声明 → 退回 profiles[0] 走 ffprobe 兜底（探到音轨 → ok）。"""
        res, media = self._probe(
            [_prof("profile_1", 2304, 1296, audio=False)], ffprobe=True,
        )
        self.assertTrue(res.ok)
        self.assertTrue(res.used_ffprobe_fallback)
        self.assertEqual(res.profile_token, "profile_1")
        self.assertEqual(media.asked_token, "profile_1")

    def test_no_audio_confirmed_marks_no_audio(self):
        res, _ = self._probe(
            [_prof("profile_1", 2304, 1296, audio=False)], ffprobe=False,
        )
        self.assertTrue(res.no_audio)
        self.assertFalse(res.ok)

    def test_diag_profiles_carry_resolution(self):
        """诊断字段带分辨率，便于事后核对"为什么选它"。"""
        res, _ = self._probe([
            _prof("profile_1", 2304, 1296),
            _prof("profile_2", 640, 480),
        ])
        self.assertEqual(res.profiles[0]["resolution"], "2304x1296")
        self.assertEqual(res.profiles[1]["resolution"], "640x480")


# ===========================================================================
# Phase 2：标签映射（spec §4.1/§4.2）
# ===========================================================================
def _raw_labels():
    """DEFAULT_RAW_LABELS 的深拷贝（frozen dataclass 里放的是可变 dict）。"""
    import copy

    return copy.deepcopy(DEFAULT_RAW_LABELS)


def _real_class_names():
    from django.conf import settings

    return {
        MODEL_YAMNET: read_class_map(settings.BABYCARE_AUDIO_YAMNET_CLASS_MAP),
        MODEL_PANNS: read_class_map(settings.BABYCARE_AUDIO_PANNS_CLASS_MAP),
    }


class LabelMapTests(SimpleTestCase):
    """映射表的精确匹配与启动强校验。"""

    def test_vendored_class_maps_exist_and_validate(self):
        """默认映射表必须能在 vendor 的两份类表上通过（spec §4.2）。"""
        lm = AudioLabelMap.from_settings()
        self.assertEqual(lm.enabled_labels(), ("cry", "speech"))
        self.assertEqual(lm.disabled_labels(), {})
        for model in MODELS:
            self.assertEqual(len(lm.raw_indices(model, "cry")), 3)
            self.assertEqual(len(lm.raw_indices(model, "speech")), 1)

    def test_baby_crying_is_not_a_real_class(self):
        """早期文档写的 `Baby crying` 不存在，必须被校验拦下。"""
        raw = _raw_labels()
        raw["cry"][MODEL_YAMNET] = ("Baby crying",)
        with self.assertRaises(LabelMapError) as ctx:
            AudioLabelMap(
                raw_labels=raw, class_names=_real_class_names(),
            ).validated()
        self.assertIn("Baby crying", str(ctx.exception))

    def test_exact_match_rejects_substring_only_hit(self):
        """`Whimper` 不能被 `Whimper (dog)` 子串命中。"""
        raw = _raw_labels()
        raw["cry"][MODEL_YAMNET] = ("Whimper",)
        names = dict(_real_class_names())
        names[MODEL_YAMNET] = ["Whimper (dog)", "Speech"]
        with self.assertRaises(LabelMapError) as ctx:
            AudioLabelMap(raw_labels=raw, class_names=names).validated()
        msg = str(ctx.exception)
        self.assertIn("Whimper", msg)
        self.assertIn("Whimper (dog)", msg)     # 提示近似类名，便于排查

    def test_empty_side_disables_label_without_raising(self):
        """任一侧 raw_labels 为空 → 该标签 disabled，不报错（spec §4.2）。"""
        raw = _raw_labels()
        raw["speech"][MODEL_PANNS] = ()
        lm = AudioLabelMap(raw_labels=raw, class_names=_real_class_names()).validated()
        self.assertEqual(lm.enabled_labels(), ("cry",))
        self.assertIn("speech", lm.disabled_labels())

    def test_score_business_takes_max_over_mapped_classes_only(self):
        """先跨标签取 max，且**不受未映射类高分影响**（完整向量按索引取）。"""
        raw = {
            "cry": {
                MODEL_YAMNET: ("cry A", "cry B"),
                MODEL_PANNS: ("cry A", "cry B"),
            },
            "speech": {MODEL_YAMNET: ("speech A",), MODEL_PANNS: ("speech A",)},
        }
        names = {m: ["cry A", "cry B", "other", "speech A"] for m in MODELS}
        lm = AudioLabelMap(raw_labels=raw, class_names=names).validated()
        vector = [0.1, 0.7, 0.95, 0.2]      # "other" 0.95 不属于任何业务标签
        self.assertEqual(
            lm.score_business(vector, MODEL_YAMNET),
            {"cry": 0.7, "speech": 0.2},
        )

    def test_raw_indices_follow_class_map_order(self):
        """索引来自类表顺序，不能是排序后的位置。"""
        raw = {
            "cry": {MODEL_YAMNET: ("z",), MODEL_PANNS: ("z",)},
        }
        names = {m: ["a", "b", "z"] for m in MODELS}
        lm = AudioLabelMap(raw_labels=raw, class_names=names).validated()
        self.assertEqual(lm.raw_indices(MODEL_YAMNET, "cry"), (2,))

    def test_raw_score_table_reports_per_class_scores(self):
        raw = {
            "cry": {MODEL_YAMNET: ("cry A", "cry B"), MODEL_PANNS: ("cry A", "cry B")},
        }
        names = {m: ["cry A", "cry B"] for m in MODELS}
        lm = AudioLabelMap(raw_labels=raw, class_names=names).validated()
        table = lm.raw_score_table([0.2, 0.6], MODEL_YAMNET)
        self.assertEqual(table["cry"], {"cry A": 0.2, "cry B": 0.6})


# ===========================================================================
# Phase 2：两级聚合（spec §4.1）
# ===========================================================================
class FrameAggregationTests(SimpleTestCase):
    def setUp(self):
        import numpy as np

        self.matrix = np.array(
            [[0.1, 0.5], [0.9, 0.2], [0.3, 0.4], [0.2, 0.1]], dtype="float32",
        )

    def test_max_aggregation(self):
        import numpy as np

        out = aggregate_frames(self.matrix, FRAME_AGG_MAX)
        self.assertTrue(np.allclose(out, [0.9, 0.5]))

    def test_mean_aggregation(self):
        import numpy as np

        out = aggregate_frames(self.matrix, FRAME_AGG_MEAN)
        self.assertTrue(np.allclose(out, [0.375, 0.3]))

    def test_yamnet_shape_2s_window(self):
        """实测 2 秒 16kHz → (4, 521)，聚合成 (521,)。"""
        import numpy as np

        scores = np.random.default_rng(0).random((4, 521)).astype("float32")
        self.assertEqual(aggregate_frames(scores, FRAME_AGG_MAX).shape, (521,))

    def test_one_dimensional_passthrough(self):
        import numpy as np

        vector = np.array([0.1, 0.2], dtype="float32")
        self.assertEqual(aggregate_frames(vector).shape, (2,))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            aggregate_frames(self.matrix, "median")

    def test_bad_ndim_raises(self):
        import numpy as np

        with self.assertRaises(ValueError):
            aggregate_frames(np.zeros((2, 3, 4), dtype="float32"))

    def test_top_k_only_for_display(self):
        out = top_k_names([0.1, 0.9, 0.5], ["a", "b", "c"], k=2)
        self.assertEqual(out, [("b", 0.9), ("c", 0.5)])

    def test_unloaded_model_returns_not_ok(self):
        model = YamnetModel(model_dir="x", class_map="y")
        out = model.infer([0.0], AudioLabelMap(raw_labels={}, class_names={}))
        self.assertFalse(out.ok)
        self.assertIn("未加载", out.error)


# ===========================================================================
# Phase 2：共识与三态（spec §4.3）
# ===========================================================================
def _engine(strategy=STRATEGY_AND, fallback=False, thr=0.3, labels=("cry", "speech"),
            override=0.8):
    return ConsensusEngine(
        thresholds=Thresholds(
            values={
                MODEL_YAMNET: {"cry": thr, "speech": thr},
                MODEL_PANNS: {"cry": thr, "speech": thr},
            },
            high_confidence_override=override,
        ),
        labels=labels,
        strategy=strategy,
        single_model_fallback=fallback,
    )


class ConsensusTests(SimpleTestCase):
    def test_and_both_above_threshold_is_positive(self):
        v = _engine().evaluate({"cry": 0.5, "speech": 0.1}, {"cry": 0.4, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v.decision, DECISION_POSITIVE)
        self.assertEqual(v.positive_labels(), ("cry",))

    def test_and_single_side_only_is_negative_with_reason(self):
        v = _engine().evaluate({"cry": 0.5, "speech": 0.1}, {"cry": 0.1, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_NEGATIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_SINGLE_SIDE_ONLY)
        self.assertEqual(v.decision, DECISION_NEGATIVE)

    def test_high_confidence_single_side_is_positive(self):
        """and 共识下单侧 ≥ 0.8 → 也算阳性（spec §4.3.5）。"""
        v = _engine().evaluate({"cry": 0.9, "speech": 0.1}, {"cry": 0.1, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_HIGH_CONFIDENCE)
        self.assertEqual(v.decision, DECISION_POSITIVE)
        self.assertEqual(v.positive_labels(), ("cry",))

    def test_high_confidence_boundary_is_inclusive(self):
        """恰好等于覆盖阈值 → 判阳（用 ≥ 而非 >）。"""
        v = _engine().evaluate({"cry": 0.8, "speech": 0.1}, {"cry": 0.1, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_HIGH_CONFIDENCE)

    def test_below_high_confidence_stays_single_side_only(self):
        """0.79 < 0.8 → 维持原行为（negative + single_side_only）。"""
        v = _engine().evaluate({"cry": 0.79, "speech": 0.1}, {"cry": 0.1, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_NEGATIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_SINGLE_SIDE_ONLY)

    def test_high_confidence_override_can_be_disabled(self):
        """覆盖阈值 > 1 → 关闭该规则，0.95 单侧仍 negative。"""
        v = _engine(override=1.1).evaluate(
            {"cry": 0.95, "speech": 0.1}, {"cry": 0.1, "speech": 0.1},
        )
        self.assertEqual(v.labels["cry"].state, STATE_NEGATIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_SINGLE_SIDE_ONLY)

    def test_high_confidence_override_custom_value(self):
        """覆盖阈值可调：0.5 → 0.6 单侧即判阳。"""
        v = _engine(override=0.5).evaluate(
            {"cry": 0.6, "speech": 0.1}, {"cry": 0.1, "speech": 0.1},
        )
        self.assertEqual(v.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v.labels["cry"].reason, REASON_HIGH_CONFIDENCE)

    def test_or_single_side_is_positive(self):
        v = _engine(strategy=STRATEGY_OR).evaluate(
            {"cry": 0.5, "speech": 0.1}, {"cry": 0.1, "speech": 0.1},
        )
        self.assertEqual(v.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v.decision, DECISION_POSITIVE)

    def test_missing_model_is_abstain_not_negative(self):
        """abstain ≠ negative（spec §4.3.4）。"""
        v = _engine().evaluate(None, {"cry": 0.9, "speech": 0.1})
        self.assertEqual(v.labels["cry"].state, STATE_ABSTAIN)
        self.assertEqual(v.decision, DECISION_DEGRADED)
        self.assertTrue(v.degraded)

    def test_all_models_missing(self):
        v = _engine().evaluate(None, None)
        self.assertEqual(v.labels["cry"].reason, REASON_ALL_ABSTAIN)
        self.assertEqual(v.decision, DECISION_DEGRADED)
        self.assertEqual(v.failure_reason, REASON_ALL_ABSTAIN)

    def test_fallback_off_does_not_produce_positive(self):
        v = _engine(fallback=False).evaluate({"cry": 0.99, "speech": 0.1}, None)
        self.assertEqual(v.labels["cry"].state, STATE_ABSTAIN)
        self.assertEqual(v.positive_labels(), ())

    def test_fallback_on_raises_threshold(self):
        """单模型降级时阈值上浮 ×1.3：0.35 过不了 0.3×1.3=0.39。"""
        v = _engine(fallback=True).evaluate({"cry": 0.35, "speech": 0.1}, None)
        self.assertEqual(v.labels["cry"].state, STATE_NEGATIVE)
        self.assertAlmostEqual(v.labels["cry"].thresholds[MODEL_YAMNET], 0.39)

        v2 = _engine(fallback=True).evaluate({"cry": 0.5, "speech": 0.1}, None)
        self.assertEqual(v2.labels["cry"].state, STATE_POSITIVE)
        self.assertEqual(v2.labels["cry"].reason, REASON_FALLBACK)

    def test_label_missing_from_model_scores_is_abstain(self):
        v = _engine(labels=("cry",)).evaluate({"cry": 0.9}, {})
        self.assertEqual(v.labels["cry"].state, STATE_ABSTAIN)

    def test_window_log_reason_positive(self):
        eng = _engine()
        v = eng.evaluate({"cry": 0.5, "speech": 0.1}, {"cry": 0.5, "speech": 0.1})
        self.assertEqual(eng.log_reason(v), LOG_POSITIVE)

    def test_window_log_reason_near_threshold(self):
        """0.25 ≥ 0.3 × 0.7 = 0.21 → 落库（供事后调阈值）。"""
        eng = _engine()
        v = eng.evaluate({"cry": 0.25, "speech": 0.01}, {"cry": 0.01, "speech": 0.01})
        self.assertEqual(v.decision, DECISION_NEGATIVE)
        self.assertEqual(eng.log_reason(v), LOG_NEAR_THRESHOLD)

    def test_window_log_reason_empty_for_ordinary_window(self):
        eng = _engine()
        v = eng.evaluate({"cry": 0.05, "speech": 0.05}, {"cry": 0.05, "speech": 0.05})
        self.assertEqual(eng.log_reason(v), "")

    def test_consensus_payload_has_threshold_snapshot(self):
        eng = _engine()
        v = eng.evaluate({"cry": 0.5, "speech": 0.1}, {"cry": 0.5, "speech": 0.1})
        payload = eng.consensus_payload(v)
        self.assertEqual(payload["strategy"], STRATEGY_AND)
        self.assertEqual(payload["thresholds"][MODEL_YAMNET]["cry"], 0.3)
        self.assertEqual(payload["high_confidence_override"], 0.8)
        self.assertEqual(payload["labels"]["cry"]["state"], STATE_POSITIVE)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            _engine(strategy="majority")

    def test_as_log_shows_reason(self):
        """每窗日志要能看出判定原因，否则覆盖判阳与真双阳无法区分。"""
        v = _engine().evaluate({"cry": 0.9, "speech": 0.1}, {"cry": 0.1, "speech": 0.1})
        self.assertIn(REASON_HIGH_CONFIDENCE, v.as_log())

    def test_as_log_contains_both_model_scores(self):
        v = _engine().evaluate({"cry": 0.5, "speech": 0.1}, {"cry": 0.5, "speech": 0.1})
        text = v.as_log()
        self.assertIn("y=", text)
        self.assertIn("p=", text)
        self.assertIn("cry", text)


# ===========================================================================
# Phase 2：推理服务与稀疏落库（spec §5.1）
# ===========================================================================
def _model_output(model: str, scores: dict, ok: bool = True, error: str = ""):
    return ModelOutput(
        model=model, ok=ok, scores=scores, version="test-v1",
        error=error, elapsed_ms=1.0,
    )


class _FakeCapture:
    def __init__(self, cam_id: int = 1, samples=None):
        self.camera_id = cam_id
        self.sample_rate = 16000
        self.name = f"cam{cam_id}"
        self._samples = samples

    def snapshot(self):
        from datetime import datetime

        return CaptureSnapshot(
            status="ready", last_packet_at=datetime.now(), coverage_ok_from_ts=0,
            bytes_total=0, bytes_per_sec=0.0, profile_token="p1",
            audio_encoding="G711", error="",
        )

    def last_samples(self, seconds: float):
        import numpy as np

        if self._samples is not None:
            return self._samples
        return np.zeros(int(seconds * self.sample_rate), dtype="float32")


class _FakeDetector:
    def __init__(self, y_scores=None, p_scores=None, y_ok=True, p_ok=True):
        self._y = y_scores or {"cry": 0.05, "speech": 0.05}
        self._p = p_scores or {"cry": 0.05, "speech": 0.05}
        self._y_ok = y_ok
        self._p_ok = p_ok
        self.calls = 0

    def infer(self, audio):
        self.calls += 1
        return (
            _model_output(MODEL_YAMNET, self._y, self._y_ok, "" if self._y_ok else "boom"),
            _model_output(MODEL_PANNS, self._p, self._p_ok, "" if self._p_ok else "boom"),
        )

    @property
    def status(self):
        return {
            "yamnet": {"loaded": self._y_ok, "version": "test-v1", "fail_count": 0},
            "panns": {
                "loaded": self._p_ok, "version": "test-v1",
                "device": "cpu", "fail_count": 0,
            },
        }

    # InferenceService 只在 announce_models_ready 里用
    yamnet = SimpleNamespace(version="test-v1")
    panns = SimpleNamespace(version="test-v1")


def _service(detector, captures=None, enable_events=False, **cfg_kwargs):
    cfg = InferenceConfig(
        window_sec=cfg_kwargs.pop("window_sec", 2.0),
        hop_sec=cfg_kwargs.pop("hop_sec", 1.0),
        min_audio_ratio=cfg_kwargs.pop("min_audio_ratio", 0.9),
        log_every_sec=cfg_kwargs.pop("log_every_sec", 0.0),
    )
    return InferenceService(
        captures=captures if captures is not None else {1: _FakeCapture(1)},
        detector=detector,
        engine=_engine(),
        label_map=AudioLabelMap(
            raw_labels={
                "cry": {m: ("Crying, sobbing",) for m in MODELS},
                "speech": {m: ("Speech",) for m in MODELS},
            },
            class_names={m: ["Crying, sobbing", "Speech"] for m in MODELS},
        ).validated(),
        config=cfg,
        enable_events=enable_events,
    )


class InferencePersistenceTests(TestCase):
    """落库策略：只写阳性 / 近阈值 / 状态变化三类窗。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_audio_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def test_positive_window_is_persisted(self):
        svc = _service(_FakeDetector({"cry": 0.9, "speech": 0.1},
                                     {"cry": 0.9, "speech": 0.1}))
        outcome = svc.process_window(self.camera.id, None, 100, 102)
        self.assertEqual(outcome.log_reason, LOG_POSITIVE)
        row = SoundDetectionLog.objects.get()
        self.assertEqual(row.decision, SoundDetectionLog.DECISION_POSITIVE)
        self.assertEqual(row.log_reason, LOG_POSITIVE)
        self.assertEqual(row.window_start_ts, 100)
        self.assertEqual(row.camera_id, self.camera.id)
        self.assertIn("thresholds", row.consensus_payload)

    def test_ordinary_window_is_not_persisted(self):
        svc = _service(_FakeDetector())
        outcome = svc.process_window(self.camera.id, None, 100, 102)
        self.assertEqual(outcome.log_reason, "")
        self.assertEqual(SoundDetectionLog.objects.count(), 0)

    def test_near_threshold_window_is_persisted(self):
        svc = _service(_FakeDetector({"cry": 0.25, "speech": 0.01},
                                     {"cry": 0.01, "speech": 0.01}))
        outcome = svc.process_window(self.camera.id, None, 100, 102)
        self.assertEqual(outcome.log_reason, LOG_NEAR_THRESHOLD)
        self.assertEqual(SoundDetectionLog.objects.count(), 1)

    def test_state_change_is_persisted_once_per_status(self):
        svc = _service(_FakeDetector())
        svc.notify_state_change(self.camera.id, "ready")
        svc.notify_state_change(self.camera.id, "ready")
        svc.notify_state_change(self.camera.id, "capture_error")
        rows = SoundDetectionLog.objects.filter(log_reason=LOG_STATE_CHANGE)
        self.assertEqual(rows.count(), 2)

    def test_forget_camera_allows_new_state_change(self):
        svc = _service(_FakeDetector())
        svc.notify_state_change(self.camera.id, "ready")
        svc.forget_camera(self.camera.id)
        svc.notify_state_change(self.camera.id, "ready")
        self.assertEqual(
            SoundDetectionLog.objects.filter(log_reason=LOG_STATE_CHANGE).count(), 2,
        )

    def test_model_failure_marks_window_degraded(self):
        """单模型挂掉 → 严格模式判 abstain，**不产生阳性**。

        这种窗仍要落库：YAMNet 单侧 0.9 属于近阈值（≥ 0.3×0.7），正是排查
        "为什么只有一侧在高分"的关键数据（spec §5.1）。
        """
        svc = _service(_FakeDetector({"cry": 0.9, "speech": 0.1}, {}, p_ok=False))
        outcome = svc.process_window(self.camera.id, None, 100, 102)
        self.assertEqual(outcome.verdict.decision, DECISION_DEGRADED)
        self.assertEqual(outcome.verdict.positive_labels(), ())
        self.assertEqual(outcome.log_reason, LOG_NEAR_THRESHOLD)
        row = SoundDetectionLog.objects.get()
        self.assertEqual(row.decision, SoundDetectionLog.DECISION_DEGRADED)
        self.assertEqual(row.failure_reason, "model_abstain")

    def test_degraded_low_score_window_is_not_persisted(self):
        """模型挂了但分数很低（非近阈值）→ 不落库，只改状态。"""
        svc = _service(_FakeDetector({"cry": 0.01, "speech": 0.01}, {}, p_ok=False))
        outcome = svc.process_window(self.camera.id, None, 100, 102)
        self.assertEqual(outcome.verdict.decision, DECISION_DEGRADED)
        self.assertEqual(outcome.log_reason, "")
        self.assertEqual(SoundDetectionLog.objects.count(), 0)


class InferenceTickTests(TestCase):
    """窗口调度：缓冲没喂满 / 采集未就绪都要跳过，而不是当阴性结论。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_tick_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def test_tick_runs_all_captures(self):
        import numpy as np

        detector = _FakeDetector()
        caps = {
            1: _FakeCapture(1, np.zeros(32000, dtype="float32")),
            2: _FakeCapture(2, np.zeros(32000, dtype="float32")),
        }
        svc = _service(detector, captures=caps)
        outcomes = svc.tick(now_ts=1000.0)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(outcomes[0].window_end_ts, 1000)
        self.assertEqual(outcomes[0].window_start_ts, 998)

    def test_starved_window_is_skipped(self):
        """窗口音频不足 → 跳过（不能当阴性结论，spec §4.1）。"""
        import numpy as np

        detector = _FakeDetector()
        caps = {1: _FakeCapture(1, np.zeros(100, dtype="float32"))}
        svc = _service(detector, captures=caps)
        self.assertEqual(svc.tick(now_ts=1000.0), [])
        self.assertEqual(detector.calls, 0)
        self.assertEqual(svc.status["starved"], 1)

    def test_not_ready_capture_is_skipped(self):
        import numpy as np

        detector = _FakeDetector()
        cap = _FakeCapture(1, np.zeros(32000, dtype="float32"))
        cap.snapshot = lambda: CaptureSnapshot(
            status="capture_error", last_packet_at=None, coverage_ok_from_ts=0,
            bytes_total=0, bytes_per_sec=0.0, profile_token="", audio_encoding="",
            error="boom",
        )
        svc = _service(detector, captures={1: cap})
        self.assertEqual(svc.tick(now_ts=1000.0), [])
        self.assertEqual(detector.calls, 0)


# ===========================================================================
# Phase 2：模型状态叠加（spec §10：采集异常优先于模型状态）
# ===========================================================================
class _StubCapture:
    """只提供 _publish_one 用到的接口。"""

    def __init__(self, cam_id: int, status: str, error: str = ""):
        from datetime import datetime

        self.camera_id = cam_id
        self._snap = CaptureSnapshot(
            status=status, last_packet_at=datetime.now() if status == "ready" else None,
            coverage_ok_from_ts=0, bytes_total=0, bytes_per_sec=0.0,
            profile_token="p", audio_encoding="G711", error=error,
        )

    def snapshot(self):
        return self._snap


class ManagerStatusOverlayTests(TestCase):
    """web 侧只看 AudioRuntimeState——模型状态的叠加逻辑必须有测试。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_overlay_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def _manager(self) -> AudioCaptureManager:
        return AudioCaptureManager(enable_inference=False)

    def _publish(
        self,
        capture_status: str,
        model_override: str = "",
        status_override: str | None = None,
    ) -> str:
        mgr = self._manager()
        mgr._model_override = model_override
        mgr._model_error = "yamnet: boom"
        cap = _StubCapture(self.camera.id, capture_status)
        mgr._publish_one(cap, status_override=status_override)
        return AudioRuntimeState.objects.get(camera_id=self.camera.id).status

    def test_ready_with_healthy_models_stays_ready(self):
        self.assertEqual(self._publish("ready"), AudioRuntimeState.STATUS_READY)

    def test_ready_with_all_models_down_becomes_model_error(self):
        self.assertEqual(
            self._publish("ready", AudioRuntimeState.STATUS_MODEL_ERROR),
            AudioRuntimeState.STATUS_MODEL_ERROR,
        )
        row = AudioRuntimeState.objects.get(camera_id=self.camera.id)
        self.assertIn("boom", row.last_error)

    def test_ready_with_one_model_down_becomes_degraded(self):
        self.assertEqual(
            self._publish("ready", AudioRuntimeState.STATUS_DEGRADED),
            AudioRuntimeState.STATUS_DEGRADED,
        )

    def test_capture_error_wins_over_model_error(self):
        """采集异常优先：没有音频就谈不上检测（spec §10）。"""
        self.assertEqual(
            self._publish("capture_error", AudioRuntimeState.STATUS_MODEL_ERROR),
            AudioRuntimeState.STATUS_CAPTURE_ERROR,
        )

    def test_status_override_wins_over_everything(self):
        """stop()/移除摄像头时传的 STOPPED 必须原样落库。"""
        self.assertEqual(
            self._publish(
                "ready", AudioRuntimeState.STATUS_MODEL_ERROR,
                status_override=AudioRuntimeState.STATUS_STOPPED,
            ),
            AudioRuntimeState.STATUS_STOPPED,
        )


# ===========================================================================
# Phase 3：自适应静音（spec §6.3）
# ===========================================================================
class EnergyTrackerTests(SimpleTestCase):
    def _audio(self, amp: float, seconds: float = 2.0):
        import numpy as np

        return np.full(int(seconds * 16000), amp, dtype="float32")

    def test_threshold_none_until_enough_samples(self):
        """阈值是"分布的分位数"，样本太少时分位数没有意义 → 返回 None。"""
        t = EnergyTracker(window_sec=30, hop_sec=1.0)
        self.assertIsNone(t.threshold_db)
        for _ in range(4):
            t.push(self._audio(0.1))
            self.assertIsNone(t.threshold_db)
        t.push(self._audio(0.1))
        self.assertIsNotNone(t.threshold_db)     # 第 5 个样本起才有分布可谈

    def test_p20_of_distribution(self):
        """P20 分位数：分布已知时阈值应落在低分位。"""
        import math

        t = EnergyTracker(window_sec=30, hop_sec=1.0)
        # 25 个安静的 + 5 个响的 → P20 是安静电平
        for _ in range(25):
            t.push(self._audio(0.01))   # rms_db ≈ -40
        for _ in range(5):
            t.push(self._audio(0.3))    # rms_db ≈ -10
        thr = t.threshold_db
        self.assertIsNotNone(thr)
        self.assertLess(thr, -20.0)      # 接近安静侧（-40dB），不是响侧（-10dB）
        self.assertGreater(thr, -50.0)

    def test_window_rolls_oldest_out(self):
        t = EnergyTracker(window_sec=3, hop_sec=1.0)
        for _ in range(10):
            t.push(self._audio(0.5))
        self.assertEqual(len(t._samples), 3)   # 只留最近 3s


class FindSilenceRangesTests(SimpleTestCase):
    def test_quiet_middle_section(self):
        import numpy as np

        # 1s 响 + 1s 静 + 1s 响，阈值介于两者之间
        loud = np.full(16000, 0.5, dtype="float32")
        quiet = np.zeros(16000, dtype="float32")
        pcm = np.concatenate([loud, quiet, loud])
        ranges = find_silence_ranges(
            pcm, sample_rate=16000, threshold_db=-20.0, min_ms=500, frame_ms=100,
        )
        self.assertEqual(len(ranges), 1)
        self.assertAlmostEqual(ranges[0][0], 1.0, places=1)
        self.assertAlmostEqual(ranges[0][1], 2.0, places=1)

    def test_short_quiet_below_min_ms_is_dropped(self):
        import numpy as np

        loud = np.full(16000, 0.5, dtype="float32")
        quiet = np.zeros(16000 // 10, dtype="float32")   # 100ms < 500ms
        pcm = np.concatenate([loud, quiet, loud])
        self.assertEqual(
            find_silence_ranges(pcm, threshold_db=-20.0, min_ms=500, frame_ms=100), [],
        )

    def test_none_threshold_returns_empty(self):
        self.assertEqual(find_silence_ranges([0.0] * 100, threshold_db=None), [])

    def test_rms_db_zero_is_very_quiet(self):
        import numpy as np

        self.assertLess(rms_db(np.zeros(16000, dtype="float32")), -100.0)


# ===========================================================================
# Phase 3：事件状态机（照 p0_7 的 8 个合成场景，含边界）
# ===========================================================================
def _verdict(kind: str, labels=("cry",)):
    """构造三态 verdict：'1'=阳性 '0'=阴性 '-'=abstain。"""
    if kind == "1":
        v = WindowVerdict(decision=DECISION_POSITIVE)
        v.labels = {
            label: LabelVerdict(label=label, state=STATE_POSITIVE)
            for label in labels
        }
    elif kind == "-":
        v = WindowVerdict(decision=DECISION_DEGRADED)
        v.labels = {label: LabelVerdict(label=label, state=STATE_ABSTAIN) for label in labels}
    else:
        v = WindowVerdict(decision=DECISION_NEGATIVE)
        v.labels = {label: LabelVerdict(label=label, state=STATE_NEGATIVE) for label in labels}
    return v


class _AsmCapture:
    """capture 替身：预录快照 + tap 管理 + 手动喂 PCM。"""

    def __init__(self, pre_bytes: bytes = b""):
        self._taps: dict[str, object] = {}
        self._pre = pre_bytes
        self.sample_rate = 16000

    def pre_roll_snapshot(self) -> bytes:
        return self._pre

    def add_tap(self, key, buf):
        self._taps[key] = buf

    def remove_tap(self, key):
        return self._taps.pop(key, None)

    def feed(self, pcm: bytes) -> None:
        for buf in self._taps.values():
            buf.append(pcm)


class _FlacCollector:
    """flac_writer 替身：不写盘，记录调用（path, pcm, sample_rate）。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, path, pcm, sample_rate):
        self.calls.append((path, pcm, sample_rate))


def _assembler(cap, collector, cam_id: int = 1, worker_epoch: str = "", **cfg_kwargs):
    cfg = AssemblerConfig(
        hop_sec=1.0, window_sec=2.0, start_n=3, start_k=2,
        event_gap_sec=3.0, pre_roll_sec=5.0, post_roll_sec=3.0,
        sample_rate=16000,
        silence_window_sec=30.0, silence_percentile=20.0, silence_min_ms=500,
        **cfg_kwargs,
    )
    return EventAssembler(
        cam_id, cap, config=cfg, media_root="media-test", flac_writer=collector,
        worker_epoch=worker_epoch,
    )


# 场景时间轴基准：把 p0_7 的相对序列平移成合法 epoch（win_start 不为负，
# 避免 Windows 的 datetime.fromtimestamp 对负时间戳直接 OSError）
_BASE_TS = 10000


def _run_scenario(
    seq: str, cap, collector, cam_id: int = 1, feed_per_tick: bytes = b"",
    worker_epoch: str = "",
):
    """跑一个 p0_7 场景：'1'=阳性 '0'=阴性 '-'=abstain（hop=1s, window=2s）。"""
    asm = _assembler(cap, collector, cam_id, worker_epoch=worker_epoch)
    import numpy as np

    audio = np.zeros(32000, dtype="float32")
    for t, c in enumerate(seq):
        asm.on_window(_BASE_TS + t - 1, _BASE_TS + t + 1, _verdict(c), audio)
        if feed_per_tick:
            cap.feed(feed_per_tick)
    asm.stop()
    return asm


class EventStateMachineTests(TestCase):
    """8 个 p0_7 场景：事件数与区间必须与参考实现一致。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_asm_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def _events(self):
        return list(
            AudioEvent.objects.filter(camera_id=self.camera.id).order_by("started_at_ts")
        )

    def _run(self, seq: str):
        cap = _AsmCapture()
        collector = _FlacCollector()
        _run_scenario(
            seq, cap, collector, cam_id=self.camera.id, feed_per_tick=b"\x00" * 32000,
        )
        return cap, collector, self._events()

    def test_single_positive_window_no_event(self):
        _, _, events = self._run("0100000000")
        self.assertEqual(events, [])

    def test_two_of_three_starts_event(self):
        _, _, events = self._run("1010000000")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.started_at_ts, _BASE_TS - 1)   # 回溯到第一个阳性窗起点
        self.assertEqual(ev.ended_at_ts, _BASE_TS + 6)     # 末阳性 +3 + 后录 3
        self.assertEqual(ev.status, AudioEvent.STATUS_PENDING_DESCRIPTION)

    def test_continuous_positive_single_event(self):
        _, _, events = self._run("1111110000")
        self.assertEqual(len(events), 1)
        self.assertEqual(
            (events[0].started_at_ts, events[0].ended_at_ts),
            (_BASE_TS - 1, _BASE_TS + 9),
        )

    def test_gap_equals_threshold_same_event(self):
        """间隔恰好 = 3s（未超阈值）→ 同一事件（P0-7 边界场景）。"""
        _, _, events = self._run("1100110000")
        self.assertEqual(len(events), 1)
        self.assertEqual(
            (events[0].started_at_ts, events[0].ended_at_ts),
            (_BASE_TS - 1, _BASE_TS + 9),
        )

    def test_gap_over_threshold_splits_events(self):
        _, _, events = self._run("1100001110")
        self.assertEqual(len(events), 2)
        self.assertEqual(
            (events[0].started_at_ts, events[0].ended_at_ts),
            (_BASE_TS - 1, _BASE_TS + 5),
        )
        self.assertEqual(
            (events[1].started_at_ts, events[1].ended_at_ts),
            (_BASE_TS + 5, _BASE_TS + 12),
        )

    def test_short_single_window_is_missed(self):
        """≤1s 的极短声音被漏——2/3 规则的固有代价（P0-7 结论 3）。"""
        _, _, events = self._run("1000000000")
        self.assertEqual(events, [])

    def test_abstain_neither_advances_nor_breaks(self):
        _, _, events = self._run("1110--0000")
        self.assertEqual(len(events), 1)
        self.assertEqual(
            (events[0].started_at_ts, events[0].ended_at_ts),
            (_BASE_TS - 1, _BASE_TS + 6),
        )
        # abstain 窗不能伪装成静音结束 → degraded（spec §6.2）
        self.assertTrue(events[0].degraded)

    def test_long_silence_then_reappear(self):
        _, _, events = self._run("11000000001100000000")
        self.assertEqual(len(events), 2)
        self.assertEqual(
            (events[0].started_at_ts, events[0].ended_at_ts),
            (_BASE_TS - 1, _BASE_TS + 5),
        )
        self.assertEqual(
            (events[1].started_at_ts, events[1].ended_at_ts),
            (_BASE_TS + 9, _BASE_TS + 15),
        )


class EventRecordingTests(TestCase):
    """录音 / FLAC 落盘 / 静音区间 / degraded / 遗留回收。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_rec_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def test_recording_row_then_finalized_row(self):
        """启动时落 recording 行；关闭时回填 ended/路径/时长。"""
        cap = _AsmCapture(pre_bytes=b"\x01" * 32000)     # ring 里有 1s 预录
        collector = _FlacCollector()
        _run_scenario("1010000000", cap, collector, cam_id=self.camera.id,
                      feed_per_tick=b"\x01" * 32000)

        ev = AudioEvent.objects.get()
        self.assertEqual(ev.status, AudioEvent.STATUS_PENDING_DESCRIPTION)
        self.assertEqual(ev.audio_format, "flac")
        self.assertIn("audio_events", ev.audio_path)
        self.assertTrue(ev.audio_path.endswith(f"{self.camera.id}_{ev.started_at_ts}.flac"))
        self.assertGreater(ev.duration_sec, 0)
        # flac_writer 确实被调用；PCM 时长 = tap 内字节数 / (rate*2)
        self.assertEqual(len(collector.calls), 1)
        _path, pcm, sr = collector.calls[0]
        self.assertEqual(sr, 16000)
        self.assertGreater(len(pcm), 0)
        self.assertIsNotNone(ev.started_at)
        self.assertIsNotNone(ev.ended_at)
        self.assertEqual(ev.ended_at_ts, _BASE_TS + 6)   # 末阳性 +3 + 后录 3

    def test_detected_labels_multiple_labels(self):
        """同一事件同时含 cry 和 speech（spec §6.2）。"""
        cap = _AsmCapture()
        collector = _FlacCollector()
        asm = _assembler(cap, collector, self.camera.id)
        import numpy as np

        audio = np.zeros(32000, dtype="float32")
        seq = [
            ("1", ("cry",)), ("1", ("speech",)), ("1", ("cry", "speech")),
            ("0", ()), ("0", ()), ("0", ()), ("0", ()), ("0", ()),
            ("0", ()), ("0", ()),
        ]
        for t, (kind, labels) in enumerate(seq):
            asm.on_window(_BASE_TS + t - 1, _BASE_TS + t + 1, _verdict(kind, labels), audio)
        asm.stop()
        ev = AudioEvent.objects.get()
        self.assertEqual(sorted(ev.detected_labels), ["cry", "speech"])
        self.assertIn("positive_windows", ev.detected_labels["cry"])

    def test_capture_gap_marks_degraded(self):
        """窗与窗之间时间不连续（capture 断流）→ degraded + capture_gap。"""
        cap = _AsmCapture()
        collector = _FlacCollector()
        asm = _assembler(cap, collector, self.camera.id)
        import numpy as np

        audio = np.zeros(32000, dtype="float32")
        asm.on_window(_BASE_TS - 1, _BASE_TS + 1, _verdict("1"), audio)
        asm.on_window(_BASE_TS, _BASE_TS + 2, _verdict("1"), audio)
        asm.on_window(_BASE_TS + 1, _BASE_TS + 3, _verdict("1"), audio)
        # 断流 5s：下一个窗直接跳到 +8（窗口不连续）
        asm.on_window(_BASE_TS + 7, _BASE_TS + 9, _verdict("0"), audio)
        asm.on_window(_BASE_TS + 8, _BASE_TS + 10, _verdict("0"), audio)
        asm.stop()
        ev = AudioEvent.objects.get()
        self.assertTrue(ev.degraded)

    def test_recording_lead_sec_from_pre_roll(self):
        """ring 里有 5s 预录 → lead ≈ 事件起点 - 录音起点 > 0。"""
        cap = _AsmCapture(pre_bytes=b"\x01" * (16000 * 2 * 5))   # 满 5s
        collector = _FlacCollector()
        _run_scenario("1010000000", cap, collector, cam_id=self.camera.id)
        ev = AudioEvent.objects.get()
        # 启动于 win_end=3（t=2），录音起点 ≈ 3 - 5 = -2，事件起点 -1 → lead ≈ 1
        self.assertGreater(ev.recording_lead_sec, 0.5)
        self.assertLessEqual(ev.recording_lead_sec, 5.5)

    def test_recover_stale_events(self):
        """上次 worker 中断遗留的 recording 行 → 回收成 failed。"""
        row = AudioEvent.objects.create(
            camera_id=self.camera.id,
            status=AudioEvent.STATUS_RECORDING,
            started_at_ts=100,
        )
        n = EventAssembler.recover_stale_events()
        self.assertEqual(n, 1)
        row.refresh_from_db()
        self.assertEqual(row.status, AudioEvent.STATUS_FAILED)
        self.assertTrue(row.degraded)

    def test_created_event_carries_worker_epoch(self):
        """事件行记录产出它的 worker 租约 epoch（recover 据此限定范围）。"""
        cap = _AsmCapture()
        collector = _FlacCollector()
        _run_scenario(
            "1010000000", cap, collector, cam_id=self.camera.id,
            worker_epoch="epoch-asm",
        )
        self.assertEqual(AudioEvent.objects.get().worker_epoch, "epoch-asm")

    def test_silence_ranges_shifted_by_lead(self):
        """静音区间是相对**事件起点**的偏移（录音内偏移 - lead）。"""
        import numpy as np

        # ring 里 1s 预录（静音）+ 事件期间 2s 静音 → 静音段都在录音前部
        cap = _AsmCapture(pre_bytes=np.zeros(16000, dtype="float32").tobytes())
        collector = _FlacCollector()
        asm = _assembler(cap, collector, self.camera.id)
        audio = np.zeros(32000, dtype="float32")
        # 喂 10s 静音窗让 EnergyTracker 出阈值（全是 0 → P20 = -inf 很低）
        for _ in range(10):
            asm._energy.push(audio)
        asm.on_window(_BASE_TS - 1, _BASE_TS + 1, _verdict("1"), audio)
        cap.feed(np.zeros(16000, dtype="float32").tobytes())
        asm.on_window(_BASE_TS, _BASE_TS + 2, _verdict("1"), audio)
        cap.feed(np.zeros(16000, dtype="float32").tobytes())
        asm.on_window(_BASE_TS + 1, _BASE_TS + 3, _verdict("1"), audio)
        cap.feed(np.zeros(16000, dtype="float32").tobytes())
        asm.stop()
        ev = AudioEvent.objects.get()
        # 全部静音 → 至少一段静音区间；区间偏移不得为负
        self.assertTrue(ev.silence_ranges)
        for s, e in ev.silence_ranges:
            self.assertGreaterEqual(s, 0.0)
            self.assertGreater(e, s)


# ===========================================================================
# Phase 4：分段边界（spec §6.4）
# ===========================================================================
def _bounds(plans):
    return [(p.start_offset, p.end_offset) for p in plans]


class SegmentPlanTests(SimpleTestCase):
    def test_under_discard_returns_empty(self):
        self.assertEqual(plan_segments(0.5), [])
        self.assertEqual(plan_segments(0.99), [])

    def test_short_event_single_segment(self):
        """<10s 但 >1s 正常发送（单段，spec §6.4）。"""
        self.assertEqual(_bounds(plan_segments(5.0)), [(0.0, 5.0)])
        self.assertEqual(_bounds(plan_segments(9.9)), [(0.0, 9.9)])

    def test_exactly_max_single_segment(self):
        self.assertEqual(_bounds(plan_segments(60.0)), [(0.0, 60.0)])

    def test_just_over_max_merges_sub_second_tail(self):
        """60.5s：强制切 60 后尾巴 0.5s <1s → 并入前段（≤90s 不丢音频）。"""
        plans = plan_segments(60.5)
        self.assertEqual(_bounds(plans), [(0.0, 60.5)])

    def test_no_silence_forced_60s_cuts(self):
        """无静音点超 90s → 按 60s 强制切（spec §6.4）。"""
        plans = plan_segments(95.0)
        self.assertEqual(_bounds(plans), [(0.0, 60.0), (60.0, 95.0)])
        self.assertTrue(plans[0].forced)
        self.assertFalse(plans[1].forced)

    def test_long_event_consecutive_60s_cuts(self):
        plans = plan_segments(200.0)
        self.assertEqual(
            _bounds(plans),
            [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0), (180.0, 200.0)],
        )

    def test_silence_point_preferred(self):
        """150s + 静音中点 45 → [0,45],[45,105],[105,150]，第二段标记 has_silence_before。"""
        plans = plan_segments(150.0, [[44.0, 46.0]])
        self.assertEqual(_bounds(plans), [(0.0, 45.0), (45.0, 105.0), (105.0, 150.0)])
        self.assertEqual(
            [p.has_silence_before for p in plans], [False, True, False],
        )
        self.assertFalse(plans[0].forced)
        self.assertTrue(plans[1].forced)     # 105 处是无静音点的强制切

    def test_silence_past_max_cut_at_earliest(self):
        """静音点在 60~90 区间 → 取最早一个（尽快落刀，远离 90s 硬上限）。"""
        plans = plan_segments(150.0, [[69.0, 71.0]])
        self.assertEqual(_bounds(plans), [(0.0, 70.0), (70.0, 130.0), (130.0, 150.0)])

    def test_silence_before_min_not_used(self):
        """静音中点 < pos+10 不切（避免碎段）。"""
        plans = plan_segments(150.0, [[4.0, 6.0]])
        self.assertEqual(
            _bounds(plans), [(0.0, 60.0), (60.0, 120.0), (120.0, 150.0)],
        )

    def test_hard_cap_90_respected_with_dense_silence(self):
        """静音点间距 75s 的长事件：所有段 ≤90s。"""
        plans = plan_segments(300.0, [[74.0, 76.0], [149.0, 151.0], [224.0, 226.0]])
        self.assertEqual(
            _bounds(plans),
            [(0.0, 75.0), (75.0, 150.0), (150.0, 225.0), (225.0, 285.0), (285.0, 300.0)],
        )
        for p in plans:
            self.assertLessEqual(p.duration, 90.0)

    def test_sub_second_tail_dropped_when_merge_would_exceed_hard_cap(self):
        """尾巴 <1s 且并入后超 90s → 丢弃尾巴（spec §6.4 <1s 丢弃）。"""
        plans = plan_segments(90.7, [[89.6, 90.2]])
        self.assertEqual(_bounds(plans), [(0.0, 89.9)])

    def test_every_segment_within_hard_cap(self):
        """边界扫荡：多种时长 × 有/无静音，任何段都不超 90s、不短于 1s。"""
        for dur in (1.0, 10.0, 60.0, 61.0, 90.0, 91.0, 150.0, 300.0, 600.0):
            for sil in ([], [[30.0, 31.0]], [[45.0, 47.0], [130.0, 132.0]]):
                for p in plan_segments(dur, sil):
                    self.assertGreaterEqual(p.duration, 1.0, (dur, sil, p))
                    self.assertLessEqual(p.duration, 90.0 + 1e-6, (dur, sil, p))


# ===========================================================================
# Phase 4：描述 JSON 解析与校验（spec §6.5）
# ===========================================================================
_VALID_DESC = {
    "description": "成人在安抚婴儿，中间有哭声",
    "has_cry": True,
    "cry_start_offset_sec": 3.2,
    "cry_duration_sec": 12.4,
    "has_adult_speech": True,
    "adult_speech_summary": "成人在安抚婴儿",
    "speech_confidence": 0.41,
    "background_sounds": ["风扇", "电视"],
}

#: 当前提示词的输出形态：只要求一段自然语言描述
_MINIMAL_DESC = {"description": "妈妈在轻声说话，背景有风扇声"}


class ParseDescriptionJsonTests(SimpleTestCase):
    def test_plain_json(self):
        obj, err = parse_description_json(json.dumps(_VALID_DESC))
        self.assertEqual(err, "")
        self.assertEqual(obj["has_cry"], True)

    def test_fenced_json(self):
        obj, err = parse_description_json(
            "```json\n" + json.dumps(_VALID_DESC) + "\n```",
        )
        self.assertEqual(err, "")
        self.assertEqual(obj["speech_confidence"], 0.41)

    def test_leading_trailing_text(self):
        obj, err = parse_description_json(
            "分析结果如下：\n" + json.dumps(_VALID_DESC) + "\n以上。",
        )
        self.assertEqual(err, "")
        self.assertIsNotNone(obj)

    def test_garbage_returns_error(self):
        obj, err = parse_description_json("这不是 JSON")
        self.assertIsNone(obj)
        self.assertTrue(err)

    def test_non_object_json_returns_error(self):
        obj, err = parse_description_json("[1, 2, 3]")
        self.assertIsNone(obj)
        self.assertIn("not an object", err)

    def test_empty_returns_error(self):
        obj, err = parse_description_json("")
        self.assertIsNone(obj)
        self.assertEqual(err, "empty response")


class ValidateDescriptionTests(SimpleTestCase):
    def test_minimal_description_only(self):
        """契约：只有 description 也能通过；其余字段缺了就写 **None（未判定）**。

        刻意不是 ``False``：``False`` 是"判定为没有"这个断言，而描述模型根本没做
        这类判断（判定归声学侧）。拿未判定冒充否定就是灌脏数据。
        """
        clean, err = validate_description(dict(_MINIMAL_DESC), 20.0)
        self.assertEqual(err, "")
        self.assertEqual(clean["description"], "妈妈在轻声说话，背景有风扇声")
        self.assertIsNone(clean["has_cry"])
        self.assertIsNone(clean["has_adult_speech"])
        self.assertIsNone(clean["cry_start_offset_sec"])
        self.assertIsNone(clean["speech_confidence"])
        self.assertIsNone(clean["background_sounds"])

    def test_null_judgement_fields_are_unknown_not_dirty(self):
        """显式 ``null`` 必须当成"未提供"——不能走 isinstance 判成类型错。

        回归：`field in obj` 对 ``{"has_cry": null}`` 是 True，旧实现会把它当
        "不是 bool" 判脏数据，于是一次本来正常的描述被反复重试到事件 failed。
        这也是 transcript 方言的常态输入（判定字段全是 null）。
        """
        desc = {
            "description": "妈妈说该吃饭了",
            "has_cry": None,
            "has_adult_speech": None,
            "cry_start_offset_sec": None,
            "cry_duration_sec": None,
            "speech_confidence": None,
            "background_sounds": None,
        }
        clean, err = validate_description(desc, 20.0)
        self.assertEqual(err, "")
        self.assertIsNone(clean["has_cry"])
        self.assertIsNone(clean["has_adult_speech"])
        self.assertIsNone(clean["background_sounds"])

    def test_description_missing_or_blank(self):
        for bad in ({}, {"description": ""}, {"description": "   "},
                    {"description": None}, {"description": ["x"]}):
            clean, err = validate_description(bad, 20.0)
            self.assertIsNone(clean, bad)
            self.assertIn("description", err, bad)

    def test_require_description_false_allows_empty(self):
        """transcript 方言：空描述是合法结果（这段确实没有可转写的人声）。"""
        for obj in ({"description": ""}, {"description": None}, {}):
            clean, err = validate_description(obj, 20.0, require_description=False)
            self.assertEqual(err, "", obj)
            self.assertEqual(clean["description"], "", obj)
            self.assertIsNone(clean["has_cry"], obj)

    def test_require_description_false_still_rejects_wrong_type(self):
        """放宽"非空"不等于放宽类型：列表/数字还是要拦。"""
        for bad in ({"description": ["x"]}, {"description": 3}):
            clean, err = validate_description(bad, 20.0, require_description=False)
            self.assertIsNone(clean, bad)
            self.assertIn("description", err, bad)

    def test_description_is_stripped(self):
        clean, err = validate_description({"description": "  有哭声  "}, 20.0)
        self.assertEqual(err, "")
        self.assertEqual(clean["description"], "有哭声")

    def test_legacy_full_payload_still_valid(self):
        """老输出（问卷式全字段）继续被接受，历史数据不会变脏。"""
        clean, err = validate_description(dict(_VALID_DESC), 20.0)
        self.assertEqual(err, "")
        self.assertEqual(clean["cry_start_offset_sec"], 3.2)
        # 老格式 background_sounds=list[str] → 归一成一句描述
        self.assertEqual(clean["background_sounds"], "风扇、电视")

    def test_int_values_coerced_to_float(self):
        desc = dict(_VALID_DESC, cry_start_offset_sec=3, cry_duration_sec=12,
                    speech_confidence=1)
        clean, err = validate_description(desc, 20.0)
        self.assertEqual(err, "")
        self.assertIsInstance(clean["cry_start_offset_sec"], float)
        self.assertEqual(clean["speech_confidence"], 1.0)

    def test_extra_keys_dropped(self):
        clean, err = validate_description(
            dict(_VALID_DESC, unexpected="x", another={"a": 1}), 20.0,
        )
        self.assertEqual(err, "")
        self.assertNotIn("unexpected", clean)
        self.assertNotIn("another", clean)

    def test_optional_legacy_fields_may_be_absent(self):
        """问卷式字段已降级为可选：缺了不判失败（新提示词本来就不给）。"""
        for field in ("has_cry", "has_adult_speech", "background_sounds",
                      "cry_start_offset_sec", "cry_duration_sec",
                      "speech_confidence", "adult_speech_summary"):
            desc = dict(_MINIMAL_DESC)
            desc.pop(field, None)
            clean, err = validate_description(desc, 20.0)
            self.assertEqual(err, "", field)
            self.assertIsNotNone(clean, field)

    def test_bool_field_wrong_type(self):
        clean, err = validate_description(dict(_VALID_DESC, has_cry=1), 20.0)
        self.assertIsNone(clean)
        self.assertIn("不是 bool", err)

    def test_cry_offsets_optional_even_when_has_cry(self):
        """has_cry=true 但没给时间 → 合法（时间置 None），不再逼模型硬猜。"""
        desc = dict(_VALID_DESC)
        desc.pop("cry_start_offset_sec")
        clean, err = validate_description(desc, 20.0)
        self.assertEqual(err, "")
        self.assertTrue(clean["has_cry"])
        self.assertIsNone(clean["cry_start_offset_sec"])

    def test_half_given_cry_offsets_ignored(self):
        """只给一半时间 → 按未提供处理，不判失败也不写半截数据。"""
        desc = dict(_VALID_DESC)
        desc.pop("cry_duration_sec")
        clean, err = validate_description(desc, 20.0)
        self.assertEqual(err, "")
        self.assertIsNone(clean["cry_start_offset_sec"])
        self.assertIsNone(clean["cry_duration_sec"])

    def test_cry_start_out_of_range(self):
        clean, err = validate_description(
            dict(_VALID_DESC, cry_start_offset_sec=25.0), 20.0,
        )
        self.assertIsNone(clean)
        self.assertIn("越界", err)

    def test_cry_start_plus_duration_out_of_range(self):
        clean, err = validate_description(
            dict(_VALID_DESC, cry_start_offset_sec=10.0, cry_duration_sec=15.0), 20.0,
        )
        self.assertIsNone(clean)
        self.assertIn("越界", err)

    def test_cry_duration_must_be_positive(self):
        clean, err = validate_description(
            dict(_VALID_DESC, cry_duration_sec=0.0), 20.0,
        )
        self.assertIsNone(clean)

    def test_confidence_out_of_range(self):
        for bad in (-0.1, 1.5, "high"):
            clean, err = validate_description(
                dict(_VALID_DESC, speech_confidence=bad), 20.0,
            )
            self.assertIsNone(clean, bad)

    def test_legacy_summary_is_ignored(self):
        """adult_speech_summary 不再读写（聚合层才做老数据兜底），脏了也不该拦。"""
        clean, err = validate_description(
            dict(_VALID_DESC, adult_speech_summary=["x"]), 20.0,
        )
        self.assertEqual(err, "")
        self.assertNotIn("adult_speech_summary", clean)

    def test_background_sounds_accepts_string_and_legacy_list(self):
        """新格式是一句描述；老格式 list[str] 归一（都别判脏数据）。"""
        clean, err = validate_description(
            dict(_VALID_DESC, background_sounds="电视声、洗衣机声"), 20.0,
        )
        self.assertEqual(err, "")
        self.assertEqual(clean["background_sounds"], "电视声、洗衣机声")

        clean, err = validate_description(
            dict(_VALID_DESC, background_sounds=["风扇", "电视"]), 20.0,
        )
        self.assertEqual(err, "")
        self.assertEqual(clean["background_sounds"], "风扇、电视")

    def test_background_sounds_type(self):
        for bad in ([1, 2], {"a": 1}, 3):
            clean, err = validate_description(
                dict(_VALID_DESC, background_sounds=bad), 20.0,
            )
            self.assertIsNone(clean, bad)
        # null 视为未提供（模型常这么写），不算脏数据
        clean, err = validate_description(dict(_VALID_DESC, background_sounds=None), 20.0)
        self.assertEqual(err, "")
        self.assertIsNone(clean["background_sounds"])

    def test_explicit_false_claim_is_kept_as_false(self):
        """模型**明确**给了 false 就照原样留着 —— 只有"没给"才归 None。"""
        clean, err = validate_description(
            {"description": "只有风声", "has_cry": False, "has_adult_speech": False}, 20.0,
        )
        self.assertEqual(err, "")
        self.assertIs(clean["has_cry"], False)
        self.assertIs(clean["has_adult_speech"], False)
        self.assertIsNone(clean["cry_start_offset_sec"])
        self.assertIsNone(clean["speech_confidence"])
        self.assertIsNone(clean["background_sounds"])


# ===========================================================================
# Phase 4：跨分段聚合（spec §5.2）
# ===========================================================================
def _seg(sequence, start, end, desc):
    return SimpleNamespace(
        sequence=sequence, start_offset=start, end_offset=end, description_json=desc,
    )


class AggregateDescriptionTests(SimpleTestCase):
    def test_offsets_shifted_to_event_space(self):
        """段内偏移 + 段起点 = 事件内偏移。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 45.0, dict(_VALID_DESC)),
            _seg(2, 45.0, 90.0, dict(
                _VALID_DESC, cry_start_offset_sec=2.0, cry_duration_sec=5.0,
            )),
        ])
        self.assertTrue(agg["has_cry"])
        self.assertEqual(agg["cry_start_offset_sec"], 3.2)       # min(0+3.2, 45+2)
        # end = max(3.2+12.4, 45+2+5) = 52 → duration = 52-3.2
        self.assertAlmostEqual(agg["cry_duration_sec"], 48.8)

    def test_description_dedup_and_confidence_max(self):
        """各段描述去重后拼接；置信度取最大。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 60.0, dict(
                _VALID_DESC, description="在安抚", has_cry=False, speech_confidence=0.3,
            )),
            _seg(2, 60.0, 120.0, dict(
                _VALID_DESC, description="在安抚", has_cry=False, speech_confidence=0.8,
            )),
            _seg(3, 120.0, 150.0, dict(
                _VALID_DESC, description="在讲故事", has_cry=False, speech_confidence=0.5,
            )),
        ])
        self.assertEqual(agg["description"], "在安抚；在讲故事")
        self.assertEqual(agg["speech_confidence"], 0.8)

    def test_legacy_summary_folded_into_description(self):
        """老数据没有 description 时，用 adult_speech_summary 兜（升级不丢文本）。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 60.0, {"has_cry": False, "has_adult_speech": True,
                                "adult_speech_summary": "妈妈在哄睡"}),
        ])
        self.assertEqual(agg["description"], "妈妈在哄睡")

    def test_has_cry_without_offsets_does_not_crash(self):
        """回归：has_cry=true 但没给时间（新提示词的常态）不能抛 KeyError。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 60.0, {"description": "有哭声", "has_cry": True}),
        ])
        self.assertTrue(agg["has_cry"])
        self.assertIsNone(agg["cry_start_offset_sec"])
        self.assertIsNone(agg["cry_duration_sec"])

    def test_background_descriptions_joined_deduped_and_placeholder_dropped(self):
        """背景音描述：去重拼接，"无"这类占位写法不进结果。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 60.0, {"description": "a", "background_sounds": "电视声、洗衣机声"}),
            _seg(2, 60.0, 120.0, {"description": "b", "background_sounds": "电视声、洗衣机声"}),
            _seg(3, 120.0, 150.0, {"description": "c", "background_sounds": "无"}),
        ])
        self.assertEqual(agg["background_sounds"], "电视声、洗衣机声")

    def test_legacy_background_list_normalized(self):
        """老数据的 list[str] 背景音，聚合后也是一句话。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 60.0, {"description": "a", "background_sounds": ["风扇", "电视"]}),
            _seg(2, 60.0, 120.0, {"description": "b", "background_sounds": ["空调"]}),
        ])
        self.assertEqual(agg["background_sounds"], "风扇、电视；空调")

    def test_no_cry_no_speech(self):
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {
                "description": "只有风扇声",
                "has_cry": False, "has_adult_speech": False, "background_sounds": [],
            }),
        ])
        self.assertEqual(agg["description"], "只有风扇声")
        self.assertFalse(agg["has_cry"])
        self.assertIsNone(agg["cry_start_offset_sec"])
        self.assertIsNone(agg["cry_duration_sec"])
        self.assertEqual(agg["segments"], [{
            "sequence": 1, "start_offset": 0.0, "end_offset": 20.0,
        }])

    # ------------------------------------------------------------------
    # 三值合并（transcript 方言：判定字段恒为 None）
    # ------------------------------------------------------------------
    def test_all_unknown_stays_unknown(self):
        """没有任何段做过判定 → 事件级也是 None，**不能回落成 False**。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {"description": "妈妈说该吃饭了", "has_cry": None,
                                "has_adult_speech": None}),
            _seg(2, 20.0, 40.0, {"description": "电视在响"}),
        ])
        self.assertIsNone(agg["has_cry"])
        self.assertIsNone(agg["has_adult_speech"])
        self.assertIsNone(agg["cry_start_offset_sec"])
        self.assertIsNone(agg["background_sounds"])

    def test_unknown_does_not_swallow_known_claim(self):
        """有段给了判定就按"任一为真即真"；只有 False 的段参与也不影响。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {"description": "a"}),                 # 未判定
            _seg(2, 20.0, 40.0, {"description": "b", "has_cry": True,
                                 "has_adult_speech": False}),
        ])
        self.assertTrue(agg["has_cry"])
        self.assertIs(agg["has_adult_speech"], False)                 # 给了 false 就记 false

    def test_empty_descriptions_are_dropped(self):
        """空描述（这段没人声）不参与拼接：事件描述里不留空档/多余分号。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {"description": "妈妈说该吃饭了"}),
            _seg(2, 20.0, 40.0, {"description": ""}),
            _seg(3, 40.0, 60.0, {"description": "电视在响"}),
        ])
        self.assertEqual(agg["description"], "妈妈说该吃饭了；电视在响")
        self.assertEqual(len(agg["segments"]), 3)                     # 审计列表仍保留全段

    def test_all_empty_descriptions_yield_empty_event_description(self):
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {"description": ""}),
            _seg(2, 20.0, 40.0, {"description": ""}),
        ])
        self.assertEqual(agg["description"], "")

    def test_dirty_description_type_does_not_crash_aggregation(self):
        """历史行里 description 可能是 list/dict：聚合不能因此崩掉整个事件。"""
        agg = aggregate_descriptions([
            _seg(1, 0.0, 20.0, {"description": ["老旧", "格式"]}),
            _seg(2, 20.0, 40.0, {"description": "正常文本"}),
        ])
        self.assertIn("正常文本", agg["description"])


# ===========================================================================
# Phase 4：raw 暂存上限（防止事件长期 describing 时内存无界增长）
# ===========================================================================
class RawStoreTests(SimpleTestCase):
    def test_single_text_truncated(self):
        store = _RawStore(text_limit=10, total_limit=1000)
        store.record(1, 1, "x" * 50)
        self.assertEqual(store.pop(1), {1: "x" * 10})

    def test_pop_removes_entry(self):
        store = _RawStore()
        store.record(7, 1, "x")
        self.assertEqual(store.pop(7), {1: "x"})
        self.assertEqual(store.pop(7), {})
        self.assertEqual(len(store), 0)

    def test_same_sequence_overwritten(self):
        store = _RawStore(text_limit=100, total_limit=1000)
        store.record(1, 1, "old")
        store.record(1, 1, "new")
        self.assertEqual(store.pop(1), {1: "new"})

    def test_eviction_prefers_least_recently_used_event(self):
        """总量超限时淘汰最久未写入的事件；刚追加过分段的事件保留。"""
        store = _RawStore(text_limit=5, total_limit=20)
        store.record(1, 1, "a" * 5)
        store.record(2, 1, "b" * 5)
        store.record(1, 2, "a" * 5)     # 事件 1 刷新为最新
        store.record(3, 1, "c" * 5)     # 20 字符，未超
        store.record(4, 1, "d" * 5)     # 25 字符超限 → 淘汰最久未用的事件 2

        self.assertEqual(store.pop(2), {})
        self.assertEqual(sorted(store.pop(1)), [1, 2])
        self.assertEqual(sorted(store.pop(3)), [1])
        self.assertEqual(sorted(store.pop(4)), [1])

    def test_single_oversized_event_kept(self):
        """只有一个事件时不被淘汰，避免把正在推进的事件自己丢掉。"""
        store = _RawStore(text_limit=100, total_limit=100)
        store.record(1, 1, "x" * 60)
        store.record(1, 2, "y" * 60)    # 共 120 字符 > 上限，但只有一个事件
        self.assertEqual(sorted(store.pop(1)), [1, 2])


# ===========================================================================
# Phase 4：AudioDescClient（HTTP 行为，mock session）
# ===========================================================================
class AudioDescClientTests(SimpleTestCase):
    def _client(self, **kw):
        return AudioDescClient(base_url="http://127.0.0.1:8083", model="asr-test", **kw)

    def _mock_ok(self, client, content="{}"):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": content}}]}
        resp.text = content
        client._session = MagicMock()
        client._session.post.return_value = resp
        return client._session

    def test_payload_uses_input_audio_base64_wav(self):
        client = self._client()
        session = self._mock_ok(client)
        client.describe(b"RIFF-fake-wav", "prompt-1", max_tokens=128)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "asr-test")
        self.assertEqual(payload["max_tokens"], 128)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "prompt-1"})
        audio = content[1]
        self.assertEqual(audio["type"], "input_audio")
        self.assertEqual(audio["input_audio"]["format"], "wav")
        import base64 as b64mod

        self.assertEqual(
            b64mod.b64decode(audio["input_audio"]["data"]), b"RIFF-fake-wav",
        )
        url = session.post.call_args.args[0]
        self.assertEqual(url, "http://127.0.0.1:8083/v1/chat/completions")

    def test_api_key_sets_bearer_header(self):
        client = self._client(api_key="k-123")
        self.assertEqual(
            client._session.headers["Authorization"], "Bearer k-123",
        )

    def test_success_returns_content(self):
        client = self._client()
        self._mock_ok(client, content='{"has_cry": true}')
        self.assertEqual(client.describe(b"w", "p"), '{"has_cry": true}')

    def test_timeout_is_split_into_connect_and_read(self):
        """requests 收到的是 `(connect, read)` 元组，不是标量。

        标量会让 connect 与 read 共用一个值：服务停掉后连接会挂在 `SYN_SENT`
        上等超时（**不是**立刻 RST），本该 1~3s 失败的 connect 被拖成整个 read
        超时（默认 30s），再乘 `max_retries+1` —— 一次 `describe()` 要 ~92s 才
        回报熔断器，"连续 3 次失败熔断"因此变成 ~4.6 分钟才发现服务不可用。
        """
        client = self._client(timeout_sec=30, connect_timeout_sec=3.0)
        session = self._mock_ok(client)
        client.describe(b"w", "p")
        self.assertEqual(session.post.call_args.kwargs["timeout"], (3.0, 30))

    def test_probe_short_timeout_only_shortens_read(self):
        """半开探测的短超时只压 **read**；connect 仍用 `connect_timeout_sec`。

        把 connect 一起压到几秒的话，网络稍慢时探测会先被自己的 connect 超时
        判死 —— 探测失败 → 继续熔断，服务其实早就好了也恢复不了。
        """
        client = self._client(connect_timeout_sec=3.0)
        session = self._mock_ok(client)
        client.describe(b"w", "p", timeout_sec=5, max_retries=0)
        self.assertEqual(session.post.call_args.kwargs["timeout"], (3.0, 5.0))

    def test_connect_phase_failure_is_unreachable(self):
        """connect 阶段失败 → `AudioDescUnreachableError`（"服务不在"）。

        `requests.exceptions.ConnectTimeout` **同时**继承 `ConnectionError` 与
        `Timeout`；不显式先捕获它，就会被 `except Timeout` 吃掉 —— 于是"端口
        不通"在日志/last_error 里显示成"读超时（模型慢）"，排查被带偏。
        """
        import requests
        from unittest.mock import MagicMock

        cases = (
            requests.exceptions.ConnectTimeout("connect timed out"),
            requests.exceptions.ConnectionError("refused"),
        )
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                client = self._client(max_retries=0)
                client._session = MagicMock()
                client._session.post.side_effect = exc
                with self.assertRaises(AudioDescUnreachableError) as cm:
                    client.describe(b"w", "p")
                # 必须是 NetworkError 子类 —— 既有 except 分支才不用改
                self.assertIsInstance(cm.exception, AudioDescNetworkError)

    def test_read_timeout_is_not_unreachable(self):
        """连接已建立、模型没回 → `AudioDescTimeoutError`，**不是**不可达。

        两者处置不同：不可达（连接没建立）一次即定论；读超时可能是"慢"，
        一次不能定论。
        """
        import requests
        from unittest.mock import MagicMock

        client = self._client(max_retries=0)
        client._session = MagicMock()
        client._session.post.side_effect = requests.exceptions.ReadTimeout("slow")
        with self.assertRaises(AudioDescTimeoutError) as cm:
            client.describe(b"w", "p")
        self.assertNotIsInstance(cm.exception, AudioDescUnreachableError)

    def test_timeout_retried_then_success(self):
        import requests
        from unittest.mock import MagicMock

        client = self._client(max_retries=2, retry_sleep_sec=0)
        session = self._mock_ok(client)
        session.post.side_effect = [
            requests.exceptions.Timeout("t1"),
            requests.exceptions.Timeout("t2"),
            session.post.return_value,
        ]
        self.assertEqual(client.describe(b"w", "p"), "{}")
        self.assertEqual(session.post.call_count, 3)

    def test_timeout_exhausted_raises(self):
        import requests
        from unittest.mock import MagicMock

        client = self._client(max_retries=1, retry_sleep_sec=0)
        client._session = MagicMock()
        client._session.post.side_effect = requests.exceptions.Timeout("t")
        with self.assertRaises(AudioDescTimeoutError):
            client.describe(b"w", "p")
        self.assertEqual(client._session.post.call_count, 2)

    def test_connection_error_is_network_error(self):
        import requests
        from unittest.mock import MagicMock

        client = self._client(max_retries=0)
        client._session = MagicMock()
        client._session.post.side_effect = requests.exceptions.ConnectionError("refused")
        with self.assertRaises(AudioDescNetworkError):
            client.describe(b"w", "p")

    def test_5xx_is_network_error(self):
        from unittest.mock import MagicMock

        client = self._client(max_retries=0)
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        client._session = MagicMock()
        client._session.post.return_value = resp
        with self.assertRaises(AudioDescNetworkError):
            client.describe(b"w", "p")

    def test_4xx_is_parse_error_without_retry(self):
        from unittest.mock import MagicMock

        client = self._client(max_retries=3, retry_sleep_sec=0)
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "bad request"
        client._session = MagicMock()
        client._session.post.return_value = resp
        with self.assertRaises(AudioDescParseError):
            client.describe(b"w", "p")
        self.assertEqual(client._session.post.call_count, 1)

    def test_non_json_response_is_parse_error(self):
        from unittest.mock import MagicMock

        client = self._client(max_retries=0)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("no json")
        resp.text = "<html>"
        client._session = MagicMock()
        client._session.post.return_value = resp
        with self.assertRaises(AudioDescParseError):
            client.describe(b"w", "p")

    def test_empty_choices_is_parse_error(self):
        from unittest.mock import MagicMock

        client = self._client(max_retries=0)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"choices": []}
        resp.text = "{}"
        client._session = MagicMock()
        client._session.post.return_value = resp
        with self.assertRaises(AudioDescParseError):
            client.describe(b"w", "p")

    def test_empty_wav_rejected(self):
        client = self._client()
        with self.assertRaises(ValueError):
            client.describe(b"", "p")


# ===========================================================================
# Phase 4：DescribeService（分段落盘 + 重试 + 聚合 + 脏数据隔离）
# ===========================================================================
class _FakeAudioDescClient:
    """按调用序返回预设行为：str = 原始返回；Exception 实例 = 抛出。"""

    def __init__(self, behaviors):
        self.model = "asr-test"
        self.calls: list[tuple] = []
        #: 每次调用的覆盖参数（半开探测要断言"短超时 + 不重试"）
        self.call_kwargs: list[dict] = []
        self._behaviors = list(behaviors)

    def describe(self, wav_bytes, prompt, max_tokens=512,
                 timeout_sec=None, max_retries=None):
        self.calls.append((wav_bytes, prompt, max_tokens))
        self.call_kwargs.append({
            "timeout_sec": timeout_sec, "max_retries": max_retries,
        })
        idx = min(len(self.calls) - 1, len(self._behaviors) - 1)
        behavior = self._behaviors[idx]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class _FakeFlacIO:
    """内存版 FLAC 读写：事件文件给合成 PCM，分段文件走 writer 记录。"""

    def __init__(self, recording_sec: float = 30.0, sample_rate: int = 16000):
        self.recording_sec = recording_sec
        self.sample_rate = sample_rate
        self.files: dict[str, tuple] = {}

    def reader(self, path):
        import numpy as np

        key = str(path)
        if key in self.files:
            return self.files[key]
        return (
            np.zeros(int(self.recording_sec * self.sample_rate), dtype="float32"),
            self.sample_rate,
        )

    def writer(self, path, pcm, sample_rate):
        self.files[str(path)] = (pcm, sample_rate)


class DescribeServiceTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_desc_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def setUp(self):
        super().setUp()
        # 熔断器是**进程内单例**（`LLMBreaker.for_service`），状态会跨用例残留：
        # 某个用例连撞阈值后，后面的用例会看到 `is_blocking()=True` → run_once()
        # 直接返回 []、一次都不调模型。每个用例进场前清一次。
        from apps.core.llm_breaker import LLMBreaker

        LLMBreaker.reset_instances()

    def _make_event(self, duration=20.0, silence=None, **kw):
        defaults = dict(
            camera_id=self.camera.id,
            status=AudioEvent.STATUS_PENDING_DESCRIPTION,
            started_at_ts=1000,
            ended_at_ts=1000 + int(duration),
            duration_sec=duration + 8.0,
            audio_path="fake-event.flac",
            recording_lead_sec=5.0,
            silence_ranges=silence or [],
        )
        defaults.update(kw)
        return AudioEvent.objects.create(**defaults)

    def _make_service(
        self, client, flac, max_retries=2, retry_backoff_sec=0.0, worker_epoch="",
        provider=DESC_PROVIDER_JSON, max_tokens=512,
    ):
        # 本类的替身大多喂 JSON 字符串，故默认走 json 方言；
        # transcript 方言单独在 TranscriptDescribeTests 里验。
        return DescribeService(
            client=client,
            worker_epoch=worker_epoch,
            config=DescriberConfig(
                poll_sec=5.0, retry_backoff_sec=retry_backoff_sec,
                max_segment_retries=max_retries, max_tokens=max_tokens,
                provider=provider,
            ),
            segmenter=SegmenterConfig(),
            media_root="fake-media",
            flac_reader=flac.reader,
            flac_writer=flac.writer,
            wav_encoder=lambda pcm, sr: b"wav-bytes",
        )

    # ------------------------------------------------------------------
    def test_success_single_segment(self):
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        self.assertEqual(svc.run_once(), [event.id])
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(event.description_status, AudioEvent.DESC_COMPLETED)
        self.assertEqual(event.moss_model, "asr-test")
        self.assertTrue(event.description_json["has_cry"])
        self.assertEqual(event.description_json["cry_start_offset_sec"], 3.2)
        self.assertIn('"1"', event.description_raw)      # 原始返回按段号留存

        segs = list(event.segments.all())
        self.assertEqual(len(segs), 1)
        self.assertEqual((segs[0].start_offset, segs[0].end_offset), (0.0, 20.0))
        self.assertEqual(segs[0].description_status, AudioEvent.DESC_COMPLETED)
        self.assertEqual(segs[0].description_json["has_cry"], True)
        self.assertIn("audio_segments", segs[0].audio_path)
        # 分段 FLAC 确实写过（fake writer 记录）
        self.assertIn(segs[0].audio_path, flac.files)
        # 客户端只被调一次，且 prompt 带了事件上下文
        self.assertEqual(len(client.calls), 1)
        self.assertIn("0.0 ~ 20.0 秒", client.calls[0][1])

    def test_created_segments_carry_worker_epoch(self):
        """分段行记录产出它的 worker 租约 epoch（recover 据此限定范围）。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac, worker_epoch="epoch-desc")
        event = self._make_event(20.0)

        svc.run_once()
        self.assertEqual(event.segments.get().worker_epoch, "epoch-desc")

    def test_multi_segment_aggregation(self):
        """150s + 静音 45s → 3 段；段内偏移聚合成事件内偏移。"""
        flac = _FakeFlacIO(recording_sec=158.0)
        desc_cry = dict(                                            # 只有哭声
            _VALID_DESC, description="宝宝在哭", has_adult_speech=False,
            speech_confidence=None,
        )
        desc_speech = dict(
            _VALID_DESC, description="妈妈在哄睡", has_cry=False,
            cry_start_offset_sec=None, cry_duration_sec=None, speech_confidence=0.9,
        )
        desc_quiet = {
            "description": "只剩风扇声",
            "has_cry": False, "has_adult_speech": False, "background_sounds": "无",
        }
        client = _FakeAudioDescClient([
            json.dumps(desc_cry), json.dumps(desc_speech), json.dumps(desc_quiet),
        ])
        svc = self._make_service(client, flac)
        event = self._make_event(150.0, silence=[[44.0, 46.0]])

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)

        segs = list(event.segments.order_by("sequence"))
        self.assertEqual(
            [(s.start_offset, s.end_offset) for s in segs],
            [(0.0, 45.0), (45.0, 105.0), (105.0, 150.0)],
        )
        self.assertEqual([s.has_silence_before for s in segs], [False, True, False])

        agg = event.description_json
        self.assertEqual(agg["description"], "宝宝在哭；妈妈在哄睡；只剩风扇声")
        self.assertTrue(agg["has_cry"])
        self.assertEqual(agg["cry_start_offset_sec"], 3.2)
        self.assertTrue(agg["has_adult_speech"])
        self.assertEqual(agg["speech_confidence"], 0.9)
        self.assertEqual(agg["background_sounds"], "风扇、电视")
        self.assertEqual(len(agg["segments"]), 3)

    def test_desc_service_down_then_retry_recovers(self):
        """描述服务不可用 → 分段标 failed、事件留 describing；**冷却后**追上。

        注意 `retry_count` 是 **0**：**服务侧失败不消耗配额**（Phase 4 / checklist §8.4）。
        配额是给"内容类失败"用的 —— 服务不可达算进去的话，8 次 ×
        `retry_backoff_sec(60s)` ≈ 8 分钟就把事件判成永久 failed，"攒着"就不成立了。

        **阈值改成 1（用户 2026-09-16 定）后，"恢复"不再是一句话的事**：
        失败一次即熔断，恢复要走「冷却结束 → 放一条探测 → 探测成功」。
        所以这里显式模拟"冷却已过"（`reset_instances()` = 冷却后用全新状态
        重新判断服务是否可用），而不是假设下一轮就能追上。
        """
        from apps.core.llm_breaker import LLMBreaker

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([AudioDescNetworkError("connection refused")])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_DESCRIBING)
        self.assertEqual(event.description_json, {})          # 不写脏数据
        seg = event.segments.get()
        self.assertEqual(seg.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(seg.retry_count, 0)                  # 不占配额
        self.assertIn("AudioDescNetworkError", seg.last_error)
        self.assertTrue(
            svc._breaker().is_blocking(),
            "阈值 1：一次失败即熔断，后续请求先攒着而不是马上再撞",
        )

        # 服务恢复 + 冷却已过 → 下一轮追上
        LLMBreaker.reset_instances()
        client._behaviors = [json.dumps(_VALID_DESC)]
        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertTrue(event.description_json["has_cry"])

    def test_invalid_json_not_persisted(self):
        """非 JSON 响应 → 记失败而不是写脏数据（spec §6.5 / Phase 4 验证）。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient(["这不是 JSON"])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        self.assertNotEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(event.description_json, {})
        seg = event.segments.get()
        self.assertEqual(seg.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(seg.description_json, {})
        self.assertIn("解析失败", seg.last_error)

    def test_schema_violation_not_persisted(self):
        """数值越界（哭声时间超出片段时长）→ 记失败，不入库。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        bad = dict(_VALID_DESC, cry_start_offset_sec=30.0)     # 片段只有 20s
        client = _FakeAudioDescClient([json.dumps(bad)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        svc.run_once()
        seg = event.segments.get()
        self.assertEqual(seg.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(seg.description_json, {})
        self.assertIn("越界", seg.last_error)

    def test_retry_exhausted_event_failed_but_recording_kept(self):
        """**内容类**失败重试耗尽 → 事件 failed；录音不动（失败不删录音，spec §12）。

        刻意用"永远返回非 JSON"（内容问题），而不是连接失败：
        连接类失败**永不消耗配额**（见下一条用例），
        若在这里喂网络错，事件会一直停在 describing、根本进不到 exhausted 分支。
        """
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient(["这不是 JSON"])
        svc = self._make_service(client, flac, max_retries=2)
        event = self._make_event(20.0)

        svc.run_once()     # retry_count 1
        svc.run_once()     # retry_count 2 = max → 之后不再重试
        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_FAILED)
        self.assertEqual(event.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(event.audio_path, "fake-event.flac")
        seg = event.segments.get()
        self.assertEqual(seg.retry_count, 2)
        self.assertTrue(seg.audio_path)                       # 分段文件也保留

    def test_connection_failure_never_consumes_retry_quota(self):
        """连接类失败**永不**耗尽配额 → 事件长期停在 describing。**这就是"攒着"**。

        (d) 的核心保证：ASR/llama 停 1 小时，这 1 小时的事件不能被判成永久 failed。
        旧行为下 8 次 × `retry_backoff_sec(60s)` ≈ 8 分钟就废（checklist §8.4）。
        """
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([AudioDescNetworkError("down")])
        svc = self._make_service(client, flac, max_retries=2)
        event = self._make_event(20.0)

        for _ in range(10):                      # 远超 max_segment_retries=2
            svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_DESCRIBING)
        self.assertNotEqual(event.status, AudioEvent.STATUS_FAILED)
        self.assertEqual(seg.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(seg.retry_count, 0)
        self.assertTrue(seg.audio_path)          # 录音/分段都留着，等回放

    def test_breaker_stops_attempts_when_service_down(self):
        """连撞阈值 → 熔断打开，`run_once()` **整轮不尝试**（连事件都不挑）。

        没有这道闸的两个坏处：每条分段白等一个超时；且涨的都是无意义的 DB 写。
        """
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([AudioDescNetworkError("down")])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        for _ in range(5):
            svc.run_once()
        calls_after_open = len(client.calls)
        self.assertTrue(svc._breaker().is_blocking(), "连撞 5 次应已熔断")
        self.assertEqual(svc.run_once(), [], "熔断打开时不该挑事件")
        self.assertEqual(len(client.calls), calls_after_open, "熔断打开时不该再发请求")

        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_DESCRIBING)
        self.assertEqual(seg.retry_count, 0)

    def test_candidates_are_newest_first(self):
        """候选按 **newest-first** 取（`-started_at_ts`）：告警时效 > 完整性。

        造 25 个事件（> `_MAX_EVENTS_PER_ROUND=20`），跑一轮 —— 处理到的必须是
        **最新的 20 个**。旧行为是 FIFO（最老的 20 个），停 3 小时后回来先补
        3 小时前的窗口，最新、还可能有意义的排最后。
        """
        from apps.audio_detect.describer import _MAX_EVENTS_PER_ROUND

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)

        events = [
            self._make_event(
                20.0, started_at_ts=1000 + i, ended_at_ts=1020 + i,
            )
            for i in range(25)
        ]
        done = svc.run_once()
        self.assertEqual(len(done), _MAX_EVENTS_PER_ROUND)
        expected = {
            e.id for e in sorted(events, key=lambda x: -x.started_at_ts)[
                :_MAX_EVENTS_PER_ROUND
            ]
        }
        self.assertEqual(set(done), expected)

    def test_expired_events_reaped_when_max_age_set(self):
        """超期收割：`QUEUE_MAX_AGE_SEC` 生效时老事件标 `expired`，不再永远挂着。

        为什么必须有它：newest-first 之后候选窗口只有 `20 × 10 = 200` 条，
        积压更多时老事件**永远进不了窗口** → 永远停在 `describing`，
        控制页"待描述"数字永不归零（分不清"还在攒"还是"卡死"）。
        """
        import time

        from django.test import override_settings

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)

        old = self._make_event(20.0, started_at_ts=1000)          # 1970 年的 ts
        fresh_ts = int(time.time())
        fresh = self._make_event(
            20.0, started_at_ts=fresh_ts, ended_at_ts=fresh_ts + 20,
        )

        with override_settings(BABYCARE_LLM_QUEUE_MAX_AGE_SEC=3600):
            svc.run_once()

        old.refresh_from_db()
        fresh.refresh_from_db()
        self.assertEqual(old.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(old.description_json, {"skipped": "expired"})
        self.assertNotEqual(
            fresh.description_json, {"skipped": "expired"}, "新事件不该被收割",
        )

    def test_no_reaping_when_max_age_zero(self):
        """默认 `QUEUE_MAX_AGE_SEC=0` → **不收割**（用户定"先给 0，上线再调"）。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        old = self._make_event(20.0, started_at_ts=1000)

        svc.run_once()
        old.refresh_from_db()
        self.assertNotEqual(old.description_json, {"skipped": "expired"})

    def test_probe_uses_short_timeout_and_no_retry(self):
        """半开探测必须**短超时 + 不重试** —— 否则探测自己就把链路堵死。

        ASR 默认是 `TIMEOUT_SEC(30) × (max_retries(2)+1) = 最坏 90s`
        （VLM 侧更狠：`timeout_sec_first=60` 且 `_chat_lock` 全局串行）。
        """
        from django.test import override_settings

        from apps.core.llm_breaker import LLMBreaker

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])

        # 冷却设 0 = "冷却已过"，让 run_once 这一格正好落在探测窗口
        with override_settings(BABYCARE_LLM_BREAKER_COOLDOWN_SEC=0):
            LLMBreaker.reset_instances()          # 新配置要在单例创建前生效
            svc = self._make_service(client, flac)
            self._make_event(20.0)
            breaker = svc._breaker()
            for _ in range(3):
                breaker.record_failure("down")
            self.assertTrue(svc._breaker().is_probing(), "这一格应当是探测")

            svc.run_once()

        self.assertEqual(len(client.call_kwargs), 1)
        self.assertEqual(client.call_kwargs[0]["max_retries"], 0, "探测不该重试")
        self.assertEqual(
            client.call_kwargs[0]["timeout_sec"], breaker.probe_timeout_sec,
        )

    def test_normal_attempt_keeps_default_timeout_and_retries(self):
        """单机回归：非探测时**不覆盖**超时/重试（用客户端自己的默认值）。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        self._make_event(20.0)

        svc.run_once()
        self.assertEqual(len(client.call_kwargs), 1)
        self.assertIsNone(client.call_kwargs[0]["timeout_sec"])
        self.assertIsNone(client.call_kwargs[0]["max_retries"])

    def test_success_closes_breaker(self):
        """成功必须回报熔断器 —— 否则熔断**永久自锁**。

        病灶：成功路径漏了 `record_success()`，唯一那句写在兜底的
        `except Exception` 里。于是熔断一旦打开，冷却结束后 `is_probing()`
        恒为真 —— 之后每条请求都按"探测"跑（5s 超时 + 0 重试），
        再也回不到正常超时，等于把描述链路永久降级。
        """
        from django.test import override_settings

        from apps.core.llm_breaker import LLMBreaker

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])

        # 冷却设 0 = "冷却已过"，让这一轮正好落在探测窗口
        with override_settings(BABYCARE_LLM_BREAKER_COOLDOWN_SEC=0):
            LLMBreaker.reset_instances()
            svc = self._make_service(client, flac)
            self._make_event(20.0)
            breaker = svc._breaker()
            for _ in range(3):
                breaker.record_failure("down")
            self.assertTrue(breaker.is_probing(), "冷却已过 → 这一格是探测")

            svc.run_once()

            self.assertFalse(
                breaker.is_probing(),
                "成功后必须回 CLOSED，否则熔断永久自锁（永远 5s 超时 + 不重试）",
            )
            self.assertFalse(breaker.is_blocking())
            self.assertEqual(breaker.snapshot()["consecutive_failures"], 0)

    def test_content_error_does_not_close_breaker(self):
        """**内容/本地类**失败不能当成"服务活着"的证据去关熔断。

        分段音频缺失、切片落盘失败这类错误压根没碰服务。旧代码把它们和
        "模型回了脏数据"混在同一个 `except Exception` 里，还补了一句
        `record_success()` —— 一个真的不可用的服务会被**假关闭**。
        服务是否活着，只由成功路径与连接类分支各自回报。
        """
        from django.test import override_settings

        from apps.core.llm_breaker import LLMBreaker

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([ValueError("模型回了脏数据")])

        with override_settings(BABYCARE_LLM_BREAKER_COOLDOWN_SEC=0):
            LLMBreaker.reset_instances()
            svc = self._make_service(client, flac)
            event = self._make_event(20.0)
            breaker = svc._breaker()
            for _ in range(3):
                breaker.record_failure("down")

            svc.run_once()

            self.assertTrue(breaker.is_probing(), "内容类失败不该把熔断关掉")

        seg = event.segments.get()
        self.assertEqual(seg.retry_count, 1, "内容类失败才消耗配额")
        self.assertNotEqual(seg.description_status, AudioEvent.DESC_COMPLETED)

    def test_breaker_gate_leaves_segment_untouched(self):
        """`allow_attempt()` 为假 → **不碰分段状态**、不调模型。

        这是"半开态严格单条探测"的落点。分段若被先改成 `processing` 再半途
        退出，会被 recover 逻辑当成"上次中断"来回折腾；所以闸门必须在改状态
        **之前**。
        """
        from unittest.mock import MagicMock, patch

        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        fake = MagicMock()
        fake.is_blocking.return_value = False      # 冷却已过 → 进到分段循环
        fake.allow_attempt.return_value = False    # 但这一格不放行
        with patch.object(type(svc), "_breaker", return_value=fake):
            self.assertEqual(svc.run_once(), [event.id])

        self.assertEqual(client.calls, [], "被熔断拦下时不该发请求")
        seg = event.segments.get()
        self.assertNotEqual(
            seg.description_status, AudioEvent.DESC_PROCESSING,
            "闸门必须在改分段状态之前 —— 否则留下 processing 残行",
        )
        self.assertNotEqual(seg.description_status, AudioEvent.DESC_COMPLETED)

    def test_one_failure_opens_breaker_by_default(self):
        """**默认阈值 1**：失败一次就熔断 —— 请求攒着，60s 后放一条探测。

        用户 2026-09-16 定：不要"撞 3 次"。之所以 1 够用，是因为 connect 超时
        已独立压到 3s（误判一次的代价只是"一个 3s 请求 + 最多 60s 排队"，而
        排队的工作会被回放，**不丢数据**）。

        本条同时钉住阈值 1 下**只发一次请求** —— 第一个分段失败后，本轮后续
        分段会被 `allow_attempt()` 拦下，不会把冷却窗口整批烧掉（§12.3）。
        """
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([
            AudioDescUnreachableError("connect refused"),
        ])
        svc = self._make_service(client, flac)
        self._make_event(20.0)

        self.assertEqual(
            svc._breaker().snapshot()["fail_threshold"], 1,
            "默认阈值应为 1（settings.BABYCARE_LLM_FAIL_THRESHOLD）",
        )
        svc.run_once()

        self.assertTrue(svc._breaker().is_blocking(), "一次失败就该熔断")
        self.assertEqual(len(client.calls), 1, "熔断后本轮不该再发请求")

    def test_too_short_event_skipped_without_calling_model(self):
        """<1s 事件：直接完成、不送描述模型（spec §6.4）。"""
        flac = _FakeFlacIO()
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0, ended_at_ts=1000)      # duration = 0

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(event.description_json["skipped"], "too_short")
        self.assertEqual(event.segments.count(), 0)
        self.assertEqual(client.calls, [])

    def test_missing_audio_path_fails_permanently(self):
        flac = _FakeFlacIO()
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0, audio_path="")

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_FAILED)
        self.assertEqual(client.calls, [])

    def test_unreadable_audio_fails_permanently(self):
        flac = _FakeFlacIO()
        flac.reader = lambda path: (_ for _ in ()).throw(OSError("文件损坏"))
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_FAILED)
        self.assertEqual(client.calls, [])

    def test_recover_stale_descriptions(self):
        """崩溃遗留 describing/processing → 回收成 pending。"""
        event = self._make_event(
            20.0, status=AudioEvent.STATUS_DESCRIBING,
        )
        seg = AudioEventSegment.objects.create(
            audio_event=event, sequence=1, start_offset=0.0, end_offset=20.0,
            description_status=AudioEvent.DESC_PROCESSING,
        )
        n = DescribeService.recover_stale_descriptions()
        self.assertEqual(n, 1)
        event.refresh_from_db()
        seg.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_PENDING_DESCRIPTION)
        self.assertEqual(event.description_status, AudioEvent.DESC_PENDING)
        self.assertEqual(seg.description_status, AudioEvent.DESC_PENDING)

    def test_partial_segments_are_backfilled(self):
        """建段中断只留前半段 → 下一轮补齐剩余段，不会误判 completed 丢音频。"""
        flac = _FakeFlacIO(recording_sec=158.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(150.0, silence=[[44.0, 46.0]])
        # 模拟第一段建行后进程被打断（只剩 sequence=1，其余段从未建）
        AudioEventSegment.objects.create(
            audio_event=event, sequence=1, start_offset=0.0, end_offset=45.0,
            audio_path="", description_status=AudioEvent.DESC_PENDING,
        )

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(
            [(s.sequence, s.start_offset, s.end_offset)
             for s in event.segments.order_by("sequence")],
            [(1, 0.0, 45.0), (2, 45.0, 105.0), (3, 105.0, 150.0)],
        )
        self.assertEqual(len(client.calls), 3)

    def test_incomplete_segments_fail_after_max_rounds(self):
        """分段一直补不上（建行持续失败）→ 若干轮后 failed，不无限卡 describing。"""
        flac = _FakeFlacIO(recording_sec=158.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac)
        event = self._make_event(150.0, silence=[[44.0, 46.0]])
        AudioEventSegment.objects.create(
            audio_event=event, sequence=1, start_offset=0.0, end_offset=45.0,
        )
        svc._create_segments = lambda *a, **kw: []      # 模拟建行永远失败

        for _ in range(4):
            svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_DESCRIBING)
        self.assertEqual(client.calls, [])               # 不齐时不送描述模型

        svc.run_once()
        event.refresh_from_db()
        self.assertEqual(event.status, AudioEvent.STATUS_FAILED)
        self.assertEqual(event.description_status, AudioEvent.DESC_FAILED)

    def test_backing_off_event_lets_new_event_through(self):
        """老事件整段在退避期 → 本轮跳过，新事件不被队头阻塞。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_VALID_DESC)])
        svc = self._make_service(client, flac, retry_backoff_sec=600.0)
        old = self._make_event(
            20.0, started_at_ts=1000, status=AudioEvent.STATUS_DESCRIBING,
        )
        new = self._make_event(20.0, started_at_ts=2000)
        seg = AudioEventSegment.objects.create(
            audio_event=old, sequence=1, start_offset=0.0, end_offset=20.0,
            description_status=AudioEvent.DESC_FAILED, retry_count=1,
        )

        self.assertEqual(svc.run_once(), [new.id])
        old.refresh_from_db()
        new.refresh_from_db()
        seg.refresh_from_db()
        self.assertEqual(old.status, AudioEvent.STATUS_DESCRIBING)   # 仍在退避，未动
        self.assertEqual(seg.retry_count, 1)
        self.assertEqual(new.status, AudioEvent.STATUS_COMPLETED)

    def test_build_prompt_describes_content_without_scene_hints(self):
        """提示词：片段位置 + 具体说清说话内容 + 其他声音；**不给场景提示**、不做问卷。"""
        event = self._make_event(150.0, silence=[[44.0, 46.0]])
        seg = AudioEventSegment(
            audio_event=event, sequence=2, start_offset=45.0, end_offset=105.0,
        )
        prompt = build_prompt(seg)
        self.assertIn("家庭监控音频的分析助手", prompt)
        self.assertIn("家庭室内环境的录音", prompt)        # 中性背景（只给环境，不列人）
        self.assertIn("时长 60.0 秒", prompt)             # 片段时长
        self.assertIn("45.0 ~ 105.0 秒", prompt)          # 在事件中的位置
        self.assertIn('"description"', prompt)
        self.assertIn("说话内容写清楚", prompt)            # 说话单独一块
        self.assertIn("background_sounds", prompt)        # 背景音单独一块
        self.assertIn("洗衣机", prompt)                    # 背景音举例
        self.assertIn("不要编造", prompt)
        # 关键 1：场景提示里不列人 —— 列在题干里会被原样复读（"爸爸妈妈保姆宝宝"）
        for gone in ("家里可能有", "妈妈", "保姆", "宝宝", "不要罗列"):
            self.assertNotIn(gone, prompt, gone)
        # 关键 2：不索要"问卷式"判定字段 —— 会把模型往"哭"上带
        for gone in ("has_cry", "cry_duration_sec", "speech_confidence",
                     "16000Hz", "静音区间"):
            self.assertNotIn(gone, prompt, gone)
        # 关键 3：不给"没听到就写 X"这种捷径，也不给"名词标签式"占位 ——
        # 实测前者让模型 6~7s 直接回 "无说话"（根本没听），后者让它把占位原文抄进内容
        for gone in ("无说话", "就写", "说话描述：", "背景音描述："):
            self.assertNotIn(gone, prompt, gone)

    def test_segment_audio_path_layout(self):
        event = self._make_event(20.0)
        path = segment_audio_path("fake-media", event, 3)
        # 不依赖具体日期（本机时区），只验证目录结构与文件名
        self.assertEqual(path.parent.parent.name, "audio_segments")
        self.assertEqual(path.parent.parent.parent, Path("fake-media"))
        self.assertEqual(path.name, f"{event.id}_seg3.flac")

    def test_noncanonical_provider_still_parses_json(self):
        """回归：``provider="JSON"`` 不能走成"把 JSON 原文当转写正文存进去"。

        提示词、归一化、是否要求非空描述是三个独立分支；只要有一个没归一，就会
        出现"提示词按 json 发、解析按 transcript 收"的静默错配。
        """
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([json.dumps(_MINIMAL_DESC)])
        svc = self._make_service(client, flac, provider="JSON")
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(
            seg.description_json["description"], "妈妈在轻声说话，背景有风扇声",
        )
        self.assertNotIn("{", seg.description_json["description"])

    # ------------------------------------------------------------------
    # transcript 方言（Qwen3-ASR 等纯转写模型）
    # ------------------------------------------------------------------
    def test_transcript_provider_persists_transcript(self):
        """转写原文进 description；判定字段**全部是 null（未判定）**。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient(["妈妈说：外面那些盘子多少人用过了"])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(20.0)

        self.assertEqual(svc.run_once(), [event.id])
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(event.moss_model, "asr-test")
        self.assertEqual(
            seg.description_json["description"], "妈妈说：外面那些盘子多少人用过了",
        )
        # 不做判断：null 而不是 false（判定归声学侧 YAMNet/PANNs）
        self.assertIsNone(seg.description_json["has_adult_speech"])
        self.assertIsNone(seg.description_json["has_cry"])
        self.assertIsNone(seg.description_json["background_sounds"])
        self.assertIsNone(seg.description_json["speech_confidence"])
        self.assertIsNone(event.description_json["has_cry"])
        # 走的是转写提示词，不是 json 方言的 JSON 模板
        prompt = client.calls[0][1]
        self.assertIn(NO_SPEECH_TEXT, prompt)
        self.assertNotIn('"description"', prompt)

    def test_transcript_strips_qwen3_asr_wrapper_before_persisting(self):
        """回归（2026-09-16 现场实测返回格式）：带 ``language XX<asr_text>`` 壳入库要剥干净。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient(["language Chinese<asr_text>指日可待，就差"])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(seg.description_json["description"], "指日可待，就差")
        # 原始返回仍然完整留存（便于事后核对模型行为）
        self.assertIn("asr_text", event.description_raw)

    def test_transcript_no_speech_segment_completes(self):
        """整段没人声：正常完成，不记失败；模型回的占位文本原样留存。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([NO_SPEECH_TEXT])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(seg.description_status, AudioEvent.DESC_COMPLETED)
        self.assertEqual(seg.description_json["description"], NO_SPEECH_TEXT)
        self.assertIsNone(seg.description_json["has_adult_speech"])

    def test_transcript_bodyless_wrapper_completes_with_empty_description(self):
        """纯静音只回 ``language Chinese<asr_text>`` → 空描述、仍算完成（不判失败）。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient(["language Chinese<asr_text>"])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(seg.description_status, AudioEvent.DESC_COMPLETED)
        self.assertEqual(seg.description_json["description"], "")
        self.assertEqual(seg.retry_count, 0)          # 不是失败，不消耗重试
        # 原始返回（含传输壳）仍完整留存，便于事后核对模型行为
        self.assertIn("asr_text", event.description_raw)

    def test_transcript_empty_response_is_retried_not_completed(self):
        """空响应 → 分段 failed 走退避（不能静默当“没人声”落库），录音保留。"""
        flac = _FakeFlacIO(recording_sec=28.0)
        client = _FakeAudioDescClient([""])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(20.0)

        svc.run_once()
        event.refresh_from_db()
        seg = event.segments.get()
        self.assertEqual(event.status, AudioEvent.STATUS_DESCRIBING)   # 未完成
        self.assertEqual(seg.description_status, AudioEvent.DESC_FAILED)
        self.assertEqual(seg.retry_count, 1)
        self.assertIn("empty response", seg.last_error)
        self.assertTrue(seg.audio_path)                                # 失败不删录音

    def test_transcript_aggregates_across_segments(self):
        """多段转写按“；”拼接；中间那段没人声（空描述）不留空档，审计列表仍在。"""
        flac = _FakeFlacIO(recording_sec=158.0)
        client = _FakeAudioDescClient([
            "第一段话", "language Chinese<asr_text>", "第三段话",
        ])
        svc = self._make_service(client, flac, provider=DESC_PROVIDER_TRANSCRIPT)
        event = self._make_event(150.0, silence=[[44.0, 46.0]])

        svc.run_once()
        event.refresh_from_db()
        agg = event.description_json
        self.assertEqual(event.status, AudioEvent.STATUS_COMPLETED)
        self.assertEqual(agg["description"], "第一段话；第三段话")
        self.assertEqual(len(agg["segments"]), len(list(event.segments.all())))
        self.assertIsNone(agg["has_cry"])
        self.assertIsNone(agg["has_adult_speech"])


# ===========================================================================
# Phase 4：输出方言（provider）——transcript（Qwen3-ASR）/ json
# ===========================================================================
class NormalizeProviderTests(SimpleTestCase):
    """`.env` 的 provider 归一：非法值不能把描述服务带进未知分支。"""

    def test_known_values(self):
        self.assertEqual(normalize_provider("transcript"), DESC_PROVIDER_TRANSCRIPT)
        self.assertEqual(normalize_provider("json"), DESC_PROVIDER_JSON)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(normalize_provider("  JSON "), DESC_PROVIDER_JSON)
        self.assertEqual(normalize_provider("Transcript"), DESC_PROVIDER_TRANSCRIPT)

    def test_invalid_falls_back_to_transcript(self):
        for bad in ("", None, "moss", "qwen3-asr", 3):
            self.assertEqual(normalize_provider(bad), DESC_PROVIDER_TRANSCRIPT, bad)

    def test_config_normalizes_provider_at_construction(self):
        """方言在**构造时**归一 —— 下游有三处按它分支，不归一就会各走各的。"""
        self.assertEqual(DescriberConfig(provider="  JSON ").provider, DESC_PROVIDER_JSON)
        self.assertEqual(DescriberConfig(provider="JSON").provider, DESC_PROVIDER_JSON)
        self.assertEqual(
            DescriberConfig(provider="moss").provider, DESC_PROVIDER_TRANSCRIPT,
        )
        self.assertEqual(DescriberConfig().provider, DESC_PROVIDER_TRANSCRIPT)


class TranscriptPromptTests(SimpleTestCase):
    """transcript 方言的提示词：短、不带 JSON 模板、明确交代"没人声"怎么写。"""

    def test_transcript_prompt_has_no_json_template(self):
        prompt = build_transcript_prompt()
        self.assertIn("逐字转写", prompt)
        self.assertIn(NO_SPEECH_TEXT, prompt)
        # json 方言的 JSON 模板被喂给转写模型会被当正文念出来
        for gone in ('"description"', "background_sounds", "{", "}"):
            self.assertNotIn(gone, prompt, gone)

    def test_build_prompt_for_dispatches(self):
        seg = SimpleNamespace(start_offset=0.0, end_offset=20.0)
        self.assertIn('"description"', build_prompt_for(seg, DESC_PROVIDER_JSON))
        self.assertEqual(
            build_prompt_for(seg, DESC_PROVIDER_TRANSCRIPT), build_transcript_prompt(),
        )
        # 非法 provider 与 normalize_provider 保持一致（走 transcript）
        self.assertEqual(build_prompt_for(seg, "whatever"), build_transcript_prompt())


class NormalizeTranscriptTests(SimpleTestCase):
    """转写原文 → 待校验 dict：**只清洗协议层，不做内容加工、不做任何判定**。"""

    def test_plain_transcript(self):
        """只有 ``description`` 一个键：判定字段一个都不填（归声学侧）。"""
        obj, err = normalize_transcript("妈妈说：该吃饭了")
        self.assertEqual(err, "")
        self.assertEqual(obj, {"description": "妈妈说：该吃饭了"})

    def test_no_speech_placeholder_kept_verbatim(self):
        """模型自己回的“没人声”占位**原样留存**：那是它的输出，不改写、不识别。"""
        for raw in (NO_SPEECH_TEXT, "无说话声", "(无语音)", " 无说话声。 "):
            obj, err = normalize_transcript(raw)
            self.assertEqual(err, "", raw)
            self.assertEqual(obj["description"], raw.strip(), raw)
            self.assertNotIn("has_adult_speech", obj, raw)

    def test_strips_quotes_and_fences(self):
        for raw in ('"妈妈说该吃饭了"', "```\n妈妈说该吃饭了\n```"):
            obj, err = normalize_transcript(raw)
            self.assertEqual(err, "", raw)
            self.assertEqual(obj["description"], "妈妈说该吃饭了", raw)

    def test_strips_qwen3_asr_wrapper(self):
        """Qwen3-ASR 会把结果包成 ``language Chinese<asr_text>正文``（实测格式）。"""
        for raw in (
            "language Chinese<asr_text>这人，你不是为了奔走求人？",
            "language Chinese<|asr_text|>这人，你不是为了奔走求人？",
        ):
            obj, err = normalize_transcript(raw)
            self.assertEqual(err, "", raw)
            self.assertEqual(obj["description"], "这人，你不是为了奔走求人？", raw)
            self.assertNotIn("asr_text", obj["description"])
            self.assertNotIn("language", obj["description"])

    def test_wrapper_no_speech_placeholder(self):
        """带壳的“没人声”占位：剥掉传输壳，占位文本本身照旧留着。"""
        obj, err = normalize_transcript(f"language Chinese<asr_text>{NO_SPEECH_TEXT}")
        self.assertEqual(err, "")
        self.assertEqual(obj, {"description": NO_SPEECH_TEXT})

    def test_strip_asr_wrapper_helper(self):
        self.assertEqual(strip_asr_wrapper("纯文本"), "纯文本")
        self.assertEqual(
            strip_asr_wrapper("language English<asr_text>hi there"), "hi there",
        )
        # 只剥最后一层：正文里若出现同名标记（ASR 自己不会这么干）也不至于全丢
        self.assertEqual(
            strip_asr_wrapper("language Chinese<asr_text>a<asr_text>b"), "b",
        )

    def test_bodyless_wrapper_yields_empty_description(self):
        """模型响应了、只是壳里没内容（纯静音）→ 空描述、合法，**不塞占位文本**。"""
        for raw in ("language Chinese<asr_text>", "language Chinese<asr_text> "):
            obj, err = normalize_transcript(raw)
            self.assertEqual(err, "", raw)
            self.assertEqual(obj, {"description": ""}, raw)

    def test_collapses_newlines(self):
        obj, err = normalize_transcript("第一句\n\n  第二句  ")
        self.assertEqual(err, "")
        self.assertEqual(obj["description"], "第一句 第二句")

    def test_blank_response_is_error_not_no_speech(self):
        """整条响应是空白 = 服务/提示词异常，判失败走退避重试；不能静默当“没人声”。"""
        for raw in ("", "   ", "\n"):
            obj, err = normalize_transcript(raw)
            self.assertIsNone(obj, raw)
            self.assertTrue(err, raw)

    def test_cry_and_background_are_never_guessed(self):
        """转写方言拿不到背景音、也不替声学侧猜哭声：判定字段一个都不产出。"""
        obj, _ = normalize_transcript("有人在说话")
        self.assertEqual(obj, {"description": "有人在说话"})
        for field in ("has_cry", "has_adult_speech", "cry_start_offset_sec",
                      "cry_duration_sec", "speech_confidence", "background_sounds"):
            self.assertNotIn(field, obj, field)

    def test_result_is_accepted_by_description_validator(self):
        """归一结果必须过生产校验（转写方言用 require_description=False）。

        否则会在 _describe_segment 里被判脏数据 —— 包括"这段没人声"的空描述。
        """
        for raw in ("妈妈说该吃饭了", NO_SPEECH_TEXT, "language Chinese<asr_text>"):
            obj, err = normalize_transcript(raw)
            self.assertEqual(err, "", raw)
            clean, verr = validate_description(obj, 20.0, require_description=False)
            self.assertEqual(verr, "", raw)
            self.assertEqual(clean["description"], obj["description"], raw)
            # 判定字段全部落到 None（未判定），不是 False
            for field in ("has_cry", "has_adult_speech", "cry_start_offset_sec",
                          "cry_duration_sec", "speech_confidence",
                          "background_sounds"):
                self.assertIsNone(clean[field], f"{raw} {field}")


# ===========================================================================
# Worker 实例互斥（spec §9.3）：单行租约锁 + epoch 归属
# ===========================================================================
class PidAliveTests(SimpleTestCase):
    """进程存活判断（Windows 上有个隐蔽误报坑，见 worker_lock._pid_alive_windows）。"""

    def test_none_and_invalid_pid(self):
        self.assertFalse(_pid_alive(None))
        self.assertFalse(_pid_alive(0))
        self.assertFalse(_pid_alive(-1))

    def test_alive_child_is_detected(self):
        import subprocess
        import sys

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        try:
            self.assertTrue(_pid_alive(proc.pid))
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_exited_child_with_open_handle_is_not_alive(self):
        """回归（2026-09-12 现场）：进程已退出，但**本进程仍持有它的 Popen 句柄**，
        内核对象就不会销毁，``OpenProcess`` 依旧成功 —— 旧实现据此误报"仍存活"，
        导致 web 侧白等 10s 优雅窗口、打出"pid 仍然存活"的假错误、PID 文件不清理
        （而 worker 其实 1 秒内就优雅退出完了）。

        修复后改用 ``WaitForSingleObject(handle, 0)`` 判断进程对象是否已 signaled。
        """
        import subprocess
        import sys

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=30)          # 已退出；proc 句柄仍被本测试持有
        self.assertFalse(
            _pid_alive(proc.pid),
            "已退出的进程（本进程仍持有其 Popen 句柄）不该被判为存活",
        )


class WorkerLockTests(TestCase):
    """`AudioServiceState` 单行租约锁：第二个实例拒绝启动、过期可接管。"""

    def _state(self) -> AudioServiceState:
        return AudioServiceState.objects.get(pk=AudioServiceState.SINGLETON_PK)

    def _expire_lease(self, **extra) -> None:
        """把租约心跳改成已过期（模拟旧实例崩溃/卡死）。"""
        from datetime import timedelta

        from django.conf import settings
        from django.utils import timezone

        stale = timezone.now() - timedelta(
            seconds=settings.BABYCARE_AUDIO_HEARTBEAT_TIMEOUT_SEC + 5,
        )
        AudioServiceState.objects.filter(
            pk=AudioServiceState.SINGLETON_PK,
        ).update(updated_at=stale, **extra)

    def test_acquire_when_free(self):
        import os

        slot = acquire_worker_slot()
        self.assertTrue(slot.epoch)
        self.assertEqual(slot.stale_epochs, {""})
        row = self._state()
        self.assertEqual(row.pid, os.getpid())
        self.assertEqual(row.worker_epoch, slot.epoch)
        self.assertEqual(row.reason, AudioServiceState.REASON_AUTOSTART)

    def test_second_instance_rejected_while_heartbeat_fresh(self):
        acquire_worker_slot()
        with self.assertRaises(WorkerLockError):
            acquire_worker_slot()

    def test_takeover_after_heartbeat_timeout(self):
        first = acquire_worker_slot()
        self._expire_lease(pid=999_999_999)      # 心跳过期 + 旧 pid 已不存在

        second = acquire_worker_slot()
        self.assertNotEqual(second.epoch, first.epoch)
        self.assertEqual(second.previous_epoch, first.epoch)
        # 接管前的租约 + 历史空 epoch 才是可回收范围
        self.assertEqual(second.stale_epochs, {"", first.epoch})

    def test_release_makes_slot_immediately_acquirable(self):
        first = acquire_worker_slot()
        self.assertTrue(release_worker_slot(first.epoch))

        row = self._state()
        self.assertIsNone(row.pid)
        self.assertEqual(row.worker_epoch, first.epoch)   # epoch 保留给 recover

        # 释放后立即可接管，不用等心跳超时；上一次 epoch 进入可回收范围
        second = acquire_worker_slot()
        self.assertEqual(second.previous_epoch, first.epoch)

    def test_release_does_not_clear_newer_owner(self):
        """被接管后，旧实例的 release 不能清掉新实例的租约。"""
        first = acquire_worker_slot()
        AudioServiceState.objects.filter(pk=AudioServiceState.SINGLETON_PK).update(
            pid=4242, worker_epoch="newer-epoch",
        )

        self.assertFalse(release_worker_slot(first.epoch))
        row = self._state()
        self.assertEqual(row.pid, 4242)
        self.assertEqual(row.worker_epoch, "newer-epoch")

    def test_touch_only_refreshes_own_epoch(self):
        from django.utils import timezone

        slot = acquire_worker_slot()
        self._expire_lease()

        self.assertFalse(touch_worker_slot("someone-else"))
        self.assertTrue(touch_worker_slot(slot.epoch))
        self.assertLess(
            (timezone.now() - self._state().updated_at).total_seconds(), 5,
        )


class RecoverScopeTests(TestCase):
    """recover 只回收"接管前那个租约"的行（epoch 归属），不碰其它实例在途数据。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_epoch_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def _event(self, epoch, status=AudioEvent.STATUS_RECORDING, ts=1000):
        return AudioEvent.objects.create(
            camera_id=self.camera.id, status=status,
            started_at_ts=ts, worker_epoch=epoch,
        )

    def test_recover_events_scoped_by_epoch(self):
        mine = self._event("epoch-a")
        other = self._event("epoch-b")
        legacy = self._event("")

        n = EventAssembler.recover_stale_events({"", "epoch-a"})
        self.assertEqual(n, 2)
        mine.refresh_from_db()
        other.refresh_from_db()
        legacy.refresh_from_db()
        self.assertEqual(mine.status, AudioEvent.STATUS_FAILED)
        self.assertTrue(mine.degraded)
        self.assertEqual(legacy.status, AudioEvent.STATUS_FAILED)
        # 不属于本租约的行一律不动
        self.assertEqual(other.status, AudioEvent.STATUS_RECORDING)

    def test_recover_events_default_is_legacy_only(self):
        mine = self._event("epoch-a")
        legacy = self._event("")

        self.assertEqual(EventAssembler.recover_stale_events(), 1)
        mine.refresh_from_db()
        legacy.refresh_from_db()
        self.assertEqual(mine.status, AudioEvent.STATUS_RECORDING)
        self.assertEqual(legacy.status, AudioEvent.STATUS_FAILED)

    def test_recover_descriptions_scoped_by_epoch(self):
        a = self._event("epoch-a", status=AudioEvent.STATUS_DESCRIBING)
        seg_a = AudioEventSegment.objects.create(
            audio_event=a, sequence=1, start_offset=0.0, end_offset=10.0,
            description_status=AudioEvent.DESC_PROCESSING, worker_epoch="epoch-a",
        )
        b = self._event("epoch-b", status=AudioEvent.STATUS_DESCRIBING, ts=2000)
        seg_b = AudioEventSegment.objects.create(
            audio_event=b, sequence=1, start_offset=0.0, end_offset=10.0,
            description_status=AudioEvent.DESC_PROCESSING, worker_epoch="epoch-b",
        )

        n = DescribeService.recover_stale_descriptions({"epoch-a"})
        self.assertEqual(n, 1)
        a.refresh_from_db()
        b.refresh_from_db()
        seg_a.refresh_from_db()
        seg_b.refresh_from_db()
        self.assertEqual(a.status, AudioEvent.STATUS_PENDING_DESCRIPTION)
        self.assertEqual(a.description_status, AudioEvent.DESC_PENDING)
        self.assertEqual(seg_a.description_status, AudioEvent.DESC_PENDING)
        # 其它 epoch 的在途行不动
        self.assertEqual(b.status, AudioEvent.STATUS_DESCRIBING)
        self.assertEqual(seg_b.description_status, AudioEvent.DESC_PROCESSING)


# ===========================================================================
# 音频 worker 进程管理（spec §9.3，Phase 5）
# PID 管启停 + DB 心跳管健康 → 「卡死」= 进程活着但心跳停
# ===========================================================================
class _FakeProc:
    """假 Popen：``poll()`` 返回 None = 存活，返回 int = 已退出。"""

    def __init__(self, pid: int = 4321, alive: bool = True):
        self.pid = pid
        self.returncode = None if alive else 1
        self._rc = self.returncode

    def poll(self):
        return self._rc

    def exit_now(self, rc: int = 1) -> None:
        self._rc = rc
        self.returncode = rc


class _AudioWorkerTestBase(TestCase):
    """共用的临时 PID / 日志路径 + 独立 manager 实例（避开单例状态串味）。"""

    def setUp(self):
        from tempfile import TemporaryDirectory

        from django.test import override_settings

        from apps.audio_detect.worker_manager import AudioWorkerManager

        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pid_file = Path(self._tmp.name) / "audio_worker.pid"
        self.log_file = Path(self._tmp.name) / "audio_worker.log"
        self.stop_file = Path(self._tmp.name) / "audio_worker.stop"

        over = override_settings(
            BABYCARE_AUDIO_ENABLED=True,
            BABYCARE_AUDIO_PID_FILE=str(self.pid_file),
            BABYCARE_AUDIO_LOG_FILE=str(self.log_file),
            BABYCARE_AUDIO_STOP_FILE=str(self.stop_file),
            # 默认不等待优雅窗口，避免每个 stop 用例白等 10s；
            # 需要验证"等待 → 优雅退出"的用例自己 override 成小值。
            BABYCARE_AUDIO_STOP_GRACE_SEC=0,
        )
        over.enable()
        self.addCleanup(over.disable)

        self.mgr = AudioWorkerManager()

    def _desired(self) -> str:
        row = AudioServiceState.objects.filter(
            pk=AudioServiceState.SINGLETON_PK,
        ).first()
        return row.desired if row else ""


class AudioWorkerManagerTests(_AudioWorkerTestBase):
    """启动 / 停止：PID 文件、desired 持久化、失败不静默。"""

    def test_start_raises_when_disabled(self):
        from django.test import override_settings

        from apps.audio_detect.worker_manager import AudioWorkerStartError

        with override_settings(BABYCARE_AUDIO_ENABLED=False):
            with self.assertRaises(AudioWorkerStartError) as cm:
                self.mgr.start(wait_sec=0)
        self.assertIn("BABYCARE_AUDIO_ENABLED", str(cm.exception))

    def test_start_writes_pidfile_and_desired_on(self):
        proc = _FakeProc(pid=4321)
        with patch.object(self.mgr, "_spawn", return_value=proc):
            info = self.mgr.start(wait_sec=0)

        self.assertTrue(info["started"])
        self.assertEqual(info["pid"], 4321)
        self.assertEqual(self.pid_file.read_text(encoding="utf-8"), "4321")
        row = AudioServiceState.objects.get(pk=AudioServiceState.SINGLETON_PK)
        self.assertEqual(row.desired, AudioServiceState.DESIRED_ON)
        self.assertEqual(row.reason, AudioServiceState.REASON_MANUAL)

    def test_start_clears_stale_pidfile_before_spawn(self):
        """残留 PID 文件的进程早没了 → 不阻挡启动，且被清掉。"""
        self.pid_file.write_text("999999999", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=False,
        ), patch.object(self.mgr, "_spawn", return_value=_FakeProc(pid=777)):
            info = self.mgr.start(wait_sec=0)

        self.assertTrue(info["started"])
        self.assertEqual(self.pid_file.read_text(encoding="utf-8"), "777")

    def test_start_skips_when_already_running(self):
        self.pid_file.write_text("4321", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ), patch.object(self.mgr, "_spawn") as spawn:
            info = self.mgr.start(wait_sec=0)

        self.assertFalse(info["started"])
        self.assertEqual(info["note"], "already_running")
        spawn.assert_not_called()

    def test_spawn_uses_audio_settings_module(self):
        """音频 venv 无 daphne/channels：子进程必须用 config.settings_audio。

        若继承 web 进程的 ``config.settings``，子进程会在 Django setup 阶段
        ``ModuleNotFoundError: daphne`` 秒退 —— 网页启停根本起不来。
        """
        import os

        from django.conf import settings

        captured = {}

        def _fake_popen(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs.get("env") or {})
            captured["cwd"] = kwargs.get("cwd")
            return _FakeProc(pid=1)

        with patch.dict(os.environ, {"DJANGO_SETTINGS_MODULE": "config.settings"}), \
                patch("apps.audio_detect.worker_manager.subprocess.Popen",
                      side_effect=_fake_popen):
            info = self.mgr.start(wait_sec=0)

        self.assertTrue(info["started"])
        self.assertEqual(
            captured["env"]["DJANGO_SETTINGS_MODULE"], "config.settings_audio",
        )
        self.assertEqual(captured["cmd"][1:], ["manage.py", "audio_worker"])
        self.assertEqual(captured["cwd"], str(Path(settings.BASE_DIR)))

    def test_start_raises_when_python_missing(self):
        from django.test import override_settings

        from apps.audio_detect.worker_manager import AudioWorkerStartError

        missing = str(Path(self._tmp.name) / "nope" / "python.exe")
        with override_settings(BABYCARE_AUDIO_PYTHON=missing):
            with self.assertRaises(AudioWorkerStartError) as cm:
                self.mgr.start(wait_sec=0)
        self.assertIn("找不到音频解释器", str(cm.exception))

    def test_start_early_exit_reports_log_and_cleans_up(self):
        """子进程早退（抢不到租约 / 映射表写错）→ 抛错 + 回日志 + 不写 desired=on。"""
        from apps.audio_detect.worker_manager import AudioWorkerStartError

        proc = _FakeProc(pid=555, alive=False)

        def _spawn():
            self.log_file.write_text(
                "拒绝启动：已有音频 worker 在运行（pid=999）\n", encoding="utf-8",
            )
            return proc

        with patch.object(self.mgr, "_spawn", side_effect=_spawn), patch(
            "apps.audio_detect.worker_manager.terminate_process_tree",
        ) as kill:
            with self.assertRaises(AudioWorkerStartError) as cm:
                self.mgr.start(wait_sec=1.0)

        self.assertIn("已有音频 worker 在运行", str(cm.exception))
        kill.assert_called_once()                       # 半死进程要回收
        self.assertFalse(self.pid_file.exists())
        self.assertNotEqual(self._desired(), AudioServiceState.DESIRED_ON)

    def test_stop_kills_tree_and_records_off(self):
        """grace=0（base 默认）→ 跳过优雅等待，直接走兜底强杀。"""
        self.pid_file.write_text("4321", encoding="utf-8")
        # 判活：初次 True（要停）→ 强杀前复查 True（未响应）→ 强杀后 False（已死）
        with patch(
            "apps.audio_detect.worker_manager._pid_alive",
            side_effect=[True, True, False],
        ), patch("apps.audio_detect.worker_manager._kill_pid_tree") as kill:
            stopped = self.mgr.stop()

        self.assertTrue(stopped)
        self.assertEqual(kill.call_args[0][0], 4321)    # 杀的是 PID 文件里的 pid
        self.assertFalse(self.pid_file.exists())
        self.assertEqual(self._desired(), AudioServiceState.DESIRED_OFF)

    def test_stop_uses_terminate_tree_for_own_child(self):
        """本进程持有的 Popen → 走 terminate_process_tree（有句柄，可优雅停）。"""
        self.mgr._proc = _FakeProc(pid=4321)
        self.pid_file.write_text("4321", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ), patch("apps.audio_detect.worker_manager.terminate_process_tree") as term:
            self.mgr.stop()
        term.assert_called_once()
        self.assertEqual(term.call_args[0][0].pid, 4321)

    def test_stop_keeps_pidfile_when_process_survives(self):
        """杀不掉（权限等）→ 保留 PID 文件，页面继续如实显示"运行中"。"""
        self.pid_file.write_text("4321", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ), patch("apps.audio_detect.worker_manager._kill_pid_tree"):
            stopped = self.mgr.stop()

        self.assertFalse(stopped)
        self.assertTrue(self.pid_file.exists())
        self.assertEqual(self._desired(), AudioServiceState.DESIRED_OFF)

    def test_stop_clears_orphan_pidfile(self):
        self.pid_file.write_text("4321", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=False,
        ):
            stopped = self.mgr.stop()

        self.assertFalse(stopped)
        self.assertFalse(self.pid_file.exists())
        self.assertEqual(self._desired(), AudioServiceState.DESIRED_OFF)

    def test_current_desired_follows_switch_when_never_set(self):
        """表里没有行 = 用户从未表过态 → 跟随总开关（on），首次部署不必手点。"""
        self.assertEqual(self.mgr.current_desired(), AudioServiceState.DESIRED_ON)

    def test_current_desired_off_when_explicitly_closed(self):
        """用户手动关闭过 → 以行里的 off 为准（不跟随总开关）。"""
        self.mgr.set_desired(AudioServiceState.DESIRED_OFF)
        self.assertEqual(self.mgr.current_desired(), AudioServiceState.DESIRED_OFF)

    def test_set_desired_does_not_touch_lease_heartbeat(self):
        """web 侧写 desired **绝不能**刷 updated_at：那是 worker 租约心跳。"""
        from datetime import timedelta

        from django.utils import timezone

        AudioServiceState.objects.update_or_create(
            pk=AudioServiceState.SINGLETON_PK,
        )
        stale = timezone.now() - timedelta(seconds=99)
        AudioServiceState.objects.filter(pk=AudioServiceState.SINGLETON_PK).update(
            updated_at=stale, pid=4242, worker_epoch="epoch-x",
        )

        self.mgr.set_desired(AudioServiceState.DESIRED_ON)

        row = AudioServiceState.objects.get(pk=AudioServiceState.SINGLETON_PK)
        self.assertEqual(row.desired, AudioServiceState.DESIRED_ON)
        self.assertEqual(row.pid, 4242)                 # 租约字段未被动
        self.assertEqual(row.worker_epoch, "epoch-x")
        self.assertEqual(row.updated_at, stale)         # 心跳未被伪造


class AudioWorkerStatusTests(_AudioWorkerTestBase):
    """status()：PID 信号 + DB 心跳 → 「卡死」判定与摄像头列表。"""

    @classmethod
    def setUpTestData(cls):
        from uuid import uuid4

        from apps.streaming.models import Camera

        cls.camera = Camera.objects.create(
            name=f"t_{uuid4().hex[:8]}_audio_mgr_cam",
            source_type=Camera.SOURCE_ONVIF,
            onvif_host="127.0.0.1",
            is_active=True,
        )

    def test_no_runtime_row_is_not_ok(self):
        st = self.mgr.status()
        self.assertFalse(st["heartbeat_ok"])
        self.assertIsNone(st["heartbeat_age_sec"])
        self.assertFalse(st["running"])
        self.assertFalse(st["stale"])

    def test_fresh_heartbeat_marks_ok(self):
        AudioRuntimeState.objects.create(
            camera=self.camera, status=AudioRuntimeState.STATUS_READY,
        )
        st = self.mgr.status()
        self.assertTrue(st["heartbeat_ok"])
        self.assertFalse(st["running"])     # 没有 PID 文件

    def test_stale_when_process_alive_but_heartbeat_dead(self):
        """核心第四态：进程活着、心跳停了 = 卡死（只看 PID 会漏报）。"""
        from datetime import timedelta

        from django.utils import timezone

        state = AudioRuntimeState.objects.create(
            camera=self.camera, status=AudioRuntimeState.STATUS_READY,
        )
        AudioRuntimeState.objects.filter(pk=state.pk).update(
            updated_at=timezone.now() - timedelta(seconds=999),
        )
        self.pid_file.write_text("4321", encoding="utf-8")

        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ):
            st = self.mgr.status()

        self.assertTrue(st["running"])
        self.assertEqual(st["pid"], 4321)
        self.assertFalse(st["heartbeat_ok"])
        self.assertTrue(st["stale"])

    def test_startup_grace_suppresses_false_stale(self):
        """刚拉起、模型还在加载 → 显示「启动中」，不能报「卡死」。

        回归：没有宽限期时，用户点完「开启」3 秒后刷新就会看到误报的「卡死」
        （模型加载要 15~40s，心跳线程那时还没开始写）。
        """
        self.pid_file.write_text("4321", encoding="utf-8")
        self.mgr._started_at = time.time()              # 刚 spawn
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ):
            st = self.mgr.status()

        self.assertTrue(st["running"])
        self.assertFalse(st["heartbeat_ok"])
        self.assertTrue(st["starting"])
        self.assertFalse(st["stale"])

    def test_stale_after_startup_grace_expires(self):
        """宽限期过后仍无心跳 → 才判「卡死」。"""
        self.pid_file.write_text("4321", encoding="utf-8")
        self.mgr._started_at = time.time() - 9999
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ):
            st = self.mgr.status()

        self.assertFalse(st["starting"])
        self.assertTrue(st["stale"])

    def test_not_starting_once_heartbeat_is_fresh(self):
        """宽限期内但心跳已到 → 直接算正常，不是「启动中」。"""
        AudioRuntimeState.objects.create(
            camera=self.camera, status=AudioRuntimeState.STATUS_READY,
        )
        self.pid_file.write_text("4321", encoding="utf-8")
        self.mgr._started_at = time.time()
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ):
            st = self.mgr.status()

        self.assertTrue(st["heartbeat_ok"])
        self.assertFalse(st["starting"])
        self.assertFalse(st["stale"])

    def test_cameras_listed_with_status(self):
        AudioRuntimeState.objects.create(
            camera=self.camera,
            status=AudioRuntimeState.STATUS_CAPTURE_ERROR,
            last_error="boom",
        )
        st = self.mgr.status()
        self.assertEqual(len(st["cameras"]), 1)
        row = st["cameras"][0]
        self.assertEqual(row["camera_id"], self.camera.id)
        self.assertEqual(row["status"], AudioRuntimeState.STATUS_CAPTURE_ERROR)
        self.assertEqual(row["last_error"], "boom")

    def test_status_reports_lease_pid(self):
        """区分 PID 文件里的 venv 启动器 PID 与租约里的 worker 真实 PID。

        Windows venv 的 ``Scripts\\python.exe`` 是个启动器，真实 python 是它的
        子进程（现场实测：PID 文件 75272 → 真实 worker 81048）——不区分会让
        用户以为"页面的 PID 和进程对不上"。
        """
        self.pid_file.write_text("75272", encoding="utf-8")
        AudioServiceState.objects.update_or_create(
            pk=AudioServiceState.SINGLETON_PK,
            defaults={
                "desired": AudioServiceState.DESIRED_ON,
                "pid": 81048,
                "worker_epoch": "abcdef1234567890",
            },
        )
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ):
            st = self.mgr.status()

        self.assertEqual(st["pid"], 75272)              # PID 文件（启动器）
        self.assertEqual(st["lease_pid"], 81048)        # 租约（真实 worker）
        self.assertEqual(st["worker_epoch"], "abcdef1234567890")
        self.assertEqual(st["desired"], AudioServiceState.DESIRED_ON)


class AudioWorkerTailLogTests(_AudioWorkerTestBase):
    """日志读取：只取尾部、支持偏移（避免全量读大文件）。"""

    def test_missing_file(self):
        self.assertIn("日志文件不存在", self.mgr.tail_log())

    def test_tail_reads_appended_content(self):
        self.log_file.write_bytes(b"old-line\n")
        self.assertIn("old-line", self.mgr.tail_log())
        with open(self.log_file, "ab") as fp:
            fp.write(b"new-error\n")
        self.assertIn("new-error", self.mgr.tail_log())

    def test_since_offset_beyond_eof(self):
        self.log_file.write_bytes(b"old\n")                   # 恰好 4 字节
        self.assertEqual(self.mgr.tail_log(since=4), "（本次启动没有产生日志）")

    def test_since_offset_only_new_content(self):
        self.log_file.write_bytes(b"old\n")
        with open(self.log_file, "ab") as fp:
            fp.write(b"fresh\n")
        self.assertIn("fresh", self.mgr.tail_log(since=4))
        self.assertNotIn("old", self.mgr.tail_log(since=4))


class AudioWorkerAutostartTests(_AudioWorkerTestBase):
    """启动期自动拉起：``ENABLED and desired == "on"`` 才起（spec §9.3）。"""

    def test_off_when_disabled(self):
        from django.test import override_settings

        with override_settings(BABYCARE_AUDIO_ENABLED=False):
            self.assertFalse(self.mgr.maybe_autostart())

    def test_autostarts_when_never_set(self):
        """表里没有行（首次部署）→ 跟随总开关自动拉起，不必先去控制页点一次。"""
        with patch.object(
            self.mgr, "start", return_value={"started": True, "pid": 9},
        ) as start:
            self.assertTrue(self.mgr.maybe_autostart())
        start.assert_called_once()

    def test_off_when_user_closed_it(self):
        """用户手动关过（desired=off 的行存在）→ 重启 daphne 不再自动拉起。"""
        self.mgr.set_desired(AudioServiceState.DESIRED_OFF)
        with patch.object(self.mgr, "start") as start:
            self.assertFalse(self.mgr.maybe_autostart())
        start.assert_not_called()

    def test_starts_when_desired_on(self):
        self.mgr.set_desired(AudioServiceState.DESIRED_ON)
        with patch.object(
            self.mgr, "start", return_value={"started": True, "pid": 9},
        ) as start:
            self.assertTrue(self.mgr.maybe_autostart())
        self.assertEqual(
            start.call_args.kwargs.get("reason"),
            AudioServiceState.REASON_AUTOSTART,
        )

    def test_skips_when_already_running(self):
        self.mgr.set_desired(AudioServiceState.DESIRED_ON)
        self.pid_file.write_text("4321", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=True,
        ), patch.object(self.mgr, "start") as start:
            self.assertFalse(self.mgr.maybe_autostart())
        start.assert_not_called()

    def test_swallows_start_error(self):
        from apps.audio_detect.worker_manager import AudioWorkerStartError

        self.mgr.set_desired(AudioServiceState.DESIRED_ON)
        with patch.object(
            self.mgr, "start", side_effect=AudioWorkerStartError("boom"),
        ):
            self.assertFalse(self.mgr.maybe_autostart())


class AudioStopSentinelTests(_AudioWorkerTestBase):
    """优雅停止：web 写哨兵 → worker 自己收尾；超时才强杀（spec §9.3）。

    回归：最初实现直接 `taskkill /F /T`，worker 没机会跑 `manager.stop()` →
    租约仍指向死进程、`AudioRuntimeState` 停在陈旧 `ready`、日志里没有停止记录。
    """

    def test_paths_default_to_data_dir(self):
        from django.test import override_settings

        from apps.audio_detect.paths import log_file_path, pid_file_path, stop_file_path

        with override_settings(
            BABYCARE_AUDIO_PID_FILE="",
            BABYCARE_AUDIO_LOG_FILE="",
            BABYCARE_AUDIO_STOP_FILE="",
        ):
            self.assertEqual(pid_file_path().parent.name, "data")
            self.assertEqual(pid_file_path().name, "audio_worker.pid")
            self.assertEqual(log_file_path().name, "audio_worker.log")
            self.assertEqual(stop_file_path().name, "audio_worker.stop")

    def test_stop_writes_sentinel_before_waiting(self):
        """优雅路径：先写哨兵让 worker 自己收尾，而不是上手就强杀。"""
        from django.test import override_settings

        self.pid_file.write_text("4321", encoding="utf-8")
        seen = []

        def _alive(_pid):
            seen.append(self.stop_file.exists())
            return len(seen) == 1        # 初次判活=活着；之后=已优雅退出

        # 给一个正的优雅窗口，才会走"等待 → 自行退出"这条路径
        with override_settings(BABYCARE_AUDIO_STOP_GRACE_SEC=0.5), patch(
            "apps.audio_detect.worker_manager._pid_alive", side_effect=_alive,
        ):
            stopped = self.mgr.stop()

        self.assertFalse(seen[0], "初次判活时不该有哨兵")
        self.assertTrue(seen[1], "等待期间必须已写好哨兵（worker 靠它自停）")
        self.assertTrue(stopped)
        self.assertFalse(self.pid_file.exists())
        self.assertFalse(self.stop_file.exists(), "结束后哨兵要清掉")
        self.assertEqual(self._desired(), AudioServiceState.DESIRED_OFF)

    def test_stop_kills_only_when_grace_expires(self):
        """worker 不响应（卡死）→ 超时后才 taskkill /F /T 兜底。"""
        from django.test import override_settings

        self.pid_file.write_text("4321", encoding="utf-8")
        with override_settings(BABYCARE_AUDIO_STOP_GRACE_SEC=0), patch(
            "apps.audio_detect.worker_manager._pid_alive",
            # 初次 True（要停）→ 强杀前复查 True（没响应）→ 强杀后 False（已死）
            side_effect=[True, True, False],
        ), patch("apps.audio_detect.worker_manager._kill_pid_tree") as kill:
            stopped = self.mgr.stop()

        kill.assert_called_once()
        self.assertEqual(kill.call_args[0][0], 4321)
        self.assertTrue(stopped)
        self.assertFalse(self.pid_file.exists())
        self.assertFalse(self.stop_file.exists())

    def test_stop_without_process_clears_leftovers(self):
        """进程早没了：清 PID + 清哨兵，且如实返回"没停过任何东西"。"""
        self.pid_file.write_text("4321", encoding="utf-8")
        self.stop_file.write_text("stop", encoding="utf-8")
        with patch(
            "apps.audio_detect.worker_manager._pid_alive", return_value=False,
        ):
            stopped = self.mgr.stop()

        self.assertFalse(stopped)
        self.assertFalse(self.pid_file.exists())
        self.assertFalse(self.stop_file.exists())
        self.assertEqual(self._desired(), AudioServiceState.DESIRED_OFF)

    def test_start_clears_leftover_sentinel(self):
        """残留哨兵会让刚拉起的 worker 立刻自杀 → 启动前必须清掉。"""
        self.stop_file.write_text("stop", encoding="utf-8")
        with patch.object(self.mgr, "_spawn", return_value=_FakeProc(pid=777)):
            info = self.mgr.start(wait_sec=0)

        self.assertTrue(info["started"])
        self.assertFalse(self.stop_file.exists())

    def test_worker_side_detects_sentinel(self):
        """worker 侧：心跳发现哨兵 → 置位内部与命令侧 stop 事件。"""
        import threading

        from apps.audio_detect.manager import AudioCaptureManager

        external = threading.Event()
        worker = AudioCaptureManager(
            enable_inference=False, enable_describe=False, stop_event=external,
        )
        self.assertFalse(worker._stop_requested_by_file())
        self.assertFalse(worker._stop.is_set())

        self.stop_file.write_text("stop", encoding="utf-8")

        self.assertTrue(worker._stop_requested_by_file())
        self.assertTrue(worker._stop.is_set())
        self.assertTrue(external.is_set())


class AudioAppAutostartGuardTests(SimpleTestCase):
    """apps.ready() 的守卫：管理命令期不拉起（含 audio_worker 自己）。"""

    def test_audio_worker_in_skip_set(self):
        from apps.audio_detect.apps import _MGMT_SKIP

        self.assertIn("audio_worker", _MGMT_SKIP)

    def test_skips_management_commands(self):
        import sys

        from apps.audio_detect import apps as audio_apps

        for argv in (
            ["manage.py", "audio_worker"],
            ["manage.py", "migrate"],
            ["manage.py", "test"],
            ["manage.py", "cleanup_expired"],
        ):
            with patch.object(sys, "argv", argv):
                self.assertFalse(audio_apps._should_autostart(), argv)

    def test_true_for_daphne(self):
        import sys

        from apps.audio_detect import apps as audio_apps

        with patch.object(sys, "argv", ["daphne", "config.asgi:application"]):
            self.assertTrue(audio_apps._should_autostart())

    def test_runserver_reloader_parent_skips(self):
        """runserver 的 autoreload 父进程不拉起（否则与子进程双份互相破坏）。"""
        import os
        import sys

        from apps.audio_detect import apps as audio_apps

        with patch.object(sys, "argv", ["manage.py", "runserver"]), \
                patch.dict(os.environ, {}):
            os.environ.pop("RUN_MAIN", None)
            self.assertFalse(audio_apps._should_autostart())

    def test_runserver_child_starts(self):
        """真正服务的是子进程（RUN_MAIN=true），它必须拉起 worker。"""
        import os
        import sys

        from apps.audio_detect import apps as audio_apps

        with patch.object(sys, "argv", ["manage.py", "runserver"]), \
                patch.dict(os.environ, {"RUN_MAIN": "true"}):
            self.assertTrue(audio_apps._should_autostart())

    def test_respects_env_switch(self):
        import os
        import sys

        from apps.audio_detect import apps as audio_apps

        with patch.object(sys, "argv", ["daphne", "config.asgi:application"]), \
                patch.dict(os.environ, {"AUDIO_AUTOSTART": "0"}):
            self.assertFalse(audio_apps._should_autostart())

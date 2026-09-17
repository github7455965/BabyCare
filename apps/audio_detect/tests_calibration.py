"""Phase 8 标定报表测试（``calibration.py`` 统计核心 + 管理命令冒烟）。

重点
----
- 分位数（线性插值）与分布统计；
- 阈值扫描：and / or / single 三列计数（含模型缺失 = 单侧）；
- 单侧阳性明细取「当前生产阈值」缺省值的行为；
- 事件 / GAP 统计（含 ``recording`` 排除、near-gap 提示计数）；
- 管理命令：空库不炸、CSV 导出齐全。
"""
import os
import uuid
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402
from django.apps import apps as _django_apps  # noqa: E402
if not _django_apps.ready:
    django.setup()

from django.core.management import call_command  # noqa: E402
from django.test import TestCase  # noqa: E402

from apps.audio_detect.calibration import (  # noqa: E402
    event_report,
    quantile,
    reason_counts,
    score_distribution,
    single_side_samples,
    threshold_scan,
)
from apps.audio_detect.label_map import MODEL_PANNS, MODEL_YAMNET  # noqa: E402
from apps.audio_detect.models import AudioEvent, SoundDetectionLog  # noqa: E402
from apps.streaming.models import Camera  # noqa: E402

T0 = 1_700_000_000


def _name(base: str) -> str:
    return f"t_{uuid.uuid4().hex[:8]}_{base}"


def _payload(model: str, cry=None, speech=None, ok=True) -> dict:
    return {
        "model": model,
        "ok": ok,
        "scores": {
            "cry": cry,
            "speech": speech,
        },
    }


class _Base(TestCase):
    def setUp(self):
        self.cam = Camera.objects.create(
            name=_name("cam"), source_type=Camera.SOURCE_FILE, is_active=False,
        )

    def _log(self, y=(None, None), p=(None, None), end_ts=T0, **overrides):
        """y/p = (cry, speech)；None = 该模型该标签没有分数。"""
        defaults = {
            "camera": self.cam,
            "window_start_ts": end_ts - 2,
            "window_end_ts": end_ts,
            "yamnet_payload": _payload(MODEL_YAMNET, *y),
            "panns_payload": _payload(MODEL_PANNS, *p),
            "consensus_payload": {},
            "decision": SoundDetectionLog.DECISION_POSITIVE,
            "log_reason": SoundDetectionLog.LOG_POSITIVE,
        }
        defaults.update(overrides)
        return SoundDetectionLog.objects.create(**defaults)


class QuantileTest(TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(quantile([], 0.5))

    def test_single_value(self):
        self.assertEqual(quantile([5.0], 0.99), 5.0)

    def test_linear_interpolation(self):
        vals = [0.0, 10.0]
        # q=0.5 → 位置 0.5 → 插值 5.0
        self.assertAlmostEqual(quantile(vals, 0.5), 5.0)
        # q=0.25 → 位置 0.25 → 2.5
        self.assertAlmostEqual(quantile(vals, 0.25), 2.5)


class ScoreDistributionTest(_Base):
    def test_buckets_per_model_and_label(self):
        self._log(y=(0.6, 0.1), p=(0.5, 0.2))
        self._log(y=(0.8, 0.1), p=(0.4, 0.2))
        dist = score_distribution(list(SoundDetectionLog.objects.all()))
        self.assertEqual(dist[(MODEL_YAMNET, "cry")]["n"], 2)
        self.assertAlmostEqual(dist[(MODEL_YAMNET, "cry")]["min"], 0.6)
        self.assertAlmostEqual(dist[(MODEL_YAMNET, "cry")]["max"], 0.8)
        self.assertEqual(dist[(MODEL_PANNS, "speech")]["n"], 2)
        self.assertNotIn((MODEL_YAMNET, "missing"), dist)

    def test_empty(self):
        self.assertEqual(score_distribution([]), {})


class ThresholdScanTest(_Base):
    def test_and_or_single_counts(self):
        # 双侧都过 0.3 → and+or
        self._log(y=(0.5, None), p=(0.4, None))
        # 只有 yamnet 过 0.3 → or + single
        self._log(y=(0.5, None), p=(0.1, None), end_ts=T0 + 10)
        # 双侧都不过 → 不计
        self._log(y=(0.05, None), p=(0.05, None), end_ts=T0 + 20)

        scan = {
            (r["label"], r["threshold"]): r
            for r in threshold_scan(list(SoundDetectionLog.objects.all()), [0.3])
        }
        cry = scan[("cry", 0.3)]
        self.assertEqual(cry["and_n"], 1)
        self.assertEqual(cry["or_n"], 2)
        self.assertEqual(cry["single_n"], 1)

    def test_missing_one_model_counts_as_single(self):
        # panns payload 没有 scores（异常窗）→ yamnet 过阈值即 or+single
        log = self._log(y=(0.5, None), p=(None, None))
        log.panns_payload = {"model": MODEL_PANNS, "ok": False, "scores": {}}
        log.save()
        scan = {
            (r["label"], r["threshold"]): r
            for r in threshold_scan(list(SoundDetectionLog.objects.all()), [0.3])
        }
        self.assertEqual(scan[("cry", 0.3)]["or_n"], 1)
        self.assertEqual(scan[("cry", 0.3)]["single_n"], 1)

    def test_both_missing_not_counted(self):
        self._log(y=(None, None), p=(None, None))
        scan = threshold_scan(list(SoundDetectionLog.objects.all()), [0.3])
        self.assertEqual(scan[0]["or_n"], 0)


class SingleSideSamplesTest(_Base):
    def test_uses_current_threshold_from_consensus_payload(self):
        # 生产阈值快照 cry=0.3：y=0.35 过、p=0.25 不过 → 单侧
        log = self._log(y=(0.35, None), p=(0.25, None))
        log.consensus_payload = {
            "thresholds": {MODEL_YAMNET: {"cry": 0.3, "speech": 0.3}},
        }
        log.save()
        samples = single_side_samples(list(SoundDetectionLog.objects.all()))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["log_id"], log.id)
        self.assertEqual(samples[0]["positive_model"], MODEL_YAMNET)

    def test_explicit_threshold_overrides(self):
        self._log(y=(0.35, None), p=(0.25, None))
        # 阈值降到 0.2 → 双侧都过 → 不再是单侧
        samples = single_side_samples(list(SoundDetectionLog.objects.all()), threshold=0.2)
        self.assertEqual(samples, [])


class EventReportTest(_Base):
    def _event(self, start, end, **kw):
        defaults = {
            "camera": self.cam,
            "status": AudioEvent.STATUS_COMPLETED,
            "started_at_ts": start,
            "ended_at_ts": end,
            "duration_sec": float(end - start) if end is not None else 0.0,
        }
        defaults.update(kw)
        return AudioEvent.objects.create(**defaults)

    def test_excludes_recording_and_counts_gaps(self):
        self._event(T0, T0 + 5)
        self._event(T0 + 8, T0 + 12)       # start 间隔 8s（≤3×GAP=9）
        self._event(T0 + 100, T0 + 110)    # start 间隔 92s
        self._event(T0, None, status=AudioEvent.STATUS_RECORDING)  # 排除

        report = event_report(days=None, gap_sec=3.0)
        self.assertEqual(report["event_count"], 3)
        self.assertEqual(report["recording_count"], 1)
        self.assertAlmostEqual(report["duration"]["q50"], 5.0)
        self.assertEqual(report["gap"]["n"], 2)
        self.assertEqual(report["gap"]["near_gap_count"], 1)
        self.assertEqual(report["degraded_count"], 0)

    def test_desc_status_counter(self):
        self._event(T0, T0 + 10, description_status=AudioEvent.DESC_COMPLETED)
        self._event(T0 + 100, T0 + 110, description_status=AudioEvent.DESC_FAILED,
                    status=AudioEvent.STATUS_FAILED)
        report = event_report()
        self.assertEqual(report["desc_status"].get("completed"), 1)
        self.assertEqual(report["desc_status"].get("failed"), 1)


class ReasonCountsTest(_Base):
    def test_counts(self):
        self._log()
        self._log(end_ts=T0 + 10, log_reason=SoundDetectionLog.LOG_NEAR_THRESHOLD,
                  decision=SoundDetectionLog.DECISION_NEGATIVE)
        overview = reason_counts(list(SoundDetectionLog.objects.all()))
        self.assertEqual(overview["window_total"], 2)
        self.assertEqual(overview["log_reason"].get("positive"), 1)
        self.assertEqual(overview["log_reason"].get("near_threshold"), 1)


class CommandSmokeTest(_Base):
    """管理命令冒烟：空库不炸、有数据时输出关键段落、CSV 导出齐全。"""

    def test_empty_db_ok(self):
        call_command("audio_calibration_report", "--days", "0")

    def test_with_data_and_csv_export(self):
        import tempfile

        self._log(y=(0.5, 0.1), p=(0.1, 0.2))
        out = Path(tempfile.mkdtemp(prefix="bc_calib_"))
        self.addCleanup(lambda: _rmtree(out))
        call_command(
            "audio_calibration_report",
            "--days", "0",
            "--thresholds", "0.2,0.3",
            "--csv", str(out),
        )
        self.assertTrue((out / "scores.csv").exists())
        self.assertTrue((out / "single_side.csv").exists())
        self.assertTrue((out / "events_summary.json").exists())


def _rmtree(p: Path) -> None:
    import shutil

    shutil.rmtree(str(p), ignore_errors=True)

"""Phase 8：阈值标定报表（spec §12 P0-5 / P0-6 推迟项 + Phase 8 动作 1~3）。

用法
----
::

    # 默认最近 7 天、候选阈值 0.05 ~ 0.90
    python manage.py audio_calibration_report

    # 指定时间窗 / 阈值网格 / 导出明细 CSV
    python manage.py audio_calibration_report --days 14 --thresholds 0.2,0.3,0.4 --csv data/calib

输出四块 + 人工步骤清单：

1. 样本概况（落库原因 / 决策 / 失败原因计数）；
2. 分数分布（per 模型 × 标签分位数）→ 圈定候选阈值区间；
3. 阈值扫描（and / or / 单侧阳性窗数）→ 按可接受的通知频率反推阈值；
4. 事件与 GAP 复核（时长分布、相邻间隔、描述状态）。

**只做统计，不定值** —— 拍板（试听 + 改 ``.env``）是人工步骤，见输出末尾清单。
"""
from __future__ import annotations

import csv
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.audio_detect.calibration import (
    _model_scores,
    event_report,
    load_logs,
    quantile,
    reason_counts,
    score_distribution,
    single_side_samples,
    threshold_scan,
)
from apps.audio_detect.label_map import MODEL_PANNS, MODEL_YAMNET

DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"

LABEL_DISPLAY = {"cry": "哭声", "speech": "说话声"}


class Command(BaseCommand):
    help = "Phase 8 阈值标定报表：分数分布 + 阈值扫描 + 事件/GAP 复核（统计自动化，拍板留给人工）"

    def add_arguments(self, parser):
        parser.add_argument(
            "--days", type=int, default=7,
            help="统计最近 N 天（按 window_end_ts / started_at_ts）；0 = 全部",
        )
        parser.add_argument(
            "--thresholds", type=str, default=DEFAULT_THRESHOLDS,
            help="逗号分隔的候选阈值列表",
        )
        parser.add_argument(
            "--single-limit", type=int, default=30,
            help="单侧阳性窗明细最多列多少条",
        )
        parser.add_argument(
            "--csv", type=str, default="",
            help="导出明细 CSV 的目录（分数明细 / 单侧明细 / 事件明细）；空 = 不导出",
        )

    # ------------------------------------------------------------------
    def handle(self, *args, **opts):
        days = int(opts["days"]) or None
        try:
            thresholds = [
                float(x) for x in str(opts["thresholds"]).split(",") if x.strip()
            ]
        except ValueError as e:
            raise CommandError(f"--thresholds 解析失败: {e}") from e
        if not thresholds:
            raise CommandError("--thresholds 为空")

        logs = load_logs(days=days)
        gap_sec = float(getattr(settings, "BABYCARE_AUDIO_EVENT_GAP_SEC", 3.0))

        self._section("1. 样本概况")
        overview = reason_counts(logs)
        if not logs:
            self.stdout.write(self.style.WARNING(
                "  最近没有落库的检测窗 —— 音频线还没跑过（或跑的时间太短）。"
                "先让 worker 正常采集积累几天再跑本报表。"
            ))
        self._print_counts(overview)

        self._section("2. 分数分布（per 模型 × 标签）")
        dist = score_distribution(logs)
        if not dist:
            self.stdout.write("  （无样本）")
        self._print_distribution(dist)

        self._section("3. 阈值扫描（落库窗内按候选阈值重判）")
        scan = threshold_scan(logs, thresholds)
        self._print_scan(scan, logs_count=len(logs))

        self._section("4. 单侧阳性窗明细（人工复核误报 / 漏报的重点对象）")
        samples = single_side_samples(logs, limit=int(opts["single_limit"]))
        self._print_samples(samples)

        self._section("5. 事件与 GAP 复核")
        report = event_report(days=days, gap_sec=gap_sec)
        self._print_events(report, gap_sec)

        self._section("人工步骤（spec Phase 8，试听与拍板由你完成）")
        self._print_manual_steps(days)

        csv_dir = str(opts["csv"] or "").strip()
        if csv_dir and logs:
            self._export_csv(csv_dir, logs, samples, days)

    # ------------------------------------------------------------------
    def _section(self, title: str) -> None:
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING(f"== {title} =="))

    def _print_counts(self, overview: dict) -> None:
        self.stdout.write(f"  落库窗总数: {overview['window_total']}")
        for key in ("log_reason", "decision", "failure_reason"):
            self.stdout.write(f"  {key}: {overview[key]}")

    def _print_distribution(self, dist: dict) -> None:
        header = f"  {'model/label':<20}{'n':>6}{'min':>7}{'q25':>7}{'q50':>7}{'q75':>7}{'q90':>7}{'q95':>7}{'q99':>7}{'max':>7}"
        self.stdout.write(header)
        for (model, label), stats in sorted(dist.items()):
            name = f"{model}/{LABEL_DISPLAY.get(label, label)}"
            row = f"  {name:<20}{stats['n']:>6}"
            for key in ("min", "q25", "q50", "q75", "q90", "q95", "q99", "max"):
                v = stats.get(key)
                row += f"{v:>7.3f}" if isinstance(v, float) else f"{'-':>7}"
            self.stdout.write(row)
        self.stdout.write(self.style.WARNING(
            "  [注意] 稀疏落库：样本偏向「有动静」的窗口，分布不代表全体窗口（spec §5.1）。"
        ))

    def _print_scan(self, scan: list[dict], logs_count: int) -> None:
        self.stdout.write(
            f"  {'label':<10}{'threshold':>10}{'and':>8}{'or':>8}{'single':>8}"
        )
        for row in scan:
            name = LABEL_DISPLAY.get(row["label"], row["label"])
            self.stdout.write(
                f"  {name:<10}{row['threshold']:>10.2f}{row['and_n']:>8}"
                f"{row['or_n']:>8}{row['single_n']:>8}"
            )
        self.stdout.write(
            "  说明：and=双模型都≥t；or=任一≥t；single=or 与 and 之差（单侧阳性）。\n"
            "  阈值越低 and/or 越多 → 通知越频繁。以「每窗落库占比」估算相对变化即可，\n"
            "  绝对误报率需要人工试听单侧阳性窗（见下一节）后再判断。"
        )

    def _print_samples(self, samples: list[dict]) -> None:
        if not samples:
            self.stdout.write("  （无单侧阳性窗 —— and/or 策略差异目前不可见）")
            return
        self.stdout.write(
            f"  {'log':>8}  {'window_end':<20}{'cam':>4}  {'label':<8}{'yamnet':>8}{'panns':>8}  正侧"
        )
        for s in samples:
            self.stdout.write(
                f"  {s['log_id']:>8}  {s['window_end_ts']:<20}{str(s['camera_id']):>4}"
                f"  {s['label']:<8}"
                f"{self._fmt(s['yamnet']):>8}{self._fmt(s['panns']):>8}  {s['positive_model']}"
            )
        self.stdout.write("  → 对照「声音检测日志」页（按时间筛）与音频事件录音试听。")

    def _print_events(self, report: dict, gap_sec: float) -> None:
        dur = report["duration"]
        gap = report["gap"]
        self.stdout.write(
            f"  已结束事件: {report['event_count']}（录制中 {report['recording_count']}，"
            f"degraded {report['degraded_count']}）"
        )
        self.stdout.write(f"  描述状态: {report['desc_status']}")
        self.stdout.write(
            f"  事件时长(结束-开始): n={dur['n']} q50={self._fmt(dur['q50'])}"
            f" q90={self._fmt(dur['q90'])} max={self._fmt(dur['max'])}"
        )
        self.stdout.write(
            f"  相邻事件间隔: n={gap['n']} min={self._fmt(gap['min'])}"
            f" q50={self._fmt(gap['q50'])} max={self._fmt(gap['max'])}"
            f"（当前 EVENT_GAP_SEC={gap_sec}，≤3×GAP 的间隔 {gap['near_gap_count']} 个）"
        )
        if gap["near_gap_count"] and gap["n"]:
            ratio = gap["near_gap_count"] / gap["n"]
            if ratio > 0.3:
                self.stdout.write(self.style.WARNING(
                    f"  [注意] {ratio:.0%} 的间隔落在 3×GAP 内：可能有事件被拆散，考虑调大 EVENT_GAP_SEC。"
                ))
            else:
                self.stdout.write("  （间隔分布健康，GAP 无需动）")

    def _print_manual_steps(self, days) -> None:
        self.stdout.write(
            "\n"
            "  1) 让音频线连续跑满观察期（验收：24h 无异常、CPU/显存符合 P0-4 预期）；\n"
            "  2) 在「声音检测日志」页按「单侧阳性」筛 → 对照时间到「音频事件」详情试听录音；\n"
            "  3) 结合第 2/3 节：定 per-model 阈值（YAMNet / PANNs 分开，改 .env 的\n"
            "     BABYCARE_AUDIO_{CRY,SPEECH}_THRESHOLD_{YAMNET,PANNS}）+ 共识策略\n"
            "     （BABYCARE_AUDIO_CONSENSUS = and|or），改完重启 daphne + worker；\n"
            "  4) 复核 EVENT_GAP_SEC（第 5 节提示）/ 启动规则（2/3 窗）/ 预录后录 ——\n"
            "     有疑义时回 spec §6/P0-7 结论对照。\n"
        )

    def _export_csv(self, csv_dir: str, logs: list, samples: list, days) -> None:
        out = Path(csv_dir)
        out.mkdir(parents=True, exist_ok=True)

        with (out / "scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["log_id", "camera_id", "window_end_ts", "model", "label", "score"])
            for log in logs:
                for model in (MODEL_YAMNET, MODEL_PANNS):
                    for label, score in _model_scores(log, model).items():
                        w.writerow([
                            log.id, log.camera_id, log.window_end_ts, model, label, score,
                        ])

        with (out / "single_side.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["log_id", "camera_id", "window_end_ts", "label", "yamnet", "panns", "positive_model"])
            for s in samples:
                w.writerow([
                    s["log_id"], s["camera_id"], s["window_end_ts"], s["label"],
                    s["yamnet"], s["panns"], s["positive_model"],
                ])

        report = event_report(days=days)
        with (out / "events_summary.json").open("w", encoding="utf-8") as f:
            import json

            json.dump(report, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS(f"  明细已导出到 {out}"))

    @staticmethod
    def _fmt(v) -> str:
        return f"{v:.1f}" if isinstance(v, float) else "-"

"""音频描述服务（Phase 4，spec §6.4/§6.5）：事件分段 → 模型描述/转写 → 聚合落库。

在音频 worker 进程里跑一个后台线程，轮询 ``AudioEvent``：

.. code-block:: text

    pending_description ──► describing ──► completed
                               │
                               └──► failed（分段重试耗尽 / 音频不可读）

关键语义
--------
- **分段用事件真实起止**（``started_at_ts`` / ``ended_at_ts``），不是录音时长——
  录音含 5s 前导 + 3s 尾巴，拿录音当内容边界会把静音尾巴喂给模型（spec §6.2）；
- 分片失败 → 整个事件**不能标完成**，保留分段与错误状态，下一轮按退避重试
  （spec §6.4）；重试耗尽或音频不可读 → 事件 failed，**录音不删**（spec §12）；
- 描述服务不可用（网络错）只影响描述：事件仍可播放，描述稍后自动追上（spec §6.5）；
- 模型返回必须过 schema/类型/时间范围校验，**非法结果不入库**（spec §6.5）；
- **每轮都按规划校验分段行是否齐全**：建段中途被打断（DB 异常 / worker 被杀）后
  剩余段会被补建，不会出现"已有段全完成 → 事件误判 completed、后半段音频静默丢失"；
- **整事件处于退避期时本轮跳过**：候选按时间取前 N 个时主动让位给新事件，避免
  队头阻塞导致新事件长时间排不进来（描述服务长时间不可用时尤其明显）。

两种输出方言（provider）
------------------------
同一套分段/重试/落库流程，只换"提示词 + 返回值归一化"两步：

= ============== ==================================================== ===============
   provider       模型形态                                              归一化
= ============== ==================================================== ===============
A  ``transcript`` 纯转写（Qwen3-ASR-1.7B）：只回一段人声文字           :func:`normalize_transcript`
B  ``json``       结构化描述（MOSS-Audio-4B / Qwen2.5-Omni-7B）：回 JSON :func:`parse_description_json`
= ============== ==================================================== ===============

两条路最后都产出同一份 `description_json`（`description` / `background_sounds` /
`has_cry` / `has_adult_speech` …），再走同一个 `aggregate_descriptions`，
所以页面、通知、数据迁移都不需要知道用的是哪个模型。

``transcript`` 方言下 `has_cry` / `has_adult_speech` / `background_sounds` 等判定
字段**恒为 `None`（未判定）**，不是 `False`："有没有哭、有没有东西掉地上"本来就由
声学侧（YAMNet/PANNs + `detected_labels`）判定，描述模型不参与（spec §6.5）；
拿"没做判断"冒充"判定为否"是灌脏数据。

同一条路下 `description` **可以是空串**：转写模型什么都没转出来，就说明这一段确实
没有可转写的人声 —— 这是合法结果，不判失败、也不塞占位文本（`require_description`
开关见 `description_schema.validate_description`）。

可注入性
--------
默认 FLAC/WAV 转换走 soundfile（仅音频 venv 有）；单测注入内存替身，
因此本模块在主 venv（无 soundfile / requests）里也能完整单测。
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from django.utils import timezone

from .assembler import ts_to_dt
from .description_schema import (
    aggregate_descriptions,
    parse_description_json,
    validate_description,
)
from .models import AudioEvent, AudioEventSegment
from .audio_desc_client import (
    AudioDescError,
    AudioDescNetworkError,
    AudioDescTimeoutError,
    AudioDescUnreachableError,
)
from .segmenter import SegmentPlan, SegmenterConfig, plan_segments

logger = logging.getLogger(__name__)

#: 输出方言：纯转写模型（Qwen3-ASR 等）
DESC_PROVIDER_TRANSCRIPT = "transcript"
#: 输出方言：结构化描述模型（MOSS-Audio / Qwen2.5-Omni 等）
DESC_PROVIDER_JSON = "json"
DESC_PROVIDERS = (DESC_PROVIDER_TRANSCRIPT, DESC_PROVIDER_JSON)


def normalize_provider(value: object) -> str:
    """`.env` 里的 provider 归一：非法值按 ``transcript``（当前生产配置）处理。"""
    text = str(value or "").strip().lower()
    return text if text in DESC_PROVIDERS else DESC_PROVIDER_TRANSCRIPT


_CLOSE_CONN_EVERY = 30

#: 每轮最多处理的事件数（内部调度上限，非业务配置；候选会多取一些再筛选）
_MAX_EVENTS_PER_ROUND = 20
#: 分段补建连续失败多少轮后判定事件永久失败
_INCOMPLETE_MAX_ROUNDS = 5
#: 单条模型原始返回的留存上限（字符）：防止服务返回超长文本撑爆内存
_RAW_TEXT_LIMIT = 4000
#: 进程内 raw 暂存总量上限（字符）：超出时按事件淘汰最久未写入的
_RAW_STORE_LIMIT = 200_000


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class DescriberConfig:
    poll_sec: float = 5.0
    retry_backoff_sec: float = 60.0      # 分段失败后至少隔这么久才重试
    max_segment_retries: int = 8         # 单段最多重试次数，耗尽 → 事件 failed
    max_tokens: int = 1024
    sample_rate: int = 16000
    #: 输出方言（`BABYCARE_AUDIO_DESC_PROVIDER`），决定提示词与返回值归一化
    provider: str = DESC_PROVIDER_TRANSCRIPT

    def __post_init__(self) -> None:
        """方言**一律在入口归一**，不依赖调用方传对。

        下游有三处按 ``provider == DESC_PROVIDER_JSON`` 分支（提示词、归一化、
        是否要求非空描述）。若只在这里存原样字符串（如 ``"JSON"``），三处会各自
        比较、结果不一致：提示词走 json 方言、归一化走 transcript —— JSON 原文会被
        当成转写正文原样入库，而且静默不报错。归一放在构造时是唯一能保证一致的点。
        """
        self.provider = normalize_provider(self.provider)

    @classmethod
    def from_settings(cls) -> "DescriberConfig":
        from django.conf import settings

        return cls(
            poll_sec=float(settings.BABYCARE_AUDIO_DESCRIBE_POLL_SEC),
            retry_backoff_sec=float(settings.BABYCARE_AUDIO_DESCRIBE_RETRY_BACKOFF_SEC),
            max_segment_retries=int(settings.BABYCARE_AUDIO_DESCRIBE_MAX_SEGMENT_RETRIES),
            max_tokens=int(settings.BABYCARE_AUDIO_DESCRIBE_MAX_TOKENS),
            sample_rate=int(settings.BABYCARE_AUDIO_SAMPLE_RATE),
            provider=normalize_provider(
                getattr(settings, "BABYCARE_AUDIO_DESC_PROVIDER", "")
            ),
        )


# ---------------------------------------------------------------------------
# 默认音频 I/O（soundfile；测试注入替身）
# ---------------------------------------------------------------------------
def default_flac_reader(path: Path):
    """读 FLAC → (float32 pcm, sample_rate)。"""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    return data, int(sr)


def default_flac_writer(path: Path, pcm: Any, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sample_rate, format="FLAC", subtype="PCM_16")


def default_wav_encoder(pcm: Any, sample_rate: int) -> bytes:
    """内存转 WAV bytes（spec §6.5：不落中间文件）。"""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def segment_audio_path(media_root: str | Path, event: AudioEvent, sequence: int) -> Path:
    """分段 FLAC 落盘路径：MEDIA_ROOT/audio_segments/<YYYY-MM-DD>/<event>_seg<n>.flac。"""
    dt = ts_to_dt(event.started_at_ts)
    date_dir = dt.strftime("%Y-%m-%d") if dt else "unknown-date"
    return (
        Path(media_root) / "audio_segments" / date_dir
        / f"{event.id}_seg{sequence}.flac"
    )


# ---------------------------------------------------------------------------
# Prompt（spec §6.5）
#
# 提示词按 provider 分两套，**不要混用**：json 方言的提示词喂给纯转写模型会
# 被当成待转写正文（模型会把 JSON 模板原样念出来），反之亦然。
# ---------------------------------------------------------------------------
#: transcript 方言下"本片段没有人声"的占位文本。
#: **只用在提示词里**（告诉模型没人声时输出什么）；入库时不做识别、不做替换 ——
#: 模型回什么就存什么（:func:`normalize_transcript`）。
NO_SPEECH_TEXT = "（无说话声）"

#: Qwen3-ASR（llama.cpp 侧）把结果包成 ``language Chinese<asr_text>正文``：
#: 前面的语言标记 + 正文标签对页面/聚合都是噪声，归一化时剥掉。
_ASR_TEXT_TAGS = ("<asr_text>", "<|asr_text|>")


def build_prompt(segment: AudioEventSegment) -> str:
    """``json`` 方言（MOSS-Audio / Qwen2.5-Omni 等）的片段提示词。

    交代中性背景 + 片段位置，让模型分两块描述。两个输出字段（一个说说话、
    一个说背景音，互不混淆）：

    - ``description``：**说话内容**（谁在说、说了什么，能写原话更好）；
    - ``background_sounds``：**背景音描述**（说话以外的其他声音，如哭声、东西掉地上、
      洗衣机、电视声），没有就写"无"。

    **场景提示只给中性的一句**（"家庭室内环境的录音"）：早先版本写的是"家里可能有爸爸、
    妈妈、保姆和宝宝，也可能有电视、风扇等背景声"，实测被模型**原样复读**进输出
    （`adult_speech_summary` 直接吐出"爸爸妈妈保姆宝宝"）；完全不给背景又会让说话人归属
    退化成"一名女性"甚至转错（"妈妈说"→"我妈妈，妈妈"，见工程计划 Phase 4 补充）。

    同样刻意**不做"问卷"**（不索要 ``has_cry`` / ``cry_duration_sec`` /
    ``speech_confidence`` 这类判定）：把 ``has_cry`` 摆在第一行会诱导模型"往哭上答"
    （实测 5/5 片段报整段在哭，而同期声学侧 cry 分数只有 0.001~0.05）；判定交回声学侧
    （YAMNet/PANNs + ``detected_labels``），描述模型只负责"描述"。

    也不再喂静音区间/采样率/事件真实起止——分段已经按静音切好了，事件起止对"这个片段里
    听到了什么"没有帮助，模型不需要这些数字。**`event` 参数因此一并去掉**（新提示词只用
    得到片段自身的偏移，留着是死参数）。
    """
    seg_dur = segment.end_offset - segment.start_offset
    return (
        "你是家庭监控音频的分析助手。\n"
        "背景：这是家庭室内环境的录音。\n"
        f"这段音频是某次声音事件中的一个片段：时长 {seg_dur:.1f} 秒，"
        f"在事件中位于第 {segment.start_offset:.1f} ~ {segment.end_offset:.1f} 秒。\n"
        "请听完后只输出一个 JSON 对象：\n"
        '{"description": "具体描述听到的内容：有人在说话就把说话内容写清楚'
        '（谁在说、说了什么，尽量具体，能写出原话更好）；没人说话就描述主要是什么声音",\n'
        ' "background_sounds": "说话以外的其他声音，'
        '如哭声、东西掉地上、洗衣机、电视声、水流、关门"}\n'
        "只写你真的从音频里听到的；听不清的部分写“听不清”，不要编造。"
    )


def build_transcript_prompt() -> str:
    """``transcript`` 方言（Qwen3-ASR-1.7B 等纯转写模型）的片段提示词。

    刻意**极短**，且不带任何"片段时长/在事件中的位置"之类的上下文：

    - 转写模型不需要任务背景，喂长提示词只会挤占上下文、抬高首字延迟；
    - 只要求"逐字转写 + 不翻译不总结"，避免模型顺手做概括（概括会丢原话，
      而原话正是这一路最大的价值）；
    - 明确交代"没有人声时输出什么"（:data:`NO_SPEECH_TEXT`），否则静音片段
      会被填空成幻觉文本。它是**模型自己**的输出，入库时不会被改写/识别
      （见 :func:`normalize_transcript`）。

    **不给 JSON 模板**：json 方言的模板喂给转写模型会被当成正文朗读出来。

    不做参数是因为它真的用不到片段信息（同 :func:`build_prompt` 里"死参数"
    的取舍）；方言派发由 :func:`build_prompt_for` 负责，调用方签名保持统一。
    """
    return (
        "请把这段音频里说的话逐字转写出来，不要翻译、不要总结、不要补充说明。\n"
        f"如果这段音频里没有人说话，只输出：{NO_SPEECH_TEXT}"
    )


def build_prompt_for(segment: AudioEventSegment, provider: str) -> str:
    """按 provider 选提示词（`BABYCARE_AUDIO_DESC_PROVIDER`）。"""
    if normalize_provider(provider) == DESC_PROVIDER_JSON:
        return build_prompt(segment)
    return build_transcript_prompt()


def strip_asr_wrapper(text: str) -> str:
    """剥掉转写模型自带的 ``language XX<asr_text>`` 外壳（没有就原样返回）。

    实测 Qwen3-ASR（llama.cpp 侧）返回的是 ``language Chinese<asr_text>正文``：
    ``language Chinese`` 是它自己吐的语言标记，``<asr_text>`` 是正文起点。这层壳
    对页面和聚合都是噪声（会出现在"结构化描述"与通知文案里），所以在归一化入口
    一次性剥掉；返回纯文本的模型不受影响。
    """
    cut = -1
    for tag in _ASR_TEXT_TAGS:
        idx = text.rfind(tag)
        if idx >= 0:
            cut = max(cut, idx + len(tag))
    return text[cut:].strip() if cut >= 0 else text


def normalize_transcript(raw: str) -> tuple[dict | None, str]:
    """``transcript`` 方言：转写原文 → 与 json 方言同构的待校验 dict。

    **只做协议层清洗，不做任何内容加工**：

    - 剥掉模型自带的 ``language XX<asr_text>`` 外壳（llama.cpp 侧的输出格式，
      属于传输包装，不是内容）；
    - 去掉偶发的代码围栏、首尾引号，把换行/连续空白压成单空格（页面与聚合里
      表现更稳）。

    然后**原文照放**进 ``description``：

    - 模型回了什么就是什么。它按提示词回的“无说话声”占位也照样留着——那是它
      自己的输出，而且对读的人有信息量，不该被我们改写成别的措辞；
    - 壳里是空的（纯静音，模型只回 ``language Chinese<asr_text>``）→
      ``description`` 就是空串，这是一条**合法结果**（见
      :func:`validate_description` 的 ``require_description``），
      不塞占位文本、也不判失败。

    **不做任何判定**：不填 ``has_adult_speech`` / ``has_cry`` /
    ``background_sounds`` / 哭声时间 / ``speech_confidence`` —— 转写模型既不知道
    有没有哭，也分不清“真人说话”和“电视里的人说话”；这些判断归声学侧
    （YAMNet/PANNs + ``detected_labels``）。缺的字段由
    :func:`validate_description` 统一写成 ``None``（未判定）。

    失败与“没人声”的分界
    --------------------
    整条响应是空白 → 判失败（交给退避重试）：服务或提示词出问题了，静默当成
    “这段没人声”会把故障藏起来。模型回了内容、只是壳里空 → 正常完成、空描述。
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty response"
    text = strip_asr_wrapper(text)
    if text.startswith("```"):                   # 个别模型爱加代码围栏
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("```")
        ).strip()
    text = text.strip("“”\"'「」『』").strip()
    # 折行/多空格压成单空格：页面与聚合里表现更稳，也省得下游再去处理换行
    text = " ".join(text.split())
    return {"description": text}, ""


# ---------------------------------------------------------------------------
# 进程内 raw 暂存（必须有上限）
# ---------------------------------------------------------------------------
class _RawStore:
    """各分段模型原始返回的进程内暂存，事件收尾时写入 ``description_raw``。

    只活在描述服务进程里，所以**必须设上限**：事件长期停在 describing（分段反复
    重试 / 等退避）时，已完成分段的原始文本会一直挂着。

    - 单条 raw 截断到 ``_RAW_TEXT_LIMIT``：防止服务无视 ``max_tokens`` 返回超长文本；
    - 总量超过 ``_RAW_STORE_LIMIT`` 时按事件淘汰**最久未写入**的（记录新分段会把
      事件移到最新端），保证正在推进的事件不被淘汰；
    - 始终保留至少一个事件，避免把单个大事件自己淘汰掉。
    """

    def __init__(
        self,
        text_limit: int = _RAW_TEXT_LIMIT,
        total_limit: int = _RAW_STORE_LIMIT,
    ):
        self._text_limit = max(int(text_limit), 1)
        # 总量下限取单条上限：总量比单条还小的话，淘汰也腾不出位置
        self._total_limit = max(int(total_limit), self._text_limit)
        self._entries: OrderedDict[int, dict[int, str]] = OrderedDict()
        self._chars = 0

    def record(self, event_id: int, sequence: int, text: str) -> None:
        """记一段原始返回（同段覆盖），并把该事件刷新为最新。"""
        text = (text or "")[:self._text_limit]
        segs = self._entries.pop(event_id, None)
        if segs is None:
            segs = {}
        else:
            self._chars -= self._entry_chars(segs)
        segs[int(sequence)] = text
        self._entries[event_id] = segs
        self._chars += self._entry_chars(segs)
        self._evict()

    def pop(self, event_id: int) -> dict[int, str]:
        """取走并清除某事件的全部 raw（不存在时返回空 dict）。"""
        segs = self._entries.pop(event_id, None)
        if not segs:
            return {}
        self._chars -= self._entry_chars(segs)
        return segs

    def _evict(self) -> None:
        while self._chars > self._total_limit and len(self._entries) > 1:
            _, segs = self._entries.popitem(last=False)
            self._chars -= self._entry_chars(segs)

    @staticmethod
    def _entry_chars(segs: dict[int, str]) -> int:
        return sum(len(v) for v in segs.values())

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# 描述服务
# ---------------------------------------------------------------------------
class DescribeService:
    """后台线程：轮询待描述事件，分段送描述模型（转写 / 结构化），聚合落库。"""

    def __init__(
        self,
        client: Any,                 # AudioDescClient 或测试替身（.describe/.model）
        config: DescriberConfig | None = None,
        segmenter: SegmenterConfig | None = None,
        media_root: str | Path = "",
        flac_reader: Callable = default_flac_reader,
        flac_writer: Callable = default_flac_writer,
        wav_encoder: Callable = default_wav_encoder,
        worker_epoch: str = "",
        stale_epochs: set[str] | None = None,
    ):
        self._client = client
        self._cfg = config or DescriberConfig.from_settings()
        self._seg_cfg = segmenter or SegmenterConfig.from_settings()
        self._media_root = media_root
        #: 产出分段行的 worker 租约 epoch（写进行，供 recover 限定范围）
        self._worker_epoch = worker_epoch
        #: recover 允许回收的 epoch（默认 {""} = 引入 worker_epoch 之前的历史行）
        self._stale_epochs = {""} if stale_epochs is None else set(stale_epochs)
        self._read_flac = flac_reader
        self._write_flac = flac_writer
        self._encode_wav = wav_encoder

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: 本次进程内各分段的模型原始返回（完成后写入 event.description_raw）
        self._raws = _RawStore()
        #: 分段补建连续失败的轮数（内存计数，收尾时清理）
        self._incomplete_rounds: dict[int, int] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        recovered = self.recover_stale_descriptions(self._stale_epochs)
        if recovered:
            logger.warning("[audio] 回收了 %d 个上次中断的描述任务", recovered)
        t = threading.Thread(target=self._loop, name="audio-describe", daemon=True)
        self._thread = t
        t.start()
        b = self._breaker().snapshot()
        logger.info(
            "[audio] describe service started: model=%s provider=%s poll=%.1fs "
            "max_retries=%d breaker=%s(threshold=%d cooldown=%.0fs probe=%.0fs) "
            "connect_timeout=%.1fs",
            getattr(self._client, "model", "?"), self._cfg.provider,
            self._cfg.poll_sec, self._cfg.max_segment_retries,
            # 把熔断配置打进启动日志：分体部署时 "页面上没看到熔断" 与
            # "熔断没配" 是两回事，这一行让人一眼能分辨。
            "on" if b["enabled"] else "off", b["fail_threshold"],
            b["cooldown_sec"], b["probe_timeout_sec"],
            getattr(self._client, "connect_timeout_sec", 0.0),
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("[audio] describe service stopped")

    @staticmethod
    def recover_stale_descriptions(stale_epochs: set[str] | None = None) -> int:
        """worker 崩溃后重启：describing 事件回到待描述，processing 分段回到 pending。

        ``stale_epochs`` 限定回收范围（默认 ``{""}`` = 引入 ``worker_epoch`` 之前
        的历史行）：只回收"本次接管前那个租约"产出的行，其它 epoch 的行一律不动。
        """
        epochs = {""} if stale_epochs is None else set(stale_epochs)
        n = AudioEvent.objects.filter(
            status=AudioEvent.STATUS_DESCRIBING, worker_epoch__in=epochs,
        ).update(
            status=AudioEvent.STATUS_PENDING_DESCRIPTION,
            description_status=AudioEvent.DESC_PENDING,
        )
        AudioEventSegment.objects.filter(
            description_status=AudioEvent.DESC_PROCESSING, worker_epoch__in=epochs,
        ).update(description_status=AudioEvent.DESC_PENDING)
        return n

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        tick = 0
        while not self._stop.wait(max(self._cfg.poll_sec, 0.5)):
            tick += 1
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                logger.exception("[audio] describe cycle failed")
            if tick % _CLOSE_CONN_EVERY == 0:
                try:
                    from django.db import close_old_connections

                    close_old_connections()
                except Exception:  # noqa: BLE001
                    pass

    def _breaker(self):
        """本进程的 ASR 熔断器（与 VLM 侧共用组件，但状态各自独立）。"""
        from apps.core.llm_breaker import LLMBreaker
        from apps.core.models import LLMHealthState

        return LLMBreaker.for_service(LLMHealthState.SERVICE_ASR)

    def _reap_expired(self, now) -> int:
        """超期收割：把老得不可能再补的事件标成 `expired`。

        为什么必须有它
        -------------
        newest-first 之后，候选窗口只有 `_MAX_EVENTS_PER_ROUND × 10 = 200` 条
        （见 `run_once`）。积压超过 200 时，只要持续有新事件，**老事件永远进不了
        窗口**，会永远停在 `describing` —— 控制页"待描述"数字永不归零，分不清
        "还在攒"还是"卡死"。这里给它们一个显式出口。

        复用现成的 `skipped: too_short` 那段模式：标 `completed` + 写
        `description_json={"skipped": "expired"}`，**不动状态机、不加 status 字段**。

        阈值 `BABYCARE_LLM_QUEUE_MAX_AGE_SEC <= 0` → 不收割
        （默认，用户 2026-09-16 定："先给 0，上线再调"）。
        """
        from django.conf import settings

        max_age = float(
            getattr(settings, "BABYCARE_LLM_QUEUE_MAX_AGE_SEC", 0) or 0
        )
        if max_age <= 0:
            return 0
        cutoff = int(now.timestamp()) - int(max_age)
        n = AudioEvent.objects.filter(
            status__in=[
                AudioEvent.STATUS_PENDING_DESCRIPTION,
                AudioEvent.STATUS_DESCRIBING,
            ],
            started_at_ts__lt=cutoff,
        ).update(
            status=AudioEvent.STATUS_COMPLETED,
            description_status=AudioEvent.DESC_COMPLETED,
            description_json={"skipped": "expired"},
        )
        if n:
            logger.info(
                "[audio] 超期收割 %d 个事件（started_at_ts < %d，"
                "QUEUE_MAX_AGE_SEC=%.0f）",
                n, cutoff, max_age,
            )
        return n

    def run_once(self) -> list[int]:
        """处理一轮候选事件（可被单测直接调用）；返回处理过的事件 id。"""
        # 0) 服务不可达（熔断打开）→ **整轮直接返回**：不挑事件、不尝试。
        #
        #    这是"攒着"能成立的关键：**不尝试 → 不涨 `retry_count`** →
        #    `_finalize_event` 的 exhausted 判定自动就正确（一行都不用改）。
        #    顺带不占本轮 `_MAX_EVENTS_PER_ROUND` 名额、不产生无意义的 DB 写。
        if self._breaker().is_blocking():
            return []

        now = timezone.now()
        self._reap_expired(now)

        # 候选多取一些（退避中的会被筛掉），保证新事件能排进来。
        # **newest-first**（用户定：告警时效 > 完整性）—— `-started_at_ts`，
        # 最新的事件先描述；老的靠 `_reap_expired` 给出口。
        ids = list(
            AudioEvent.objects.filter(
                status__in=[
                    AudioEvent.STATUS_PENDING_DESCRIPTION,
                    AudioEvent.STATUS_DESCRIBING,
                ],
            ).order_by("-started_at_ts").values_list("id", flat=True)[
                :_MAX_EVENTS_PER_ROUND * 10
            ]
        )
        done = []
        for event_id in self._select_candidates(ids, now):
            try:
                if self._process_event(event_id):
                    done.append(event_id)
            except Exception:  # noqa: BLE001
                logger.exception("[audio] 事件 #%s 描述流程异常", event_id)
        return done

    def _select_candidates(self, ids: list[int], now) -> list[int]:
        """从候选里挑出本轮真正要处理的事件，最多 ``_MAX_EVENTS_PER_ROUND`` 个。

        **整事件处于退避期时本轮跳过**（未完成分段全部是 failed、且都还没到退避
        时间）：它们既不该占用处理名额，也不该把排在后面的新事件饿死。单个事件内
        只有部分分段退避时照常处理，段级退避由 :meth:`_process_event` 判断。

        未完成分段一次性按事件分组查出，避免逐个候选查库。
        """
        if not ids:
            return []
        unfinished: dict[int, list] = {event_id: [] for event_id in ids}
        for row in AudioEventSegment.objects.filter(
            audio_event_id__in=ids,
        ).exclude(
            description_status=AudioEvent.DESC_COMPLETED,
        ).values("audio_event_id", "description_status", "updated_at"):
            unfinished[row["audio_event_id"]].append(row)

        picked: list[int] = []
        for event_id in ids:
            segs = unfinished[event_id]
            backing_off = bool(segs) and all(
                s["description_status"] == AudioEvent.DESC_FAILED
                and (now - s["updated_at"]).total_seconds() < self._cfg.retry_backoff_sec
                for s in segs
            )
            if backing_off:
                continue
            picked.append(event_id)
            if len(picked) >= _MAX_EVENTS_PER_ROUND:
                break
        return picked

    # ------------------------------------------------------------------
    # 单事件处理
    # ------------------------------------------------------------------
    def _process_event(self, event_id: int) -> bool:
        """认领并推进一个事件；返回是否真正处理了它。"""
        claimed = AudioEvent.objects.filter(
            id=event_id,
            status__in=[
                AudioEvent.STATUS_PENDING_DESCRIPTION,
                AudioEvent.STATUS_DESCRIBING,
            ],
        ).update(
            status=AudioEvent.STATUS_DESCRIBING,
            description_status=AudioEvent.DESC_PROCESSING,
        )
        if not claimed:
            return False
        event = AudioEvent.objects.get(id=event_id)

        # 事件太短 → 直接完成（spec §6.4：<1s 丢弃，不送描述模型）
        duration = self._real_duration(event)
        if duration is None:
            self._fail_event_permanently(event, "缺少结束时间或音频路径，无法描述")
            return True
        if duration < self._seg_cfg.discard_sec:
            event.status = AudioEvent.STATUS_COMPLETED
            event.description_status = AudioEvent.DESC_COMPLETED
            event.description_json = {
                "skipped": "too_short",
                "duration_sec": round(duration, 3),
            }
            self._stamp_model(event)
            event.save(update_fields=[
                "status", "description_status", "description_json",
                "moss_model", "updated_at",
            ])
            logger.info(
                "[audio] 事件 #%s 时长 %.2fs < %.1fs，跳过描述",
                event.id, duration, self._seg_cfg.discard_sec,
            )
            return True

        segments = self._ensure_segments(event, duration)
        if segments is None:
            return True     # 音频不可读，已在内部标 failed

        expected = len(plan_segments(duration, event.silence_ranges, self._seg_cfg))
        if len(segments) < expected:
            # 分段行不齐（建行连续失败）：**不能 finalize**，否则会把缺段音频
            # 静默标成 completed；留 describing 下一轮继续补建。
            self._note_incomplete_segments(event, len(segments), expected)
            return True
        self._incomplete_rounds.pop(event.id, None)

        now = timezone.now()
        for seg in segments:
            if seg.description_status == AudioEvent.DESC_COMPLETED:
                continue
            if seg.retry_count >= self._cfg.max_segment_retries:
                continue
            if (
                seg.description_status == AudioEvent.DESC_FAILED
                and (now - seg.updated_at).total_seconds() < self._cfg.retry_backoff_sec
            ):
                continue    # 退避中
            if not self._describe_segment(event, seg):
                # 熔断器拦下了这一条 → 本轮到此为止（后面每条都会被同样拦下）。
                # 该分段状态未被改动，下一轮冷却结束后接着来。
                break

        self._finalize_event(event)
        return True

    @staticmethod
    def _real_duration(event: AudioEvent) -> float | None:
        """事件真实时长（真实起止，不是录音时长，spec §6.2）。"""
        if event.ended_at_ts is None or not event.audio_path:
            return None
        return float(event.ended_at_ts - event.started_at_ts)

    def _stamp_model(self, event: AudioEvent) -> None:
        """记录产出描述的模型名。

        DB 列名仍是 `moss_model`（历史命名，改它要一次无收益的表结构迁移），
        内容已经是当前 provider 对应的模型；页面按"描述模型"展示。
        """
        event.moss_model = getattr(self._client, "model", "") or ""

    # ------------------------------------------------------------------
    # 分段建立 / 补齐
    # ------------------------------------------------------------------
    def _ensure_segments(
        self, event: AudioEvent, duration: float,
    ) -> list[AudioEventSegment] | None:
        """对齐"规划分段"与实际分段行：缺的补建、文件丢的补切。

        **每轮都按规划校验段数**，不能只在"一条分段行都没有"时才规划——否则建段
        中途被打断（DB 异常 / worker 被杀）后剩余段永远不会补建，而已有段全部完成
        后事件会被误判 completed，后半段音频静默丢失且没有任何错误标记。
        音频不可读 → 事件永久失败，返回 None。
        """
        plans = plan_segments(duration, event.silence_ranges, self._seg_cfg)
        if not plans:
            # duration ≥ discard 却规划不出分段（理论上不会发生）→ 防御
            self._fail_event_permanently(event, "分段规划为空")
            return None

        segments = list(event.segments.order_by("sequence"))
        existing = {seg.sequence for seg in segments}
        missing = [
            (seq, plan) for seq, plan in enumerate(plans, start=1)
            if seq not in existing
        ]
        if missing:
            try:
                pcm, sr = self._read_flac(Path(event.audio_path))
            except Exception as e:  # noqa: BLE001
                self._fail_event_permanently(event, f"事件音频不可读: {e}")
                return None
            self._create_segments(event, missing, pcm, sr)
            segments = list(event.segments.order_by("sequence"))

        if not self._ensure_segment_files(event, segments):
            return None
        return segments

    def _create_segments(
        self,
        event: AudioEvent,
        numbered_plans: list[tuple[int, SegmentPlan]],
        pcm: Any,
        sample_rate: int,
    ) -> list[AudioEventSegment]:
        """按 ``(sequence, plan)`` 切片、落分段 FLAC、建行。

        单段失败（切片为空 / 落盘失败 / 建行失败）只记日志并继续，不中断其余段；
        缺失的段下一轮由 :meth:`_ensure_segments` 重新补建。
        """
        lead = float(event.recording_lead_sec or 0.0)
        total = len(pcm)
        created: list[AudioEventSegment] = []
        for seq, plan in numbered_plans:
            s = int(round((plan.start_offset + lead) * sample_rate))
            e = int(round((plan.end_offset + lead) * sample_rate))
            s = max(min(s, total), 0)
            e = max(min(e, total), 0)
            path = segment_audio_path(self._media_root, event, seq)
            if e <= s:
                logger.error(
                    "[audio] 事件 #%s 分段 %d 切片为空（%.1f~%.1f, lead=%.1f, 录音 %.1fs）",
                    event.id, seq, plan.start_offset, plan.end_offset,
                    lead, total / float(sample_rate),
                )
                audio_path = ""
            else:
                try:
                    self._write_flac(path, pcm[s:e], sample_rate)
                    audio_path = str(path)
                except Exception:  # noqa: BLE001
                    logger.exception("[audio] 事件 #%s 分段 %d FLAC 落盘失败", event.id, seq)
                    audio_path = ""
            try:
                created.append(AudioEventSegment.objects.create(
                    audio_event=event,
                    sequence=seq,
                    start_offset=plan.start_offset,
                    end_offset=plan.end_offset,
                    audio_path=audio_path,
                    has_silence_before=plan.has_silence_before,
                    description_status=AudioEvent.DESC_PENDING,
                    last_error="" if audio_path else "分段音频切片/落盘失败",
                    worker_epoch=self._worker_epoch,
                ))
            except Exception:  # noqa: BLE001
                logger.exception("[audio] 事件 #%s 分段 %d 建行失败", event.id, seq)
        logger.info(
            "[audio] 事件 #%s 分段补齐：新建 %d 段（%s）",
            event.id, len(created),
            ", ".join(
                f"{plan.start_offset:.0f}~{plan.end_offset:.0f}s"
                for _, plan in numbered_plans
            ),
        )
        return created

    def _note_incomplete_segments(
        self, event: AudioEvent, have: int, expected: int,
    ) -> None:
        """分段行不齐：本轮不 finalize，连续多轮仍不齐则判定事件永久失败。

        "不齐"只可能来自建行失败（``_create_segments`` 每轮都会补建）。留 describing
        让下一轮继续；若一直补不上（DB 持续异常），退化为 failed 而不是无限卡住。
        """
        rounds = self._incomplete_rounds.get(event.id, 0) + 1
        self._incomplete_rounds[event.id] = rounds
        logger.error(
            "[audio] 事件 #%s 分段不齐：%d/%d（第 %d 轮）",
            event.id, have, expected, rounds,
        )
        if rounds >= _INCOMPLETE_MAX_ROUNDS:
            self._incomplete_rounds.pop(event.id, None)
            self._fail_event_permanently(
                event, f"分段补建失败（{have}/{expected}，连续 {rounds} 轮）",
            )

    def _ensure_segment_files(
        self, event: AudioEvent, segments: list[AudioEventSegment],
    ) -> bool:
        """重试场景：分段行已存在但文件被删了 → 从事件录音重新切片补齐。"""
        missing = [
            seg for seg in segments
            if seg.description_status != AudioEvent.DESC_COMPLETED
            and (not seg.audio_path or not Path(seg.audio_path).exists())
        ]
        if not missing:
            return True
        try:
            pcm, sr = self._read_flac(Path(event.audio_path))
        except Exception as e:  # noqa: BLE001
            self._fail_event_permanently(event, f"事件音频不可读: {e}")
            return False
        lead = float(event.recording_lead_sec or 0.0)
        total = len(pcm)
        for seg in missing:
            s = max(min(int(round((seg.start_offset + lead) * sr)), total), 0)
            e = max(min(int(round((seg.end_offset + lead) * sr)), total), 0)
            path = Path(seg.audio_path) if seg.audio_path else segment_audio_path(
                self._media_root, event, seg.sequence,
            )
            if e <= s:
                seg.last_error = "分段音频切片为空"
                seg.save(update_fields=["last_error", "updated_at"])
                continue
            try:
                self._write_flac(path, pcm[s:e], sr)
                seg.audio_path = str(path)
                seg.last_error = ""
                seg.save(update_fields=["audio_path", "last_error", "updated_at"])
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[audio] 事件 #%s 分段 %d 重新落盘失败", event.id, seg.sequence,
                )
        return True

    # ------------------------------------------------------------------
    # 单分段描述
    # ------------------------------------------------------------------
    def _prompt_for(self, seg: AudioEventSegment) -> str:
        """按 provider 出提示词（方言差异只在这一个入口）。"""
        return build_prompt_for(seg, self._cfg.provider)

    def _normalize(self, raw: str) -> tuple[dict | None, str]:
        """把模型原文归一成待校验 dict（``(obj, error)``）。

        两种方言在这里汇合：之后共用同一套 :func:`validate_description` 校验与
        :func:`aggregate_descriptions` 聚合，所以脏数据拦截/重试语义对两个模型
        完全一致——换模型不会改变"什么算失败"。
        """
        if self._cfg.provider == DESC_PROVIDER_JSON:
            return parse_description_json(raw)
        return normalize_transcript(raw)

    def _describe_segment(self, event: AudioEvent, seg: AudioEventSegment) -> bool:
        """给一个分段做描述；返回**这次是否真的发出请求了**。

        返回 ``False`` = 被熔断器拦下（服务已知不可用，或半开态已有探测在飞）：
        此时**不碰分段状态**，让它原地等下一轮；调用方应立刻停止本轮。
        """
        breaker = self._breaker()
        if not breaker.allow_attempt():
            # 拦一条就够 —— `allow_attempt()` 在半开态是**严格单条**（置 inflight
            # 防重复放行），所以本轮后续分段会继续返回 False。
            # 没有这道闸时，一次冷却里会把本轮**所有**分段都当探测发出去，
            # 一边白耗冷却时间、一边把 `consecutive_failures` 刷得很难看
            # （现场实测 `consec=31`，远超阈值 3）。
            return False

        seg.description_status = AudioEvent.DESC_PROCESSING
        seg.save(update_fields=["description_status", "updated_at"])
        seg_dur = seg.end_offset - seg.start_offset
        try:
            if not seg.audio_path:
                raise AudioDescError("分段音频缺失（切片/落盘失败）")
            pcm, sr = self._read_flac(Path(seg.audio_path))
            wav = self._encode_wav(pcm, sr)
            # 半开态 = 这条请求就是熔断器的探测 → 必须**短超时 + 不重试**。
            # 默认是 `30s × (2+1) = 最坏 90s`，不压下来探测自己就把链路堵死了
            # （见 apps/core/llm_breaker.py 的调用约定 2）。
            probing = breaker.is_probing()
            raw = self._client.describe(
                wav, self._prompt_for(seg), max_tokens=self._cfg.max_tokens,
                timeout_sec=breaker.probe_timeout_sec if probing else None,
                max_retries=0 if probing else None,
            )
            obj, err = self._normalize(raw)
            if err:
                raise ValueError(f"解析失败: {err}")
            clean, err = validate_description(
                obj, seg_dur,
                # 转写方言允许空描述（这段确实没人声），结构化描述必须给内容
                require_description=self._cfg.provider == DESC_PROVIDER_JSON,
            )
            if err:
                raise ValueError(f"校验失败: {err}")

            seg.description_json = clean
            seg.description_status = AudioEvent.DESC_COMPLETED
            seg.last_error = ""
            seg.save(update_fields=[
                "description_json", "description_status", "last_error", "updated_at",
            ])
            # **成功必须回报** —— 这是熔断器唯一能关掉的路径。
            # 漏了它，熔断一旦打开就永久自锁：冷却后 `is_probing()` 恒为真，
            # 之后每条请求都按"探测"跑（5s 超时 + 0 重试），永远回不到正常超时。
            breaker.record_success()
            self._raws.record(event.id, seg.sequence, raw)
            logger.info(
                "[audio] 事件 #%s 分段 %d 描述完成：cry=%s speech=%s bg=%s desc=%d字",
                event.id, seg.sequence, clean["has_cry"],
                clean["has_adult_speech"], clean["background_sounds"],
                len(clean["description"]),
            )
        except (AudioDescTimeoutError, AudioDescNetworkError) as e:
            # **服务侧失败**（不是这条分段的问题）→ **不涨 retry_count**。
            #
            # 涨了的话：8 次 × retry_backoff_sec(60s) ≈ **8 分钟**就耗尽配额 →
            # `_finalize_event` 判 exhausted → `_fail_event_permanently`
            # → status 永久 failed、永不回补。llama 停 1 小时这 1 小时的事件全废，
            # "攒着"根本不成立（checklist §8.4 的病灶）。
            #
            # 两个子类**处置不同**，所以日志分开写：
            # - 不可达（TCP 连接没建立）= 服务不在，**一次即定论**；
            # - 其余（read 超时 / 5xx）= 服务活着但慢/坏，**偶发不能定论**。
            breaker.record_failure(e)
            unreachable = isinstance(e, AudioDescUnreachableError)
            seg.last_error = f"{type(e).__name__}: {e}"[:2000]
            seg.description_status = AudioEvent.DESC_FAILED
            seg.save(update_fields=[
                "last_error", "description_status", "updated_at",
            ])
            logger.warning(
                "[audio] 事件 #%s 分段 %d 描述服务%s（不计重试配额，等回放）: %s",
                event.id, seg.sequence,
                "不可达：连接没建立" if unreachable else "响应异常：活着但慢/坏",
                seg.last_error,
            )
        except Exception as e:  # noqa: BLE001
            # 内容/编排类失败（解析错 / 校验错 / 切片缺失）→ **才消耗配额**。
            #
            # **这里不回报熔断器**：本分支混着两类异常 —— "服务回了脏数据"
            # （服务确实活着）与"分段音频缺失 / 切片落盘失败"（本地错误，压根
            # 没碰服务）。后者若被当成存活证据去 `record_success()`，会把一个
            # 真的不可用的服务**假关闭**。服务是否活着由成功路径与连接类分支
            # 各自回报。
            seg.retry_count += 1
            seg.last_error = f"{type(e).__name__}: {e}"[:2000]
            seg.description_status = AudioEvent.DESC_FAILED
            seg.save(update_fields=[
                "retry_count", "last_error", "description_status", "updated_at",
            ])
            logger.warning(
                "[audio] 事件 #%s 分段 %d 描述失败（第 %d 次）: %s",
                event.id, seg.sequence, seg.retry_count, seg.last_error,
            )
        return True

    # ------------------------------------------------------------------
    # 事件收尾
    # ------------------------------------------------------------------
    def _finalize_event(self, event: AudioEvent) -> None:
        segments = list(event.segments.order_by("sequence"))
        if segments and all(
            s.description_status == AudioEvent.DESC_COMPLETED for s in segments
        ):
            self._incomplete_rounds.pop(event.id, None)
            event.description_json = aggregate_descriptions(segments)
            event.description_status = AudioEvent.DESC_COMPLETED
            event.status = AudioEvent.STATUS_COMPLETED
            self._stamp_model(event)
            raws = self._raws.pop(event.id)
            if raws:
                event.description_raw = json.dumps(
                    {str(k): v for k, v in sorted(raws.items())},
                    ensure_ascii=False,
                )
            event.save(update_fields=[
                "description_json", "description_status", "status",
                "moss_model", "description_raw", "updated_at",
            ])
            logger.info(
                "[audio] 事件 #%s 描述完成：cry=%s speech=%s",
                event.id, event.description_json.get("has_cry"),
                event.description_json.get("has_adult_speech"),
            )
            return

        exhausted = any(
            s.description_status != AudioEvent.DESC_COMPLETED
            and s.retry_count >= self._cfg.max_segment_retries
            for s in segments
        )
        if exhausted:
            self._fail_event_permanently(
                event, "分段重试耗尽，保留分段与错误状态（spec §6.4）",
            )
        # 否则留在 describing：下一轮按退避继续重试

    def _fail_event_permanently(self, event: AudioEvent, reason: str) -> None:
        """永久失败：**不删录音**（spec §12），只改状态。"""
        event.status = AudioEvent.STATUS_FAILED
        event.description_status = AudioEvent.DESC_FAILED
        event.save(update_fields=["status", "description_status", "updated_at"])
        self._raws.pop(event.id)
        self._incomplete_rounds.pop(event.id, None)
        logger.error("[audio] 事件 #%s 描述永久失败: %s", event.id, reason)

    # ------------------------------------------------------------------
    @property
    def status(self) -> dict[str, Any]:
        return {
            "running": self._thread is not None,
            "model": getattr(self._client, "model", ""),
            "provider": self._cfg.provider,
            "poll_sec": self._cfg.poll_sec,
        }


__all__ = [
    "DESC_PROVIDER_JSON",
    "DESC_PROVIDER_TRANSCRIPT",
    "NO_SPEECH_TEXT",
    "DescriberConfig",
    "DescribeService",
    "build_prompt",
    "build_prompt_for",
    "build_transcript_prompt",
    "normalize_provider",
    "normalize_transcript",
    "segment_audio_path",
    "strip_asr_wrapper",
]

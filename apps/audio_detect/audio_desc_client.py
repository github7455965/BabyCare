"""音频描述模型 HTTP 客户端（Phase 4，spec §6.5）。

OpenAI 兼容接口
---------------
POST {base_url}/v1/chat/completions，音频走标准 ``input_audio`` base64 WAV：

.. code-block:: json

    {
      "model": "<BABYCARE_AUDIO_DESC_MODEL>",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "<prompt>"},
        {"type": "input_audio",
         "input_audio": {"data": "<base64 wav>", "format": "wav"}}
      ]}],
      "temperature": 0,
      "max_tokens": 512
    }

一份 transport，两种输出方言
----------------------------
**方言（提示词 + 返回值归一化）不在这层**，全在 :mod:`apps.audio_detect.describer`：

- ``transcript``：纯转写模型（如 Qwen3-ASR-1.7B），content 就是一段人声文字；
- ``json``：结构化描述模型（如 MOSS-Audio-4B / Qwen2.5-Omni-7B），content 是 JSON。

本类只做"把 WAV 发出去、把 content 取回来"，不解析、不校验、不认识方言。
换模型时业务代码零改动，只改 ``.env`` 的 URL / MODEL / PROVIDER 三行。

设计约定
--------
- **纯客户端**：只做 HTTP + 内容提取，不碰 DB / 文件 / 分段（同 LlamaClient 的
  "纯函数式"约定）；重试只覆盖 timeout / 网络错误，4xx / JSON 解析失败不重试；
- 落盘是 FLAC，请求前由调用方**在内存里转 WAV**（spec §6.5，不落中间文件），
  本类只接受已经转好的 WAV bytes；
- 同机部署（``http://127.0.0.1:<port>``）或**分体部署**（局域网内另一台推理机的
  ``http://<host>:<port>``）都只是 URL 不同，本类不区分；不走公网；
  ``api_key`` 非空时带 ``Authorization: Bearer``；
- 若未来服务只提供 multipart 接口，在本类内加 transport adapter 转换，
  **不把差异散落到业务代码**（spec §6.5）。

异常分类（与 describer 的 last_error 记录对齐）
----------------------------------------------
- :class:`AudioDescUnreachableError`：**连接根本没建立**（连接被拒 / connect 超时）
  —— "服务不在"；
- :class:`AudioDescTimeoutError`：连接已建立、模型没回（read 超时）—— "服务活着但慢"；
- :class:`AudioDescNetworkError`：HTTP 5xx（服务活着但报错）；
- :class:`AudioDescParseError`：HTTP 4xx / 响应 JSON 解码失败 / content 提取失败。

> 前两者必须分清楚：**"服务不在"一次就能定论，而"慢"一次不能** —— 混在一起
> 会让熔断阈值写不出合适的值（阈值定 1，"慢"会白停整条线；定 3，"不在"要多等
> 两倍时间）。

超时分两段
----------
``timeout_sec`` 是 **read** 超时（模型要跑多久）；``connect_timeout_sec`` 是
**connect** 超时（服务在不在）。两者传给 requests 时是 ``(connect, read)`` 元组
—— 见 :meth:`_post_once` 里为什么不能合成一个标量。
"""

from __future__ import annotations

import base64
import logging
import time

import requests

logger = logging.getLogger(__name__)


class AudioDescError(Exception):
    """AudioDescClient 基异常。"""


class AudioDescTimeoutError(AudioDescError):
    """HTTP 超时。"""


class AudioDescNetworkError(AudioDescError):
    """网络错误（连接拒绝 / HTTP 5xx）。"""


class AudioDescUnreachableError(AudioDescNetworkError):
    """服务**不可达**：TCP 连接根本没建立（连接被拒 / connect 阶段超时）。

    单独分出来的唯一目的是**诊断**：``requests.exceptions.ConnectTimeout``
    同时继承 ``ConnectionError`` 与 ``Timeout``，若不显式区分，"端口不通"在
    日志/last_error 里会显示成"模型慢"，排查时被带偏（见 checklist §12）。

    它仍是 :class:`AudioDescNetworkError` 的**子类**，所以既有的
    ``except AudioDescNetworkError`` 无需改动即可继续捕获。
    """


class AudioDescParseError(AudioDescError):
    """解析错误（HTTP 4xx / 响应不是 JSON / content 缺失）。"""


class AudioDescClient:
    """OpenAI 兼容音频描述服务客户端。"""

    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str = "",
        timeout_sec: int = 60,
        max_retries: int = 2,
        retry_sleep_sec: float = 1.0,
        connect_timeout_sec: float = 3.0,
    ):
        if not base_url:
            raise ValueError("AudioDescClient: base_url 不能为空")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_sec = int(timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_sleep_sec = float(retry_sleep_sec)
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    @classmethod
    def from_settings(cls) -> "AudioDescClient":
        from django.conf import settings

        return cls(
            base_url=str(settings.BABYCARE_AUDIO_DESC_SERVER_URL),
            model=str(settings.BABYCARE_AUDIO_DESC_MODEL),
            api_key=str(settings.BABYCARE_AUDIO_DESC_API_KEY),
            timeout_sec=int(settings.BABYCARE_AUDIO_DESC_TIMEOUT_SEC),
            max_retries=int(settings.BABYCARE_AUDIO_DESC_MAX_RETRIES),
            connect_timeout_sec=float(
                getattr(settings, "BABYCARE_AUDIO_DESC_CONNECT_TIMEOUT_SEC", 3.0)
            ),
        )

    # ------------------------------------------------------------------
    def _build_payload(self, wav_bytes: bytes, prompt: str, max_tokens: int) -> dict:
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "wav"}},
                ],
            }],
            "temperature": 0,
            "max_tokens": int(max_tokens),
        }

    def _post_once(self, payload: dict, timeout_sec: float | None = None) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        read_timeout = self.timeout_sec if timeout_sec is None else float(timeout_sec)
        # **connect 与 read 必须分开**（requests 的标量 timeout 是两者共用）。
        #
        # 服务不在时（进程已停 / 端口不通），连接**不一定**立刻 RST —— 实测本机
        # `127.0.0.1:8136` 停掉后连接会挂在 `SYN_SENT` 上等超时。共用标量的话，
        # 本该 1~3s 失败的 connect 会被拖成整个 read 超时（默认 30s），再乘
        # `max_retries+1` —— 一次 `describe()` 要 ~92s 才回报熔断器，
        # "连续 3 次连接类失败 → 熔断"于是变成 ~4.6 分钟才发现服务不可用。
        # 拆开后 connect 最多 `connect_timeout_sec`，熔断在数十秒内就能打开。
        timeout = (self.connect_timeout_sec, read_timeout)
        try:
            resp = self._session.post(url, json=payload, timeout=timeout)
        # **捕获顺序有讲究**：`ConnectTimeout` 同时继承 `ConnectionError` 与
        # `Timeout`，必须最先捕获；否则它会被下面的 `Timeout` 分支吃掉，于是
        # "端口不通"在日志里显示成"读超时（模型慢）"。
        except requests.exceptions.ConnectTimeout as e:
            raise AudioDescUnreachableError(
                f"connect timeout after {self.connect_timeout_sec}s: {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise AudioDescUnreachableError(f"connect error: {e}") from e
        except requests.exceptions.Timeout as e:
            # 连接已建立、模型没在 read 超时内回 → 服务**活着**，是慢不是死
            raise AudioDescTimeoutError(f"read timeout after {read_timeout}s: {e}") from e
        except requests.exceptions.RequestException as e:
            raise AudioDescNetworkError(f"request error: {e}") from e

        if resp.status_code >= 500:
            raise AudioDescNetworkError(
                f"server 5xx: status={resp.status_code} body={resp.text[:200]}"
            )
        if resp.status_code >= 400:
            # 4xx = 参数/路由问题，重试无意义
            raise AudioDescParseError(
                f"client 4xx: status={resp.status_code} body={resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError as e:
            raise AudioDescParseError(
                f"json decode failed: {e}; body={resp.text[:200]}"
            ) from e

    @staticmethod
    def _extract_content(data: dict) -> str:
        try:
            choices = data["choices"]
            if not choices:
                raise AudioDescParseError("choices 为空")
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise AudioDescParseError(f"content 不是 str: {type(content).__name__}")
            return content
        except (KeyError, IndexError, TypeError) as e:
            raise AudioDescParseError(
                f"extract content failed: {e}; data={str(data)[:200]}"
            ) from e

    def describe(
        self,
        wav_bytes: bytes,
        prompt: str,
        max_tokens: int = 512,
        timeout_sec: float | None = None,
        max_retries: int | None = None,
    ) -> str:
        """发一段 WAV，返回模型原始文本（未做任何方言解析）。

        Args:
            timeout_sec: **单次调用**覆盖超时（不传 = 用实例默认）。
                半开探测必须传短值：默认 `30s × (max_retries+1)` 最坏 **90s**，
                探测自己就能把链路堵死（见 `apps/core/llm_breaker.py` 的调用约定 2）。
            max_retries: **单次调用**覆盖重试次数（不传 = 用实例默认）。
                探测应传 `0` —— 失败就赶紧回报熔断器，别在一条探测上耗三倍时间。

        Raises:
            AudioDescTimeoutError / AudioDescNetworkError / AudioDescParseError
        """
        if not wav_bytes:
            raise ValueError("wav_bytes 不能为空")
        payload = self._build_payload(wav_bytes, prompt, max_tokens)
        retries = self.max_retries if max_retries is None else max(0, int(max_retries))
        attempts = retries + 1
        for attempt in range(attempts):
            try:
                return self._extract_content(
                    self._post_once(payload, timeout_sec=timeout_sec)
                )
            except (AudioDescTimeoutError, AudioDescNetworkError):
                if attempt < attempts - 1:
                    time.sleep(self.retry_sleep_sec)
                    continue
                raise
            except AudioDescParseError:
                raise
        raise AudioDescNetworkError("unreachable")  # pragma: no cover


__all__ = [
    "AudioDescClient",
    "AudioDescError",
    "AudioDescTimeoutError",
    "AudioDescNetworkError",
    "AudioDescUnreachableError",
    "AudioDescParseError",
]

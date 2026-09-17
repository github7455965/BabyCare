"""
LlamaClient（Step 7）：HTTP 客户端，封装 llama-server 的 /v1/chat/completions 调用。

职责
----
- 纯客户端，不负责 GPU 调度（Step 4 GpuManager 管 acquire/release）
- 单次 chat(images, prompt, **kwargs) → str  返回 VLM 原始回答
- 内置 timeout + retry（失败一次，重试一次用 normal timeout）
- 内置解析（plain / json），返回 (hit, status)
- 抛 3 种异常：LlamaTimeoutError / LlamaNetworkError / LlamaParseError

为什么是纯函数式
---------------
Step 8 VlmWorker 调用：GpuManager.acquire_vlm() → client.chat() → GpuManager.release_vlm_if_idle()
LlamaClient 只在 client 内做 HTTP + 解析，不碰 GPU / DB / cursor。

请求格式（OpenAI 兼容）
----------------------
POST {base_url}/v1/chat/completions
{
  "model": "qwen3vl",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": prompt},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # ×3
    ]
  }],
  "max_tokens": int,
  "temperature": 0
}

返回格式（OpenAI 兼容）
----------------------
{
  "choices": [{
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop" | "length"
  }]
}

异常分类
--------
- LlamaTimeoutError: requests.exceptions.Timeout → failure_reason="timeout"
- LlamaNetworkError: ConnectionError / HTTPError(5xx) → failure_reason="network_error"
- LlamaParseError: JSON 解码失败 / HTTP 4xx / hit/status 解析失败 → failure_reason="parse_error"
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from typing import List, Optional, Tuple

import requests


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 异常类（与 VLMCheckState.failure_reason 6 值枚举对齐）
# ---------------------------------------------------------------------------
class LlamaClientError(Exception):
    """LlamaClient 基异常。"""


class LlamaTimeoutError(LlamaClientError):
    """HTTP 超时（requests.exceptions.Timeout）→ failure_reason='timeout'"""


class LlamaNetworkError(LlamaClientError):
    """网络错误（连接拒绝 / DNS / HTTP 5xx 中非 loading 的）→ failure_reason='network_error'"""


class LlamaLoadingError(LlamaClientError):
    """llama-server 正在加载模型（HTTP 503 body 含 "Loading model"）→ 不算失败。

    这是临时状态：llama-server 启了 + listen 但 model 还没加载完时 chat 请求会返回
    503。不是真的失败——下次再 chat 就好。Runner 应该 sleep 重试 + 不计 fail_count。
    """


class LlamaParseError(LlamaClientError):
    """解析错误（JSON 失败 / HTTP 4xx / 响应字段缺失）→ failure_reason='parse_error'"""


# ---------------------------------------------------------------------------
class LlamaClient:
    """llama-server HTTP 客户端（OpenAI 兼容 /v1/chat/completions）。"""

    _instance: Optional["LlamaClient"] = None
    _cls_lock_cls = None  # 延迟初始化（避免启动期 import 顺序问题）

    @classmethod
    def instance(cls) -> "LlamaClient":
        """进程内单例；DrainerThread / PromptRunnerManager 共用同一 client。

        第一次调时 new；之后返回缓存。base_url 从 .env 读一次（启动期固定）。
        """
        import threading
        if cls._cls_lock_cls is None:
            cls._cls_lock_cls = threading.Lock()
        if cls._instance is None:
            with cls._cls_lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_sec_first: int = 60,
        timeout_sec_normal: int = 15,
        max_retries: int = 1,
        retry_sleep_sec: float = 1.0,
        model: str = "qwen3vl",
    ):
        """Args:
            base_url: llama-server 地址（含 scheme + host + port；不含路径）。
                     缺省从 os.environ['BABYCARE_LLAMA_SERVER_URL'] 读，
                     仍缺省则 'http://127.0.0.1:8082'。
            timeout_sec_first: 首次请求 timeout（冷启长）
            timeout_sec_normal: 重试请求 timeout（热机短）
            max_retries: 重试次数（默认 1）
            retry_sleep_sec: 重试前 sleep 秒数
            model: 发送给 llama-server 的 model 字段值（llama-server 通常忽略）
        """
        if base_url is None:
            base_url = os.environ.get("BABYCARE_LLAMA_SERVER_URL", "http://127.0.0.1:8082")
        # 去尾斜杠，避免拼成 //v1
        self.base_url = base_url.rstrip("/")
        self.timeout_sec_first = timeout_sec_first
        self.timeout_sec_normal = timeout_sec_normal
        self.max_retries = max_retries
        self.retry_sleep_sec = retry_sleep_sec
        self.model = model
        self._session = requests.Session()
        # 串行化 llama-server 调用（mtmd 不能并发请求，否则返回 500
        # "failed to process mtmd chunk"）。LlamaClient 是单例，此锁全局生效。
        self._chat_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 构造请求体
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(jpeg_bytes: bytes) -> str:
        """bytes → base64 data URL。"""
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _build_payload(
        self,
        images: List[bytes],
        prompt: str,
        max_tokens: int,
    ) -> dict:
        """构造 OpenAI 兼容的 chat/completions 请求体。"""
        content = [{"type": "text", "text": prompt}]
        for img_bytes in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(img_bytes)},
            })
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

    # ------------------------------------------------------------------
    # HTTP 调用（带 retry）
    # ------------------------------------------------------------------
    def _post_once(
        self,
        payload: dict,
        timeout_sec: int,
    ) -> dict:
        """单次 POST；不重试。

        抛出 LlamaTimeoutError / LlamaNetworkError / LlamaParseError。
        """
        url = f"{self.base_url}/v1/chat/completions"
        try:
            resp = self._session.post(url, json=payload, timeout=timeout_sec)
        except requests.exceptions.Timeout as e:
            raise LlamaTimeoutError(f"timeout after {timeout_sec}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise LlamaNetworkError(f"connection error: {e}") from e
        except requests.exceptions.RequestException as e:
            # 其他 requests 异常 → 网络错误
            raise LlamaNetworkError(f"request error: {e}") from e

        # HTTP 错误
        if resp.status_code >= 500:
            # llama-server 加载模型中的 503 是临时状态——不算失败
            if resp.status_code == 503 and "Loading model" in resp.text:
                raise LlamaLoadingError(
                    f"server loading model: status=503 body={resp.text[:200]}"
                )
            raise LlamaNetworkError(
                f"server 5xx: status={resp.status_code} body={resp.text[:200]}"
            )
        if resp.status_code >= 400:
            # 4xx 通常是参数错或路由不存在 → 解析错误（不可重试）
            raise LlamaParseError(
                f"client 4xx: status={resp.status_code} body={resp.text[:200]}"
            )

        # 解析 JSON
        try:
            return resp.json()
        except ValueError as e:
            raise LlamaParseError(f"json decode failed: {e}; body={resp.text[:200]}") from e

    def chat(
        self,
        images: List[bytes],
        prompt: str,
        max_tokens: int = 256,
        timeout_sec: Optional[int] = None,
    ) -> str:
        """发请求（含 retry）→ 返回 VLM 原始回答文本。

        Args:
            images: 3 张 jpg bytes
            prompt: 完整 prompt 文本
            max_tokens: VLM 生成上限
            timeout_sec: 显式指定本次请求的 timeout（秒）。
                None（默认）→ 第 1 次用 timeout_sec_first，重试用 timeout_sec_normal
                指定值 → 第 1 次用该值，重试用 timeout_sec_normal（Step 8 Runner 冷启专用）

        Returns:
            VLM 回答原始文本（未解析）

        Raises:
            LlamaTimeoutError / LlamaNetworkError / LlamaParseError
        """
        if not images:
            raise ValueError("images 不能为空（v1 要求 3 张图）")
        payload = self._build_payload(images, prompt, max_tokens)

        # 串行化：mtmd 不支持并发请求（会返回 500 "failed to process mtmd chunk"）。
        # 单例 client → 进程内所有 PromptRunner / DrainerThread 共用这把锁。
        with self._chat_lock:
            attempts = self.max_retries + 1  # 总尝试次数 = 重试次数 + 1
            first_timeout = timeout_sec if timeout_sec is not None else self.timeout_sec_first
            try:
                for attempt in range(attempts):
                    # 第 1 次：timeout_sec 显式传了用传的，否则 first；之后用 normal
                    cur_timeout = first_timeout if attempt == 0 else self.timeout_sec_normal
                    try:
                        data = self._post_once(payload, cur_timeout)
                        text = self._extract_content(data)
                        return text
                    except (LlamaTimeoutError, LlamaNetworkError):
                        # 可重试错误：最后 attempt 直接抛；否则 sleep 后重试
                        if attempt < attempts - 1:
                            time.sleep(self.retry_sleep_sec)
                            continue
                        raise
                    except LlamaParseError:
                        # 4xx / JSON 失败 → 不可重试，立即抛
                        raise
            finally:
                # 每次 chat 后清空 slot KV cache，避免 llama-server 内存累积
                # （默认 slot KV 不释放，跨请求保留——常驻模式下内存只增不减）。
                # 清失败不抛（清理操作不应阻塞主流程）。
                self._erase_slot(0)

    @staticmethod
    def _extract_content(data: dict) -> str:
        """从 OpenAI 响应 JSON 提取 assistant 的 content 文本。

        data 形态：{"choices": [{"message": {"role": "assistant", "content": "..."}}, ...]}
        """
        try:
            choices = data["choices"]
            if not choices:
                raise LlamaParseError("choices 为空")
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise LlamaParseError(f"content 不是 str: {type(content).__name__}")
            return content
        except (KeyError, IndexError, TypeError) as e:
            raise LlamaParseError(f"extract content failed: {e}; data={str(data)[:200]}") from e

    def _erase_slot(self, slot_id: int = 0) -> None:
        """清空指定 slot 的 KV cache（每次 chat 后调，防止 llama-server 内存累积）。

        llama.cpp 默认 slot KV 跨请求保留，n_slots × ctx_size 大内存场景下常驻模式
        内存只增不减。调 `POST /slots/{id}?action=erase` 显式清掉 KV。

        失败不抛（清理操作不应阻塞主流程；下次请求会重新分配 slot）。
        """
        try:
            self._session.post(
                f"{self.base_url}/slots/{slot_id}?action=erase",
                timeout=2,
            )
        except Exception as e:
            logger.debug("[llama-client] erase slot %d failed: %s", slot_id, e)

    # ------------------------------------------------------------------
    # 解析
    # ------------------------------------------------------------------
    def parse(
        self,
        raw_response: str,
        positive_keyword: str,
        result_format: str = "plain",
        kind: str = "judge",
    ) -> Tuple[bool, str]:
        """从 VLM 原始回答提取 (hit, status)。

        Args:
            raw_response: VLM 原始返回文本
            positive_keyword: 命中关键词（默认"是"）
            result_format: "plain" | "json"
            kind: "judge" | "describe"
                - judge（判断型）：首字符匹配 positive_keyword → (hit, status)
                - describe（描述型）：status = 全文；hit = False（永远不报警）

        Returns:
            (hit, status)
            - hit: 描述型永远 False；判断型 = 首字符 == positive_keyword
            - status: 描述型 = 全文；判断型 = 首字符后剩余文本
        """
        text = (raw_response or "").strip()
        if not text:
            return (False, "")

        # ---- describe 描述型：全文存 status，永远不报警 ----
        if kind == "describe":
            return (False, text)

        # ---- judge 判断型 ----
        # json 格式尝试
        if result_format == "json":
            hit_json, status_json = self._parse_json(text, positive_keyword)
            if hit_json is not None:
                return (hit_json, status_json)
            # 失败 fallback 到 plain（不记日志，调用方自己处理）

        # plain 格式（首字符匹配）
        return self._parse_plain(text, positive_keyword)

    @staticmethod
    def _parse_plain(text: str, positive_keyword: str) -> Tuple[bool, str]:
        """首字符匹配 → (hit, status)。

        status = 首字符去除关键词后的剩余描述；空时为 ""。
        """
        kw = positive_keyword.strip()
        if text.startswith(kw):
            status = text[len(kw):].lstrip(" :：，,。.")
            return (True, status)
        return (False, text)

    @staticmethod
    def _parse_json(
        text: str, positive_keyword: str
    ) -> Tuple[Optional[bool], str]:
        """尝试解析 JSON：{"verdict": "是/否", "description": "..."}

        返回 (hit, status) 或 (None, "") 表示解析失败（调用方 fallback 到 plain）。
        """
        try:
            obj = json.loads(text)
        except ValueError:
            return (None, "")
        if not isinstance(obj, dict):
            return (None, "")
        verdict = obj.get("verdict", "")
        description = obj.get("description", "")
        if not isinstance(verdict, str):
            return (None, "")
        hit = verdict.strip().startswith(positive_keyword.strip())
        return (hit, str(description))
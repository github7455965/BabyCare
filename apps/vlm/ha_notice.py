"""Home Assistant 底层 HTTP + ``.env`` 加载。

只保留 ``notify.py`` 需要的最底层能力：

- :func:`load_env`：把 ``09_web_vlm_manage/.env`` 的 ``HA_URL`` / ``HA_TOKEN`` 读进
  ``os.environ``（已存在的系统环境变量优先）；
- :func:`_post_json`：带 Bearer token 的 POST，HTTP 错误统一转 :class:`HANoticeError`；
- :class:`HANoticeError`：调用失败（鉴权 / 404 / 连不上），由 ``notify.py`` 记失败。

通知目标**已不再走 .env**（B5）：早期版本用 ``HA_NOTIFY_TARGETS`` /
``HA_SPEAKER_TARGETS`` 全局别名表 + ``send_notice`` / ``play_text`` 群发；现在改为
``NotifyTarget`` 表（网页「通知目标」页）逐条直发 —— ``target_id`` 就是 HA 服务 /
实体 ID，按 ``kind`` 分发（见 ``notify.py::_deliver``）。因此别名解析、群发函数和相关
常量都已删除，**不要再往 .env 加目标别名**。
"""

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# 09_web_vlm_manage/.env：apps/vlm/ha_notice.py → ../../.env
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class HANoticeError(RuntimeError):
    """Home Assistant 调用失败。"""


def load_env(path: Path = DEFAULT_ENV_PATH) -> None:
    """加载本地 .env，已存在的系统环境变量优先。"""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _post_json(url: str, token: str, payload: dict[str, object], timeout: float) -> None:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 401:
            reason = "鉴权失败，请检查 HA_TOKEN"
        elif error.code == 404:
            reason = "Home Assistant 资源不存在，请检查服务/实体名称"
        else:
            reason = f"Home Assistant 返回 HTTP {error.code}"
        raise HANoticeError(f"{reason}；{detail}") from error
    except URLError as error:
        raise HANoticeError(f"无法连接 Home Assistant：{error.reason}") from error

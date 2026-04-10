"""OpenAI 风格 Chat Completions 兼容后端。

支持：
- OpenAI 官方 API（默认 base_url=https://api.openai.com/v1）
- 任何兼容 OpenAI Chat Completions 的网关（自定义 base_url）
"""
from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from isaaclab_logistics_vla.agent.vlm_backend import Message, VlmBackend, VlmToolSchema


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def _rgb_uint8_to_jpeg_b64(image_rgb_uint8: np.ndarray, quality: int = 85) -> str:
    arr = np.asarray(image_rgb_uint8)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3 RGB image, got shape={arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _tools_block(tools: List[VlmToolSchema]) -> str:
    if not tools:
        return ""
    lines = ["可用工具（仅可调用下列 name）："]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


def _split_system_instruction_tail(messages: List[Message]) -> tuple[str, str, List[Message]]:
    system_text = ""
    instruction_text = ""
    i = 0
    n = len(messages)
    while i < n and messages[i].get("role") == "system":
        c = messages[i].get("content", "")
        if isinstance(c, str) and c.strip():
            system_text = c.strip()
        i += 1
    if i < n and messages[i].get("role") == "user":
        c = messages[i].get("content", "")
        instruction_text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
        i += 1
    return system_text, instruction_text, messages[i:]


def _history_to_openai_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role in ("user", "assistant"):
            c = m.get("content", "")
            if not isinstance(c, str):
                c = json.dumps(c, ensure_ascii=False)
            out.append({"role": role, "content": c})
        elif role == "tool":
            name = m.get("name", "tool")
            body = m.get("content", {})
            out.append(
                {
                    "role": "user",
                    "content": f"[tool:{name}] 返回结果:\n{json.dumps(body, ensure_ascii=False)}",
                }
            )
    return out


class OpenAICompatBackend(VlmBackend):
    """OpenAI 风格 Chat Completions 后端（可带图）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError("使用 OpenAICompatBackend 需要安装 openai：pip install openai>=1.0") from e

        key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise ValueError("未找到 API Key：请设置 OPENAI_API_KEY 或在构造函数传入 api_key=")

        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self._model = (model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

        from openai import OpenAI as _OpenAI

        self._client = _OpenAI(api_key=key, base_url=self._base_url)

    @property
    def model_name(self) -> str:
        return self._model

    def infer(
        self,
        image_rgb_uint8: np.ndarray,
        messages: List[Message],
        tools: List[VlmToolSchema],
    ) -> str:
        sys0, instruction_text, tail = _split_system_instruction_tail(messages)
        system_parts: List[str] = []
        if sys0:
            system_parts.append(sys0)
        tb = _tools_block(tools)
        if tb:
            system_parts.append(tb)
        system_text = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

        api_messages = _history_to_openai_messages(tail)
        jpeg_b64 = _rgb_uint8_to_jpeg_b64(image_rgb_uint8)
        data_url = f"data:image/jpeg;base64,{jpeg_b64}"

        json_help = (
            "上图为当前仿真环境的顶视相机 RGB 画面。\n"
            "请只输出一个 JSON 对象（不要 Markdown、不要代码围栏、不要解释文字）。"
        )
        blocks: List[Dict[str, Any]] = []
        if instruction_text.strip():
            blocks.append({"type": "text", "text": instruction_text.strip()})
        blocks.append({"type": "image_url", "image_url": {"url": data_url}})
        blocks.append({"type": "text", "text": json_help})
        api_messages.append({"role": "user", "content": blocks})

        msg_list: List[Dict[str, Any]] = [{"role": "system", "content": system_text}] + api_messages
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=msg_list,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        if not resp.choices:
            return ""
        content = resp.choices[0].message.content
        return content or ""


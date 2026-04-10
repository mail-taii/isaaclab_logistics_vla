"""火山引擎 Ark Coding：Anthropic Messages API 兼容后端。

文档约定：
- Base URL（不含 /v1/messages，由 anthropic SDK 拼接）：``https://ark.cn-beijing.volces.com/api/coding``
- 默认模型：``ark-code-latest``

鉴权：
- 环境变量 ``ANTHROPIC_API_KEY`` 或 ``VOLC_ARK_API_KEY``（二选一，前者优先）

依赖：``pip install anthropic``（建议 ``anthropic>=0.34``）
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


DEFAULT_VOLC_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding"
DEFAULT_VOLC_ARK_MODEL = "ark-code-latest"


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


def _history_to_anthropic_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """将一段对话（已去掉 system 与首条 user 任务指令）转为 Anthropic ``messages``。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            c = m.get("content", "")
            if not isinstance(c, str):
                c = json.dumps(c, ensure_ascii=False)
            out.append({"role": "user", "content": c})
        elif role == "assistant":
            c = m.get("content", "")
            if not isinstance(c, str):
                c = json.dumps(c, ensure_ascii=False)
            out.append({"role": "assistant", "content": c})
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


def _split_system_instruction_tail(messages: List[Message]) -> tuple[str, str, List[Message]]:
    """返回 (system_text, instruction_text, tail_messages)。

    tail 从「首条非 system 的 user」之后开始，避免与本轮带图 user 连成两条连续 user。
    """
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


def _extract_text_from_message(message: Any) -> str:
    """兼容 anthropic SDK 的 Message 对象或 dict。"""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content is None:
        return ""
    parts: List[str] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype is None and isinstance(block, dict):
            btype = block.get("type")
        if btype == "text":
            t = getattr(block, "text", None)
            if t is None and isinstance(block, dict):
                t = block.get("text")
            if t:
                parts.append(str(t))
    return "\n".join(parts).strip()


class VolcArkAnthropicBackend(VlmBackend):
    """通过 Anthropic 兼容 HTTP 接口调用火山 Ark Coding（如 ark-code-latest）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        test_describe: bool = False,
        test_plan: bool = False,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "使用 VolcArkAnthropicBackend 需要安装 anthropic：pip install anthropic>=0.34"
            ) from e

        key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("VOLC_ARK_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "未找到 API Key：请设置环境变量 ANTHROPIC_API_KEY 或 VOLC_ARK_API_KEY，或在构造函数传入 api_key="
            )

        self._base_url = (base_url or os.environ.get("VOLC_ARK_BASE_URL") or DEFAULT_VOLC_ARK_BASE_URL).rstrip("/")
        self._model = (model or os.environ.get("VOLC_ARK_MODEL") or DEFAULT_VOLC_ARK_MODEL).strip()
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)

        import anthropic as _anthropic

        self._client = _anthropic.Anthropic(api_key=key, base_url=self._base_url)
        self._test_describe = bool(test_describe)
        self._test_plan = bool(test_plan)
        # Some endpoints/models behind Anthropic-compatible gateways are text-only.
        # If image input is rejected once, disable image blocks for subsequent turns.
        self._disable_image_input = False
        self._warned_image_not_supported = False

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

        api_messages = _history_to_anthropic_messages(tail)
        final_blocks: List[Dict[str, Any]] = []
        if instruction_text.strip():
            final_blocks.append({"type": "text", "text": instruction_text.strip()})
        if not self._disable_image_input:
            jpeg_b64 = _rgb_uint8_to_jpeg_b64(image_rgb_uint8)
            final_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": jpeg_b64,
                    },
                }
            )
        if self._test_plan:
            json_help = (
                "上图为当前仿真环境的顶视相机 RGB 画面（与 tool get_topview_image 所见一致）。\n"
                "你正在进行【任务编排测试】。请**只输出一个 JSON 对象**，格式只能是下面之一：\n"
                '1) {"tool_name":"get_topview_image","parameters":{}}  — 必须先调用一次以确认画面；\n'
                '2) {"tool_name":"task_create","parameters":{"subject":"...","description":"...","blockedBy":[...]}} — 创建任务节点；\n'
                '3) {"tool_name":"task_update","parameters":{"task_id":1,"status":"completed"}} — 更新任务状态/依赖；\n'
                '4) {"tool_name":"task_list","parameters":{"view":"ready"}} — 查看可执行任务；\n'
                '5) {"done":true,"summary":"..."} — 仅当你已经创建了一个 DAG（至少 2 个任务且存在 blockedBy 依赖边）才可结束。\n'
                "要求：先 1)，然后用 2) 创建至少两个任务，并通过 blockedBy 表达依赖（DAG），最后 done。"
            )
        elif self._test_describe:
            json_help = (
                "上图为当前仿真环境的顶视相机 RGB 画面（与 tool get_topview_image 所见一致）。\n"
                "请**只输出一个 JSON 对象**（不要 Markdown、不要代码围栏、不要其它解释），格式只能是下面之一：\n"
                '1) {"tool_name":"get_topview_image","parameters":{}}  — 需要刷新/确认顶视图时调用；\n'
                '2) {"scene_description":"..."}  — 在对话里已有 tool 返回后，用中文详细描述你看到的场景（物体、箱子、机械臂大致位置等）；\n'
                '3) {"done":true,"summary":"...","scene_description":"可选，结束前的简短总结"}  — 描述完成后结束。\n'
                "建议顺序：先 1) 再 2) 再 3)。当前仅允许 tool_name 为 get_topview_image。"
            )
        else:
            json_help = (
                "上图为当前仿真环境的顶视相机 RGB 画面。\n"
                "请**只输出一个 JSON 对象**（不要 Markdown、不要代码围栏、不要解释文字），"
                "格式二选一：\n"
                '1) {"tool_name":"get_topview_image","parameters":{}}\n'
                '2) {"done":true,"summary":"..."}\n'
                "当前仅允许 tool_name 为 get_topview_image。"
            )
        final_blocks.append({"type": "text", "text": json_help})
        api_messages.append({"role": "user", "content": final_blocks})

        try:
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_text,
                messages=api_messages,
            )
        except Exception as e:
            s = str(e)
            image_not_supported = (
                (not self._disable_image_input)
                and ("Model do not support image input" in s or "param': 'image_url'" in s or 'param": "image_url"' in s)
            )
            if not image_not_supported:
                raise
            self._disable_image_input = True
            if not self._warned_image_not_supported:
                print(
                    "[Warn] Current model rejects image input; switching this backend to text-only mode. "
                    "Set --vlm_model to a vision-capable model to re-enable direct image conditioning."
                )
                self._warned_image_not_supported = True
            # Retry once immediately without image block.
            final_blocks_no_image = [b for b in final_blocks if b.get("type") != "image"]
            api_messages[-1] = {"role": "user", "content": final_blocks_no_image}
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_text,
                messages=api_messages,
            )
        return _extract_text_from_message(msg)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import json
import time
from pathlib import Path

from isaaclab_logistics_vla.agent.json_parsing import parse_json_object_loose
from .tools import ToolManager
from isaaclab_logistics_vla.agent.vlm_backend import Message, VlmBackend, VlmToolSchema


@dataclass
class RunStats:
    steps: int = 0
    parse_errors: int = 0
    invalid_calls: int = 0
    tool_failures: int = 0
    done: bool = False
    done_reason: str = ""
    # 测试模式：模型在 tool 之后返回的 scene_description（按时间顺序追加）
    scene_descriptions: List[str] = field(default_factory=list)
    # 任务系统统计（用于 test_plan）
    task_create_calls: int = 0
    task_update_calls: int = 0
    task_get_calls: int = 0
    task_list_calls: int = 0
    first_tool_called: str = ""


def _tool_result_for_history(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """把 tool 返回结果转成可 JSON 序列化的摘要，写入对话历史。"""
    out = dict(tool_result)
    if "image_rgb_uint8" in out:
        img = out["image_rgb_uint8"]
        if isinstance(img, np.ndarray):
            out["image_rgb_uint8"] = {
                "type": "ndarray",
                "shape": list(img.shape),
                "dtype": str(img.dtype),
            }
        else:
            out["image_rgb_uint8"] = {"type": type(img).__name__}
    return out


class VlmToolUseRunner:
    """VLM-tool-use 主循环（只做 runner，不绑定具体 VLM）。"""

    def __init__(
        self,
        backend: VlmBackend,
        tool_manager: ToolManager,
        system_prompt: str,
        max_steps: int = 50,
        allow_scene_description_only: bool = False,
        require_first_tool_name: str = "",
        require_task_dag_before_done: bool = False,
        tasks_dir_for_validation: str = "",
        trace_jsonl_path: str = "",
    ):
        self.backend = backend
        self.tools = tool_manager
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.allow_scene_description_only = allow_scene_description_only
        self.require_first_tool_name = (require_first_tool_name or "").strip()
        self.require_task_dag_before_done = bool(require_task_dag_before_done)
        self.tasks_dir_for_validation = (tasks_dir_for_validation or "").strip()
        self._trace_jsonl_path = (trace_jsonl_path or "").strip()
        self._trace_fp = None
        if self._trace_jsonl_path:
            p = Path(self._trace_jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # line-buffered append; best-effort trace
            self._trace_fp = p.open("a", encoding="utf-8")
        self.messages: List[Message] = []
        self.stats = RunStats()

    def close(self) -> None:
        if self._trace_fp is not None:
            try:
                self._trace_fp.flush()
                self._trace_fp.close()
            finally:
                self._trace_fp = None

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self._trace_fp is None:
            return
        try:
            rec = {
                "ts": time.time(),
                "event": event,
                "step": int(self.stats.steps),
                "payload": payload,
            }
            self._trace_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._trace_fp.flush()
        except Exception:
            # tracing must not break evaluation loop
            return

    def _feedback_invalid_call(self, error: str, detail: Dict[str, Any] | str | None = None) -> None:
        """把校验失败原因写回对话历史，避免模型重复犯错直到 max_steps。"""
        payload: Dict[str, Any] = {"error": str(error)}
        if detail is not None:
            payload["detail"] = detail
        # Use a tool-role message so it is clearly "feedback from the environment".
        self.messages.append({"role": "tool", "name": "validator", "content": payload})
        self._trace("validator_feedback", payload)

    def reset(self, instruction: str) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]
        self.stats = RunStats()
        self._trace(
            "run_reset",
            {
                "model": self.backend.model_name,
                "instruction": instruction,
                "system_prompt": self.system_prompt,
                "tools": [t.__dict__ for t in self._tool_schemas()],
            },
        )

    def _tool_schemas(self) -> List[VlmToolSchema]:
        lst = []
        for name, meta in self.tools.list_tools().items():
            lst.append(VlmToolSchema(name=name, description=meta.get("description", "")))
        return lst

    def step(self, current_image: np.ndarray) -> Dict[str, Any]:
        """执行一次 VLM 决策 → (可选) tool → 记录历史。"""
        if self.stats.done:
            return {"done": True, "reason": self.stats.done_reason}
        if self.stats.steps >= self.max_steps:
            self.stats.done = True
            self.stats.done_reason = "max_steps"
            self._trace("done_max_steps", {"done_reason": self.stats.done_reason, "stats": self.stats.__dict__})
            return {"done": True, "reason": "max_steps"}

        raw = self.backend.infer(
            image_rgb_uint8=current_image,
            messages=self.messages,
            tools=self._tool_schemas(),
        )
        self._trace(
            "assistant_raw",
            {
                "raw": raw,
                "current_image": {"type": "ndarray", "shape": list(current_image.shape), "dtype": str(current_image.dtype)},
            },
        )
        self.messages.append({"role": "assistant", "content": raw, "model": self.backend.model_name})

        parsed = parse_json_object_loose(raw)
        if not parsed.ok or parsed.data is None:
            self.stats.parse_errors += 1
            self.stats.steps += 1
            self._trace("parse_error", {"raw": raw, "stats": self.stats.__dict__})
            return {"done": False, "parse_error": True, "raw": raw}

        data = parsed.data
        self._trace("assistant_parsed", {"data": data})

        def _record_scene_description() -> None:
            d = data.get("scene_description")
            if isinstance(d, str) and d.strip():
                self.stats.scene_descriptions.append(d.strip())

        if data.get("done") is True:
            # test_plan: done 前必须已经形成 DAG（至少 2 个任务 + 至少 1 条依赖边）
            if self.require_task_dag_before_done and self.tasks_dir_for_validation:
                try:
                    from pathlib import Path
                    from isaaclab_logistics_vla.agent.task_system import TaskManager

                    tm = TaskManager(Path(self.tasks_dir_for_validation))
                    tasks = [t.to_dict() for t in tm.list_all()]
                    n_tasks = len(tasks)
                    n_edges = sum(1 for t in tasks if (t.get("blockedBy") or []))
                    if n_tasks < 2 or n_edges < 1:
                        self.stats.invalid_calls += 1
                        self.stats.steps += 1
                        return {
                            "done": False,
                            "invalid_call": True,
                            "error": "task_dag_required_before_done",
                            "detail": {"n_tasks": n_tasks, "n_tasks_with_blockedBy": n_edges},
                            "data": data,
                        }
                except Exception as e:
                    self.stats.invalid_calls += 1
                    self.stats.steps += 1
                    return {
                        "done": False,
                        "invalid_call": True,
                        "error": "task_dag_validation_failed",
                        "detail": f"{type(e).__name__}: {e}",
                        "data": data,
                    }
            _record_scene_description()
            self.stats.done = True
            self.stats.done_reason = "vlm_done"
            self.stats.steps += 1
            self._trace("done_vlm", {"summary": data.get("summary", ""), "data": data, "stats": self.stats.__dict__})
            return {"done": True, "reason": "vlm_done", "summary": data.get("summary", "")}

        tool_name = data.get("tool_name")
        params = data.get("parameters", {})

        if tool_name is not None:
            if not isinstance(params, dict):
                self.stats.invalid_calls += 1
                self.stats.steps += 1
                self._trace("invalid_call", {"error": "parameters_not_dict", "data": data, "stats": self.stats.__dict__})
                return {"done": False, "invalid_call": True, "error": "parameters_not_dict", "data": data}

            self._trace("tool_call", {"tool_name": tool_name, "parameters": params})
            tool_result = self.tools.execute(tool_name, **params)
            if not tool_result.get("success", False):
                self.stats.tool_failures += 1
            self.messages.append({"role": "tool", "name": tool_name, "content": _tool_result_for_history(tool_result)})
            self._trace("tool_result", {"tool_name": tool_name, "tool_result": _tool_result_for_history(tool_result)})

            # 记录第一个 tool（用于强制先看图）
            if not self.stats.first_tool_called:
                self.stats.first_tool_called = str(tool_name)
                if self.require_first_tool_name and self.stats.first_tool_called != self.require_first_tool_name:
                    self.stats.invalid_calls += 1
                    self.stats.steps += 1
                    self._trace(
                        "invalid_call",
                        {
                            "error": "first_tool_must_be",
                            "required": self.require_first_tool_name,
                            "got": self.stats.first_tool_called,
                            "stats": self.stats.__dict__,
                        },
                    )
                    return {
                        "done": False,
                        "invalid_call": True,
                        "error": "first_tool_must_be",
                        "detail": {"required": self.require_first_tool_name, "got": self.stats.first_tool_called},
                    }

            # 任务 tools 计数
            if tool_name == "task_create":
                self.stats.task_create_calls += 1
            elif tool_name == "task_update":
                self.stats.task_update_calls += 1
            elif tool_name == "task_get":
                self.stats.task_get_calls += 1
            elif tool_name == "task_list":
                self.stats.task_list_calls += 1

            self.stats.steps += 1
            self._trace("step_end", {"stats": self.stats.__dict__})
            return {"done": False, "tool_called": tool_name, "tool_result": tool_result}

        # 无 tool：测试模式下允许仅输出场景描述（先 tool 再描述）
        if isinstance(data.get("scene_description"), str) and data.get("scene_description", "").strip():
            if not self.allow_scene_description_only:
                self.stats.invalid_calls += 1
                self.stats.steps += 1
                self._trace("invalid_call", {"error": "scene_description_only_disabled", "data": data, "stats": self.stats.__dict__})
                return {
                    "done": False,
                    "invalid_call": True,
                    "error": "scene_description_only_disabled",
                    "data": data,
                }
            _record_scene_description()
            desc = data["scene_description"].strip()
            self.stats.steps += 1
            self._trace("scene_description", {"scene_description": desc, "stats": self.stats.__dict__})
            return {"done": False, "description_only": True, "scene_description": desc}

        self.stats.invalid_calls += 1
        self.stats.steps += 1
        self._trace("invalid_call", {"error": "expected_tool_or_done", "data": data, "stats": self.stats.__dict__})
        return {
            "done": False,
            "invalid_call": True,
            "error": "expected tool_name or non-empty scene_description or done",
            "data": data,
        }

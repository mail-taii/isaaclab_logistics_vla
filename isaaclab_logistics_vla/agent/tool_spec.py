from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Dict[str, Any]]


class ToolManager:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name!r}")
        self._tools[spec.name] = spec

    def list_tools(self) -> Dict[str, Dict[str, str]]:
        return {k: {"description": v.description} for k, v in self._tools.items()}

    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        if name not in self._tools:
            return {"success": False, "error": f"unknown_tool: {name}"}
        try:
            return self._tools[name].handler(**kwargs)
        except Exception as e:
            return {"success": False, "error": f"tool_exception: {type(e).__name__}", "detail": str(e)}

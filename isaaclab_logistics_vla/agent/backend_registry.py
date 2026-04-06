"""根据名称创建 VlmBackend。"""
from __future__ import annotations

from typing import Any

from isaaclab_logistics_vla.agent.vlm_backend import DummyBackend, VlmBackend


def create_vlm_backend(name: str, **kwargs: Any) -> VlmBackend:
    kw = dict(kwargs)
    test_mode = str(kw.pop("test_mode", "") or "").strip().lower()
    test_describe = bool(kw.pop("test_describe", False)) or (test_mode == "describe")
    test_plan = bool(kw.pop("test_plan", False)) or (test_mode == "plan")
    n = (name or "").strip().lower()
    if n in ("dummy", ""):
        return DummyBackend(test_describe=test_describe, test_plan=test_plan)
    if n in ("volc_ark", "volc-ark", "ark", "anthropic_volc"):
        from isaaclab_logistics_vla.agent.anthropic_volc_backend import VolcArkAnthropicBackend

        return VolcArkAnthropicBackend(test_describe=test_describe, test_plan=test_plan, **kw)
    raise ValueError(f"unknown vlm backend: {name!r}. Supported: dummy, volc_ark")

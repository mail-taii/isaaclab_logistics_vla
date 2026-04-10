from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from isaaclab_logistics_vla.agent.tool_spec import ToolSpec


def build_step_action_tool(env) -> ToolSpec:
    """Execute env steps with given action vector, so robot actually moves.

    Params:
    - action: list[float] of length equal to action_dim (priority)
    - joints: dict[joint_name, target_value] (uses env.joint_index_dict mapping to action vector)
    - repeat: int number of simulation steps to repeat
    """

    def _handler(
        action: Optional[list[float]] = None,
        joints: Optional[Dict[str, float]] = None,
        repeat: int = 1,
    ) -> Dict[str, Any]:
        import torch
        from isaaclab_logistics_vla.agent.tools.gripper_control import sync_attached_object_to_gripper

        isaac_env = env.unwrapped
        adim = int(isaac_env.action_manager.total_action_dim)
        num_envs = int(getattr(isaac_env, "num_envs", 1))
        device = getattr(isaac_env, "device", "cpu")

        rep = int(repeat)
        if rep <= 0 or rep > 10_000:
            return {"success": False, "error": "invalid_repeat", "detail": str(rep)}

        # Build zero vector then fill from args
        a = np.zeros((adim,), dtype=np.float32)
        if action is not None:
            try:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
            except Exception:
                return {"success": False, "error": "invalid_action", "detail": "action must be list[float]"}
            if int(arr.shape[0]) != adim:
                return {
                    "success": False,
                    "error": "action_dim_mismatch",
                    "detail": {"expected": adim, "got": int(arr.shape[0])},
                }
            a = arr

        if joints:
            mapping = getattr(env, "joint_index_dict", {}) or {}
            if not isinstance(mapping, dict) or not mapping:
                return {
                    "success": False,
                    "error": "joint_mapping_missing",
                    "detail": "env.joint_index_dict is not available; provide full 'action' instead",
                }
            for k, v in dict(joints).items():
                if k not in mapping:
                    return {
                        "success": False,
                        "error": "unknown_joint",
                        "detail": {"joint": k, "known": list(mapping.keys())[:50]},
                    }
                idx = int(mapping[k])
                if idx < 0 or idx >= adim:
                    return {
                        "success": False,
                        "error": "joint_index_oob",
                        "detail": {"joint": k, "idx": idx, "adim": adim},
                    }
                a[idx] = float(v)

        act = torch.tensor(a, dtype=torch.float32, device=device).view(1, adim).repeat(num_envs, 1)
        last_info = None
        for _ in range(rep):
            sync_attached_object_to_gripper(env)
            _, _, _, _, info = isaac_env.step(act)
            last_info = info

        return {
            "success": True,
            "action_dim": adim,
            "repeat": rep,
            "used_joints": sorted(list(joints.keys())) if joints else [],
            "joint_mapping_keys_sample": list((getattr(env, "joint_index_dict", {}) or {}).keys())[:30],
            "info_keys": list(last_info.keys())[:50] if isinstance(last_info, dict) else [],
        }

    return ToolSpec(
        name="step_action",
        description="Step the simulation with your action vector (or joint-name targets) so robot executes motion.",
        handler=_handler,
    )

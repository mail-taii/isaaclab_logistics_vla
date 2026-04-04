"""
将 CuRobo MotionGen 的返回值转为 RoboTwin 风格的 dict（CPU numpy），供上层与仿真解耦。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


def motion_gen_batch_result_to_plan_dict(
    result: Any,
    batch_index: int = 0,
) -> dict[str, Any]:
    """从 ``plan_batch`` 的 ``MotionGenResult`` 提取单条轨迹，转为标准 dict。

    延迟导入 ``torch``，便于在无 CUDA/仅测 ``plan_grippers_linear`` 时加载本模块。

    Keys:
        - ``status``: ``\"Success\"`` | ``\"Fail\"``
        - ``position``: 成功时为 ``(T, n_dof)`` 的 float32 ndarray，失败为 ``None``
        - ``velocity``: 成功且存在时为同形状 ndarray，否则 ``None``
        - ``detail``: 可选，CuRobo 状态枚举或字符串
        - ``interpolation_dt``: 可选 float，来自 result
    """
    out: dict[str, Any] = {
        "status": "Fail",
        "position": None,
        "velocity": None,
    }
    if result is None:
        return out

    try:
        succ = result.success
        if succ is None:
            return out
        ok = bool(succ[batch_index].item()) if succ.dim() > 0 else bool(succ.item())
    except Exception:
        ok = False

    st = getattr(result, "status", None)
    if st is not None:
        out["detail"] = st.name if hasattr(st, "name") else str(st)

    if not ok:
        return out

    import torch

    interp = None
    try:
        interp = result.get_interpolated_plan()
    except Exception:
        interp = getattr(result, "interpolated_plan", None)

    if interp is None:
        return out

    pos = interp.position
    if isinstance(pos, torch.Tensor):
        if pos.dim() == 3:
            pos = pos[batch_index]
        elif pos.dim() == 2 and pos.shape[0] > 1 and getattr(result.success, "shape", [0])[0] > 1:
            pos = pos[batch_index]
        pos_np = pos.detach().float().cpu().numpy()
    else:
        pos_np = np.asarray(pos, dtype=np.float32)

    vel_np: Optional[np.ndarray] = None
    vel = getattr(interp, "velocity", None)
    if vel is not None and isinstance(vel, torch.Tensor):
        v = vel[batch_index] if vel.dim() == 3 else vel
        vel_np = v.detach().float().cpu().numpy()

    idt = getattr(result, "interpolation_dt", None)
    if idt is not None and isinstance(idt, torch.Tensor):
        idt = float(idt.flatten()[0].item())
    if idt is not None:
        out["interpolation_dt"] = idt

    out["status"] = "Success"
    out["position"] = np.asarray(pos_np, dtype=np.float32)
    out["velocity"] = np.asarray(vel_np, dtype=np.float32) if vel_np is not None else None
    return out


def plan_grippers_linear(now_val: float, target_val: float, num_step: int = 200) -> dict[str, Any]:
    """RoboTwin 风格夹爪插值（不经过 CuRobo）。"""
    dis_val = target_val - now_val
    step = dis_val / num_step if num_step else 0.0
    vals = np.linspace(now_val, target_val, num_step, dtype=np.float32)
    return {
        "num_step": num_step,
        "per_step": step,
        "result": vals,
    }

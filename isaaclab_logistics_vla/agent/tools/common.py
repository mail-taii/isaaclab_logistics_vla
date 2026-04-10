from __future__ import annotations

from typing import Any

import numpy as np


def _camera_rgb_to_uint8(frame: Any) -> np.ndarray:
    """Normalize IsaacLab camera output to uint8 HxWx3."""
    import torch

    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    # common shapes: (H,W,3) or (1,H,W,3)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def read_topview_rgb_uint8(env) -> np.ndarray | None:
    """Read current RGB frame from scene top_camera, uint8 HxWx3; return None on failure."""
    isaac_env = env.unwrapped
    sensors = getattr(isaac_env.scene, "sensors", {})
    if "top_camera" not in sensors:
        return None
    cam = sensors["top_camera"]
    data = getattr(cam, "data", None)
    output = getattr(data, "output", None) if data is not None else None
    if not isinstance(output, dict):
        return None
    if "rgb" in output and output["rgb"] is not None:
        rgb = output["rgb"]
    elif "rgba" in output and output["rgba"] is not None:
        rgb = output["rgba"][..., :3]
    else:
        return None
    return _camera_rgb_to_uint8(rgb)

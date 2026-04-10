from __future__ import annotations

from typing import Any, Dict

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from typing import Any, Dict

from .common import read_topview_rgb_uint8
from isaaclab_logistics_vla.agent.tool_spec import ToolSpec


def build_get_topview_image_tool(env) -> ToolSpec:
    """Create tool: get current top-view RGB image from the scene camera.

    Convention: camera named `top_camera` (injected by `BaseOrderSceneCfg.top_camera`).
    """

    def _handler() -> Dict[str, Any]:
        frame = read_topview_rgb_uint8(env)
        if frame is None:
            return {
                "success": False,
                "error": "camera_rgb_missing",
                "detail": "camera named 'top_camera' not found or no valid rgb output",
            }
        return {
            "success": True,
            "image_rgb_uint8": frame,  # numpy.ndarray (H,W,3)
            "height": int(frame.shape[0]),
            "width": int(frame.shape[1]),
        }

    return ToolSpec(
        name="get_topview_image",
        description="Get current top-view RGB image (uint8 HxWx3) from the scene fixed camera.",
        handler=_handler,
    )

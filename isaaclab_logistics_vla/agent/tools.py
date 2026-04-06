from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
from pathlib import Path


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


def build_get_topview_image_tool(env) -> ToolSpec:
    """创建唯一 tool：获取顶视相机 RGB 图片。

    约定：相机名为 `top_camera`（由 `BaseOrderSceneCfg.top_camera` 注入）。
    """

    def _handler() -> Dict[str, Any]:
        isaac_env = env.unwrapped
        sensors = getattr(isaac_env.scene, "sensors", {})
        if "top_camera" not in sensors:
            # 兼容：部分版本会把 CameraCfg 的属性名当作 key
            # 但这里按我们在 scene_cfg 里定义的属性名来找。
            return {
                "success": False,
                "error": "camera_not_found",
                "detail": f"scene.sensors keys={list(sensors.keys())}",
            }

        cam = sensors["top_camera"]
        data = getattr(cam, "data", None)
        output = getattr(data, "output", None) if data is not None else None
        if not isinstance(output, dict):
            return {"success": False, "error": "camera_output_missing"}

        if "rgb" in output and output["rgb"] is not None:
            rgb = output["rgb"]
        elif "rgba" in output and output["rgba"] is not None:
            rgb = output["rgba"][..., :3]
        else:
            return {"success": False, "error": "camera_rgb_missing", "detail": f"keys={list(output.keys())}"}

        frame = _camera_rgb_to_uint8(rgb)
        return {
            "success": True,
            "image_rgb_uint8": frame,  # numpy.ndarray (H,W,3)
            "height": int(frame.shape[0]),
            "width": int(frame.shape[1]),
        }

    return ToolSpec(
        name="get_topview_image",
        description="Get current top-view RGB image (uint8 HxWx3) from the scene camera.",
        handler=_handler,
    )


def build_task_system_tools(tasks_dir: str | Path) -> list[ToolSpec]:
    """为 harness 层提供持久化任务系统 tools（与仿真无关）。"""
    from isaaclab_logistics_vla.agent.task_system import TaskManager

    tm = TaskManager(Path(tasks_dir))

    def _task_create(subject: str, description: str = "", owner: str = "", blockedBy: Optional[list[int]] = None):
        t = tm.create(subject=subject, description=description, owner=owner, blocked_by=blockedBy)
        return {"success": True, "task": t.to_dict()}

    def _task_update(
        task_id: int,
        status: Optional[str] = None,
        add_blocked_by: Optional[list[int]] = None,
        remove_blocked_by: Optional[list[int]] = None,
        owner: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
    ):
        out = tm.update(
            task_id=task_id,
            status=status,
            add_blocked_by=add_blocked_by,
            remove_blocked_by=remove_blocked_by,
            owner=owner,
            subject=subject,
            description=description,
        )
        return {"success": True, **out}

    def _task_get(task_id: int):
        t = tm.get(task_id)
        return {"success": True, "task": t.to_dict()}

    def _task_list(view: str = "all"):
        if view == "ready":
            ts = tm.ready_tasks()
        elif view == "blocked":
            ts = tm.blocked_tasks()
        else:
            ts = tm.list_all()
        return {"success": True, "tasks": [t.to_dict() for t in ts], "view": view}

    return [
        ToolSpec(
            name="task_create",
            description="Create a persistent task node: {subject, description?, owner?, blockedBy?}.",
            handler=_task_create,
        ),
        ToolSpec(
            name="task_update",
            description="Update a task: status/owner/subject/description and dependency edges (add_blocked_by/remove_blocked_by). Completing a task auto-unblocks dependents.",
            handler=_task_update,
        ),
        ToolSpec(
            name="task_get",
            description="Get a task by id.",
            handler=_task_get,
        ),
        ToolSpec(
            name="task_list",
            description="List tasks. Optional view: all|ready|blocked.",
            handler=_task_list,
        ),
    ]


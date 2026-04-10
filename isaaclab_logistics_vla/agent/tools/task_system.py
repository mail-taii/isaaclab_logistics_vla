from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List

from isaaclab_logistics_vla.agent.tool_spec import ToolSpec


def build_task_system_tools(tasks_dir: str | Path) -> List[ToolSpec]:
    """Task-system persistent tools: create/update/get/list tasks in persistent JSON files on disk."""
    from isaaclab_logistics_vla.agent.task_system import TaskManager

    tm = TaskManager(Path(tasks_dir))

    def _task_create(
        subject: str,
        description: str = "",
        owner: str = "",
        blockedBy: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        t = tm.create(
            subject=subject,
            description=description,
            owner=owner,
            blocked_by=blockedBy,
        )
        return {"success": True, "task": t.to_dict()}

    def _task_update(
        task_id: int,
        status: Optional[str] = None,
        add_blocked_by: Optional[List[int]] = None,
        remove_blocked_by: Optional[List[int]] = None,
        owner: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
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

    def _task_get(task_id: int) -> Dict[str, Any]:
        t = tm.get(task_id)
        return {"success": True, "task": t.to_dict()}

    def _task_list(view: str = "all") -> Dict[str, Any]:
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
            description="Create a new persistent task node: {subject, description?, owner?, blockedBy?}.",
            handler=_task_create,
        ),
        ToolSpec(
            name="task_update",
            description="Update a task: status/owner/subject/description and dependency edges (add_blocked_by/remove_blocked_by). Completing a task auto-unblocks its dependents.",
            handler=_task_update,
        ),
        ToolSpec(
            name="task_get",
            description="Get the current dictionary of a single task by id.",
            handler=_task_get,
        ),
        ToolSpec(
            name="task_list",
            description="List tasks, optional filter by view: all|ready|blocked.",
            handler=_task_list,
        ),
    ]

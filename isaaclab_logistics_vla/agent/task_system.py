from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TaskStatus = str  # "pending" | "in_progress" | "completed" | "cancelled"


def _now_s() -> float:
    return float(time.time())


def _task_path(tasks_dir: Path, task_id: int) -> Path:
    return tasks_dir / f"task_{task_id}.json"


def _safe_int(x: Any) -> int:
    if isinstance(x, bool):
        raise ValueError("bool is not a valid task id")
    return int(x)


@dataclass
class Task:
    id: int
    subject: str
    description: str = ""
    status: TaskStatus = "pending"
    blockedBy: List[int] = None  # noqa: N815  (keep field name per spec)
    owner: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "blockedBy": list(self.blockedBy or []),
            "owner": self.owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Task":
        return Task(
            id=_safe_int(d["id"]),
            subject=str(d.get("subject", "")),
            description=str(d.get("description", "")),
            status=str(d.get("status", "pending")),
            blockedBy=[_safe_int(x) for x in d.get("blockedBy", [])],
            owner=str(d.get("owner", "")),
            created_at=float(d.get("created_at", 0.0)),
            updated_at=float(d.get("updated_at", 0.0)),
        )


class TaskManager:
    """磁盘持久化任务图：每个任务一个 JSON 文件，支持依赖 blockedBy 与自动解锁。"""

    def __init__(self, tasks_dir: Path):
        self.dir = Path(tasks_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        mx = 0
        for p in self.dir.glob("task_*.json"):
            try:
                tid = int(p.stem.split("_")[-1])
                mx = max(mx, tid)
            except Exception:
                continue
        return mx

    def _save(self, task: Task) -> None:
        task.updated_at = _now_s()
        if task.created_at <= 0:
            task.created_at = task.updated_at
        _task_path(self.dir, task.id).write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self, task_id: int) -> Task:
        p = _task_path(self.dir, task_id)
        if not p.exists():
            raise FileNotFoundError(f"task not found: {task_id}")
        return Task.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def create(self, subject: str, description: str = "", owner: str = "", blocked_by: Optional[List[int]] = None) -> Task:
        t = Task(
            id=self._next_id,
            subject=str(subject),
            description=str(description or ""),
            status="pending",
            blockedBy=[_safe_int(x) for x in (blocked_by or [])],
            owner=str(owner or ""),
        )
        self._save(t)
        self._next_id += 1
        return t

    def get(self, task_id: int) -> Task:
        return self._load(_safe_int(task_id))

    def list_all(self) -> List[Task]:
        tasks: List[Task] = []
        for p in sorted(self.dir.glob("task_*.json")):
            try:
                tasks.append(Task.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue
        tasks.sort(key=lambda t: t.id)
        return tasks

    def _clear_dependency(self, completed_id: int) -> int:
        """从其他任务的 blockedBy 中移除 completed_id。返回被解锁/更新的任务数量。"""
        updated = 0
        for t in self.list_all():
            if completed_id in (t.blockedBy or []):
                t.blockedBy = [x for x in t.blockedBy if x != completed_id]
                self._save(t)
                updated += 1
        return updated

    def update(
        self,
        task_id: int,
        status: Optional[TaskStatus] = None,
        add_blocked_by: Optional[List[int]] = None,
        remove_blocked_by: Optional[List[int]] = None,
        owner: Optional[str] = None,
        subject: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        t = self._load(_safe_int(task_id))

        if subject is not None:
            t.subject = str(subject)
        if description is not None:
            t.description = str(description)
        if owner is not None:
            t.owner = str(owner)

        if add_blocked_by:
            add_ids = {_safe_int(x) for x in add_blocked_by}
            t.blockedBy = sorted(set((t.blockedBy or []) + list(add_ids)))
        if remove_blocked_by:
            rm = {_safe_int(x) for x in remove_blocked_by}
            t.blockedBy = [x for x in (t.blockedBy or []) if x not in rm]

        unlocked = 0
        if status:
            t.status = str(status)
            self._save(t)
            if t.status == "completed":
                unlocked = self._clear_dependency(t.id)
        else:
            self._save(t)

        return {"task": t.to_dict(), "unlocked_tasks_updated": unlocked}

    def ready_tasks(self) -> List[Task]:
        """状态 pending 且没有 blockedBy 的任务。"""
        out = []
        for t in self.list_all():
            if t.status == "pending" and not (t.blockedBy or []):
                out.append(t)
        return out

    def blocked_tasks(self) -> List[Task]:
        """pending 但被依赖卡住的任务。"""
        out = []
        for t in self.list_all():
            if t.status == "pending" and (t.blockedBy or []):
                out.append(t)
        return out


def format_task_board(tasks: List[Dict[str, Any]] | List[Task]) -> str:
    """把任务列表渲染成控制台看板（类似 todo board）。"""
    norm: List[Dict[str, Any]] = []
    for t in tasks:
        if isinstance(t, Task):
            norm.append(t.to_dict())
        else:
            norm.append(dict(t))

    def _k(x: Dict[str, Any]) -> int:
        try:
            return int(x.get("id", 0))
        except Exception:
            return 0

    norm.sort(key=_k)
    ready, blocked, doing, done, other = [], [], [], [], []
    for t in norm:
        st = str(t.get("status", "pending"))
        bb = t.get("blockedBy", []) or []
        if st == "completed":
            done.append(t)
        elif st == "in_progress":
            doing.append(t)
        elif st == "pending" and not bb:
            ready.append(t)
        elif st == "pending" and bb:
            blocked.append(t)
        else:
            other.append(t)

    def _status_mark(st: str) -> str:
        if st == "completed":
            return "✓"
        if st == "in_progress":
            return "▶"
        if st == "cancelled":
            return "✗"
        # pending / unknown
        return " "

    def _fmt_list(title: str, items: List[Dict[str, Any]]) -> List[str]:
        lines = [f"{title} ({len(items)}):"]
        for t in items:
            tid = t.get("id")
            subj = str(t.get("subject", "")).strip()
            owner = str(t.get("owner", "")).strip()
            bb = t.get("blockedBy", []) or []
            st = str(t.get("status", "pending"))
            extra = []
            if owner:
                extra.append(f"owner={owner}")
            if bb:
                extra.append(f"blockedBy={bb}")
            suffix = f"  ({', '.join(extra)})" if extra else ""
            lines.append(f"  [{_status_mark(st)}] [{tid}] {subj}{suffix}")
        return lines

    out_lines: List[str] = []
    out_lines += _fmt_list("READY", ready)
    out_lines += _fmt_list("IN_PROGRESS", doing)
    out_lines += _fmt_list("BLOCKED", blocked)
    out_lines += _fmt_list("DONE", done)
    if other:
        out_lines += _fmt_list("OTHER", other)
    return "\n".join(out_lines)


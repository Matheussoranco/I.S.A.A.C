"""Kanban agent tools — let in-graph agents collaborate on the shared board.

Registered via :func:`register_all_tools` alongside the other built-ins.
All tools operate on the default shared :class:`KanbanBoard` (SQLite at
``~/.isaac/kanban.db``) unless ``ISAAC_KANBAN_DB`` points elsewhere.
"""

from __future__ import annotations

from typing import Any

from isaac.background.kanban import KanbanError, get_board
from isaac.tools.base import IsaacTool, ToolResult


def _current_task_id() -> str:
    import os

    return os.environ.get("ISAAC_KANBAN_TASK", "")


def _current_agent() -> str:
    try:
        from isaac.config.profiles import get_active_profile

        return get_active_profile()
    except Exception:
        return "agent"


def _err(exc: Exception) -> ToolResult:
    return ToolResult(success=False, error=str(exc))


def _render_task(t: Any, board: Any) -> str:
    deps = board.dependencies_of(t.id)
    lines = [
        f"[{t.id}] {t.title}",
        f"  status:   {t.status}",
        f"  priority: {t.priority}",
    ]
    if t.description:
        lines.append(f"  desc:     {t.description}")
    if t.assignee:
        lines.append(f"  assignee: {t.assignee}")
    if deps:
        lines.append(f"  blocked by: {', '.join(deps)}")
    if t.blocked_reason:
        lines.append(f"  blocked reason: {t.blocked_reason}")
    return "\n".join(lines)


class KanbanShowTool(IsaacTool):
    """Show the board (or one task) to the agent."""

    name = "kanban_show"
    description = (
        "Show the shared kanban board: columns (backlog/ready/running/review/"
        "complete/blocked), counts, and task details. Pass a task_id for one "
        "task including its comment thread."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "Optional task id to inspect."},
        },
    }

    async def execute(self, task_id: str = "", **_: Any) -> ToolResult:
        board = get_board()
        try:
            if task_id:
                t = board.get_task(task_id)
                if t is None:
                    return _err(KanbanError(f"task not found: {task_id}"))
                out = [_render_task(t, board)]
                comments = board.list_comments(task_id)
                if comments:
                    out.append("  comments:")
                    out.extend(f"    - [{c['author']}] {c['body']}" for c in comments)
                return ToolResult(success=True, output="\n".join(out))
            state = board.get_board_state()
            lines = [f"Kanban board — {state['total']} tasks"]
            for col, items in state["columns"].items():
                lines.append(f"\n== {col} ({state['counts'][col]}) ==")
                for t in items:
                    assignee = f" @{t['assignee']}" if t["assignee"] else ""
                    lines.append(f"  [{t['id']}] (p{t['priority']}) {t['title']}{assignee}")
            return ToolResult(success=True, output="\n".join(lines), metadata=state)
        except Exception as exc:
            return _err(exc)


class KanbanCreateTool(IsaacTool):
    """Create a new card on the board."""

    name = "kanban_create"
    description = (
        "Create a kanban task. title required; optional description, priority "
        "(1=highest, default 5), assignee (profile name), and depends_on "
        "(list of task ids that must complete first). Tasks with unfinished "
        "dependencies stay in backlog until unblocked."
    )
    risk_level = 2
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string", "default": ""},
            "priority": {"type": "integer", "default": 5},
            "assignee": {"type": "string", "default": ""},
            "depends_on": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["title"],
    }

    async def execute(  # type: ignore[override]
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        assignee: str = "",
        depends_on: list[str] | None = None,
        **_: Any,
    ) -> ToolResult:
        board = get_board()
        try:
            t = board.create_task(
                title,
                description,
                priority=priority,
                assignee=assignee,
                depends_on=depends_on or [],
            )
            return ToolResult(
                success=True,
                output=f"Created task [{t.id}] {t.title} ({t.status})",
                metadata={"task_id": t.id, "status": t.status},
            )
        except KanbanError as exc:
            return _err(exc)


class KanbanCompleteTool(IsaacTool):
    """Mark a task complete."""

    name = "kanban_complete"
    description = (
        "Complete a kanban task (defaults to the task this agent was "
        "dispatched for via ISAAC_KANBAN_TASK). Optionally include a result "
        "summary recorded as a comment. Unblocks dependent tasks."
    )
    risk_level = 2
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "default": ""},
            "result": {"type": "string", "default": ""},
        },
    }

    async def execute(self, task_id: str = "", result: str = "", **_: Any) -> ToolResult:
        task_id = task_id or _current_task_id()
        if not task_id:
            return _err(KanbanError("no task_id given and ISAAC_KANBAN_TASK not set"))
        try:
            t = get_board().complete_task(task_id, result=result)
            return ToolResult(success=True, output=f"Completed task [{t.id}] {t.title}")
        except KanbanError as exc:
            return _err(exc)


class KanbanBlockTool(IsaacTool):
    """Mark a task blocked."""

    name = "kanban_block"
    description = (
        "Mark a kanban task as blocked because you cannot proceed (missing "
        "info, needs approval, external failure). Give a clear reason so a "
        "human or another agent can unblock it."
    )
    risk_level = 2
    parameters = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why the task is blocked."},
            "task_id": {"type": "string", "default": ""},
        },
        "required": ["reason"],
    }

    async def execute(  # type: ignore[override]
        self, reason: str, task_id: str = "", **_: Any
    ) -> ToolResult:
        task_id = task_id or _current_task_id()
        if not task_id:
            return _err(KanbanError("no task_id given and ISAAC_KANBAN_TASK not set"))
        try:
            t = get_board().block_task(task_id, reason)
            return ToolResult(success=True, output=f"Blocked task [{t.id}]: {reason}")
        except KanbanError as exc:
            return _err(exc)


class KanbanHeartbeatTool(IsaacTool):
    """Refresh the running claim's heartbeat."""

    name = "kanban_heartbeat"
    description = (
        "Refresh the heartbeat on a running kanban task (defaults to the "
        "dispatched task). Call every few minutes while working; claims with "
        "no heartbeat for 5 minutes are reclaimed by the dispatcher for "
        "another agent."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"task_id": {"type": "string", "default": ""}},
    }

    async def execute(self, task_id: str = "", **_: Any) -> ToolResult:
        task_id = task_id or _current_task_id()
        if not task_id:
            return _err(KanbanError("no task_id given and ISAAC_KANBAN_TASK not set"))
        try:
            t = get_board().heartbeat(task_id, agent=_current_agent())
            return ToolResult(
                success=True,
                output=f"Heartbeat refreshed for [{t.id}] at {t.heartbeat_at}",
            )
        except KanbanError as exc:
            return _err(exc)


class KanbanCommentTool(IsaacTool):
    """Attach a comment to a task."""

    name = "kanban_comment"
    description = (
        "Add a comment to a kanban task (progress note, question, or context "
        "for the next agent). Defaults to the dispatched task."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {
            "body": {"type": "string"},
            "task_id": {"type": "string", "default": ""},
        },
        "required": ["body"],
    }

    async def execute(  # type: ignore[override]
        self, body: str, task_id: str = "", **_: Any
    ) -> ToolResult:
        task_id = task_id or _current_task_id()
        if not task_id:
            return _err(KanbanError("no task_id given and ISAAC_KANBAN_TASK not set"))
        try:
            c = get_board().add_comment(task_id, body, author=_current_agent())
            return ToolResult(success=True, output=f"Comment #{c['id']} added to [{task_id}]")
        except KanbanError as exc:
            return _err(exc)


class KanbanLinkTool(IsaacTool):
    """Declare a dependency between tasks."""

    name = "kanban_link"
    description = (
        "Declare that task_id depends on another task (depends_on) — it will "
        "not become ready until the dependency completes. Cycles are rejected."
    )
    risk_level = 2
    parameters = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "depends_on": {"type": "string"},
        },
        "required": ["task_id", "depends_on"],
    }

    async def execute(  # type: ignore[override]
        self, task_id: str, depends_on: str, **_: Any
    ) -> ToolResult:
        try:
            get_board().link_tasks(task_id, depends_on)
            return ToolResult(
                success=True,
                output=f"Linked: [{task_id}] now depends on [{depends_on}]",
            )
        except KanbanError as exc:
            return _err(exc)


KANBAN_TOOL_CLASSES = (
    KanbanShowTool,
    KanbanCreateTool,
    KanbanCompleteTool,
    KanbanBlockTool,
    KanbanHeartbeatTool,
    KanbanCommentTool,
    KanbanLinkTool,
)

"""Kanban multi-agent queue — SQLite-backed task board.

Columns: ``backlog``, ``ready``, ``running``, ``review``, ``complete``
(plus ``blocked`` for tasks that need human intervention).  Tasks can have
assignees, dependencies on other tasks, and a comment thread.  Agents
*claim* ready tasks; claims carry a heartbeat timestamp so the dispatcher
(:mod:`isaac.background.kanban_dispatcher`) can reclaim stale claims.

Storage lives at ``~/.isaac/kanban.db`` by default (overridable with the
``db_path`` argument or the ``ISAAC_KANBAN_DB`` env var) so multiple
gateways / agent processes share one board.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

#: Valid board columns (status values).
COLUMNS: tuple[str, ...] = ("backlog", "ready", "running", "review", "complete", "blocked")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'backlog',
    assignee      TEXT DEFAULT '',
    priority      INTEGER DEFAULT 5,
    claimed_at    TEXT DEFAULT '',
    heartbeat_at  TEXT DEFAULT '',
    blocked_reason TEXT DEFAULT '',
    workspace     TEXT DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dependencies (
    task_id      TEXT NOT NULL,
    depends_on   TEXT NOT NULL,
    PRIMARY KEY (task_id, depends_on)
);
CREATE TABLE IF NOT EXISTS comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT DEFAULT 'agent',
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_comments_task ON comments(task_id);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A single card on the kanban board."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    title: str = ""
    description: str = ""
    status: str = "backlog"
    assignee: str = ""
    priority: int = 5
    claimed_at: str = ""
    heartbeat_at: str = ""
    blocked_reason: str = ""
    workspace: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"


def _default_db_path() -> Path:
    env = os.environ.get("ISAAC_KANBAN_DB", "")
    if env:
        return Path(env)
    return _isaac_home() / "kanban.db"


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


class KanbanError(Exception):
    """Raised for invalid board operations (bad transitions, missing tasks)."""


class KanbanBoard:
    """SQLite-backed kanban board, safe to share between processes."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self._local = threading.local()

    # -- connection handling ------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- schema -------------------------------------------------------------

    def init_board(self) -> Path:
        """Create the database schema (idempotent).  Returns the db path."""
        conn = self._conn()
        conn.executescript(_SCHEMA)
        conn.commit()
        logger.info("Kanban board initialised at %s", self.db_path)
        return self.db_path

    # -- CRUD ---------------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str = "",
        *,
        status: str = "backlog",
        priority: int = 5,
        assignee: str = "",
        depends_on: list[str] | None = None,
    ) -> Task:
        """Create a task.  Dependencies starting satisfied keep the given
        status; the dispatcher later promotes backlog cards to ready."""
        if status not in COLUMNS:
            raise KanbanError(f"invalid status: {status!r}")
        task = Task(title=title, description=description, status=status,
                    priority=priority, assignee=assignee)
        conn = self._conn()
        conn.execute(
            "INSERT INTO tasks (id,title,description,status,assignee,priority,"
            "claimed_at,heartbeat_at,blocked_reason,workspace,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (task.id, task.title, task.description, task.status, task.assignee,
             task.priority, "", "", "", "", task.created_at, task.updated_at),
        )
        for dep in depends_on or []:
            if not self.get_task(dep):
                raise KanbanError(f"dependency task not found: {dep}")
            conn.execute(
                "INSERT INTO dependencies (task_id, depends_on) VALUES (?,?)",
                (task.id, dep),
            )
        conn.commit()
        return task

    def get_task(self, task_id: str) -> Task | None:
        row = self._conn().execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return Task(**{k: row[k] for k in row.keys()})

    def list_tasks(self, status: str | None = None) -> list[Task]:
        conn = self._conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority, created_at",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY priority, created_at"
            ).fetchall()
        return [Task(**{k: r[k] for k in r.keys()}) for r in rows]

    # -- state transitions --------------------------------------------------

    def _update(self, task_id: str, **fields: Any) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise KanbanError(f"task not found: {task_id}")
        fields["updated_at"] = _now()
        sets = ", ".join(f"{k}=?" for k in fields)
        self._conn().execute(
            f"UPDATE tasks SET {sets} WHERE id=?", (*fields.values(), task_id)
        )
        self._conn().commit()
        return self.get_task(task_id)  # type: ignore[return-value]

    def move_task(self, task_id: str, column: str) -> Task:
        """Move a task to any column (validated)."""
        if column not in COLUMNS:
            raise KanbanError(f"invalid column: {column!r}")
        return self._update(task_id, status=column)

    def assign_task(self, task_id: str, assignee: str) -> Task:
        """Assign (or reassign) a task to an agent/profile name."""
        return self._update(task_id, assignee=assignee)

    def claim_task(self, task_id: str, agent: str) -> Task:
        """Atomically claim a ready (or unclaimed assigned) task for an agent.

        Moves the task to ``running`` and stamps claim + heartbeat times.
        Raises :class:`KanbanError` if already claimed by someone else.
        """
        conn = self._conn()
        task = self.get_task(task_id)
        if task is None:
            raise KanbanError(f"task not found: {task_id}")
        if task.status == "running" and task.assignee and task.assignee != agent:
            raise KanbanError(f"task {task_id} already claimed by {task.assignee}")
        if task.status in ("complete",):
            raise KanbanError(f"task {task_id} is complete")
        now = _now()
        cur = conn.execute(
            "UPDATE tasks SET status='running', assignee=?, claimed_at=?,"
            " heartbeat_at=?, updated_at=? WHERE id=?"
            " AND NOT (status='running' AND assignee != '' AND assignee != ?)",
            (agent, now, now, now, task_id, agent),
        )
        conn.commit()
        if cur.rowcount == 0:
            raise KanbanError(f"task {task_id} could not be claimed (race)")
        return self.get_task(task_id)  # type: ignore[return-value]

    def heartbeat(self, task_id: str, agent: str | None = None) -> Task:
        """Refresh the claim heartbeat (keeps a running claim alive)."""
        task = self.get_task(task_id)
        if task is None:
            raise KanbanError(f"task not found: {task_id}")
        if task.status != "running":
            raise KanbanError(f"task {task_id} is not running")
        if agent and task.assignee and task.assignee != agent:
            raise KanbanError(f"task {task_id} owned by {task.assignee}")
        return self._update(task_id, heartbeat_at=_now())

    def complete_task(self, task_id: str, *, result: str = "") -> Task:
        """Move a task to ``complete`` and record an optional result comment."""
        task = self.get_task(task_id)
        if task is None:
            raise KanbanError(f"task not found: {task_id}")
        if task.status == "complete":
            return task
        done = self._update(task_id, status="complete", blocked_reason="")
        if result:
            self.add_comment(task_id, author=task.assignee or "agent", body=result)
        # promote dependents whose deps are now all satisfied
        self._promote_dependents(task_id)
        return done

    def block_task(self, task_id: str, reason: str = "") -> Task:
        """Move a task to ``blocked`` with a reason (needs human help)."""
        return self._update(task_id, status="blocked", blocked_reason=reason)

    # -- dependencies -------------------------------------------------------

    def link_tasks(self, task_id: str, depends_on: str) -> None:
        """Declare that ``task_id`` depends on ``depends_on`` being complete."""
        if task_id == depends_on:
            raise KanbanError("a task cannot depend on itself")
        if not self.get_task(task_id):
            raise KanbanError(f"task not found: {task_id}")
        if not self.get_task(depends_on):
            raise KanbanError(f"dependency not found: {depends_on}")
        if self._creates_cycle(task_id, depends_on):
            raise KanbanError("dependency would create a cycle")
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO dependencies (task_id, depends_on) VALUES (?,?)",
            (task_id, depends_on),
        )
        conn.commit()

    def dependencies_of(self, task_id: str) -> list[str]:
        rows = self._conn().execute(
            "SELECT depends_on FROM dependencies WHERE task_id=?", (task_id,)
        ).fetchall()
        return [r["depends_on"] for r in rows]

    def dependents_of(self, task_id: str) -> list[str]:
        rows = self._conn().execute(
            "SELECT task_id FROM dependencies WHERE depends_on=?", (task_id,)
        ).fetchall()
        return [r["task_id"] for r in rows]

    def _creates_cycle(self, task_id: str, depends_on: str) -> bool:
        """True if adding task_id->depends_on closes a dependency cycle."""
        seen, stack = set(), [depends_on]
        while stack:
            cur = stack.pop()
            if cur == task_id:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.dependencies_of(cur))
        return False

    def _deps_satisfied(self, task_id: str) -> bool:
        deps = self.dependencies_of(task_id)
        return all(
            (t := self.get_task(d)) is not None and t.status == "complete"
            for d in deps
        )

    def _promote_dependents(self, completed_id: str) -> None:
        for dep_id in self.dependents_of(completed_id):
            t = self.get_task(dep_id)
            if t and t.status == "backlog" and self._deps_satisfied(dep_id):
                self._update(dep_id, status="ready")

    # -- readiness / queries ------------------------------------------------

    def list_ready(self) -> list[Task]:
        """Backlog/ready tasks whose dependencies are all complete,
        promoted to ``ready`` on the fly.  Ordered by priority then age."""
        ready: list[Task] = []
        for task in self.list_tasks():
            if task.status == "ready" and self._deps_satisfied(task.id):
                ready.append(task)
            elif task.status == "backlog" and self._deps_satisfied(task.id):
                ready.append(self._update(task.id, status="ready"))
        ready.sort(key=lambda t: (t.priority, t.created_at))
        return ready

    # -- comments -----------------------------------------------------------

    def add_comment(self, task_id: str, body: str, author: str = "agent") -> dict[str, Any]:
        if not self.get_task(task_id):
            raise KanbanError(f"task not found: {task_id}")
        now = _now()
        cur = self._conn().execute(
            "INSERT INTO comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
            (task_id, author, body, now),
        )
        self._conn().commit()
        return {"id": cur.lastrowid, "task_id": task_id, "author": author,
                "body": body, "created_at": now}

    def list_comments(self, task_id: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM comments WHERE task_id=? ORDER BY id", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- board state --------------------------------------------------------

    def get_board_state(self) -> dict[str, Any]:
        """Snapshot: per-column task lists + counts (JSON-serialisable)."""
        tasks = self.list_tasks()
        columns: dict[str, list[dict[str, Any]]] = {c: [] for c in COLUMNS}
        for t in tasks:
            d = asdict(t)
            d["depends_on"] = self.dependencies_of(t.id)
            columns.setdefault(t.status, []).append(d)
        counts = {c: len(v) for c, v in columns.items()}
        return {"columns": columns, "counts": counts, "total": len(tasks)}

    def board_json(self) -> str:
        return json.dumps(self.get_board_state(), indent=2)


# ---------------------------------------------------------------------------
# Module-level convenience (shared default board)
# ---------------------------------------------------------------------------

_default_board: KanbanBoard | None = None


def get_board(db_path: Path | str | None = None) -> KanbanBoard:
    """Return the shared default board (or one at ``db_path``)."""
    global _default_board
    if db_path is not None:
        return KanbanBoard(db_path)
    if _default_board is None:
        _default_board = KanbanBoard()
        _default_board.init_board()
    return _default_board


def reset_default_board() -> None:
    """Close and discard the shared default board (used by tests)."""
    global _default_board
    if _default_board is not None:
        _default_board.close()
    _default_board = None

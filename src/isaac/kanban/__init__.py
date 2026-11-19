"""Kanban board persistence — SQLite-backed task boards with deps, claims, comments.

Schema
------
board (id, name, created_at, updated_at)
column (id, board_id, name, position, wip_limit)
task (id, board_id, column_id, title, description, priority,
      assignee, created_at, updated_at, due_at, blocked, tags)
task_dep (task_id, depends_on_task_id)
claim (id, task_id, worker_id, profile, claimed_at, released_at, stale_threshold_minutes)
comment (id, task_id, author, body, created_at)
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Dispatcher exports
from isaac.kanban.dispatcher import (
    DispatchConfig,
    Dispatcher,
    HTTPGateway,
    LoggingGateway,
    TaskGateway,
    WorkerProfile,
    WorkerRegistry,
    get_dispatcher,
    get_worker_registry,
    run_dispatcher_once,
    start_dispatcher,
    stop_dispatcher,
)

# Dashboard exports
try:
    from fastapi import FastAPI

    from isaac.kanban.dashboard import app
except ImportError:
    app = None  # type: ignore[assignment]
dashboard_app: FastAPI | None = app  # type: ignore[assignment]

DB_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS board (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS column (
    id TEXT PRIMARY KEY,
    board_id TEXT NOT NULL REFERENCES board(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    wip_limit INTEGER DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS task (
    id TEXT PRIMARY KEY,
    board_id TEXT NOT NULL REFERENCES board(id) ON DELETE CASCADE,
    column_id TEXT NOT NULL REFERENCES column(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    assignee TEXT DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    due_at TEXT DEFAULT NULL,
    blocked INTEGER NOT NULL DEFAULT 0,
    tags TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS task_dep (
    task_id TEXT NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    depends_on_task_id TEXT NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    PRIMARY KEY (task_id, depends_on_task_id)
);

CREATE TABLE IF NOT EXISTS claim (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    worker_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    claimed_at TEXT NOT NULL DEFAULT (datetime('now')),
    released_at TEXT DEFAULT NULL,
    stale_threshold_minutes INTEGER NOT NULL DEFAULT 60
);

CREATE TABLE IF NOT EXISTS comment (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_task_board ON task(board_id);
CREATE INDEX IF NOT EXISTS idx_task_column ON task(column_id);
CREATE INDEX IF NOT EXISTS idx_task_assignee ON task(assignee);
CREATE INDEX IF NOT EXISTS idx_claim_task ON claim(task_id);
CREATE INDEX IF NOT EXISTS idx_claim_worker ON claim(worker_id);
CREATE INDEX IF NOT EXISTS idx_comment_task ON comment(task_id);
"""


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def gen_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(DB_SCHEMA)
        conn.commit()


# Board operations


@dataclass
class Board:
    id: str
    name: str
    created_at: str
    updated_at: str


def create_board(db_path: Path, name: str) -> Board:
    with connect(db_path) as conn:
        board_id = gen_id("b_")
        now = now_iso()
        conn.execute(
            "INSERT INTO board (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (board_id, name, now, now),
        )
        conn.commit()
        return Board(id=board_id, name=name, created_at=now, updated_at=now)


def get_board(db_path: Path, board_id_or_name: str) -> Board | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM board WHERE id = ? OR name = ?",
            (board_id_or_name, board_id_or_name),
        ).fetchone()
        if row:
            return Board(**row)
        return None


def list_boards(db_path: Path) -> list[Board]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM board ORDER BY updated_at DESC").fetchall()
        return [Board(**row) for row in rows]


def update_board_timestamp(db_path: Path, board_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE board SET updated_at = ? WHERE id = ?",
            (now_iso(), board_id),
        )
        conn.commit()


# Column operations


@dataclass
class Column:
    id: str
    board_id: str
    name: str
    position: int
    wip_limit: int | None


DEFAULT_COLUMNS = [
    ("Backlog", 0, None),
    ("Ready", 1, None),
    ("In Progress", 2, 5),
    ("Review", 3, None),
    ("Done", 4, None),
]


def create_default_columns(db_path: Path, board_id: str) -> list[Column]:
    with connect(db_path) as conn:
        columns = []
        for name, position, wip_limit in DEFAULT_COLUMNS:
            col_id = gen_id("c_")
            conn.execute(
                "INSERT INTO column (id, board_id, name, position, wip_limit) "
                "VALUES (?, ?, ?, ?, ?)",
                (col_id, board_id, name, position, wip_limit),
            )
            columns.append(
                Column(
                    id=col_id, board_id=board_id, name=name, position=position, wip_limit=wip_limit
                )
            )
        conn.commit()
        return columns


def get_columns(db_path: Path, board_id: str) -> list[Column]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM column WHERE board_id = ? ORDER BY position",
            (board_id,),
        ).fetchall()
        return [Column(**row) for row in rows]


def create_column(
    db_path: Path,
    board_id: str,
    name: str,
    position: int | None = None,
    wip_limit: int | None = None,
) -> Column:
    with connect(db_path) as conn:
        col_id = gen_id("c_")
        if position is None:
            row = conn.execute(
                "SELECT MAX(position) FROM column WHERE board_id = ?",
                (board_id,),
            ).fetchone()
            position = (row[0] or -1) + 1
        conn.execute(
            "INSERT INTO column (id, board_id, name, position, wip_limit) VALUES (?, ?, ?, ?, ?)",
            (col_id, board_id, name, position, wip_limit),
        )
        conn.commit()
        return Column(
            id=col_id, board_id=board_id, name=name, position=position, wip_limit=wip_limit
        )


# Task operations


@dataclass
class Task:
    id: str
    board_id: str
    column_id: str
    title: str
    description: str
    priority: int
    assignee: str | None
    created_at: str
    updated_at: str
    due_at: str | None
    blocked: int
    tags: str

    @property
    def is_blocked(self) -> bool:
        return self.blocked == 1

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in self.tags.split(",") if t.strip()]


def create_task(
    db_path: Path,
    board_id: str,
    title: str,
    description: str = "",
    priority: int = 0,
    assignee: str | None = None,
    due_at: str | None = None,
    tags: str = "",
) -> Task:
    with connect(db_path) as conn:
        # Find the first column (Backlog)
        col_row = conn.execute(
            "SELECT id FROM column WHERE board_id = ? ORDER BY position LIMIT 1",
            (board_id,),
        ).fetchone()
        if not col_row:
            raise ValueError("Board has no columns")
        column_id = col_row[0]

        task_id = gen_id("t_")
        now = now_iso()
        conn.execute(
            (
                "INSERT INTO task (id, board_id, column_id, title, description, "
                "priority, assignee, created_at, updated_at, due_at, blocked, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                task_id,
                board_id,
                column_id,
                title,
                description,
                priority,
                assignee,
                now,
                now,
                due_at,
                0,
                tags,
            ),
        )
        # Update board timestamp in the same transaction
        conn.execute(
            "UPDATE board SET updated_at = ? WHERE id = ?",
            (now, board_id),
        )
        conn.commit()
        return Task(
            id=task_id,
            board_id=board_id,
            column_id=column_id,
            title=title,
            description=description,
            priority=priority,
            assignee=assignee,
            created_at=now,
            updated_at=now,
            due_at=due_at,
            blocked=0,
            tags=tags,
        )


def get_task(db_path: Path, task_id: str) -> Task | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM task WHERE id = ?", (task_id,)).fetchone()
        if row:
            return Task(**row)
        return None


def list_tasks(db_path: Path, board_id: str, column_id: str | None = None) -> list[Task]:
    with connect(db_path) as conn:
        if column_id:
            rows = conn.execute(
                "SELECT * FROM task WHERE board_id = ? AND column_id = ? ORDER BY priority DESC, created_at",  # noqa: E501
                (board_id, column_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM task WHERE board_id = ? ORDER BY column_id, priority DESC, created_at",  # noqa: E501
                (board_id,),
            ).fetchall()
        return [Task(**row) for row in rows]


def get_tasks(db_path: Path, board_id: str, column_id: str | None = None) -> list[Task]:
    """Alias for list_tasks for backward compatibility."""
    return list_tasks(db_path, board_id, column_id)


def move_task(db_path: Path, task_id: str, new_column_id: str) -> Task:
    with connect(db_path) as conn:
        now = now_iso()
        conn.execute(
            "UPDATE task SET column_id = ?, updated_at = ? WHERE id = ?",
            (new_column_id, now, task_id),
        )
        conn.execute(
            "UPDATE board SET updated_at = ? WHERE id = (SELECT board_id FROM task WHERE id = ?)",
            (now, task_id),
        )
        conn.commit()
    return get_task(db_path, task_id)  # type: ignore[return-value]


def update_task(
    db_path: Path,
    task_id: str,
    title: str | None = None,
    description: str | None = None,
    priority: int | None = None,
    assignee: str | None = None,
    due_at: str | None = None,
    blocked: int | None = None,
    tags: str | None = None,
) -> Task:
    with connect(db_path) as conn:
        now = now_iso()
        fields: list[str] = []
        params: list[str | int] = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if description is not None:
            fields.append("description = ?")
            params.append(description)
        if priority is not None:
            fields.append("priority = ?")
            params.append(priority)
        if assignee is not None:
            fields.append("assignee = ?")
            params.append(assignee)
        if due_at is not None:
            fields.append("due_at = ?")
            params.append(due_at)
        if blocked is not None:
            fields.append("blocked = ?")
            params.append(blocked)
        if tags is not None:
            fields.append("tags = ?")
            params.append(tags)
        if fields:
            fields.append("updated_at = ?")
            params.append(now)
            params.append(task_id)
            conn.execute(f"UPDATE task SET {', '.join(fields)} WHERE id = ?", params)
            conn.execute(
                "UPDATE board SET updated_at = ? WHERE id = (SELECT board_id FROM task WHERE id = ?)",  # noqa: E501
                (now, task_id),
            )
            conn.commit()
    return get_task(db_path, task_id)  # type: ignore[return-value]


def add_task_dep(db_path: Path, task_id: str, depends_on_task_id: str) -> None:
    if task_id == depends_on_task_id:
        raise ValueError("Task cannot depend on itself")
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO task_dep (task_id, depends_on_task_id) VALUES (?, ?)",
            (task_id, depends_on_task_id),
        )
        conn.commit()


def remove_task_dep(db_path: Path, task_id: str, depends_on_task_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "DELETE FROM task_dep WHERE task_id = ? AND depends_on_task_id = ?",
            (task_id, depends_on_task_id),
        )
        conn.commit()


def get_task_deps(db_path: Path, task_id: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT depends_on_task_id FROM task_dep WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        return [row[0] for row in rows]


def get_dependents(db_path: Path, task_id: str) -> list[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT task_id FROM task_dep WHERE depends_on_task_id = ?",
            (task_id,),
        ).fetchall()
        return [row[0] for row in rows]


# Claim operations (for dispatcher)


@dataclass
class Claim:
    id: str
    task_id: str
    worker_id: str
    profile: str
    claimed_at: str
    released_at: str | None
    stale_threshold_minutes: int


def claim_task(
    db_path: Path, task_id: str, worker_id: str, profile: str, stale_threshold_minutes: int = 60
) -> Claim:
    with connect(db_path) as conn:
        # Check if task is already claimed and not stale
        existing = conn.execute(
            "SELECT * FROM claim WHERE task_id = ? AND released_at IS NULL",
            (task_id,),
        ).fetchone()
        if existing:
            claim = Claim(**existing)
            if not is_claim_stale(claim):
                raise ValueError(f"Task {task_id} already claimed by {claim.worker_id}")

        # Create new claim
        claim_id = gen_id("cl_")
        now = now_iso()
        conn.execute(
            (
                "INSERT INTO claim (id, task_id, worker_id, profile, "
                "claimed_at, stale_threshold_minutes) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            ),
        )
        return Claim(
            id=claim_id,
            task_id=task_id,
            worker_id=worker_id,
            profile=profile,
            claimed_at=now,
            released_at=None,
            stale_threshold_minutes=stale_threshold_minutes,
        )


def release_claim(db_path: Path, claim_id: str) -> bool:
    with connect(db_path) as conn:
        now = now_iso()
        cur = conn.execute(
            "UPDATE claim SET released_at = ? WHERE id = ? AND released_at IS NULL",
            (now, claim_id),
        )
        conn.commit()
        return cur.rowcount > 0


def release_task_claims(db_path: Path, task_id: str) -> int:
    with connect(db_path) as conn:
        now = now_iso()
        cur = conn.execute(
            "UPDATE claim SET released_at = ? WHERE task_id = ? AND released_at IS NULL",
            (now, task_id),
        )
        conn.commit()
        return cur.rowcount


def get_active_claim(db_path: Path, task_id: str) -> Claim | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM claim WHERE task_id = ? AND released_at IS NULL",
            (task_id,),
        ).fetchone()
        if row:
            return Claim(**row)
        return None


def is_claim_stale(claim: Claim) -> bool:
    claimed = datetime.fromisoformat(claim.claimed_at)
    threshold = claimed + timedelta(minutes=claim.stale_threshold_minutes)
    return datetime.utcnow() > threshold


def reclaim_stale_claims(db_path: Path) -> list[Claim]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM claim WHERE released_at IS NULL").fetchall()
        reclaimed = []
        for row in rows:
            claim = Claim(**row)
            if is_claim_stale(claim):
                release_claim(db_path, claim.id)
                reclaimed.append(claim)
        return reclaimed


# Comment operations


@dataclass
class Comment:
    id: str
    task_id: str
    author: str
    body: str
    created_at: str


def add_comment(db_path: Path, task_id: str, author: str, body: str) -> Comment:
    with connect(db_path) as conn:
        comment_id = gen_id("cm_")
        now = now_iso()
        conn.execute(
            "INSERT INTO comment (id, task_id, author, body, created_at) VALUES (?, ?, ?, ?, ?)",
            (comment_id, task_id, author, body, now),
        )
        conn.commit()
        return Comment(id=comment_id, task_id=task_id, author=author, body=body, created_at=now)


def get_comments(db_path: Path, task_id: str) -> list[Comment]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM comment WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
        return [Comment(**row) for row in rows]


# Statistics


@dataclass
class BoardStats:
    board_id: str
    board_name: str
    total_tasks: int
    by_column: dict[str, int]
    by_priority: dict[int, int]
    blocked_count: int
    claimed_count: int
    stale_claims: int


def get_board_stats(db_path: Path, board_id: str) -> BoardStats:
    with connect(db_path) as conn:
        board = conn.execute("SELECT name FROM board WHERE id = ?", (board_id,)).fetchone()
        if not board:
            raise ValueError(f"Board {board_id} not found")
        board_name = board[0]

        total = conn.execute(
            "SELECT COUNT(*) FROM task WHERE board_id = ?", (board_id,)
        ).fetchone()[0]

        col_rows = conn.execute(
            """SELECT c.name, COUNT(t.id)
               FROM column c LEFT JOIN task t ON t.column_id = c.id
               WHERE c.board_id = ?
               GROUP BY c.name, c.position ORDER BY c.position""",
            (board_id,),
        ).fetchall()
        by_column = {name: count for name, count in col_rows}

        pri_rows = conn.execute(
            "SELECT priority, COUNT(*) FROM task WHERE board_id = ? GROUP BY priority",
            (board_id,),
        ).fetchall()
        by_priority = {p: c for p, c in pri_rows}

        blocked = conn.execute(
            "SELECT COUNT(*) FROM task WHERE board_id = ? AND blocked = 1",
            (board_id,),
        ).fetchone()[0]

        claimed = conn.execute(
            """SELECT COUNT(DISTINCT t.id) FROM task t
               JOIN claim c ON c.task_id = t.id
               WHERE t.board_id = ? AND c.released_at IS NULL""",
            (board_id,),
        ).fetchone()[0]

        # Count stale claims
        claim_rows = conn.execute(
            """SELECT c.claimed_at, c.stale_threshold_minutes
               FROM claim c JOIN task t ON t.id = c.task_id
               WHERE t.board_id = ? AND c.released_at IS NULL""",
            (board_id,),
        ).fetchall()
        stale = sum(
            1
            for claimed_at, threshold in claim_rows
            if is_claim_stale(
                Claim(
                    id="",
                    task_id="",
                    worker_id="",
                    profile="",
                    claimed_at=claimed_at,
                    released_at=None,
                    stale_threshold_minutes=threshold,
                )
            )
        )

        return BoardStats(
            board_id=board_id,
            board_name=board_name,
            total_tasks=total,
            by_column=by_column,
            by_priority=by_priority,
            blocked_count=blocked,
            claimed_count=claimed,
            stale_claims=stale,
        )


try:
    from isaac.kanban.dashboard import app
except ImportError:
    app = None  # type: ignore[assignment]

__all__ = [
    "Board",
    "BoardStats",
    "Claim",
    "Column",
    "Comment",
    "DispatchConfig",
    "Dispatcher",
    "HTTPGateway",
    "LoggingGateway",
    "Task",
    "TaskGateway",
    "WorkerProfile",
    "WorkerRegistry",
    "add_comment",
    "add_task_dep",
    "claim_task",
    "create_board",
    "create_column",
    "create_task",
    "get_active_claim",
    "get_board",
    "get_board_stats",
    "get_columns",
    "get_comments",
    "get_dependents",
    "get_dispatcher",
    "get_task",
    "get_task_deps",
    "get_worker_registry",
    "init_db",
    "init_kanban_board",
    "is_claim_stale",
    "list_boards",
    "list_tasks",
    "move_task",
    "reclaim_stale_claims",
    "release_claim",
    "release_task_claims",
    "remove_task_dep",
    "run_dispatcher_once",
    "start_dispatcher",
    "stop_dispatcher",
    "update_task",
]

if dashboard_app:
    __all__.append("dashboard_app")


# High-level board initialization


def init_kanban_board(db_path: Path, name: str) -> tuple[Board, list[Column]]:
    """Create a new board with default columns."""
    board = create_board(db_path, name)
    columns = create_default_columns(db_path, board.id)
    return board, columns

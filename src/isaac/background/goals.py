"""Persistent Goals — long-lived objectives stored in SQLite.

Each goal has free-form text, an optional due date, a status
(``active`` / ``paused`` / ``completed``) and monotonically-increasing
progress notes.  Goals are injected into the agent context on wake by
:mod:`isaac.agents.agent_loop` (see ``due_goals_context``).

Storage: ``<isaac_home>/goals.db`` (SQLite).
CLI: ``isaac goal set|list|show|complete|pause`` (see ``isaac.cli``).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_ACTIVE = "active"
_PAUSED = "paused"
_COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    """A single persistent goal."""

    id: str
    text: str
    status: str = _ACTIVE
    due: str = ""  # ISO datetime or ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    progress: int = 0  # count of progress marks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"


def _db_path(isaac_home: Path | None = None) -> Path:
    return (isaac_home if isaac_home is not None else _isaac_home()) / "goals.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_goal(row: tuple[Any, ...]) -> Goal:
    return Goal(
        id=row[0],
        text=row[1],
        status=row[2],
        due=row[3] or "",
        notes=row[4] or "",
        created_at=row[5],
        updated_at=row[6],
        completed_at=row[7] or "",
        progress=int(row[8]),
    )


_COLUMNS = "id, text, status, due, notes, created_at, updated_at, completed_at, progress"


def _connect(isaac_home: Path | None = None) -> sqlite3.Connection:
    path = _db_path(isaac_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            due TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT DEFAULT '',
            progress INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    con.commit()
    return con


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_goal(
    text: str,
    *,
    due: str = "",
    notes: str = "",
    isaac_home: Path | None = None,
) -> Goal:
    """Create a new active goal and persist it."""
    now = _now()
    goal = Goal(
        id=uuid4().hex[:12],
        text=text.strip(),
        status=_ACTIVE,
        due=due,
        notes=notes,
        created_at=now,
        updated_at=now,
        completed_at="",
        progress=0,
    )
    with _connect(isaac_home) as con:
        con.execute(
            f"INSERT INTO goals ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal.id,
                goal.text,
                goal.status,
                goal.due,
                goal.notes,
                goal.created_at,
                goal.updated_at,
                goal.completed_at,
                goal.progress,
            ),
        )
    logger.info("Goal added: %s (%s)", goal.id, goal.text[:60])
    return goal


def _get_goal(goal_id: str, isaac_home: Path | None = None) -> Goal | None:
    with _connect(isaac_home) as con:
        cursor = con.execute(f"SELECT {_COLUMNS} FROM goals WHERE id = ?", (goal_id,))
        row = cursor.fetchone()
        if row is None:
            rows = con.execute(
                f"SELECT {_COLUMNS} FROM goals WHERE id LIKE ?", (f"{goal_id}%",)
            ).fetchall()
            if len(rows) == 1:
                row = rows[0]
    return _row_to_goal(row) if row else None


def get_goal(goal_id: str, *, isaac_home: Path | None = None) -> Goal | None:
    """Fetch a goal by full id (supports unique-prefix shorthand)."""
    return _get_goal(goal_id, isaac_home)


def list_goals(
    *, status: str | None = None, isaac_home: Path | None = None
) -> list[Goal]:
    """List goals, optionally filtered by status."""
    with _connect(isaac_home) as con:
        if status:
            rows = con.execute(
                f"SELECT {_COLUMNS} FROM goals WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        else:
            rows = con.execute(
                f"SELECT {_COLUMNS} FROM goals ORDER BY created_at"
            ).fetchall()
    return [_row_to_goal(r) for r in rows]


def complete_goal(goal_id: str, *, isaac_home: Path | None = None) -> bool:
    """Mark a goal completed. Returns True if found."""
    goal = _get_goal(goal_id, isaac_home)
    if goal is None:
        return False
    now = _now()
    with _connect(isaac_home) as con:
        con.execute(
            "UPDATE goals SET status = ?, completed_at = ?, updated_at = ? WHERE id = ?",
            (_COMPLETED, now, now, goal.id),
        )
    logger.info("Goal completed: %s", goal.id)
    return True


def pause_goal(goal_id: str, *, isaac_home: Path | None = None) -> bool:
    """Pause a goal (temporarily exclude it from the agent context)."""
    goal = _get_goal(goal_id, isaac_home)
    if goal is None:
        return False
    with _connect(isaac_home) as con:
        con.execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
            (_PAUSED, _now(), goal.id),
        )
    logger.info("Goal paused: %s", goal.id)
    return True


def resume_goal(goal_id: str, *, isaac_home: Path | None = None) -> bool:
    """Resume a paused goal."""
    goal = _get_goal(goal_id, isaac_home)
    if goal is None:
        return False
    with _connect(isaac_home) as con:
        con.execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE id = ?",
            (_ACTIVE, _now(), goal.id),
        )
    logger.info("Goal resumed: %s", goal.id)
    return True


def mark_goal_progress(
    goal_id: str, note: str = "", *, isaac_home: Path | None = None
) -> bool:
    """Increment the progress counter (and optionally append a note)."""
    goal = _get_goal(goal_id, isaac_home)
    if goal is None:
        return False
    notes = goal.notes
    if note:
        notes = (notes + "\n" if notes else "") + f"[{_now()}] {note}"
    with _connect(isaac_home) as con:
        con.execute(
            "UPDATE goals SET progress = ?, notes = ?, updated_at = ? WHERE id = ?",
            (goal.progress + 1, notes, _now(), goal.id),
        )
    logger.info("Goal progress marked: %s (%d)", goal.id, goal.progress + 1)
    return True


def due_goals(*, isaac_home: Path | None = None) -> list[Goal]:
    """Return active goals whose due date has passed (or which have none)."""
    now = datetime.now(UTC)
    due: list[Goal] = []
    for goal in list_goals(status=_ACTIVE, isaac_home=isaac_home):
        if not goal.due:
            due.append(goal)
            continue
        try:
            when = datetime.fromisoformat(goal.due)
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
        except ValueError:
            due.append(goal)
            continue
        if when <= now:
            due.append(goal)
    return due


def due_goals_context(*, isaac_home: Path | None = None) -> str:
    """Render due active goals as a context block for agent injection.

    Returns an empty string when there is nothing to inject.
    """
    goals = due_goals(isaac_home=isaac_home)
    if not goals:
        return ""
    lines = ["Active goals (persist across sessions):"]
    for g in goals:
        due = f" — due {g.due}" if g.due else ""
        prog = f" (progress marks: {g.progress})" if g.progress else ""
        lines.append(f"- [{g.id}] {g.text}{due}{prog}")
    return "\n".join(lines)

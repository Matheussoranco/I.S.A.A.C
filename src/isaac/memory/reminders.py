from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class Reminder:
    id: str
    text: str
    created_at: str
    due_at: str = ""
    done: bool = False
    source: str = "user"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReminderNotification:
    reminder: Reminder
    token: str


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _datetime(value: str) -> datetime:
    value = _nonempty(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc


def _now(value: str) -> datetime:
    return datetime.now(UTC) if value == "" else _datetime(value)


def _store_path(isaac_home: Path | None = None) -> Path:
    if isaac_home is None:
        from isaac.config.settings import get_settings

        home_dir = get_settings().isaac_home
        return Path(home_dir) / "reminders.sqlite3"
    return Path(isaac_home) / "reminders.sqlite3"


def _insert(conn: sqlite3.Connection, reminder: Reminder) -> None:
    due_ts = _datetime(reminder.due_at).timestamp() if reminder.due_at else None
    conn.execute(
        "INSERT INTO reminders (id, text, created_at, due_at, due_ts, done, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            reminder.id,
            reminder.text,
            reminder.created_at,
            reminder.due_at,
            due_ts,
            int(reminder.done),
            reminder.source,
        ),
    )


def _migrate(conn: sqlite3.Connection, path: Path) -> None:
    if conn.execute("SELECT 1 FROM metadata WHERE key = 'json_migrated'").fetchone():
        return
    legacy = path.with_name("reminders.json")
    if legacy.exists():
        items = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("legacy reminders.json must contain a list")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("done", False), bool):
                raise ValueError("invalid legacy reminder")
            due = item.get("due_at", "")
            _insert(
                conn,
                Reminder(
                    id=_nonempty(str(item.get("id") or ""), "id"),
                    text=_nonempty(str(item.get("text") or ""), "text"),
                    created_at=_datetime(str(item.get("created_at") or "")).isoformat(),
                    due_at=_datetime(str(due)).isoformat() if due != "" else "",
                    done=bool(item.get("done", False)),
                    source=_nonempty(str(item.get("source") or "user"), "source"),
                ),
            )
    conn.execute("INSERT INTO metadata (key) VALUES ('json_migrated')")


@contextmanager
def _transaction(isaac_home: Path | None) -> Iterator[sqlite3.Connection]:
    path = _store_path(isaac_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reminders ("
            "id TEXT PRIMARY KEY, text TEXT NOT NULL, created_at TEXT NOT NULL, "
            "due_at TEXT NOT NULL, due_ts REAL, done INTEGER NOT NULL, source TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS reminders_due ON reminders(done, due_ts)")
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS notifications ("
            "reminder_id TEXT NOT NULL REFERENCES reminders(id), channel TEXT NOT NULL, "
            "token TEXT NOT NULL, lease_until REAL NOT NULL, notified_at TEXT, "
            "attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT NOT NULL DEFAULT '', "
            "PRIMARY KEY (reminder_id, channel))"
        )
        _migrate(conn, path)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _reminder(row: sqlite3.Row) -> Reminder:
    return Reminder(
        id=str(row["id"]),
        text=str(row["text"]),
        created_at=str(row["created_at"]),
        due_at=str(row["due_at"]),
        done=bool(row["done"]),
        source=str(row["source"]),
    )


def add_reminder(text: str, due_at: str = "", *, isaac_home: Path | None = None) -> Reminder:
    reminder = Reminder(
        id=uuid.uuid4().hex,
        text=_nonempty(text, "text"),
        created_at=datetime.now(UTC).isoformat(),
        due_at=_datetime(due_at).isoformat() if due_at != "" else "",
    )
    with _transaction(isaac_home) as conn:
        _insert(conn, reminder)
    return reminder


def list_reminders(*, include_done: bool = False, isaac_home: Path | None = None) -> list[Reminder]:
    with _transaction(isaac_home) as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE (? OR done = 0) ORDER BY rowid", (include_done,)
        ).fetchall()
    return [_reminder(row) for row in rows]


def complete_reminder(reminder_id: str, *, isaac_home: Path | None = None) -> bool:
    reminder_id = _nonempty(reminder_id, "reminder_id")
    with _transaction(isaac_home) as conn:
        rows = conn.execute("SELECT id FROM reminders WHERE id = ?", (reminder_id,)).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT id FROM reminders WHERE substr(id, 1, ?) = ?",
                (len(reminder_id), reminder_id),
            ).fetchall()
        if len(rows) > 1:
            raise ValueError("ambiguous reminder id prefix")
        if not rows:
            return False
        conn.execute("UPDATE reminders SET done = 1 WHERE id = ?", (str(rows[0]["id"]),))
    return True


def due_reminders(*, isaac_home: Path | None = None, now: str = "") -> list[Reminder]:
    current = _now(now).timestamp()
    with _transaction(isaac_home) as conn:
        rows = conn.execute(
            "SELECT * FROM reminders WHERE done = 0 AND due_ts <= ? ORDER BY due_ts, rowid",
            (current,),
        ).fetchall()
    return [_reminder(row) for row in rows]


def claim_due_reminders(
    channel: str,
    *,
    isaac_home: Path | None = None,
    now: str = "",
    limit: int = 10,
    lease_seconds: float = 300,
) -> list[ReminderNotification]:
    channel = _nonempty(channel, "channel")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive and finite")
    current = _now(now).timestamp()
    claimed: list[ReminderNotification] = []
    with _transaction(isaac_home) as conn:
        rows = conn.execute(
            "SELECT r.* FROM reminders r LEFT JOIN notifications n "
            "ON n.reminder_id = r.id AND n.channel = ? "
            "WHERE r.done = 0 AND r.due_ts <= ? AND n.notified_at IS NULL "
            "AND (n.reminder_id IS NULL OR n.lease_until <= ?) "
            "ORDER BY r.due_ts, r.rowid LIMIT ?",
            (channel, current, current, limit),
        ).fetchall()
        for row in rows:
            token = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO notifications (reminder_id, channel, token, lease_until, attempts) "
                "VALUES (?, ?, ?, ?, 1) ON CONFLICT(reminder_id, channel) DO UPDATE SET "
                "token = excluded.token, lease_until = excluded.lease_until, "
                "attempts = attempts + 1",
                (str(row["id"]), channel, token, current + lease_seconds),
            )
            claimed.append(ReminderNotification(_reminder(row), token))
    return claimed


def finish_reminder_notification(
    reminder_id: str,
    channel: str,
    token: str,
    *,
    delivered: bool,
    error: str = "",
    isaac_home: Path | None = None,
    now: str = "",
) -> None:
    reminder_id = _nonempty(reminder_id, "reminder_id")
    channel = _nonempty(channel, "channel")
    token = _nonempty(token, "token")
    if not isinstance(delivered, bool):
        raise ValueError("delivered must be a boolean")
    current = _now(now)
    with _transaction(isaac_home) as conn:
        changed = conn.execute(
            "UPDATE notifications SET notified_at = ?, lease_until = 0, token = '', last_error = ? "
            "WHERE reminder_id = ? AND channel = ? AND token = ? AND notified_at IS NULL",
            (
                current.isoformat() if delivered else None,
                "" if delivered else str(error)[:1000],
                reminder_id,
                channel,
                token,
            ),
        ).rowcount
        if changed != 1:
            raise ValueError("notification claim is stale or unknown")


def parse_remind_args(text: str) -> tuple[str, str]:
    text = _nonempty(text, "text")
    if "@" not in text:
        return text, ""
    body, _, when = text.rpartition("@")
    body = _nonempty(body, "text")
    when = _nonempty(when, "due_at")
    if when.lower().startswith("in "):
        match = re.fullmatch(r"in\s+(\d+(?:\.\d+)?)\s*([hmd])", when.lower())
        if not match:
            raise ValueError("relative due time must be 'in <positive number>h|m|d'")
        multipliers = {"h": 3600, "m": 60, "d": 86400}
        seconds = float(match.group(1)) * multipliers[match.group(2)]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("relative due time must be positive and finite")
        try:
            due = datetime.now(UTC) + timedelta(seconds=seconds)
        except OverflowError as exc:
            raise ValueError("relative due time is out of range") from exc
    else:
        due = _datetime(when)
    return body, due.isoformat()

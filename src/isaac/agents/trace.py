"""Persistent run traces — every agent run leaves an inspectable record.

The ``AgentLoop`` already streams events (``iteration``, ``tool_call``,
``tool_result``, ``final``, ...) through its ``on_event`` hook; the
:class:`TraceStore` persists that stream to SQLite so a run can be inspected
after the fact (``isaac trace`` / ``isaac trace <run_id>``) — the
observability requirement of ROADMAP-1.0 ("per-run traces persisted").
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path

from isaac.security.redact import redact_secrets

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    stopped_reason TEXT NOT NULL DEFAULT '',
    iterations  INTEGER NOT NULL DEFAULT 0,
    output      TEXT NOT NULL DEFAULT '',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_latency_ms REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_events (
    run_id   TEXT NOT NULL REFERENCES agent_runs(run_id),
    seq      INTEGER NOT NULL,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS agent_runs_started_at_idx ON agent_runs(started_at);
"""

_MAX_FIELD = 8_000
_SAFE_EVENT_FIELDS = frozenset({"n", "success", "iterations"})


def default_trace_db() -> Path:
    from isaac.config.settings import get_settings

    return get_settings().isaac_home / "traces.db"


class TraceStore:
    """SQLite-backed store of agent runs and their event streams."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        include_content: bool = False,
        retention_days: int | None = 30,
    ) -> None:
        if retention_days is not None and retention_days < 1:
            raise ValueError("retention_days must be positive or None")
        self._path = Path(db_path) if db_path else default_trace_db()
        self._include_content = include_content
        self._retention_days = retention_days
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")}
            for name, declaration in (
                ("prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("completion_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("total_latency_ms", "REAL NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE agent_runs ADD COLUMN {name} {declaration}")
            if conn.execute("PRAGMA user_version").fetchone()[0] < 1:
                # Existing trace databases may contain unredacted private text.
                conn.execute(
                    "UPDATE agent_runs SET task='[content omitted]', output='[content omitted]'"
                )
                rows = conn.execute("SELECT run_id, seq, data_json FROM agent_events").fetchall()
                for row in rows:
                    try:
                        data = json.loads(row["data_json"])
                        safe = self._safe_event_data(data) if isinstance(data, dict) else {}
                    except (TypeError, ValueError):
                        safe = {}
                    conn.execute(
                        "UPDATE agent_events SET data_json=? WHERE run_id=? AND seq=?",
                        (json.dumps(safe), row["run_id"], row["seq"]),
                    )
                conn.execute("PRAGMA user_version=1")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------

    def start_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            if self._retention_days is not None:
                cutoff = time.time() - self._retention_days * 86_400
                conn.execute(
                    "DELETE FROM agent_events WHERE run_id IN "
                    "(SELECT run_id FROM agent_runs WHERE started_at < ?)",
                    (cutoff,),
                )
                conn.execute("DELETE FROM agent_runs WHERE started_at < ?", (cutoff,))
            conn.execute(
                "INSERT INTO agent_runs (run_id, task, started_at) VALUES (?,?,?)",
                (run_id, self._stored_text(task), time.time()),
            )
        return run_id

    def record_event(self, run_id: str, kind: str, data: dict) -> None:
        try:
            stored = data if self._include_content else self._safe_event_data(data)
            payload = redact_secrets(json.dumps(stored, ensure_ascii=False, default=str))[
                :_MAX_FIELD
            ]
        except Exception:
            payload = "{}"
        with self._connect() as conn:
            (seq,) = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO agent_events VALUES (?,?,?,?,?)",
                (run_id, seq, time.time(), kind, payload),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        stopped_reason: str,
        iterations: int,
        output: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_latency_ms: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agent_runs SET finished_at=?, stopped_reason=?, iterations=?, output=?, "
                "prompt_tokens=?, completion_tokens=?, total_latency_ms=? "
                "WHERE run_id=?",
                (
                    time.time(),
                    stopped_reason,
                    iterations,
                    self._stored_text(output),
                    prompt_tokens,
                    completion_tokens,
                    total_latency_ms,
                    run_id,
                ),
            )

    def _stored_text(self, value: str) -> str:
        return redact_secrets(value)[:_MAX_FIELD] if self._include_content else "[content omitted]"

    @staticmethod
    def _safe_event_data(data: dict) -> dict:
        safe: dict[str, str | int | float | bool] = {
            key: value
            for key, value in data.items()
            if key in _SAFE_EVENT_FIELDS and isinstance(value, (int, float, bool))
        }
        name = data.get("name")
        if isinstance(name, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", name):
            safe["name"] = name
        return safe

    # ------------------------------------------------------------------

    def recent_runs(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def run_events(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, ts, kind, data_json FROM agent_events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

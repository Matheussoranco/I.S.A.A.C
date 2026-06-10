"""Persistent run traces — every agent run leaves an inspectable record.

The ``AgentLoop`` already streams events (``iteration``, ``tool_call``,
``tool_result``, ``final``, ...) through its ``on_event`` hook; the
:class:`TraceStore` persists that stream to SQLite so a run can be inspected
after the fact (``isaac trace`` / ``isaac trace <run_id>``) — the
observability requirement of ROADMAP-1.0 ("per-run traces persisted").
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id      TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    stopped_reason TEXT NOT NULL DEFAULT '',
    iterations  INTEGER NOT NULL DEFAULT 0,
    output      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS agent_events (
    run_id   TEXT NOT NULL REFERENCES agent_runs(run_id),
    seq      INTEGER NOT NULL,
    ts       REAL NOT NULL,
    kind     TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, seq)
);
"""

_MAX_FIELD = 8_000


def default_trace_db() -> Path:
    from isaac.config.settings import get_settings

    return get_settings().isaac_home / "traces.db"


class TraceStore:
    """SQLite-backed store of agent runs and their event streams."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path) if db_path else default_trace_db()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------

    def start_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_runs (run_id, task, started_at) VALUES (?,?,?)",
                (run_id, task[:_MAX_FIELD], time.time()),
            )
        return run_id

    def record_event(self, run_id: str, kind: str, data: dict) -> None:
        try:
            payload = json.dumps(data, ensure_ascii=False, default=str)[:_MAX_FIELD]
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

    def finish_run(self, run_id: str, *, stopped_reason: str, iterations: int, output: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agent_runs SET finished_at=?, stopped_reason=?, iterations=?, output=? "
                "WHERE run_id=?",
                (time.time(), stopped_reason, iterations, output[:_MAX_FIELD], run_id),
            )

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

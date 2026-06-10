"""SQLite results store — every eval run is a durable, comparable record.

Schema:
    eval_runs          one row per ``isaac eval`` invocation
    eval_task_results  one row per task within a run

The row records everything needed to reproduce the number: suite name +
content hash, model + provider, runner kind, git revision, and timestamps.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      TEXT PRIMARY KEY,
    suite       TEXT NOT NULL,
    suite_hash  TEXT NOT NULL,
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    runner      TEXT NOT NULL,
    git_rev     TEXT NOT NULL DEFAULT '',
    started_at  REAL NOT NULL,
    finished_at REAL NOT NULL,
    total       INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    accuracy    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_task_results (
    run_id      TEXT NOT NULL REFERENCES eval_runs(run_id),
    task_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    stopped_reason TEXT NOT NULL DEFAULT '',
    duration_ms REAL NOT NULL DEFAULT 0,
    answer      TEXT NOT NULL DEFAULT '',
    checks_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (run_id, task_id)
);
"""

_MAX_STORED_ANSWER = 4_000


@dataclass
class RunRecord:
    """One persisted eval run (header row)."""

    run_id: str
    suite: str
    suite_hash: str
    model: str
    provider: str
    runner: str
    git_rev: str
    started_at: float
    finished_at: float
    total: int
    passed: int
    accuracy: float
    task_rows: list[dict] = field(default_factory=list)


class EvalStore:
    """Thin SQLite wrapper for eval results."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------

    def new_run_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def record_run(
        self,
        *,
        run_id: str,
        suite: str,
        suite_hash: str,
        model: str,
        provider: str,
        runner: str,
        git_rev: str,
        started_at: float,
        task_rows: list[dict],
    ) -> RunRecord:
        """Persist a finished run and its per-task rows."""
        total = len(task_rows)
        passed = sum(1 for r in task_rows if r["passed"])
        accuracy = passed / total if total else 0.0
        finished_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    suite,
                    suite_hash,
                    model,
                    provider,
                    runner,
                    git_rev,
                    started_at,
                    finished_at,
                    total,
                    passed,
                    accuracy,
                ),
            )
            conn.executemany(
                "INSERT INTO eval_task_results VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        run_id,
                        r["task_id"],
                        r.get("category", "general"),
                        1 if r["passed"] else 0,
                        r.get("stopped_reason", ""),
                        float(r.get("duration_ms", 0.0)),
                        str(r.get("answer", ""))[:_MAX_STORED_ANSWER],
                        json.dumps(r.get("checks", []), ensure_ascii=False),
                    )
                    for r in task_rows
                ],
            )
        return RunRecord(
            run_id=run_id,
            suite=suite,
            suite_hash=suite_hash,
            model=model,
            provider=provider,
            runner=runner,
            git_rev=git_rev,
            started_at=started_at,
            finished_at=finished_at,
            total=total,
            passed=passed,
            accuracy=accuracy,
            task_rows=task_rows,
        )

    def recent_runs(self, limit: int = 10) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [RunRecord(**dict(r)) for r in rows]

    def run_details(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_task_results WHERE run_id = ? ORDER BY task_id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

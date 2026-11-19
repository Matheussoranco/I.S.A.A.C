"""Performance tracker — SQLite-backed metrics store.

Records every node invocation and every skill execution so the
:pymod:`isaac.improvement.engine` can identify systematic failure modes,
slow nodes, and underperforming skills.

Schema
------
``node_runs``    — one row per cognitive-graph node execution.
``skill_runs``   — one row per skill invocation.
``prompt_runs``  — one row per prompt-variant invocation (for evolution).

All columns are append-only.  Aggregations are computed on demand.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS node_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    node         TEXT    NOT NULL,
    duration_ms  REAL    NOT NULL,
    success      INTEGER NOT NULL,
    iteration    INTEGER NOT NULL DEFAULT 0,
    session_id   TEXT    NOT NULL DEFAULT '',
    error        TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_node_runs_node    ON node_runs(node);
CREATE INDEX IF NOT EXISTS idx_node_runs_session ON node_runs(session_id);

CREATE TABLE IF NOT EXISTS skill_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    skill_name   TEXT    NOT NULL,
    duration_ms  REAL    NOT NULL,
    success      INTEGER NOT NULL,
    error        TEXT    NOT NULL DEFAULT '',
    task_context TEXT    NOT NULL DEFAULT '',
    task_type    TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_skill_runs_name ON skill_runs(skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_runs_type ON skill_runs(task_type);

CREATE TABLE IF NOT EXISTS prompt_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    prompt_id    TEXT    NOT NULL,
    variant      TEXT    NOT NULL,
    success      INTEGER NOT NULL,
    score        REAL    NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_prompt_runs_id ON prompt_runs(prompt_id);
"""


@dataclass
class NodeStats:
    node: str
    runs: int
    success_rate: float
    avg_duration_ms: float
    p95_duration_ms: float
    common_errors: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class SkillStats:
    skill_name: str
    runs: int
    success_rate: float
    avg_duration_ms: float
    last_seen_ts: float


class PerformanceTracker:
    """Append-only metrics store backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            from isaac.config.settings import get_settings

            db_path = get_settings().isaac_home / "performance-v2.db"
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- Recording ----------------------------------------------------------

    def record_node(
        self,
        node: str,
        duration_ms: float,
        success: bool | None,
        iteration: int = 0,
        session_id: str = "",
        error: str = "",
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO node_runs "
                "(ts, node, duration_ms, success, iteration, session_id, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    node,
                    duration_ms,
                    -1 if success is None else int(success),
                    iteration,
                    session_id,
                    error[:500],
                ),
            )
            self._conn.commit()

    def record_skill(
            self,
            skill_name: str,
            duration_ms: float,
            success: bool | None,
            error: str = "",
            task_context: str = "",
            task_type: str = "",
        ) -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO skill_runs "
                    "(ts, skill_name, duration_ms, success, error, task_context, task_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(),
                        skill_name,
                        duration_ms,
                        -1 if success is None else int(success),
                        error[:500],
                        task_context[:500],
                        task_type[:100],
                    ),
                )
                self._conn.commit()

    def record_prompt(
        self,
        prompt_id: str,
        variant: str,
        success: bool | None,
        score: float = 0.0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO prompt_runs (ts, prompt_id, variant, success, score) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), prompt_id, variant, -1 if success is None else int(success), score),
            )
            self._conn.commit()

    # -- Aggregations -------------------------------------------------------

    def node_stats(self, node: str | None = None, since_ts: float = 0.0) -> list[NodeStats]:
        """Return aggregated stats per node (optionally filtered)."""
        with self._lock:
            base_query = (
                "SELECT node, COUNT(*) as runs, "
                "AVG(CASE WHEN success >= 0 THEN success END) as sr, AVG(duration_ms) as avg_d "
                "FROM node_runs WHERE ts >= ? "
            )
            params: tuple[Any, ...] = (since_ts,)
            if node:
                base_query += "AND node = ? "
                params = (since_ts, node)
            base_query += "GROUP BY node"
            rows = self._conn.execute(base_query, params).fetchall()

            out: list[NodeStats] = []
            for r_node, runs, sr, avg_d in rows:
                # P95 — quantile via Python (SQLite has no native percentile)
                durations = [
                    row[0]
                    for row in self._conn.execute(
                        "SELECT duration_ms FROM node_runs WHERE node=? AND ts>=?",
                        (r_node, since_ts),
                    )
                ]
                p95 = _percentile(durations, 0.95) if durations else 0.0

                # Top 3 errors
                err_rows = self._conn.execute(
                    "SELECT error, COUNT(*) c FROM node_runs "
                    "WHERE node=? AND ts>=? AND success=0 AND error<>'' "
                    "GROUP BY error ORDER BY c DESC LIMIT 3",
                    (r_node, since_ts),
                ).fetchall()

                out.append(
                    NodeStats(
                        node=r_node,
                        runs=int(runs),
                        success_rate=float(sr or 0.0),
                        avg_duration_ms=float(avg_d or 0.0),
                        p95_duration_ms=p95,
                        common_errors=[(e, int(c)) for e, c in err_rows],
                    )
                )
            return out

    def skill_stats(self, since_ts: float = 0.0) -> list[SkillStats]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT skill_name, COUNT(*) runs, AVG(success) sr, "
                "AVG(duration_ms) avg_d, MAX(ts) last_ts "
                "FROM skill_runs WHERE ts >= ? AND success >= 0 "
                "GROUP BY skill_name ORDER BY runs DESC",
                (since_ts,),
            ).fetchall()
            return [
                SkillStats(
                    skill_name=name,
                    runs=int(runs),
                    success_rate=float(sr or 0.0),
                    avg_duration_ms=float(avg_d or 0.0),
                    last_seen_ts=float(last_ts or 0.0),
                )
                for name, runs, sr, avg_d, last_ts in rows
            ]

    def skill_stats_by_type(
        self, since_ts: float = 0.0, alpha: float = 1.0, beta: float = 1.0
    ) -> dict[str, dict[str, float]]:
        """Return hierarchical Bayesian win-rates per (skill, task_type).

        Uses a Beta(alpha, beta) prior for each skill's win-rate per task type.
        The posterior is Beta(alpha + wins, beta + losses). Returns the posterior
        mean for each (skill, task_type) pair.

        Args:
            since_ts: Only consider runs after this timestamp.
            alpha: Prior pseudo-count for successes (default 1.0 = uniform).
            beta: Prior pseudo-count for failures (default 1.0 = uniform).

        Returns:
            Dict mapping skill_name -> {task_type: posterior_mean_win_rate}.
            Task types with zero runs are omitted.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT skill_name, task_type, COUNT(*) runs, SUM(success) wins "
                "FROM skill_runs WHERE ts >= ? AND success >= 0 "
                "GROUP BY skill_name, task_type",
                (since_ts,),
            ).fetchall()

        result: dict[str, dict[str, float]] = {}
        for skill_name, task_type, runs, wins in rows:
            if runs == 0 or not task_type:
                continue
            wins = int(wins or 0)
            # Posterior mean of Beta(alpha + wins, beta + losses)
            post_mean = (alpha + wins) / (alpha + beta + runs)
            if skill_name not in result:
                result[skill_name] = {}
            result[skill_name][task_type] = post_mean

        return result

    def task_type_summary(self, since_ts: float = 0.0) -> dict[str, dict]:
        """Return summary statistics per task type across all skills."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT task_type, COUNT(*) runs, AVG(success) sr, "
                "COUNT(DISTINCT skill_name) skills "
                "FROM skill_runs WHERE ts >= ? AND success >= 0 AND task_type <> '' "
                "GROUP BY task_type ORDER BY runs DESC",
                (since_ts,),
            ).fetchall()

        return {
            task_type: {"runs": runs, "success_rate": float(sr or 0.0), "skills": skills}
            for task_type, runs, sr, skills in rows
        }

    def prompt_leaderboard(self, prompt_id: str) -> list[tuple[str, float, int]]:
        """Return ``[(variant, score, runs), ...]`` for a prompt."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT variant, AVG(score) score, COUNT(*) runs "
                "FROM prompt_runs WHERE prompt_id=? GROUP BY variant "
                "ORDER BY score DESC",
                (prompt_id,),
            ).fetchall()
            return [(v, float(s or 0.0), int(r)) for v, s, r in rows]

    # -- Maintenance --------------------------------------------------------

    def prune(self, older_than_days: int = 90) -> int:
        cutoff = time.time() - older_than_days * 86400
        with self._lock:
            c1 = self._conn.execute("DELETE FROM node_runs WHERE ts < ?", (cutoff,)).rowcount
            c2 = self._conn.execute("DELETE FROM skill_runs WHERE ts < ?", (cutoff,)).rowcount
            c3 = self._conn.execute("DELETE FROM prompt_runs WHERE ts < ?", (cutoff,)).rowcount
            self._conn.commit()
            return int(c1 + c2 + c3)

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._conn.close()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = round((len(s) - 1) * q)
    return float(s[max(0, min(k, len(s) - 1))])


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracker: PerformanceTracker | None = None


def get_tracker() -> PerformanceTracker:
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker


def reset_tracker() -> None:
    """Reset the singleton (used in tests)."""
    global _tracker
    if _tracker is not None:
        _tracker.close()
    _tracker = None

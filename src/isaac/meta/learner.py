"""Self-Improvement Meta-Learner — tracks task outcomes and drives adaptive strategy selection.

Stores every task execution in SQLite with:
  - task description (embedding-friendly)
  - strategy used (analogy/beam/object/llm/refinement/direct)
  - success/failure + error type
  - duration and token cost
  - iteration count

Provides:
  - get_best_strategy(task_description) → ranked strategy list
  - get_stats(task_type) → aggregated metrics
  - record(…) → write outcome after each graph run
  - analyse_failures() → cluster common error patterns
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".isaac" / "meta_learner.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    task_desc   TEXT NOT NULL,
    task_type   TEXT NOT NULL DEFAULT 'general',
    strategy    TEXT NOT NULL DEFAULT 'unknown',
    success     INTEGER NOT NULL,
    error_type  TEXT,
    error_msg   TEXT,
    iterations  INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    session_id  TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_task_type ON task_outcomes(task_type);
CREATE INDEX IF NOT EXISTS idx_strategy  ON task_outcomes(strategy);
CREATE INDEX IF NOT EXISTS idx_success   ON task_outcomes(success);

CREATE TABLE IF NOT EXISTS strategy_scores (
    task_type   TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    wins        INTEGER DEFAULT 0,
    losses      INTEGER DEFAULT 0,
    avg_ms      REAL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (task_type, strategy)
);
"""


class MetaLearner:
    """SQLite-backed task outcome tracker with adaptive strategy ranking."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        task_desc: str,
        task_type: str = "general",
        strategy: str = "unknown",
        success: bool,
        error_type: str = "",
        error_msg: str = "",
        iterations: int = 0,
        duration_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        session_id: str = "",
    ) -> None:
        """Persist a single task outcome."""
        ts = datetime.utcnow().isoformat()
        self._conn.execute(
            """INSERT INTO task_outcomes
               (ts, task_desc, task_type, strategy, success, error_type, error_msg,
                iterations, duration_ms, input_tokens, output_tokens, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                task_desc[:500],
                task_type,
                strategy,
                int(success),
                error_type,
                error_msg[:500],
                iterations,
                duration_ms,
                input_tokens,
                output_tokens,
                session_id,
            ),
        )
        self._conn.commit()
        self._update_scores(task_type, strategy, success, duration_ms)

    def _update_scores(
        self, task_type: str, strategy: str, success: bool, duration_ms: float
    ) -> None:
        row = self._conn.execute(
            "SELECT wins, losses, avg_ms FROM strategy_scores WHERE task_type=? AND strategy=?",
            (task_type, strategy),
        ).fetchone()

        ts = datetime.utcnow().isoformat()
        if row is None:
            wins = 1 if success else 0
            losses = 0 if success else 1
            avg_ms = duration_ms
            self._conn.execute(
                "INSERT INTO strategy_scores VALUES (?,?,?,?,?,?)",
                (task_type, strategy, wins, losses, avg_ms, ts),
            )
        else:
            wins = row["wins"] + (1 if success else 0)
            losses = row["losses"] + (0 if success else 1)
            total = wins + losses
            avg_ms = (row["avg_ms"] * (total - 1) + duration_ms) / total
            self._conn.execute(
                "UPDATE strategy_scores "
                "SET wins=?, losses=?, avg_ms=?, updated_at=? "
                "WHERE task_type=? AND strategy=?",
                (wins, losses, avg_ms, ts, task_type, strategy),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_best_strategy(self, task_type: str = "general") -> list[dict[str, Any]]:
        """Return strategies ranked by success rate (win-rate then speed)."""
        rows = self._conn.execute(
            """SELECT strategy, wins, losses,
                      CAST(wins AS REAL)/(wins+losses+1) AS win_rate,
                      avg_ms
               FROM strategy_scores
               WHERE task_type=?
               ORDER BY win_rate DESC, avg_ms ASC""",
            (task_type,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, task_type: str | None = None) -> dict[str, Any]:
        """Return aggregated performance statistics."""
        where = "WHERE task_type=?" if task_type else ""
        params: tuple = (task_type,) if task_type else ()

        rows = self._conn.execute(
            f"SELECT * FROM task_outcomes {where} ORDER BY ts DESC LIMIT 500", params
        ).fetchall()

        total = len(rows)
        successes = sum(1 for r in rows if r["success"])
        avg_ms = sum(r["duration_ms"] for r in rows) / max(total, 1)

        error_counts: dict[str, int] = {}
        for r in rows:
            if r["error_type"]:
                error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1

        strategy_stats: dict[str, Any] = {}
        for r in rows:
            s = r["strategy"]
            if s not in strategy_stats:
                strategy_stats[s] = {"total": 0, "success": 0}
            strategy_stats[s]["total"] += 1
            if r["success"]:
                strategy_stats[s]["success"] += 1

        best = self.get_best_strategy(task_type or "general")

        return {
            "task_type": task_type or "all",
            "total_tasks": total,
            "success_rate": round(successes / max(total, 1), 3),
            "avg_duration_ms": round(avg_ms, 1),
            "top_errors": sorted(error_counts.items(), key=lambda x: -x[1])[:5],
            "strategy_breakdown": strategy_stats,
            "best_strategies": best[:5],
        }

    def analyse_failures(self, limit: int = 100) -> dict[str, Any]:
        """Cluster recent failures by error type and return patterns."""
        rows = self._conn.execute(
            "SELECT error_type, error_msg, strategy, task_type FROM task_outcomes "
            "WHERE success=0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()

        patterns: dict[str, dict[str, Any]] = {}
        for r in rows:
            key = r["error_type"] or "unknown"
            if key not in patterns:
                patterns[key] = {
                    "count": 0,
                    "strategies": set(),
                    "task_types": set(),
                    "samples": [],
                }
            patterns[key]["count"] += 1
            patterns[key]["strategies"].add(r["strategy"])
            patterns[key]["task_types"].add(r["task_type"])
            if len(patterns[key]["samples"]) < 3:
                patterns[key]["samples"].append(r["error_msg"][:200] if r["error_msg"] else "")

        # Convert sets to lists for JSON serialisation
        for p in patterns.values():
            p["strategies"] = list(p["strategies"])
            p["task_types"] = list(p["task_types"])

        return {
            "failure_count": len(rows),
            "patterns": patterns,
        }

    def close(self) -> None:
        self._conn.close()


# Module-level singleton
_learner: MetaLearner | None = None


def _configured_db_path() -> Path | None:
    """Return ``ISAAC_META_LEARNER_DB_PATH`` when set, else ``None``.

    The setting existed since 0.4.0 but was never consulted — every process
    opened the hardcoded default.  Honouring it is what lets the ablation
    harness give each arm its own isolated history.
    """
    try:
        from isaac.config.settings import get_settings

        configured = get_settings().meta_learner_db_path
    except Exception:  # pragma: no cover - defensive
        return None
    return Path(configured) if configured else None


def get_learner() -> MetaLearner:
    """Return the process-wide :class:`MetaLearner` (created on first use)."""
    global _learner
    if _learner is None:
        _learner = MetaLearner(_configured_db_path())
    return _learner


def reset_learner() -> None:
    """Close and drop the cached learner so the next call re-reads settings."""
    global _learner
    if _learner is not None:
        try:
            _learner.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("MetaLearner close failed during reset", exc_info=True)
    _learner = None

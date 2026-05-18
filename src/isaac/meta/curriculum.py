"""Self-play curriculum — auto-generates training tasks from past failures.

Goal: turn the MetaLearner's failure log into actionable practice tasks,
drilled offline so the agent improves without user intervention.

Two generation strategies:

1. **Mutation** — take a failed task and apply small mutations
   (rephrase, change parameters, add a constraint). The mutated task is
   stored as a *practice task*. The agent attempts it; success/failure is
   recorded back in the MetaLearner.

2. **Synthesis** — ask the language expert to invent a slightly easier
   task that exercises the same skill the failure required. Useful when
   the failure was due to missing knowledge rather than parameter choice.

Practice tasks live in ``~/.isaac/curriculum.db`` and the heartbeat
scheduler can run a fixed number per cycle so the agent improves
continuously when idle.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".isaac" / "curriculum.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS practice_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  REAL NOT NULL,
    derived_from TEXT,
    task_type   TEXT NOT NULL DEFAULT 'general',
    description TEXT NOT NULL,
    difficulty  REAL DEFAULT 0.5,
    attempts    INTEGER DEFAULT 0,
    successes   INTEGER DEFAULT 0,
    last_attempt REAL,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_type ON practice_tasks(task_type);
"""


@dataclass
class PracticeTask:
    id: int
    description: str
    task_type: str = "general"
    difficulty: float = 0.5
    attempts: int = 0
    successes: int = 0
    derived_from: str = ""

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.attempts, 1)


class Curriculum:
    """Persistent curriculum store + generator."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_from_failures(self, *, limit: int = 5) -> list[PracticeTask]:
        """Walk recent failures and synthesise practice tasks for each."""
        try:
            from isaac.meta.learner import get_learner
        except Exception as exc:
            logger.info("MetaLearner unavailable: %s", exc)
            return []

        learner = get_learner()
        rows = learner._conn.execute(
            "SELECT task_desc, task_type, error_type, error_msg "
            "FROM task_outcomes WHERE success=0 ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        if not rows:
            return []

        new_tasks: list[PracticeTask] = []
        for row in rows:
            for variant in self._mutate(row["task_desc"]):
                new_tasks.append(
                    self._store(
                        description=variant,
                        derived_from=row["task_desc"][:200],
                        task_type=row["task_type"] or "general",
                        difficulty=0.4,
                    )
                )
            try:
                synth = self._synthesise(row["task_desc"], row["error_msg"] or "")
                if synth:
                    new_tasks.append(
                        self._store(
                            description=synth,
                            derived_from=row["task_desc"][:200],
                            task_type=row["task_type"] or "general",
                            difficulty=0.3,
                        )
                    )
            except Exception as exc:
                logger.debug("synth failed: %s", exc)
        return new_tasks

    @staticmethod
    def _mutate(task_desc: str) -> list[str]:
        """Apply simple textual mutations — rephrasings, parameter changes."""
        out: list[str] = []
        templates = [
            "Try a simpler version: {t}",
            "Solve this step by step: {t}",
            "Explain your reasoning, then solve: {t}",
        ]
        for tmpl in templates:
            out.append(tmpl.format(t=task_desc.strip()))
        return out

    @staticmethod
    def _synthesise(failed_task: str, error: str) -> str | None:
        """Ask the language expert to invent a related, slightly easier task."""
        try:
            from langchain_core.messages import HumanMessage

            from isaac.llm.provider import get_llm

            llm = get_llm("fast")
            prompt = (
                "I.S.A.A.C. recently failed at this task:\n"
                f"{failed_task}\n\n"
                f"Error: {error[:300]}\n\n"
                "Invent a *related but slightly easier* practice task that "
                "drills the same underlying skill. Respond with ONLY the new "
                "task statement — no preamble."
            )
            raw = str(llm.invoke([HumanMessage(content=prompt)]).content).strip()
            return raw[:500] if raw else None
        except Exception as exc:
            logger.debug("synthesise failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _store(
        self,
        *,
        description: str,
        derived_from: str = "",
        task_type: str = "general",
        difficulty: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> PracticeTask:
        cur = self._conn.execute(
            "INSERT INTO practice_tasks "
            "(created_at, derived_from, task_type, description, difficulty, metadata) "
            "VALUES (?,?,?,?,?,?)",
            (
                time.time(),
                derived_from,
                task_type,
                description,
                difficulty,
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return PracticeTask(
            id=cur.lastrowid or 0,
            description=description,
            task_type=task_type,
            difficulty=difficulty,
            derived_from=derived_from,
        )

    def next_task(self, task_type: str | None = None) -> PracticeTask | None:
        """Return the next under-attempted task (round-robin by attempt count)."""
        where = "WHERE task_type = ?" if task_type else ""
        params = (task_type,) if task_type else ()
        row = self._conn.execute(
            f"SELECT * FROM practice_tasks {where} ORDER BY attempts ASC, difficulty ASC LIMIT 1",
            params,
        ).fetchone()
        if row is None:
            return None
        return PracticeTask(
            id=row["id"],
            description=row["description"],
            task_type=row["task_type"],
            difficulty=row["difficulty"],
            attempts=row["attempts"],
            successes=row["successes"],
            derived_from=row["derived_from"] or "",
        )

    def record_attempt(self, task_id: int, *, success: bool) -> None:
        self._conn.execute(
            "UPDATE practice_tasks SET attempts = attempts + 1, "
            "successes = successes + ?, last_attempt = ? WHERE id = ?",
            (1 if success else 0, time.time(), task_id),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, SUM(attempts) AS attempts, "
            "SUM(successes) AS successes FROM practice_tasks"
        ).fetchone()
        total = row["total"] or 0
        attempts = row["attempts"] or 0
        successes = row["successes"] or 0
        return {
            "tasks": total,
            "attempts": attempts,
            "successes": successes,
            "success_rate": round(successes / max(attempts, 1), 3),
        }


# ---------------------------------------------------------------------------
# Singleton + scheduling
# ---------------------------------------------------------------------------


_instance: Curriculum | None = None


def get_curriculum() -> Curriculum:
    global _instance
    if _instance is None:
        _instance = Curriculum()
    return _instance


def schedule_self_play(every_seconds: int = 1800, batch_size: int = 3) -> Any:
    """Register a heartbeat callback that runs ``batch_size`` practice tasks
    each cycle and records the outcomes."""
    try:
        from isaac.scheduler.heartbeat import register_callback
    except Exception as exc:
        logger.info("Heartbeat unavailable for self-play: %s", exc)
        return None

    def _tick() -> None:
        curriculum = get_curriculum()
        # Generate fresh tasks from recent failures
        with contextlib.suppress(Exception):
            curriculum.generate_from_failures(limit=2)
        # Practice a batch
        for _ in range(batch_size):
            task = curriculum.next_task()
            if task is None:
                break
            try:
                from isaac.experts import answer

                result = answer(task.description)
                curriculum.record_attempt(task.id, success=result.confidence >= 0.6)
            except Exception as exc:
                logger.debug("self-play attempt failed: %s", exc)

    register_callback(_tick, interval_seconds=every_seconds, name="self_play_curriculum")
    return _tick

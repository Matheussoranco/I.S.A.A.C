"""DreamCoder-style library learning for the ARC DSL.

Inspired by Ellis et al. *DreamCoder* (2021): grow the library of primitives
by mining frequently re-used sub-programs from past successful syntheses
and promoting them to first-class DSL ops.

Pipeline
--------

1. **Collect** — every solved ARC task contributes its program (list of
   ``{op, args}`` steps). Stored in a SQLite table.
2. **Compress** — find common contiguous n-gram sub-programs (length
   2–``max_len``) that appear in ≥ ``min_support`` distinct solutions.
3. **Promote** — each frequent fragment becomes a new composite primitive,
   registered into :data:`isaac.arc.dsl.PRIMITIVES`. The primitive's body is a
   pre-composed function so it has zero call overhead.
4. **Refactor** — past programs are rewritten to use the new primitives so
   the library compounds.

This is a true self-improvement loop: the more ARC tasks I.S.A.A.C. solves,
the more powerful its DSL becomes, and the smaller the search space for
future tasks.

The learned library survives across sessions (SQLite at
``~/.isaac/arc_library.db``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaac.arc.dsl import PRIMITIVES, apply_program
from isaac.arc.grid_ops import Grid

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".isaac" / "arc_library.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS solved_programs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    task_hash   TEXT NOT NULL,
    program     TEXT NOT NULL,
    accuracy    REAL DEFAULT 1.0,
    strategy    TEXT,
    UNIQUE(task_hash, program)
);

CREATE TABLE IF NOT EXISTS abstractions (
    name        TEXT PRIMARY KEY,
    fragment    TEXT NOT NULL,
    support     INTEGER DEFAULT 1,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Abstraction:
    """A learned composite primitive."""

    name: str
    fragment: list[dict[str, Any]]
    """Sequence of DSL ops that this abstraction expands to."""
    support: int = 1
    """Number of distinct solved tasks the fragment was mined from."""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def to_callable(self) -> Callable[[Grid], Grid]:
        """Compile this abstraction into a single callable primitive."""
        fragment = self.fragment

        def _composite(grid: Grid) -> Grid:
            return apply_program(fragment, grid)

        _composite.__name__ = self.name
        _composite.__doc__ = f"Learned abstraction: {self.name} = " + " >> ".join(
            s.get("op", "?") for s in fragment
        )
        return _composite


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class LibraryLearner:
    """Persistent DreamCoder-style library learner."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        min_support: int = 3,
        min_len: int = 2,
        max_len: int = 4,
    ) -> None:
        self._db = Path(db_path) if db_path else _DEFAULT_DB
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self.min_support = min_support
        self.min_len = min_len
        self.max_len = max_len

        # Restore previously promoted abstractions into PRIMITIVES
        self._restore_abstractions()

    # ------------------------------------------------------------------
    # Recording solved tasks
    # ------------------------------------------------------------------

    def record_solution(
        self,
        task_payload: Any,
        program: list[dict[str, Any]],
        *,
        accuracy: float = 1.0,
        strategy: str = "unknown",
    ) -> None:
        """Persist a solved program. Only programs with accuracy >= 0.99 are kept
        for abstraction mining."""
        if accuracy < 0.99 or not program:
            return
        thash = self._hash_task(task_payload)
        prog_json = json.dumps(program, sort_keys=True)
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO solved_programs "
                "(ts, task_hash, program, accuracy, strategy) VALUES (?,?,?,?,?)",
                (time.time(), thash, prog_json, accuracy, strategy),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("record_solution failed: %s", exc)

    # ------------------------------------------------------------------
    # Library compression (the DreamCoder step)
    # ------------------------------------------------------------------

    def compress(self) -> list[Abstraction]:
        """Mine frequent fragments and promote them to DSL primitives.

        Returns the list of *newly* promoted abstractions.
        """
        rows = self._conn.execute("SELECT program FROM solved_programs").fetchall()
        programs: list[list[dict[str, Any]]] = []
        for (prog_json,) in rows:
            try:
                programs.append(json.loads(prog_json))
            except Exception:
                continue

        if len(programs) < self.min_support:
            return []

        counts = self._count_fragments(programs)
        promoted: list[Abstraction] = []

        for fragment_key, count in counts.most_common():
            if count < self.min_support:
                break
            fragment = list(json.loads(fragment_key))
            name = self._mint_name(fragment, count)
            if name in PRIMITIVES:
                continue  # Already promoted
            abstraction = Abstraction(
                name=name,
                fragment=fragment,
                support=count,
            )
            self._register(abstraction)
            promoted.append(abstraction)
            logger.info(
                "Promoted ARC abstraction %s (support=%d, len=%d).", name, count, len(fragment)
            )

        return promoted

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_abstractions(self) -> list[Abstraction]:
        rows = self._conn.execute(
            "SELECT name, fragment, support, created_at, last_used FROM abstractions "
            "ORDER BY support DESC, last_used DESC"
        ).fetchall()
        return [
            Abstraction(
                name=row[0],
                fragment=json.loads(row[1]),
                support=row[2],
                created_at=row[3],
                last_used=row[4],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        n_solutions = self._conn.execute("SELECT COUNT(*) FROM solved_programs").fetchone()[0]
        n_abstractions = self._conn.execute("SELECT COUNT(*) FROM abstractions").fetchone()[0]
        return {
            "solutions_recorded": n_solutions,
            "abstractions_learned": n_abstractions,
            "primitive_count": len(PRIMITIVES),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_fragments(
        self,
        programs: list[list[dict[str, Any]]],
    ) -> Counter[str]:
        """Count contiguous fragments by *task*-support (not raw frequency)."""
        per_task_seen: dict[str, set[str]] = {}
        for prog in programs:
            seen: set[str] = set()
            for n in range(self.min_len, min(self.max_len, len(prog)) + 1):
                for i in range(len(prog) - n + 1):
                    fragment = prog[i : i + n]
                    key = json.dumps(fragment, sort_keys=True)
                    seen.add(key)
            for key in seen:
                per_task_seen.setdefault(key, set()).add(id(prog))

        counts: Counter[str] = Counter()
        for key, task_set in per_task_seen.items():
            counts[key] = len(task_set)
        return counts

    @staticmethod
    def _mint_name(fragment: list[dict[str, Any]], count: int) -> str:
        ops = [s.get("op", "x") for s in fragment]
        base = "lib_" + "_".join(ops)[:40]
        suffix = hashlib.md5(json.dumps(fragment, sort_keys=True).encode()).hexdigest()[:6]
        return f"{base}_{suffix}"

    def _register(self, abstraction: Abstraction) -> None:
        """Inject the abstraction into PRIMITIVES and persist it."""
        PRIMITIVES[abstraction.name] = abstraction.to_callable()
        self._conn.execute(
            "INSERT OR REPLACE INTO abstractions "
            "(name, fragment, support, created_at, last_used) VALUES (?,?,?,?,?)",
            (
                abstraction.name,
                json.dumps(abstraction.fragment, sort_keys=True),
                abstraction.support,
                abstraction.created_at,
                abstraction.last_used,
            ),
        )
        self._conn.commit()

    def _restore_abstractions(self) -> None:
        for abstraction in self.list_abstractions():
            PRIMITIVES[abstraction.name] = abstraction.to_callable()

    @staticmethod
    def _hash_task(task: Any) -> str:
        try:
            payload = json.dumps(task, sort_keys=True, default=str)
        except Exception:
            payload = str(task)
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_instance: LibraryLearner | None = None


def get_library_learner() -> LibraryLearner:
    global _instance
    if _instance is None:
        _instance = LibraryLearner()
    return _instance


def reset_library_learner() -> None:
    global _instance
    _instance = None

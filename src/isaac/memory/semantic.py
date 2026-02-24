"""Semantic Memory — knowledge graph backed by NetworkX + SQLite.

Stores factual knowledge as a directed graph of (subject, predicate, object)
triples with confidence scores, timestamps, and provenance.  Supports
transitive inference for reasoning without LLM calls.

Persistence: SQLite at ``~/.isaac/memory/semantic.db``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class Fact:
    """A single knowledge triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
        }


class SemanticMemory:
    """Knowledge graph backed by NetworkX with SQLite persistence.

    Parameters
    ----------
    db_path:
        Path to the SQLite database for persistent storage.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (Path.home() / ".isaac" / "memory" / "semantic.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph: nx.DiGraph = nx.DiGraph()
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        self._load_from_db()

    # -- Database setup -----------------------------------------------------

    def _init_db(self) -> None:
        """Create the facts table if it doesn't exist."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp TEXT DEFAULT '',
                source TEXT DEFAULT '',
                UNIQUE(subject, predicate, object)
            )
        """)
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load all facts from SQLite into the NetworkX graph."""
        if self._conn is None:
            return
        cursor = self._conn.execute("SELECT subject, predicate, object, confidence, timestamp, source FROM facts")
        for row in cursor.fetchall():
            subj, pred, obj, conf, ts, src = row
            self._graph.add_edge(
                subj, obj,
                predicate=pred,
                confidence=conf,
                timestamp=ts,
                source=src,
            )
        logger.info("SemanticMemory: loaded %d facts from %s.", self._graph.number_of_edges(), self._db_path)

    # -- Write --------------------------------------------------------------

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """Add or update a fact in the knowledge graph.

        Parameters
        ----------
        subject:
            The entity or concept that is the source of the relationship.
        predicate:
            The relationship type (e.g. 'is_a', 'has_property', 'depends_on').
        object:
            The target entity or value.
        confidence:
            Confidence score (0.0–1.0).
        source:
            Provenance of the fact (e.g. 'perception', 'user', 'inference').
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        self._graph.add_edge(
            subject, object,
            predicate=predicate,
            confidence=confidence,
            timestamp=ts,
            source=source,
        )
        if self._conn is not None:
            self._conn.execute(
                """INSERT OR REPLACE INTO facts 
                   (subject, predicate, object, confidence, timestamp, source) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (subject, predicate, object, confidence, ts, source),
            )
            self._conn.commit()
        logger.debug("SemanticMemory: added fact (%s, %s, %s).", subject, predicate, object)

    # -- Read ---------------------------------------------------------------

    def query_facts(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
    ) -> list[Fact]:
        """Query facts matching the given filters.

        All parameters are optional.  ``None`` means "match any".
        """
        results: list[Fact] = []
        for u, v, data in self._graph.edges(data=True):
            edge_pred = data.get("predicate", "")
            if subject is not None and u != subject:
                continue
            if predicate is not None and edge_pred != predicate:
                continue
            if object is not None and v != object:
                continue
            results.append(Fact(
                subject=u,
                predicate=edge_pred,
                object=v,
                confidence=data.get("confidence", 1.0),
                timestamp=data.get("timestamp", ""),
                source=data.get("source", ""),
            ))
        return results

    def infer_transitive(
        self,
        subject: str,
        predicate: str,
        depth: int = 3,
    ) -> list[Fact]:
        """Infer transitive relationships up to ``depth`` hops.

        For example, if A --is_a--> B --is_a--> C, then
        ``infer_transitive('A', 'is_a', depth=2)`` returns facts for
        both B and C.
        """
        results: list[Fact] = []
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(subject, 0)]

        while queue:
            current, current_depth = queue.pop(0)
            if current in visited or current_depth >= depth:
                continue
            visited.add(current)

            for _u, v, data in self._graph.out_edges(current, data=True):
                if data.get("predicate") == predicate:
                    results.append(Fact(
                        subject=current,
                        predicate=predicate,
                        object=v,
                        confidence=data.get("confidence", 1.0) * (0.9 ** current_depth),
                        timestamp=data.get("timestamp", ""),
                        source=f"inferred(depth={current_depth + 1})",
                    ))
                    queue.append((v, current_depth + 1))

        return results

    def get_entity_context(self, entity: str) -> list[Fact]:
        """Get all facts related to an entity (as subject or object)."""
        results: list[Fact] = []

        # As subject
        for _u, v, data in self._graph.out_edges(entity, data=True):
            results.append(Fact(
                subject=entity,
                predicate=data.get("predicate", ""),
                object=v,
                confidence=data.get("confidence", 1.0),
            ))

        # As object
        for u, _v, data in self._graph.in_edges(entity, data=True):
            results.append(Fact(
                subject=u,
                predicate=data.get("predicate", ""),
                object=entity,
                confidence=data.get("confidence", 1.0),
            ))

        return results

    def contradicts(self, subject: str, predicate: str, object: str) -> bool:
        """Check if a new fact contradicts existing knowledge.

        Simple heuristic: if the same (subject, predicate) points to a
        *different* object with high confidence, it's a contradiction.
        """
        existing = self.query_facts(subject=subject, predicate=predicate)
        for fact in existing:
            if fact.object != object and fact.confidence >= 0.8:
                return True
        return False

    def to_context_string(self, max_facts: int = 50) -> str:
        """Compact string representation for LLM prompts."""
        facts = list(self._graph.edges(data=True))[:max_facts]
        if not facts:
            return "No knowledge available."
        lines: list[str] = []
        for u, v, data in facts:
            pred = data.get("predicate", "related_to")
            conf = data.get("confidence", 1.0)
            lines.append(f"  ({u}) --[{pred}]--> ({v}) [conf={conf:.2f}]")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        """Number of facts in the knowledge graph."""
        return self._graph.number_of_edges()

    @property
    def entity_count(self) -> int:
        """Number of unique entities."""
        return self._graph.number_of_nodes()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    """Return the session-scoped semantic memory singleton."""
    global _semantic_memory  # noqa: PLW0603
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory


def reset_semantic_memory() -> None:
    """Reset the singleton (used in tests)."""
    global _semantic_memory  # noqa: PLW0603
    if _semantic_memory is not None:
        _semantic_memory.close()
    _semantic_memory = None

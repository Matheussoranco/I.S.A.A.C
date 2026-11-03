"""Semantic Memory — knowledge graph backed by NetworkX + SQLite.

Stores factual knowledge as a directed graph of (subject, predicate, object)
triples with confidence scores, timestamps, and provenance.  Supports
transitive inference for reasoning without LLM calls.

Persistence: SQLite at ``~/.isaac/memory/semantic.db``.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _fact_id(subject: str, predicate: str, object: str) -> str:
    """Stable, collision-free Chroma document ID for a knowledge triple."""
    raw = "\x1f".join((subject, predicate, object)).encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Fact:
    """A single knowledge triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: str = ""
    source: str = ""
    vector_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
            "vector_id": self.vector_id,
        }


class SemanticMemory:
    """Knowledge graph backed by NetworkX with SQLite persistence.

    Parameters
    ----------
    db_path:
        Path to the SQLite database for persistent storage.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            from isaac.config.settings import get_settings

            db_path = get_settings().isaac_home / "memory" / "semantic.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._conn: sqlite3.Connection | None = None

        # ChromaDB — optional; if unavailable we fall back to exact-match only
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        try:
            import chromadb  # type: ignore[import-untyped]

            chroma_path = self._db_path.parent / "chroma_db"
            self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="semantic_facts",
            )
            logger.debug("SemanticMemory: ChromaDB initialised at %s.", chroma_path)
        except Exception as exc:
            logger.warning(
                "SemanticMemory: ChromaDB unavailable (%s) — "
                "falling back to in-memory exact-match only.",
                exc,
            )

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
        cursor = self._conn.execute(
            "SELECT subject, predicate, object, confidence, timestamp, source FROM facts"
        )
        for row in cursor.fetchall():
            subj, pred, obj, conf, ts, src = row
            self._graph.add_edge(
                subj,
                obj,
                key=pred,
                predicate=pred,
                confidence=conf,
                timestamp=ts,
                source=src,
            )
        logger.info(
            "SemanticMemory: loaded %d facts from %s.", self._graph.number_of_edges(), self._db_path
        )

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
        ts = datetime.now(tz=UTC).isoformat()
        self._graph.add_edge(
            subject,
            object,
            key=predicate,
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

        if self._chroma_collection is not None:
            fact_str = f"{subject} {predicate} {object}"
            fact_id = _fact_id(subject, predicate, object)
            self._chroma_collection.upsert(
                documents=[fact_str],
                metadatas=[
                    {
                        "subject": subject,
                        "predicate": predicate,
                        "object": object,
                    }
                ],
                ids=[fact_id],
            )

        logger.debug("SemanticMemory: added fact (%s, %s, %s).", subject, predicate, object)

    def search_similar_facts(self, query: str, top_k: int = 5) -> list[Fact]:
        """Search for facts semantically similar to the given query.

        Uses the underlying Chroma database to retrieve related knowledge graph facts.
        """
        if self._chroma_collection is None:
            logger.debug("Chroma collection not initialized — semantic search unavailable.")
            return []

        results = self._chroma_collection.query(query_texts=[query], n_results=top_k)

        matches: list[Fact] = []
        if results and results.get("metadatas"):
            for meta in results["metadatas"][0]:
                fact = self.get_fact(
                    str(meta.get("subject", "")),
                    str(meta.get("predicate", "")),
                    str(meta.get("object", "")),
                )
                if fact is not None:
                    matches.append(fact)

        return matches

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
        for u, v, edge_pred, data in self._graph.edges(keys=True, data=True):
            if subject is not None and u != subject:
                continue
            if predicate is not None and edge_pred != predicate:
                continue
            if object is not None and v != object:
                continue
            results.append(
                Fact(
                    subject=u,
                    predicate=edge_pred,
                    object=v,
                    confidence=data.get("confidence", 1.0),
                    timestamp=data.get("timestamp", ""),
                    source=data.get("source", ""),
                )
            )
        return results

    def get_fact(self, subject: str, predicate: str, object: str) -> Fact | None:
        """Return a single fact by exact triple, or ``None``."""
        data = self._graph.get_edge_data(subject, object, key=predicate)
        if data is None:
            return None
        return Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=data.get("confidence", 1.0),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", ""),
        )

    def all_facts(self) -> list[Fact]:
        """Return every stored fact."""
        return self.query_facts()

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

            for _u, v, key, data in self._graph.out_edges(current, keys=True, data=True):
                if key == predicate:
                    results.append(
                        Fact(
                            subject=current,
                            predicate=predicate,
                            object=v,
                            confidence=data.get("confidence", 1.0) * (0.9**current_depth),
                            timestamp=data.get("timestamp", ""),
                            source=f"inferred(depth={current_depth + 1})",
                        )
                    )
                    queue.append((v, current_depth + 1))

        return results

    def get_entity_context(self, entity: str) -> list[Fact]:
        """Get all facts related to an entity (as subject or object)."""
        results: list[Fact] = []

        # As subject
        for _u, v, key, data in self._graph.out_edges(entity, keys=True, data=True):
            results.append(
                Fact(
                    subject=entity,
                    predicate=key,
                    object=v,
                    confidence=data.get("confidence", 1.0),
                )
            )

        # As object
        for u, _v, key, data in self._graph.in_edges(entity, keys=True, data=True):
            results.append(
                Fact(
                    subject=u,
                    predicate=key,
                    object=entity,
                    confidence=data.get("confidence", 1.0),
                )
            )

        return results

    def contradicts(self, subject: str, predicate: str, object: str) -> bool:
        """Check if a new fact contradicts existing knowledge.

        Simple heuristic: if the same (subject, predicate) points to a
        *different* object with high confidence, it's a contradiction.
        """
        existing = self.query_facts(subject=subject, predicate=predicate)
        return any(fact.object != object and fact.confidence >= 0.8 for fact in existing)

    def decay_and_prune(
        self,
        *,
        decay: float = 0.98,
        prune_below: float = 0.15,
        exclude: set[tuple[str, str, str]] | None = None,
    ) -> int:
        """Decay confidence of stale facts and prune those below ``prune_below``.

        Facts in ``exclude`` (``(subject, predicate, object)`` triples touched
        during this pass) keep their confidence.  Changes are persisted to
        SQLite and ChromaDB.  Returns the number of pruned facts.
        """
        excluded = exclude or set()
        updates: list[tuple[float, str, str, str]] = []
        deletions: list[tuple[str, str, str]] = []

        for subject, obj, predicate, data in list(self._graph.edges(keys=True, data=True)):
            if (subject, predicate, obj) in excluded:
                continue
            new_conf = float(data.get("confidence", 0.5)) * decay
            if new_conf < prune_below:
                deletions.append((subject, predicate, obj))
            else:
                data["confidence"] = new_conf
                updates.append((new_conf, subject, predicate, obj))

        for subject, predicate, obj in deletions:
            self._graph.remove_edge(subject, obj, key=predicate)

        if self._conn is not None:
            self._conn.executemany(
                "UPDATE facts SET confidence = ? "
                "WHERE subject = ? AND predicate = ? AND object = ?",
                updates,
            )
            self._conn.executemany(
                "DELETE FROM facts WHERE subject = ? AND predicate = ? AND object = ?",
                deletions,
            )
            self._conn.commit()

        if self._chroma_collection is not None and deletions:
            ids = [_fact_id(s, p, o) for s, p, o in deletions]
            try:
                self._chroma_collection.delete(ids=ids)
            except Exception as exc:
                logger.debug("SemanticMemory: failed to prune Chroma facts: %s", exc)

        if deletions:
            logger.debug("SemanticMemory: pruned %d low-confidence facts.", len(deletions))
        return len(deletions)

    def to_context_string(self, max_facts: int = 50) -> str:
        """Compact string representation for LLM prompts."""
        facts = self.all_facts()[:max_facts]
        if not facts:
            return "No knowledge available."
        return "\n".join(
            f"  ({f.subject}) --[{f.predicate}]--> ({f.object}) [conf={f.confidence:.2f}]"
            for f in facts
        )

    def graph_snapshot(self) -> dict[str, Any]:
        """Return nodes and predicate-keyed edges for dashboards and tooling."""
        return {
            "nodes": [str(node) for node in self._graph.nodes],
            "edges": [
                {"subject": str(subject), "object": str(obj), "predicate": str(key)}
                for subject, obj, key in self._graph.edges(keys=True)
            ],
        }

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
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory


def reset_semantic_memory() -> None:
    """Reset the singleton (used in tests)."""
    global _semantic_memory
    if _semantic_memory is not None:
        _semantic_memory.close()
    _semantic_memory = None

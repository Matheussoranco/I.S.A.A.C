"""Persistent Long-Term Memory backed by SQLite FTS5.

Stores cross-session memories with full-text search for recall.
Each memory has a type (fact, preference, event, skill_outcome),
an importance score, access tracking, and a preview for FTS indexing.

Consolidation merges duplicate or low-importance memories automatically
after every N interactions (configurable via settings).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LongTermMemory:
    """SQLite-backed persistent memory with FTS5 full-text search.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            try:
                from isaac.config.settings import settings
                db_path = Path(settings.memory_db_path).expanduser()
            except Exception:
                db_path = Path.home() / ".isaac" / "memory.db"

        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._interaction_count: int = 0
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and FTS5 virtual table if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'fact',
                content TEXT NOT NULL,
                embedding_preview TEXT NOT NULL DEFAULT '',
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT NOT NULL DEFAULT ''
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                embedding_preview,
                content='memories',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, embedding_preview)
                VALUES (new.rowid, new.content, new.embedding_preview);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, embedding_preview)
                VALUES ('delete', old.rowid, old.content, old.embedding_preview);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, embedding_preview)
                VALUES ('delete', old.rowid, old.content, old.embedding_preview);
                INSERT INTO memories_fts(rowid, content, embedding_preview)
                VALUES (new.rowid, new.content, new.embedding_preview);
            END;
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        type: str = "fact",
        importance: float = 0.5,
    ) -> str:
        """Store a new memory.

        Parameters
        ----------
        content:
            The memory text to store.
        type:
            One of ``'fact'``, ``'preference'``, ``'event'``, ``'skill_outcome'``.
        importance:
            Float 0–1 indicating how important this memory is.

        Returns
        -------
        str
            The UUID of the newly created memory.
        """
        memory_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()
        preview = content[:200]

        self._conn.execute(
            """INSERT INTO memories (id, timestamp, type, content, embedding_preview,
               importance, access_count, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (memory_id, now, type, content, preview, max(0.0, min(1.0, importance)), now),
        )
        self._conn.commit()

        self._interaction_count += 1
        self._maybe_consolidate()

        logger.debug(
            "LongTermMemory: remembered [%s] (type=%s, importance=%.2f) id=%s",
            preview[:60], type, importance, memory_id,
        )
        return memory_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search memories using FTS5 full-text search.

        Parameters
        ----------
        query:
            Natural language search query.
        top_k:
            Maximum number of results.

        Returns
        -------
        list[dict]
            Matching memories sorted by relevance, each containing
            id, timestamp, type, content, importance, access_count.
        """
        if not query.strip():
            return self.recent(top_k)

        # Tokenise and join with OR for broader matching
        tokens = [t.strip() for t in query.split() if t.strip()]
        fts_query = " OR ".join(f'"{t}"' for t in tokens[:10])

        try:
            cursor = self._conn.execute(
                """SELECT m.id, m.timestamp, m.type, m.content, m.importance,
                          m.access_count, m.last_accessed
                   FROM memories m
                   JOIN memories_fts f ON m.rowid = f.rowid
                   WHERE memories_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_query, top_k),
            )
            results = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            logger.debug("FTS query failed for '%s' — falling back to LIKE.", query)
            results = self._recall_fallback(query, top_k)

        # Update access counts
        now = datetime.now(tz=timezone.utc).isoformat()
        for mem in results:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, mem["id"]),
            )
        if results:
            self._conn.commit()

        return results

    def _recall_fallback(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """LIKE-based fallback when FTS5 fails."""
        pattern = f"%{query}%"
        cursor = self._conn.execute(
            """SELECT id, timestamp, type, content, importance, access_count, last_accessed
               FROM memories
               WHERE content LIKE ?
               ORDER BY importance DESC, timestamp DESC
               LIMIT ?""",
            (pattern, top_k),
        )
        return [dict(row) for row in cursor.fetchall()]

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the *n* most recent memories."""
        cursor = self._conn.execute(
            """SELECT id, timestamp, type, content, importance, access_count, last_accessed
               FROM memories
               ORDER BY timestamp DESC
               LIMIT ?""",
            (n,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by ID."""
        cursor = self._conn.execute(
            """SELECT id, timestamp, type, content, importance, access_count, last_accessed
               FROM memories WHERE id = ?""",
            (memory_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Returns
        -------
        bool
            ``True`` if a memory was deleted.
        """
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("LongTermMemory: forgot memory %s.", memory_id)
        return deleted

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self) -> int:
        """Merge duplicate and prune low-importance memories.

        * Exact-duplicate content → keep highest importance, remove rest.
        * Memories with importance < 0.1 and access_count == 0 → delete.

        Returns
        -------
        int
            Number of memories removed.
        """
        removed = 0

        # Remove exact duplicates (keep the row with highest importance)
        cursor = self._conn.execute(
            """SELECT content, COUNT(*) as cnt
               FROM memories
               GROUP BY content
               HAVING cnt > 1"""
        )
        for row in cursor.fetchall():
            dups = self._conn.execute(
                """SELECT id FROM memories
                   WHERE content = ?
                   ORDER BY importance DESC, timestamp DESC""",
                (row["content"],),
            ).fetchall()
            # Keep first (highest importance), delete rest
            for dup in dups[1:]:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (dup["id"],))
                removed += 1

        # Prune low-importance, never-accessed memories
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE importance < 0.1 AND access_count = 0"
        )
        removed += cursor.rowcount

        if removed > 0:
            self._conn.commit()
            logger.info("LongTermMemory: consolidation removed %d memories.", removed)

        return removed

    def _maybe_consolidate(self) -> None:
        """Consolidate if interaction count has reached the threshold."""
        try:
            from isaac.config.settings import settings
            interval = settings.memory_consolidation_interval
        except Exception:
            interval = 10

        if self._interaction_count > 0 and self._interaction_count % interval == 0:
            self.consolidate()

    # ------------------------------------------------------------------
    # Stats & context
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of stored memories."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def to_context_string(self, query: str, top_k: int = 3) -> str:
        """Format top-k recalled memories as a prompt-injectable string."""
        memories = self.recall(query, top_k=top_k)
        if not memories:
            return ""
        lines: list[str] = []
        for m in memories:
            lines.append(f"  [{m['type']}] {m['content'][:200]}")
        return "## Long-term memories\n" + "\n".join(lines)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Return the singleton LongTermMemory instance."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = LongTermMemory()
    return _instance


def reset_long_term_memory() -> None:
    """Reset the singleton (used in tests)."""
    global _instance  # noqa: PLW0603
    if _instance is not None:
        _instance.close()
    _instance = None

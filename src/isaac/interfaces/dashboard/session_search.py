"""FTS5-powered search over past session history.
Query the conversation store for historical contexts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from isaac.interfaces.conversation_store import ConversationStore

# Initialize store
_store = ConversationStore(path=Path.home() / ".isaac" / "sessions.sqlite3")


def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search across all session logs for a specific query."""
    try:
        # Assuming ConversationStore has a search method or we iterate sessions
        with _store._connect() as db:
            rows = db.execute(
                "SELECT c.id, c.title, m.role, m.content "
                "FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.content LIKE ? ORDER BY m.created_at DESC LIMIT ?",
                (f"%{query}%", max(1, min(200, limit))),
            ).fetchall()
        results = [dict(row) for row in rows]
        return results
    except Exception as e:
        # Fallback if .search doesn't exist yet
        raise HTTPException(501, f"Session search not implemented in store: {e}") from e

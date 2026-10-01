"""FTS5-powered search over past session history.
Query the conversation store for historical contexts.
"""
from __future__ import annotations
from typing import Any
from fastapi import HTTPException
from isaac.interfaces.conversation_store import ConversationStore
from isaac.config.settings import settings

# Initialize store
_store = ConversationStore(store_dir=settings.store_dir)

def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search across all session logs for a specific query."""
    try:
        # Assuming ConversationStore has a search method or we iterate sessions
        results = _store.search(query, limit=limit)
        return results
    except Exception as e:
        # Fallback if .search doesn't exist yet
        raise HTTPException(501, f"Session search not implemented in store: {e}")

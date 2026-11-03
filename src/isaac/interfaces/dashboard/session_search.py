"""Search over persisted native-app conversations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from isaac.config.settings import get_settings
from isaac.interfaces.conversation_store import ConversationStore

_store: ConversationStore | None = None
_store_path: Path | None = None


def _get_store() -> ConversationStore:
    global _store, _store_path
    path = get_settings().isaac_home / "ui" / "conversations.sqlite3"
    if _store is None or _store_path != path:
        _store = ConversationStore(path)
        _store_path = path
    return _store


def search_sessions(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search across all stored conversations for *query*."""
    if not query.strip():
        return []
    try:
        return _get_store().search(query, limit=limit)
    except Exception as e:
        raise HTTPException(500, f"Session search failed: {e}") from e

"""Persistent native-app conversations backed by a small local SQLite file."""

from __future__ import annotations

import builtins
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class ConversationStore:
    """Store chat titles and messages locally without provider-side state."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def create(self, title: str = "Nova conversa") -> str:
        conversation_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO conversations(id, title, created_at, updated_at) VALUES(?,?,?,?)",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def ensure_latest(self) -> str:
        items = self.list()
        return str(items[0]["id"]) if items else self.create()

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(200, limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def load(self, conversation_id: str) -> builtins.list[dict[str, str]]:
        with self._lock, self._connect() as db:
            exists = db.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(conversation_id)
            rows = db.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    def add(self, conversation_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {role}")
        now = time.time()
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            db.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
                (conversation_id, role, content, now),
            )
            title = str(row["title"])
            if role == "user" and title == "Nova conversa":
                title = _title_from_message(content)
            db.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id),
            )

    def _initialize(self) -> None:
        with self._lock, self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def _title_from_message(message: str) -> str:
    compact = " ".join(message.split()).strip()
    return (compact[:45] + "…") if len(compact) > 46 else (compact or "Nova conversa")


__all__ = ["ConversationStore"]

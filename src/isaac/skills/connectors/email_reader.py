"""EmailConnector — Read-only IMAP email access.

Requires ``EMAIL_IMAP_HOST``, ``EMAIL_USER``, and ``EMAIL_PASSWORD``
environment variables.  Provides **read-only** operations: listing,
reading, and searching messages.
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import logging
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class EmailConnector(BaseConnector):
    """Read-only IMAP email connector."""

    name = "email"
    description = (
        "Read-only IMAP email access.  List, read, and search messages. "
        "Requires EMAIL_IMAP_HOST, EMAIL_USER, and EMAIL_PASSWORD."
    )
    requires_env: list[str] = ["EMAIL_IMAP_HOST", "EMAIL_USER", "EMAIL_PASSWORD"]

    def _connect(self) -> imaplib.IMAP4_SSL:
        import os

        host = os.environ["EMAIL_IMAP_HOST"]
        user = os.environ["EMAIL_USER"]
        password = os.environ["EMAIL_PASSWORD"]
        port = int(os.environ.get("EMAIL_IMAP_PORT", "993"))

        conn = imaplib.IMAP4_SSL(host, port)
        conn.login(user, password)
        return conn

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run an email operation.

        Parameters
        ----------
        action : str
            ``"list"`` — list recent messages, ``"read"`` — read a specific
            message, ``"search"`` — search messages.
        folder : str
            IMAP folder (default ``INBOX``).
        uid : str
            Message UID (for ``read``).
        query : str
            Search query string (for ``search``).
        limit : int
            Max messages to return (default 10).
        """
        action: str = kwargs.get("action", "list")

        try:
            if action == "list":
                return self._list_messages(**kwargs)
            if action == "read":
                return self._read_message(**kwargs)
            if action == "search":
                return self._search_messages(**kwargs)
            return {"error": f"Unknown action: {action}"}
        except Exception as exc:
            logger.error("Email %s failed: %s", action, exc)
            return {"error": str(exc)}

    def _list_messages(self, **kwargs: Any) -> dict[str, Any]:
        folder = kwargs.get("folder", "INBOX")
        limit = min(int(kwargs.get("limit", 10)), 30)

        conn = self._connect()
        try:
            conn.select(folder, readonly=True)
            _status, data = conn.search(None, "ALL")
            msg_ids = data[0].split()
            recent = msg_ids[-limit:] if msg_ids else []
            recent.reverse()  # newest first

            messages: list[dict[str, str]] = []
            for mid in recent:
                _status, msg_data = conn.fetch(mid, "(RFC822.HEADER)")
                if not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw)  # type: ignore[arg-type]
                messages.append(
                    {
                        "id": mid.decode(),
                        "from": msg.get("From", ""),
                        "subject": msg.get("Subject", ""),
                        "date": msg.get("Date", ""),
                    }
                )
            return {"folder": folder, "messages": messages}
        finally:
            conn.logout()

    def _read_message(self, **kwargs: Any) -> dict[str, Any]:
        folder = kwargs.get("folder", "INBOX")
        uid = kwargs.get("uid", "")
        if not uid:
            return {"error": "Missing 'uid'"}

        conn = self._connect()
        try:
            conn.select(folder, readonly=True)
            _status, msg_data = conn.fetch(uid.encode(), "(RFC822)")
            if not msg_data or not msg_data[0]:
                return {"error": f"Message {uid} not found"}
            raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
            msg = email.message_from_bytes(raw)  # type: ignore[arg-type]

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    if ct == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="replace")
                            break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="replace")

            return {
                "uid": uid,
                "from": msg.get("From", ""),
                "to": msg.get("To", ""),
                "subject": msg.get("Subject", ""),
                "date": msg.get("Date", ""),
                "body": body[:10_000],
            }
        finally:
            conn.logout()

    def _search_messages(self, **kwargs: Any) -> dict[str, Any]:
        folder = kwargs.get("folder", "INBOX")
        query = kwargs.get("query", "")
        limit = min(int(kwargs.get("limit", 10)), 30)
        if not query:
            return {"error": "Missing 'query'"}

        conn = self._connect()
        try:
            conn.select(folder, readonly=True)
            # IMAP SEARCH with SUBJECT or TEXT
            _status, data = conn.search(None, f'(OR SUBJECT "{query}" BODY "{query}")')
            msg_ids = data[0].split()
            recent = msg_ids[-limit:] if msg_ids else []
            recent.reverse()

            messages: list[dict[str, str]] = []
            for mid in recent:
                _status2, msg_data = conn.fetch(mid, "(RFC822.HEADER)")
                if not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1] if isinstance(msg_data[0], tuple) else msg_data[0]
                msg = email.message_from_bytes(raw)  # type: ignore[arg-type]
                messages.append(
                    {
                        "id": mid.decode(),
                        "from": msg.get("From", ""),
                        "subject": msg.get("Subject", ""),
                        "date": msg.get("Date", ""),
                    }
                )
            return {"folder": folder, "query": query, "messages": messages}
        finally:
            conn.logout()

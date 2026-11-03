"""Email Tool — IMAP / SMTP with optional OAuth2.

Send operations carry risk_level 4 and always require approval.
Content of outgoing messages is passed through the Guard node.
"""

from __future__ import annotations

import asyncio
import email as email_lib
import email.mime.text
import imaplib
import logging
import smtplib
import ssl
from contextlib import suppress
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


def _email_settings() -> dict[str, Any]:
    """Retrieve email-related settings from the environment."""
    from isaac.config.settings import get_settings

    settings = get_settings()
    return {
        "imap_host": settings.email_imap_host,
        "imap_port": settings.email_imap_port,
        "imap_user": settings.email_user,
        "imap_password": settings.email_password,
        "smtp_host": settings.email_smtp_host,
        "smtp_port": settings.email_smtp_port,
        "smtp_user": settings.email_smtp_user or settings.email_user,
        "smtp_password": settings.email_smtp_password or settings.email_password,
    }


def _decode_part(part: Any) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _read_imap(cfg: dict[str, Any], folder: str, limit: int) -> str:
    context = ssl.create_default_context()
    connection = imaplib.IMAP4_SSL(
        cfg["imap_host"],
        int(cfg["imap_port"]),
        ssl_context=context,
        timeout=30,
    )
    try:
        connection.login(cfg["imap_user"], cfg["imap_password"])
        status, _ = connection.select(folder, readonly=True)
        if status != "OK":
            raise RuntimeError(f"IMAP folder is unavailable: {folder}")
        status, data = connection.search(None, "ALL")
        if status != "OK" or not data or not data[0]:
            return "No messages found."
        message_ids = data[0].split()
        recent_ids = message_ids[-limit:]

        messages: list[str] = []
        for message_id in reversed(recent_ids):
            status, message_data = connection.fetch(message_id, "(RFC822)")
            if status != "OK" or not message_data:
                continue
            if not message_data[0] or not isinstance(message_data[0], tuple):
                continue
            message = email_lib.message_from_bytes(message_data[0][1])
            subject = message.get("Subject", "(no subject)")
            sender = message.get("From", "(unknown)")
            date = message.get("Date", "")
            body = ""
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        body = _decode_part(part)[:500]
                        break
            else:
                body = _decode_part(message)[:500]
            messages.append(f"From: {sender}\nDate: {date}\nSubject: {subject}\n{body}\n---")
        return "\n".join(messages) or "No messages found."
    finally:
        with suppress(Exception):
            connection.logout()


def _send_smtp(cfg: dict[str, Any], to_addr: str, subject: str, body: str) -> None:
    context = ssl.create_default_context()
    message = email.mime.text.MIMEText(body, "plain", "utf-8")
    message["Subject"] = subject
    message["From"] = cfg["smtp_user"]
    message["To"] = to_addr

    host = cfg["smtp_host"]
    port = int(cfg["smtp_port"])
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as smtp:
            smtp.login(cfg["smtp_user"], cfg["smtp_password"])
            smtp.send_message(message)
        return
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        smtp.login(cfg["smtp_user"], cfg["smtp_password"])
        smtp.send_message(message)


class EmailReadTool(IsaacTool):
    """Read emails via IMAP."""

    name = "email_read"
    description = "Read recent emails from an IMAP mailbox."
    risk_level = 2
    requires_approval = False
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "IMAP folder (default 'INBOX')."},
            "limit": {"type": "integer", "description": "Number of messages (default 5)."},
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch recent emails.

        Parameters
        ----------
        folder:
            IMAP folder (default ``"INBOX"``).
        limit:
            Number of messages to fetch (default 5).
        """
        cfg = _email_settings()
        if not cfg.get("imap_host") or not cfg.get("imap_user"):
            return ToolResult(
                success=False,
                error=(
                    "Email IMAP settings are not configured. "
                    "Set ISAAC_EMAIL_IMAP_HOST, ISAAC_EMAIL_USER, ISAAC_EMAIL_PASSWORD."
                ),
            )

        folder = str(kwargs.get("folder") or "INBOX")
        try:
            limit = int(kwargs.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 100))

        try:
            output = await asyncio.to_thread(_read_imap, cfg, folder, limit)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            logger.error("IMAP read error: %s", exc)
            return ToolResult(success=False, error=str(exc))


class EmailSendTool(IsaacTool):
    """Send an email via SMTP.  Risk 4 — always requires approval."""

    name = "email_send"
    description = "Send an email via SMTP. High risk — requires human approval."
    risk_level = 4
    requires_approval = True
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address."},
            "subject": {"type": "string", "description": "Email subject line."},
            "body": {"type": "string", "description": "Plain-text email body."},
        },
        "required": ["to", "body"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send an email.

        Parameters
        ----------
        to:
            Recipient address.
        subject:
            Email subject.
        body:
            Plain-text body.
        """
        cfg = _email_settings()
        if not cfg.get("smtp_host") or not cfg.get("smtp_user"):
            return ToolResult(
                success=False,
                error="Email SMTP settings are not configured.",
            )

        to_addr = str(kwargs.get("to", "")).strip()
        subject = str(kwargs.get("subject", "")).strip()
        body = str(kwargs.get("body", ""))

        if not to_addr:
            return ToolResult(success=False, error="Missing 'to' parameter.")
        if not body:
            return ToolResult(success=False, error="Missing 'body' parameter.")
        for label, value in (("to", to_addr), ("subject", subject)):
            if "\r" in value or "\n" in value:
                return ToolResult(
                    success=False,
                    error=f"Invalid {label}: line breaks are not allowed.",
                )

        body = await self._guard_screen(body)

        try:
            await asyncio.to_thread(_send_smtp, cfg, to_addr, subject, body)
            return ToolResult(success=True, output=f"Email sent to {to_addr}.")
        except Exception as exc:
            logger.error("SMTP send error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    async def _guard_screen(self, text: str) -> str:
        """Run outgoing text through the prompt-injection guard.

        If the guard flags suspicious content the text is returned unchanged
        but a warning is logged.  The guard is best-effort — import errors
        are silently ignored.
        """
        try:
            from isaac.nodes.guard import PromptInjectionGuard

            guard = PromptInjectionGuard()
            result = guard.analyse(text)
            if result.flagged_patterns:
                logger.warning(
                    "Outgoing email flagged by guard: %s",
                    result.flagged_patterns,
                )
        except Exception:
            pass
        return text

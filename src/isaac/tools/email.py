"""Email Tool — IMAP / SMTP with optional OAuth2.

Send operations carry risk_level 4 and always require approval.
Content of outgoing messages is passed through the Guard node.
"""

from __future__ import annotations

import email as email_lib
import email.mime.text
import imaplib
import logging
import smtplib
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


def _email_settings() -> dict[str, str]:
    """Retrieve email-related settings from the environment."""
    try:
        from isaac.config.settings import get_settings

        s = get_settings()
        return {
            "imap_host": getattr(s, "email_imap_host", ""),
            "imap_port": str(getattr(s, "email_imap_port", "993")),
            "smtp_host": getattr(s, "email_smtp_host", ""),
            "smtp_port": str(getattr(s, "email_smtp_port", "587")),
            "username": getattr(s, "email_username", ""),
            "password": getattr(s, "email_password", ""),
        }
    except Exception:
        return {}


class EmailReadTool(IsaacTool):
    """Read emails via IMAP."""

    name = "email_read"
    description = "Read recent emails from an IMAP mailbox."
    risk_level = 2
    requires_approval = False
    sandbox_required = False

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
        if not cfg.get("imap_host") or not cfg.get("username"):
            return ToolResult(
                success=False,
                error="Email IMAP settings are not configured. Set EMAIL_IMAP_HOST, EMAIL_USERNAME, EMAIL_PASSWORD.",
            )

        folder: str = kwargs.get("folder", "INBOX")
        limit: int = int(kwargs.get("limit", 5))

        try:
            conn = imaplib.IMAP4_SSL(cfg["imap_host"], int(cfg.get("imap_port", "993")))
            conn.login(cfg["username"], cfg["password"])
            conn.select(folder, readonly=True)

            _status, data = conn.search(None, "ALL")
            msg_ids = data[0].split()
            recent_ids = msg_ids[-limit:] if msg_ids else []

            messages: list[str] = []
            for mid in reversed(recent_ids):
                _status, msg_data = conn.fetch(mid, "(RFC822)")
                if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                    raw = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw)  # type: ignore[arg-type]
                    subject = msg.get("Subject", "(no subject)")
                    sender = msg.get("From", "(unknown)")
                    date = msg.get("Date", "")
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                payload = part.get_payload(decode=True)
                                if isinstance(payload, bytes):
                                    body = payload.decode("utf-8", errors="replace")[:500]
                                break
                    else:
                        payload = msg.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode("utf-8", errors="replace")[:500]

                    messages.append(
                        f"From: {sender}\nDate: {date}\nSubject: {subject}\n{body}\n---"
                    )

            conn.logout()
            return ToolResult(
                success=True,
                output="\n".join(messages) or "No messages found.",
            )
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
        if not cfg.get("smtp_host") or not cfg.get("username"):
            return ToolResult(
                success=False,
                error="Email SMTP settings are not configured.",
            )

        to_addr: str = kwargs.get("to", "")
        subject: str = kwargs.get("subject", "")
        body: str = kwargs.get("body", "")

        if not to_addr:
            return ToolResult(success=False, error="Missing 'to' parameter.")
        if not body:
            return ToolResult(success=False, error="Missing 'body' parameter.")

        # Pass outgoing body through the Guard for injection screening
        body = await self._guard_screen(body)

        try:
            msg = email.mime.text.MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = cfg["username"]
            msg["To"] = to_addr

            with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port", "587"))) as smtp:
                smtp.starttls()
                smtp.login(cfg["username"], cfg["password"])
                smtp.send_message(msg)

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

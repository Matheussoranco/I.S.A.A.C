"""Delivery of background-job results to operator channels."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage

logger = logging.getLogger(__name__)


def deliver(target: str, text: str) -> None:
    """Deliver *text* to *target*.

    Supported targets are ``telegram``, ``email`` or ``email:<address>``,
    ``discord`` (via ``DISCORD_WEBHOOK_URL``), and explicit ``http(s)://``
    webhook URLs.  Raises ``RuntimeError`` when the channel is unavailable so
    callers can record the failure.
    """
    kind, _, arg = target.strip().partition(":")
    kind = kind.lower()
    if kind == "telegram":
        from isaac.interfaces.telegram_gateway import send_message_to_all

        send_message_to_all(text)
    elif kind == "email":
        _send_email(arg or None, text)
    elif kind == "discord":
        url = arg or os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        if not url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is not set")
        _post_json(url, {"content": text[:1900]})
    elif target.startswith(("http://", "https://")):
        _post_json(target, {"text": text})
    else:
        raise RuntimeError(f"Unsupported delivery target: {target!r}")


def _post_json(url: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise RuntimeError(f"delivery webhook returned HTTP {status}")


def _send_email(to: str | None, text: str) -> None:
    from isaac.config.settings import get_settings

    settings = get_settings()
    host = settings.email_smtp_host
    user = settings.email_smtp_user or settings.email_user
    password = settings.email_smtp_password or settings.email_password
    recipient = to or user
    if not (host and user and password and recipient):
        raise RuntimeError("Email SMTP settings are incomplete")

    message = EmailMessage()
    message["Subject"] = "I.S.A.A.C. job result"
    message["From"] = user
    message["To"] = recipient
    message.set_content(text)
    with smtplib.SMTP(host, settings.email_smtp_port, timeout=15) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(message)

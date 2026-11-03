"""Tests for IMAP/SMTP transport selection, limits, and header safety."""

from __future__ import annotations

import asyncio
import email.message
from typing import Any

import pytest

from isaac.config.settings import get_settings
from isaac.tools import email as email_module
from isaac.tools.email import EmailReadTool, EmailSendTool, _email_settings, _send_smtp


def _run(coro):
    return asyncio.run(coro)


def _message(subject: str, sender: str, body: str) -> bytes:
    message = email.message.EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message.set_content(body)
    return message.as_bytes()


class _FakeIMAP:
    instances: list[_FakeIMAP] = []

    def __init__(self, host: str, port: int, *, ssl_context: Any, timeout: int) -> None:
        self.host = host
        self.port = port
        self.logged_out = False
        self.fetched: list[bytes] = []
        type(self).instances.append(self)

    def login(self, user: str, password: str) -> None:
        self.user = user

    def select(self, folder: str, readonly: bool = True) -> tuple[str, list[bytes]]:
        return "OK", [b"1"]

    def search(self, charset: Any, criterion: str) -> tuple[str, list[bytes]]:
        return "OK", [b"1 2 3"]

    def fetch(self, message_id: bytes, spec: str) -> tuple[str, list[Any]]:
        self.fetched.append(message_id)
        payload = _message(f"Subject {message_id.decode()}", "a@example.com", "body")
        return "OK", [(b"header", payload)]

    def logout(self) -> None:
        self.logged_out = True


class _FakeSMTP:
    instances: list[_FakeSMTP] = []
    starttls_calls = 0
    sent: list[email.message.Message] = []

    def __init__(self, host: str, port: int, *, timeout: int, context: Any = None) -> None:
        self.host = host
        self.port = port
        type(self).instances.append(self)

    def __enter__(self) -> _FakeSMTP:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def ehlo(self) -> None:
        return None

    def starttls(self, context: Any) -> None:
        type(self).starttls_calls += 1

    def login(self, user: str, password: str) -> None:
        self.user = user

    def send_message(self, message: email.message.Message) -> None:
        type(self).sent.append(message)


def _configure(monkeypatch, **values: Any) -> None:
    settings = get_settings()
    for key, value in values.items():
        monkeypatch.setattr(settings, key, value, raising=False)


def test_settings_mapping_uses_dedicated_smtp_credentials(monkeypatch) -> None:
    _configure(
        monkeypatch,
        email_imap_host="imap.example.com",
        email_user="imap-user",
        email_password="imap-secret",
        email_smtp_host="smtp.example.com",
        email_smtp_user="smtp-user",
        email_smtp_password="smtp-secret",
    )

    cfg = _email_settings()

    assert cfg["imap_user"] == "imap-user"
    assert cfg["imap_password"] == "imap-secret"
    assert cfg["smtp_user"] == "smtp-user"
    assert cfg["smtp_password"] == "smtp-secret"


def test_settings_mapping_falls_back_to_imap_credentials(monkeypatch) -> None:
    _configure(
        monkeypatch,
        email_user="shared-user",
        email_password="shared-secret",
        email_smtp_user="",
        email_smtp_password="",
    )

    cfg = _email_settings()

    assert cfg["smtp_user"] == "shared-user"
    assert cfg["smtp_password"] == "shared-secret"


def test_smtp_uses_starttls_on_submission_port(monkeypatch) -> None:
    _FakeSMTP.instances = []
    _FakeSMTP.sent = []
    _FakeSMTP.starttls_calls = 0
    monkeypatch.setattr(email_module.smtplib, "SMTP", _FakeSMTP)
    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_user": "user",
        "smtp_password": "secret",
    }

    _send_smtp(cfg, "to@example.com", "Hi", "Body")

    smtp = _FakeSMTP.instances[0]
    assert smtp.port == 587
    assert _FakeSMTP.starttls_calls == 1
    assert _FakeSMTP.sent[0]["To"] == "to@example.com"


def test_smtp_uses_implicit_tls_on_port_465(monkeypatch) -> None:
    _FakeSMTP.instances = []
    _FakeSMTP.sent = []
    _FakeSMTP.starttls_calls = 0
    monkeypatch.setattr(email_module.smtplib, "SMTP_SSL", _FakeSMTP)
    cfg = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 465,
        "smtp_user": "user",
        "smtp_password": "secret",
    }

    _send_smtp(cfg, "to@example.com", "Hi", "Body")

    assert _FakeSMTP.instances[0].port == 465
    assert _FakeSMTP.starttls_calls == 0


def test_send_rejects_header_injection(monkeypatch) -> None:
    _configure(monkeypatch, email_smtp_host="smtp.example.com", email_smtp_user="user")

    result = _run(
        EmailSendTool().execute(
            to="to@example.com", subject="Hello\r\nBcc: victim@example.com", body="Body"
        )
    )

    assert result.success is False
    assert "line breaks" in result.error


def test_read_clamps_limit_and_logs_out(monkeypatch) -> None:
    _FakeIMAP.instances = []
    monkeypatch.setattr(email_module.imaplib, "IMAP4_SSL", _FakeIMAP)
    _configure(
        monkeypatch,
        email_imap_host="imap.example.com",
        email_user="user",
        email_password="secret",
    )

    result = _run(EmailReadTool().execute(limit=0))

    assert result.success, result.error
    connection = _FakeIMAP.instances[0]
    assert connection.fetched == [b"3"]
    assert connection.logged_out is True
    assert "Subject 3" in result.output


def test_read_logs_out_when_login_fails(monkeypatch) -> None:
    _FakeIMAP.instances = []

    class _FailingIMAP(_FakeIMAP):
        def login(self, user: str, password: str) -> None:
            raise RuntimeError("bad credentials")

    monkeypatch.setattr(email_module.imaplib, "IMAP4_SSL", _FailingIMAP)
    _configure(
        monkeypatch,
        email_imap_host="imap.example.com",
        email_user="user",
        email_password="wrong",
    )

    result = _run(EmailReadTool().execute(limit=5))

    assert result.success is False
    assert "bad credentials" in result.error
    assert _FailingIMAP.instances[0].logged_out is True


@pytest.mark.parametrize("limit", [1, 5, 100])
def test_read_fetches_at_most_the_requested_limit(monkeypatch, limit: int) -> None:
    _FakeIMAP.instances = []
    monkeypatch.setattr(email_module.imaplib, "IMAP4_SSL", _FakeIMAP)
    _configure(
        monkeypatch,
        email_imap_host="imap.example.com",
        email_user="user",
        email_password="secret",
    )

    result = _run(EmailReadTool().execute(limit=limit))

    assert result.success, result.error
    assert len(_FakeIMAP.instances[0].fetched) == min(limit, 3)

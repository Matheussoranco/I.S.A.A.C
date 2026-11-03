"""Tests for CalDAV datetime handling and iCalendar escaping."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from isaac.tools import calendar as calendar_module
from isaac.tools.calendar import CalendarWriteTool


class _Calendar:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_event(self, payload: str) -> None:
        self.saved.append(payload)


def _run(coro):
    return asyncio.run(coro)


def _fake_client(monkeypatch) -> _Calendar:
    cal = _Calendar()
    principal = SimpleNamespace(calendars=lambda: [cal])
    client = SimpleNamespace(principal=lambda: principal)
    monkeypatch.setattr(calendar_module, "_caldav_client", lambda: client)
    return cal


def test_write_converts_offsets_to_utc(monkeypatch) -> None:
    cal = _fake_client(monkeypatch)

    result = _run(
        CalendarWriteTool().execute(
            summary="Standup",
            start="2026-10-28T10:00:00+02:00",
            end="2026-10-28T11:00:00+02:00",
        )
    )

    assert result.success, result.error
    assert "DTSTART:20261028T080000Z" in cal.saved[0]
    assert "DTEND:20261028T090000Z" in cal.saved[0]


def test_write_escapes_ical_text(monkeypatch) -> None:
    cal = _fake_client(monkeypatch)

    result = _run(
        CalendarWriteTool().execute(
            summary="Lunch, with; team\\x",
            start="2026-10-28T10:00:00Z",
            description="line one\nline two",
        )
    )

    assert result.success, result.error
    payload = cal.saved[0]
    assert "SUMMARY:Lunch\\, with\\; team\\\\x" in payload
    description_line = next(
        line for line in payload.split("\r\n") if line.startswith("DESCRIPTION:")
    )
    assert description_line == "DESCRIPTION:line one\\nline two"


def test_write_rejects_end_before_start(monkeypatch) -> None:
    cal = _fake_client(monkeypatch)

    result = _run(
        CalendarWriteTool().execute(
            summary="Backwards",
            start="2026-10-28T10:00:00Z",
            end="2026-10-28T09:00:00Z",
        )
    )

    assert result.success is False
    assert "precede" in result.error
    assert cal.saved == []


def test_read_rejects_non_integer_days(monkeypatch) -> None:
    _fake_client(monkeypatch)
    result: Any = _run(calendar_module.CalendarReadTool().execute(days="soon"))
    assert result.success is False
    assert "integer" in result.error

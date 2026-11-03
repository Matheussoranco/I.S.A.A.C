"""Calendar Tool — CalDAV / Google Calendar integration.

Write operations carry risk_level 4 and require approval.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


def _caldav_client() -> Any | None:
    """Create a CalDAV client from environment settings."""
    try:
        import caldav  # type: ignore[import-untyped]

        from isaac.config.settings import get_settings

        settings = get_settings()
        url = settings.caldav_url
        if not url:
            return None
        return caldav.DAVClient(
            url=url,
            username=settings.caldav_username,
            password=settings.caldav_password,
        )
    except Exception as exc:
        logger.debug("CalDAV client unavailable: %s", exc)
        return None


def _escape_ical(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {label} datetime: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(UTC)


def _read_events(client: Any, days: int) -> str:
    principal = client.principal()
    calendars = principal.calendars()
    if not calendars:
        return "No calendars found."

    now = datetime.now(UTC)
    end = now + timedelta(days=days)
    events_out: list[str] = []
    for cal in calendars:
        cal_name = getattr(cal, "name", "Unnamed")
        for event in cal.date_search(start=now, end=end, expand=True):
            vevent = event.vobject_instance.vevent  # type: ignore[attr-defined]
            summary = str(getattr(vevent, "summary", "(no title)"))
            dtstart = str(getattr(vevent, "dtstart", ""))
            dtend = str(getattr(vevent, "dtend", ""))
            events_out.append(f"[{cal_name}] {summary} | {dtstart} \u2192 {dtend}")
    return "\n".join(events_out) or "No upcoming events."


def _save_event(client: Any, vcal: str) -> None:
    calendars = client.principal().calendars()
    if not calendars:
        raise RuntimeError("No calendars available.")
    calendars[0].save_event(vcal)


class CalendarReadTool(IsaacTool):
    """Read upcoming calendar events via CalDAV."""

    name = "calendar_read"
    description = "Read upcoming events from a CalDAV calendar."
    risk_level = 1
    requires_approval = False
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "description": "Look-ahead window in days (default 7)."},
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch upcoming events.

        Parameters
        ----------
        days:
            Look-ahead window in days (default 7).
        """
        client = _caldav_client()
        if client is None:
            return ToolResult(
                success=False,
                error="CalDAV not configured. Set CALDAV_URL, CALDAV_USERNAME, CALDAV_PASSWORD.",
            )

        try:
            days = int(kwargs.get("days", 7))
        except (TypeError, ValueError):
            return ToolResult(success=False, error="'days' must be an integer.")
        days = max(1, min(days, 365))

        try:
            output = await asyncio.to_thread(_read_events, client, days)
            return ToolResult(success=True, output=output)
        except Exception as exc:
            logger.error("CalDAV read error: %s", exc)
            return ToolResult(success=False, error=str(exc))


class CalendarWriteTool(IsaacTool):
    """Create a calendar event.  Risk 4 — requires approval."""

    name = "calendar_write"
    description = "Create a calendar event via CalDAV. Requires human approval."
    risk_level = 4
    requires_approval = True
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Event title."},
            "start": {"type": "string", "description": "Start datetime (ISO 8601)."},
            "end": {"type": "string", "description": "End datetime (ISO 8601); default start+1h."},
            "description": {"type": "string", "description": "Optional event description."},
        },
        "required": ["summary", "start"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Create an event.

        Parameters
        ----------
        summary:
            Event title.
        start:
            Start datetime (ISO 8601).
        end:
            End datetime (ISO 8601).  Defaults to start + 1h.
        description:
            Optional event description.
        """
        client = _caldav_client()
        if client is None:
            return ToolResult(success=False, error="CalDAV not configured.")

        summary = str(kwargs.get("summary", "")).strip()
        start_str = str(kwargs.get("start", "")).strip()
        end_str = str(kwargs.get("end", "")).strip()
        description = str(kwargs.get("description", ""))

        if not summary or not start_str:
            return ToolResult(
                success=False,
                error="Missing 'summary' and/or 'start' parameter.",
            )

        try:
            start_dt = _parse_utc(start_str, "start")
            end_dt = _parse_utc(end_str, "end") if end_str else start_dt + timedelta(hours=1)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if end_dt < start_dt:
            return ToolResult(success=False, error="Event end must not precede its start.")

        vcal = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//ISAAC//EN\r\n"
            "BEGIN:VEVENT\r\n"
            f"SUMMARY:{_escape_ical(summary)}\r\n"
            f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DESCRIPTION:{_escape_ical(description)}\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        try:
            await asyncio.to_thread(_save_event, client, vcal)
            return ToolResult(
                success=True,
                output=f"Event '{summary}' created on {start_dt.isoformat()}.",
            )
        except Exception as exc:
            logger.error("CalDAV write error: %s", exc)
            return ToolResult(success=False, error=str(exc))

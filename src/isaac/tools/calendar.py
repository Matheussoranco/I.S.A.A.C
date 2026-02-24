"""Calendar Tool — CalDAV / Google Calendar integration.

Write operations carry risk_level 4 and require approval.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


def _caldav_client() -> Any | None:
    """Create a CalDAV client from environment settings."""
    try:
        import caldav  # type: ignore[import-untyped]
        from isaac.config.settings import get_settings

        s = get_settings()
        url = getattr(s, "caldav_url", "")
        username = getattr(s, "caldav_username", "")
        password = getattr(s, "caldav_password", "")

        if not url:
            return None

        return caldav.DAVClient(url=url, username=username, password=password)
    except Exception:
        return None


class CalendarReadTool(IsaacTool):
    """Read upcoming calendar events via CalDAV."""

    name = "calendar_read"
    description = "Read upcoming events from a CalDAV calendar."
    risk_level = 1
    requires_approval = False
    sandbox_required = False

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

        days: int = int(kwargs.get("days", 7))

        try:
            principal = client.principal()
            calendars = principal.calendars()

            if not calendars:
                return ToolResult(success=True, output="No calendars found.")

            now = datetime.utcnow()
            end = now + timedelta(days=days)

            events_out: list[str] = []
            for cal in calendars:
                cal_name = getattr(cal, "name", "Unnamed")
                results = cal.date_search(start=now, end=end, expand=True)
                for event in results:
                    vevent = event.vobject_instance.vevent  # type: ignore[attr-defined]
                    summary = str(getattr(vevent, "summary", "(no title)"))
                    dtstart = str(getattr(vevent, "dtstart", ""))
                    dtend = str(getattr(vevent, "dtend", ""))
                    events_out.append(
                        f"[{cal_name}] {summary} | {dtstart} → {dtend}"
                    )

            return ToolResult(
                success=True,
                output="\n".join(events_out) or "No upcoming events.",
            )
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
            return ToolResult(
                success=False,
                error="CalDAV not configured.",
            )

        summary: str = kwargs.get("summary", "")
        start_str: str = kwargs.get("start", "")
        end_str: str = kwargs.get("end", "")
        description: str = kwargs.get("description", "")

        if not summary or not start_str:
            return ToolResult(
                success=False,
                error="Missing 'summary' and/or 'start' parameter.",
            )

        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = (
                datetime.fromisoformat(end_str)
                if end_str
                else start_dt + timedelta(hours=1)
            )
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid datetime: {exc}")

        vcal = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//ISAAC//EN\r\n"
            "BEGIN:VEVENT\r\n"
            f"SUMMARY:{summary}\r\n"
            f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DESCRIPTION:{description}\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        try:
            principal = client.principal()
            calendars = principal.calendars()
            if not calendars:
                return ToolResult(success=False, error="No calendars available.")

            cal = calendars[0]
            cal.save_event(vcal)
            return ToolResult(
                success=True,
                output=f"Event '{summary}' created on {start_dt.isoformat()}.",
            )
        except Exception as exc:
            logger.error("CalDAV write error: %s", exc)
            return ToolResult(success=False, error=str(exc))

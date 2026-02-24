"""CalendarConnector — Read / write local .ics calendar files.

Uses the ``icalendar`` package to parse and create iCalendar events.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_DEFAULT_ICS = Path.home() / ".isaac" / "calendar.ics"


class CalendarConnector(BaseConnector):
    """Read and write events in local .ics calendar files."""

    name = "calendar"
    description = (
        "Read or add events to a local .ics calendar file. "
        "Uses the icalendar package."
    )
    requires_env: list[str] = []

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run a calendar operation.

        Parameters
        ----------
        action : str
            ``"read"`` — list upcoming events, ``"add"`` — create a new event.
        ics_path : str | None
            Path to the .ics file (default ``~/.isaac/calendar.ics``).
        summary : str
            Event summary / title (for ``add``).
        start : str
            ISO-format start datetime (for ``add``).
        duration_minutes : int
            Duration in minutes (default 60, for ``add``).
        description : str
            Optional event description (for ``add``).
        """
        action: str = kwargs.get("action", "read")
        ics_path = Path(kwargs.get("ics_path", "") or str(_DEFAULT_ICS))

        try:
            if action == "read":
                return self._read(ics_path)
            if action == "add":
                return self._add(ics_path, **kwargs)
            return {"error": f"Unknown action: {action}"}
        except Exception as exc:
            logger.error("Calendar %s failed: %s", action, exc)
            return {"error": str(exc)}

    def _read(self, ics_path: Path) -> dict[str, Any]:
        if not ics_path.exists():
            return {"events": [], "note": "No calendar file found"}

        from icalendar import Calendar  # type: ignore[import-untyped]

        cal = Calendar.from_ical(ics_path.read_bytes())
        events: list[dict[str, str]] = []
        for component in cal.walk():
            if component.name == "VEVENT":
                dtstart = component.get("dtstart")
                dtend = component.get("dtend")
                events.append(
                    {
                        "summary": str(component.get("summary", "")),
                        "start": str(dtstart.dt) if dtstart else "",
                        "end": str(dtend.dt) if dtend else "",
                        "description": str(component.get("description", "")),
                        "uid": str(component.get("uid", "")),
                    }
                )
        # Sort by start ascending
        events.sort(key=lambda e: e["start"])
        return {"ics_path": str(ics_path), "events": events[:50]}

    def _add(self, ics_path: Path, **kwargs: Any) -> dict[str, Any]:
        from icalendar import Calendar, Event  # type: ignore[import-untyped]

        summary = kwargs.get("summary", "Untitled Event")
        start_str = kwargs.get("start", "")
        duration_minutes = int(kwargs.get("duration_minutes", 60))
        description = kwargs.get("description", "")

        if not start_str:
            return {"error": "Missing 'start' datetime (ISO format)"}

        dt_start = datetime.fromisoformat(start_str)
        dt_end = dt_start + timedelta(minutes=duration_minutes)

        # Load existing or create new calendar
        if ics_path.exists():
            cal = Calendar.from_ical(ics_path.read_bytes())
        else:
            cal = Calendar()
            cal.add("prodid", "-//I.S.A.A.C.//Calendar//EN")
            cal.add("version", "2.0")

        event = Event()
        event.add("summary", summary)
        event.add("dtstart", dt_start)
        event.add("dtend", dt_end)
        event.add("description", description)
        uid = f"{uuid.uuid4()}@isaac"
        event.add("uid", uid)
        event.add("dtstamp", datetime.utcnow())

        cal.add_component(event)

        ics_path.parent.mkdir(parents=True, exist_ok=True)
        ics_path.write_bytes(cal.to_ical())

        return {
            "status": "created",
            "uid": uid,
            "summary": summary,
            "start": str(dt_start),
            "end": str(dt_end),
            "ics_path": str(ics_path),
        }

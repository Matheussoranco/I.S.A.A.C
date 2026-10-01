"""Natural Language to Cron Expression Parser.
Provides logic to convert colloquial schedules into standard 5-field cron expressions.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Mapping of common weekdays to cron day-of-week indices (0=Sunday)
_WEEKDAY_CRON = {
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
    "sunday": "0",
}


def schedule_from_nl(text: str) -> str:
    """
    Convert a short colloquial schedule into a cron expression.
    Returns the input unchanged when it already looks like a 5-field cron expression
    or cannot be parsed.

    Examples:
    - "every 30m" -> "*/30 * * * *"
    - "every monday 9am" -> "0 9 * * 1"
    - "daily" -> "0 0 * * *"
    """
    t = text.strip().lower()
    if not t:
        return "0 * * * *"

    # Already a 5-field cron expression?
    if re.fullmatch(r"\S+(\s+\S+){4}", t):
        return t

    # 'every X seconds' (sub-minute handled via specialized marker or treated as every minute)
    m = re.fullmatch(r"every\s+(\d+)\s*(s|sec|secs|seconds)", t)
    if m:
        # Standard cron has no seconds field; the engine handles this marker.
        return f"* * * * * */{m.group(1)}"

    # 'every X minutes'
    m = re.fullmatch(r"every\s+(\d+)\s*(m|min|mins|minutes)", t)
    if m:
        return f"*/{m.group(1)} * * * *"

    # 'every X hours'
    m = re.fullmatch(r"every\s+(\d+)\s*(h|hr|hrs|hours)", t)
    if m:
        return f"0 */{m.group(1)} * * *"

    # 'every X days'
    m = re.fullmatch(r"every\s+(\d+)\s*(d|day|days)", t)
    if m:
        return f"0 0 */{m.group(1)} * *"

    # 'every [weekday] [at] [time]'
    m = re.fullmatch(
        r"every\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"(?:\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
        t,
    )
    if m:
        day = m.group(1)
        hour_str, minute_str, ampm = m.group(2), m.group(3), m.group(4)

        hh = int(hour_str) if hour_str else 0
        mm = minute_str if minute_str else "0"

        if ampm == "pm" and hh < 12:
            hh += 12
        elif ampm == "am" and hh == 12:
            hh = 0

        return f"{mm} {hh} * * {_WEEKDAY_CRON[day]}"

    # Common presets
    presets = {
        "hourly": "0 * * * *",
        "daily": "0 0 * * *",
        "weekly": "0 0 * * 0",
        "monthly": "0 0 1 * *",
    }
    if t in presets:
        return presets[t]

    logger.debug("Could not parse NL schedule '%s', returning as-is", text)
    return text

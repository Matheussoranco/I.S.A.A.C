"""Proactive Heartbeat Scheduler — periodic self-initiated actions.

Uses APScheduler to run background jobs:
- **Heartbeat**: periodic status check + Telegram notification.
- **TASKS.md scan**: parse ``~/.isaac/TASKS.md`` and surface due items.
- **Memory consolidation**: periodic compression / dedup of episodic memory.

The scheduler is started once by the CLI or ``__main__`` entry point.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_scheduler: Any | None = None


def _get_settings() -> Any:
    from isaac.config.settings import get_settings
    return get_settings()


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------


def heartbeat_job() -> None:
    """Periodic heartbeat — logs status and notifies operator."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    message = f"💓 Heartbeat at {now} — I.S.A.A.C. is running."

    logger.info(message)

    try:
        from isaac.interfaces.telegram_gateway import send_notification
        send_notification(message)
    except Exception:
        pass


def tasks_scan_job() -> None:
    """Parse ``TASKS.md`` and surface due/overdue items via Telegram."""
    settings = _get_settings()
    tasks_path = settings.isaac_home / "TASKS.md"

    if not tasks_path.exists():
        return

    try:
        content = tasks_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read TASKS.md: %s", exc)
        return

    due_items: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        # Look for unchecked items: - [ ] ...
        if stripped.startswith("- [ ]"):
            due_items.append(stripped)

    if not due_items:
        return

    text = "📋 Pending Tasks:\n" + "\n".join(due_items[:10])
    if len(due_items) > 10:
        text += f"\n... and {len(due_items) - 10} more."

    try:
        from isaac.interfaces.telegram_gateway import send_notification
        send_notification(text)
    except Exception:
        pass


def memory_consolidation_job() -> None:
    """Periodic memory maintenance — deduplicate / compress episodic entries."""
    try:
        from isaac.memory.manager import get_memory_manager

        mm = get_memory_manager()
        # Access episodic layer and compact if available
        episodic = mm.episodic
        if hasattr(episodic, "compact"):
            episodic.compact()
            logger.info("Episodic memory compacted.")
    except Exception as exc:
        logger.debug("Memory consolidation skipped: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------


def start_scheduler() -> None:
    """Start the APScheduler background scheduler.

    Jobs are registered based on the ``heartbeat_interval_minutes``
    setting.  Safe to call multiple times — only one scheduler runs.
    """
    global _scheduler

    if _scheduler is not None:
        logger.debug("Scheduler already running.")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore[import-untyped]
        from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("APScheduler not installed — heartbeat disabled.")
        return

    settings = _get_settings()
    interval = getattr(settings, "heartbeat_interval_minutes", 30)

    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        heartbeat_job,
        IntervalTrigger(minutes=interval),
        id="heartbeat",
        replace_existing=True,
    )

    scheduler.add_job(
        tasks_scan_job,
        IntervalTrigger(minutes=max(interval, 60)),
        id="tasks_scan",
        replace_existing=True,
    )

    scheduler.add_job(
        memory_consolidation_job,
        IntervalTrigger(hours=6),
        id="memory_consolidation",
        replace_existing=True,
    )

    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Heartbeat scheduler started (interval=%d min, tasks_scan=%d min).",
        interval,
        max(interval, 60),
    )


def stop_scheduler() -> None:
    """Shutdown the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
        _scheduler = None
        logger.info("Heartbeat scheduler stopped.")

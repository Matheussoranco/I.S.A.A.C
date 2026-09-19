"""Proactive Heartbeat Scheduler — periodic self-initiated actions.

Uses APScheduler to run background jobs:
- **Heartbeat**: periodic status check + Telegram notification.
- **TASKS.md scan**: parse ``~/.isaac/TASKS.md`` and surface due items.
- **Memory consolidation**: periodic compression / dedup of episodic memory.

The scheduler is started once by the CLI or ``__main__`` entry point.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

_scheduler: Any | None = None
_scheduler_lock = RLock()


def _synchronized(function: Any) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _scheduler_lock:
            return function(*args, **kwargs)

    return wrapped


def _get_settings() -> Any:
    from isaac.config.settings import get_settings

    return get_settings()


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------


def heartbeat_job() -> None:
    """Periodic heartbeat — logs status and notifies operator."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
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


def improvement_job() -> None:
    """Periodic self-improvement cycle — only runs when explicitly enabled."""
    try:
        from isaac.improvement import run_improvement_cycle

        result = run_improvement_cycle()
        promoted = sum(1 for d in result.curation_decisions if d.get("action") == "promote")
        deprecated = sum(1 for d in result.curation_decisions if d.get("action") == "deprecate")
        logger.info(
            "Improvement cycle: promoted=%d deprecated=%d critique=%r",
            promoted,
            deprecated,
            (result.critique_summary or "")[:120],
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Improvement cycle failed: %s", exc)


def _send_reminder_notification(text: str, chat_id: str, token: str) -> None:
    import httpx

    try:
        response = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000]},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("Telegram did not confirm delivery")
    except Exception:
        raise RuntimeError("Telegram reminder delivery failed") from None


def reminders_job(*, isaac_home: Path | None = None) -> None:
    from isaac.memory.reminders import claim_due_reminders, finish_reminder_notification

    settings = _get_settings()
    home = isaac_home if isaac_home is not None else settings.isaac_home
    token = getattr(settings, "telegram_bot_token", "")
    recipients = getattr(settings, "telegram_allowed_users", [])
    channels = [("log", "")]
    if token and recipients:
        channels.extend((f"telegram:{user}", str(user)) for user in dict.fromkeys(recipients))
    errors = []
    for channel, recipient in channels:
        for claim in claim_due_reminders(channel, isaac_home=home):
            reminder = claim.reminder
            text = f"⏰ Due reminder [{reminder.id}]: {reminder.text}"
            try:
                if channel == "log":
                    logger.info(text)
                else:
                    _send_reminder_notification(text, recipient, token)
            except Exception as exc:
                finish_reminder_notification(
                    reminder.id,
                    channel,
                    claim.token,
                    delivered=False,
                    error=str(exc),
                    isaac_home=home,
                )
                errors.append(f"{reminder.id} ({channel}): {exc}")
            else:
                finish_reminder_notification(
                    reminder.id,
                    channel,
                    claim.token,
                    delivered=True,
                    isaac_home=home,
                )
    if errors:
        raise RuntimeError("Reminder delivery failed: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------


@_synchronized
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
        from apscheduler.schedulers.background import (
            BackgroundScheduler,  # type: ignore[import-untyped]
        )
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

    scheduler.add_job(
        reminders_job,
        IntervalTrigger(minutes=5),
        id="reminders",
        replace_existing=True,
    )

    if getattr(settings, "improvement_enabled", False):
        scheduler.add_job(
            improvement_job,
            IntervalTrigger(minutes=settings.improvement_interval_minutes),
            id="self_improvement",
            replace_existing=True,
        )
        logger.info(
            "Self-improvement scheduled every %d minutes.",
            settings.improvement_interval_minutes,
        )

    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "Heartbeat scheduler started (interval=%d min, tasks_scan=%d min).",
        interval,
        max(interval, 60),
    )


@_synchronized
def stop_scheduler() -> None:
    """Shutdown the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        with contextlib.suppress(Exception):
            _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Heartbeat scheduler stopped.")


@_synchronized
def register_callback(
    callback: Any,
    *,
    interval_seconds: int = 900,
    name: str = "",
) -> bool:
    """Attach an arbitrary callable to the running scheduler.

    Used by self-improvement modules (``memory.consolidation``,
    ``meta.curriculum``) to run on idle cycles. Starts the scheduler if
    needed. Returns ``True`` on success.
    """
    global _scheduler
    if _scheduler is None:
        try:
            start_scheduler()
        except Exception as exc:
            logger.debug("scheduler start failed: %s", exc)
            return False
    if _scheduler is None:
        return False
    try:
        from apscheduler.triggers.interval import IntervalTrigger  # type: ignore[import-untyped]
    except ImportError:
        return False
    job_id = name or f"cb_{id(callback):x}"
    _scheduler.add_job(
        callback,
        IntervalTrigger(seconds=interval_seconds),
        id=job_id,
        replace_existing=True,
    )
    logger.info("Registered scheduler callback %r every %ds.", job_id, interval_seconds)
    return True

"""Cron Engine — schedule and execute recurring background tasks.

Tasks are defined by simple dataclasses stored in a JSON manifest at
``~/.isaac/cron_tasks.json``.  The engine runs in a daemon thread and
uses ``croniter`` to evaluate cron expressions.

Features
--------
* Add / remove / list / pause tasks via the public API.
* Each task is a free-form *command description* that gets routed through
  the I.S.A.A.C. cognitive graph (or a simpler connector call).
* PID-file based singleton guard (``~/.isaac/cron.pid``).
* Execution log at ``~/.isaac/cron_execution.log``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CronTask:
    """A single cron-scheduled task."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = ""
    schedule: str = "0 * * * *"  # cron expression (every hour default)
    command: str = ""  # free-form description or connector call
    enabled: bool = True
    last_run: str = ""  # ISO datetime
    last_status: str = ""  # "ok" | "error" | ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _task_from_dict(d: dict[str, Any]) -> CronTask:
    return CronTask(**{k: v for k, v in d.items() if k in CronTask.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"


def _manifest_path() -> Path:
    return _isaac_home() / "cron_tasks.json"


def _pid_path() -> Path:
    return _isaac_home() / "cron.pid"


def _log_path() -> Path:
    return _isaac_home() / "cron_execution.log"


def _append_log(task_id: str, status: str, detail: str = "") -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts}  task={task_id}  status={status}"
        if detail:
            line += f"  detail={detail[:300]}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_tasks() -> list[CronTask]:
    """Load tasks from the JSON manifest."""
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [_task_from_dict(d) for d in data]
    except Exception as exc:
        logger.error("Failed to load cron tasks: %s", exc)
        return []


def save_tasks(tasks: list[CronTask]) -> None:
    """Persist tasks to the JSON manifest."""
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(t) for t in tasks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CRUD API
# ---------------------------------------------------------------------------


def add_task(
    name: str,
    schedule: str,
    command: str,
    *,
    enabled: bool = True,
) -> CronTask:
    """Create and persist a new cron task.  Returns the created task."""
    task = CronTask(name=name, schedule=schedule, command=command, enabled=enabled)
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)
    logger.info("Cron task added: %s (%s)", task.id, name)
    return task


def remove_task(task_id: str) -> bool:
    """Remove a task by id.  Returns True if found and removed."""
    tasks = load_tasks()
    filtered = [t for t in tasks if t.id != task_id]
    if len(filtered) == len(tasks):
        return False
    save_tasks(filtered)
    logger.info("Cron task removed: %s", task_id)
    return True


def pause_task(task_id: str) -> bool:
    """Disable a task by id.  Returns True if found."""
    tasks = load_tasks()
    for t in tasks:
        if t.id == task_id:
            t.enabled = False
            save_tasks(tasks)
            return True
    return False


def resume_task(task_id: str) -> bool:
    """Re-enable a task by id.  Returns True if found."""
    tasks = load_tasks()
    for t in tasks:
        if t.id == task_id:
            t.enabled = True
            save_tasks(tasks)
            return True
    return False


def list_tasks() -> list[dict[str, Any]]:
    """Return all tasks as dicts."""
    return [asdict(t) for t in load_tasks()]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute_task(task: CronTask) -> str:
    """Execute a single cron task.  Returns status string."""
    logger.info("Cron executing: %s (%s)", task.id, task.name)

    # Try connector-style execution first
    try:
        from isaac.skills.connectors.registry import run_connector

        # If the command looks like `connector:action key=val`, parse it
        if ":" in task.command and not task.command.startswith("http"):
            parts = task.command.split(":", 1)
            connector_name = parts[0].strip()
            rest = parts[1].strip()

            kwargs: dict[str, Any] = {}
            for token in rest.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    kwargs[k] = v
                else:
                    kwargs.setdefault("query", token)

            result = run_connector(connector_name, **kwargs)
            _append_log(task.id, "ok", json.dumps(result)[:300])
            return "ok"
    except Exception as exc:
        logger.debug("Connector-style cron exec failed: %s", exc)

    # Fallback: run as shell command if shell connector is available
    try:
        from isaac.skills.connectors.registry import run_connector

        result = run_connector("shell", command=task.command)
        status = "ok" if result.get("exit_code", 1) == 0 else "error"
        _append_log(task.id, status, json.dumps(result)[:300])
        return status
    except Exception as exc:
        _append_log(task.id, "error", str(exc))
        return "error"


def _is_due(task: CronTask) -> bool:
    """Check if *task* is due based on its cron schedule and last_run."""
    try:
        from croniter import croniter  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("croniter not installed — cron tasks will not fire.")
        return False

    now = datetime.now(timezone.utc)
    if task.last_run:
        last = datetime.fromisoformat(task.last_run)
    else:
        # Never ran: consider it due immediately
        return True

    cron = croniter(task.schedule, last)
    next_run = cron.get_next(datetime)
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=timezone.utc)
    return now >= next_run


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------

_stop_event = threading.Event()
_daemon_thread: threading.Thread | None = None


def _daemon_loop(poll_seconds: int = 30) -> None:
    """Main loop: polls tasks file and executes due tasks."""
    logger.info("Cron daemon loop started (poll=%ds).", poll_seconds)
    while not _stop_event.is_set():
        try:
            tasks = load_tasks()
            for task in tasks:
                if _stop_event.is_set():
                    break
                if not task.enabled:
                    continue
                if _is_due(task):
                    status = _execute_task(task)
                    # reload & update persisted state
                    all_tasks = load_tasks()
                    for t in all_tasks:
                        if t.id == task.id:
                            t.last_run = datetime.now(timezone.utc).isoformat()
                            t.last_status = status
                            break
                    save_tasks(all_tasks)
        except Exception as exc:
            logger.error("Cron daemon tick error: %s", exc)

        _stop_event.wait(poll_seconds)

    logger.info("Cron daemon loop stopped.")


def start_cron_daemon(poll_seconds: int = 30) -> None:
    """Start the cron daemon in a background thread.

    Uses a PID file to prevent multiple daemons.
    """
    global _daemon_thread

    pid_path = _pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing PID
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text().strip())
            # On Windows, os.kill(pid, 0) checks existence
            os.kill(existing_pid, 0)
            logger.warning("Cron daemon already running (PID %d).", existing_pid)
            return
        except (OSError, ValueError):
            # Stale PID file
            pid_path.unlink(missing_ok=True)

    _stop_event.clear()
    _daemon_thread = threading.Thread(
        target=_daemon_loop,
        args=(poll_seconds,),
        daemon=True,
        name="isaac-cron",
    )
    _daemon_thread.start()

    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    logger.info("Cron daemon started (PID %d).", os.getpid())


def stop_cron_daemon() -> None:
    """Signal the cron daemon to stop."""
    global _daemon_thread
    _stop_event.set()
    if _daemon_thread is not None:
        _daemon_thread.join(timeout=5)
        _daemon_thread = None

    pid_path = _pid_path()
    pid_path.unlink(missing_ok=True)
    logger.info("Cron daemon stopped.")


def is_cron_running() -> bool:
    """Return True if the cron daemon thread is alive."""
    return _daemon_thread is not None and _daemon_thread.is_alive()

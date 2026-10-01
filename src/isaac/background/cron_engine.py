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
import re
import threading
import time as _time

try:  # POSIX advisory locking; on Windows tick dedup falls back to best-effort create
    import fcntl as _fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # type: ignore[assignment]
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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
    approved: bool = False
    """Explicit authorization for unattended connector/host execution."""
    last_run: str = ""  # ISO datetime
    last_status: str = ""  # "ok" | "error" | ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    # Phase 4.2.1: colloquial schedule preserved alongside cron expression
    schedule_nl: str = ""
    # Phase 4.2.2: skill attached to this job
    skill: str = ""
    # Phase 4.2.3 / 4.2.4: model / provider override for agent runs
    model: str = ""
    provider: str = ""
    """Optional pre-run hook; when ``no_agent`` is set the job runs *only*
    the script and skips agent/connector routing."""
    pre_run_script: str = ""
    no_agent: bool = False
    """Phase 4.2.5: list of task ids whose latest run output is injected
    into this job's command as ``{context}``."""
    context_from: list[str] = field(default_factory=list)
    """Multi-platform delivery targets, e.g. ``["telegram", "discord"]``.
    Results are forwarded to ``isaac.interfaces.delivery.deliver`` when
    the module is available; otherwise they are logged."""
    deliver_to: list[str] = field(default_factory=list)
    timeout_seconds: int = 180  # roadmap 4.2: hard interrupt after 3 min


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
        ts = datetime.now(UTC).isoformat()
        line = f"{ts}  task={task_id}  status={status}"
        if detail:
            line += f"  detail={detail[:300]}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Natural-language schedules (roadmap 4.2.1)
# ---------------------------------------------------------------------------

_WEEKDAY_CRON = {
    "monday": "* * * * 1",
    "tuesday": "* * * * 2",
    "wednesday": "* * * * 3",
    "thursday": "* * * * 4",
    "friday": "* * * * 5",
    "saturday": "* * * * 6",
    "sunday": "* * * * 0",
}


from isaac.scheduler.cron_parser import schedule_from_nl
from isaac.background.cron_locks import _acquire_lock, _release_lock


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

_TASK_KEY_RE = re.compile(r"^[\w@]{1,64}$", re.UNICODE)


def _step(task_id: str, note: str) -> None:
    """Append a lifecycle note to the execution log for *task_id*.

    The task id is validated and the note capped so log output stays
    single-line and injection-safe."""
    if not _TASK_KEY_RE.fullmatch(task_id):
        task_id = task_id[:64]
    _append_log(task_id, "step", note)


# ---------------------------------------------------------------------------
# CRUD API
# ---------------------------------------------------------------------------


def add_task(
    name: str,
    schedule: str,
    command: str,
    *,
    enabled: bool = True,
    approved: bool = False,
    skill: str = "",
    model: str = "",
    provider: str = "",
    pre_run_script: str = "",
    no_agent: bool = False,
    context_from: list[str] | None = None,
    deliver_to: list[str] | None = None,
    timeout_seconds: int = 180,
) -> CronTask:
    """Create and persist a new cron task.  Returns the created task.

    ``schedule`` accepts a 5-field cron expression or a short natural
    language schedule ("every 30m", "every monday 9am"); the latter is
    converted via :func:`schedule_from_nl` and the original text kept in
    ``CronTask.schedule_nl``.
    """
    schedule_nl = ""
    if not re.fullmatch(r"\S+(\s+\S+){4,5}", schedule.strip()):
        schedule_nl = schedule
        schedule = schedule_from_nl(schedule)
    task = CronTask(
        name=name,
        schedule=schedule,
        schedule_nl=schedule_nl,
        command=command,
        enabled=enabled,
        approved=approved,
        skill=skill,
        model=model,
        provider=provider,
        pre_run_script=pre_run_script,
        no_agent=no_agent,
        context_from=list(context_from or []),
        deliver_to=list(deliver_to or []),
        timeout_seconds=timeout_seconds,
    )
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


# ---------------------------------------------------------------------------
# Phase 4.2 execution helpers
# ---------------------------------------------------------------------------

# Long-running records: task_id -> {"pid": int, "future": Future, "cancel_after": float}
_running: dict[str, dict[str, Any]] = {}


def _outputs_path() -> Path:
    return _isaac_home() / "cron_outputs.json"


def record_run_output(task_id: str, task_name: str, result_text: str) -> None:
    """Store (rolling, per-task) the latest run output so other jobs can
    chain on it via ``context_from``."""
    if not _TASK_KEY_RE.fullmatch(task_id):
        task_id = task_id[:64]
    path = _outputs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[task_id] = {
        "task_id": task_id,
        "name": task_name,
        "output": result_text[:4000],
        "ts": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _gather_context(task: CronTask) -> str:
    """Collect latest outputs from ``context_from`` jobs (4.2.5)."""
    if not task.context_from:
        return ""
    path = _outputs_path()
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    lines = []
    for tid in task.context_from:
        rec = data.get(tid)
        if rec:
            lines.append(f"[{rec.get('name', tid)}]\n{rec.get('output', '')}")
    return "\n\n".join(lines)


def run_agent_job(
    command: str,
    *,
    skill: str = "",
    model: str = "",
    provider: str = "",
) -> str:
    """Run *command* through the agent graph with optional skill / model /
    provider overrides (4.2.2-4.2.4).  Returns text output or raises."""
    from isaac.agents.runner import run_text  # noqa: PLC0415 - lazy heavy import

    result = run_text(
        command,
        skill=skill or None,
        model=model or None,
        provider=provider or None,
    )
    return str(result)


def _run_script(script: str, timeout: int) -> str:
    """Run the pre-run script / script-only job with a hard timeout."""
    import subprocess

    proc = subprocess.run(
        script,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"script exited {proc.returncode}: {out[:300]}")
    return out


def mark_running(task_id: str, pid: int, future: Any, cancel_after: float) -> None:
    if _TASK_KEY_RE.fullmatch(task_id):
        _running[task_id] = {"pid": pid, "future": future, "cancel_after": cancel_after}


def reap_long_running(now: float | None = None) -> list[str]:
    """Hard interrupt (roadmap 4.2): fence tasks running longer than their
    ``cancel_after`` deadline.  Returns ids that were signalled."""
    now = now if now is not None else _time.time()
    killed = []
    for tid, rec in list(_running.items()):
        if now > rec["cancel_after"]:
            try:
                rec["future"].cancel()
            except Exception:
                pass
            _step(tid, "hard_interrupted(timeout)")
            killed.append(tid)
            _running.pop(tid, None)
    return killed


def _deliver(task: CronTask, status: str, output: str) -> None:
    """Multi-platform delivery targets per job (roadmap 4.2)."""
    text = f"[cron:{task.name or task.id}] {status}: {output[:800]}"
    for target in task.deliver_to:
        try:
            from isaac.interfaces.delivery import deliver  # type: ignore[attr-defined]

            deliver(target, text)
        except Exception as exc:
            logger.debug("cron delivery to %s failed: %s", target, exc)
            _step(task.id, f"deliver_unavailable:{target}")


def _execute_task(task: CronTask) -> str:
    """Execute a single cron task.  Returns status string."""
    logger.info("Cron executing: %s (%s)", task.id, task.name)

    if not task.approved:
        detail = "unattended execution was not explicitly approved when the task was created"
        _append_log(task.id, "approval_required", detail)
        return "approval_required"

    # --- Phase 4.2: context chaining -------------------------------------
    context = _gather_context(task)
    if "{context}" in task.command:
        command = task.command.replace("{context}", context or "")
    else:
        command = (task.command + "\n\nContext from previous jobs:\n" + context) if context else task.command

    timeout = max(int(task.timeout_seconds or 180), 1)

    # --- Phase 4.2: pre-run script / script-only jobs ---------------------
    script_out = ""
    if task.pre_run_script:
        try:
            _step(task.id, "pre_run_script:start")
            script_out = _run_script(task.pre_run_script, timeout=timeout)
            _step(task.id, "pre_run_script:ok")
        except Exception as exc:
            _append_log(task.id, "error", f"pre_run_script failed: {exc}")
            record_run_output(task.id, task.name, f"pre_run_script error: {exc}")
            return "error"
        if task.no_agent:
            record_run_output(task.id, task.name, script_out)
            _append_log(task.id, "ok", "script-only job (no_agent)")
            _deliver(task, "ok", script_out)
            return "ok"
        if script_out.strip():
            command = f"{command}\n\nPre-run output:\n{script_out.strip()}"

    # --- Phase 4.2: agent-run path with per-job overrides -----------------
    if task.skill or task.model or task.provider:
        try:
            output = run_agent_job(
                command,
                skill=task.skill,
                model=task.model,
                provider=task.provider,
            )
            record_run_output(task.id, task.name, output)
            _append_log(task.id, "ok", "agent job completed")
            _deliver(task, "ok", output)
            return "ok"
        except Exception as exc:
            _append_log(task.id, "error", f"agent job failed: {exc}")
            record_run_output(task.id, task.name, f"agent error: {exc}")
            return "error"

    def issue_token(connector_name: str) -> str:
        from isaac.security.capabilities import get_token_store

        token = get_token_store().issue(
            connector_name,
            action="execute",
            ttl_hours=1 / 60,
            issued_by=f"cron:{task.id}",
            max_uses=1,
        )
        return token.token_id

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

            if connector_name == "shell":
                from isaac.security.constitution import review

                command = str(kwargs.get("command", ""))
                decision = review("shell", command, context={"cron_task": task.id}, use_llm=False)
                if not decision.allow:
                    _append_log(task.id, "blocked", decision.reason)
                    return "blocked"

            result = run_connector(
                connector_name,
                capability_token=issue_token(connector_name),
                **kwargs,
            )
            status = "error" if result.get("error") else "ok"
            _append_log(task.id, status, json.dumps(result)[:300])
            return status
    except Exception as exc:
        logger.debug("Connector-style cron exec failed: %s", exc)
        if ":" in task.command and not task.command.startswith("http"):
            _append_log(task.id, "error", str(exc))
            return "error"

    # Fallback: run as shell command if shell connector is available
    try:
        from isaac.security.constitution import review
        from isaac.skills.connectors.registry import run_connector

        decision = review(
            "shell",
            task.command,
            context={"cron_task": task.id},
            use_llm=False,
        )
        if not decision.allow:
            _append_log(task.id, "blocked", decision.reason)
            return "blocked"
        result = run_connector(
            "shell",
            capability_token=issue_token("shell"),
            command=task.command,
        )
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

    now = datetime.now(UTC)
    if task.last_run:
        last = datetime.fromisoformat(task.last_run)
    else:
        # Never ran: consider it due immediately
        return True

    cron = croniter(task.schedule, last)
    next_run = cron.get_next(datetime)
    if next_run.tzinfo is None:
        next_run = next_run.replace(tzinfo=UTC)
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
            reap_long_running()  # 4.2: hard interrupt watchdog
            tasks = load_tasks()
            for task in tasks:
                if _stop_event.is_set():
                    break
                if not task.enabled:
                    continue
                if _is_due(task):
                    if not _acquire_lock(task.id):
                        logger.info("Task %s is already running (lock held). Skipping.", task.id)
                        continue
                    try:
                        status = _execute_task(task)
                        # reload & update persisted state
                        all_tasks = load_tasks()
                        for t in all_tasks:
                            if t.id == task.id:
                                t.last_run = datetime.now(UTC).isoformat()
                                t.last_status = status
                                break
                        save_tasks(all_tasks)
                    finally:
                        _release_lock(task.id)
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

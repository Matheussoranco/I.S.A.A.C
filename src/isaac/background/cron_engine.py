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
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
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


def task_logs(task_id: str, limit: int = 200) -> list[str]:
    """Return the most recent execution-log lines for *task_id*."""
    path = _log_path()
    if not path.exists():
        return []
    marker = f"task={task_id}"
    matches: deque[str] = deque(maxlen=max(1, limit))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if marker in line:
                    matches.append(line.rstrip())
    except OSError:
        return []
    return list(matches)


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


from isaac.background.cron_locks import _acquire_lock, _release_lock  # noqa: E402
from isaac.scheduler.cron_parser import schedule_from_nl  # noqa: E402

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class CronManifestError(RuntimeError):
    """Raised when the cron manifest exists but cannot be decoded."""


def load_tasks() -> list[CronTask]:
    """Load tasks from the JSON manifest.

    A corrupt manifest raises instead of silently yielding an empty list,
    which would let the next write erase every scheduled job.
    """
    path = _manifest_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CronManifestError(f"cron manifest is unreadable ({path}): {exc}") from exc
    if not isinstance(data, list):
        raise CronManifestError(f"cron manifest must contain a list: {path}")
    try:
        return [_task_from_dict(d) for d in data]
    except Exception as exc:
        raise CronManifestError(f"cron manifest contains an invalid task: {exc}") from exc


def save_tasks(tasks: list[CronTask]) -> None:
    """Persist tasks atomically so a crash cannot truncate the manifest."""
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps([asdict(t) for t in tasks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temp, path)


@contextmanager
def _manifest_guard(timeout: float = 10.0) -> Iterator[None]:
    """Serialise read-modify-write cycles across CLI and daemon processes."""
    lock_path = _manifest_path().with_name("cron_tasks.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = _time.monotonic() + timeout
    while True:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                stale = _time.time() - lock_path.stat().st_mtime > 30
            except OSError:
                stale = False
            if stale:
                with suppress(OSError):
                    lock_path.unlink()
                continue
            if _time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for cron manifest lock: {lock_path}"
                ) from None
            _time.sleep(0.05)
    try:
        yield
    finally:
        with suppress(OSError):
            lock_path.unlink()


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
    with _manifest_guard():
        tasks = load_tasks()
        tasks.append(task)
        save_tasks(tasks)
    logger.info("Cron task added: %s (%s)", task.id, name)
    return task


def remove_task(task_id: str) -> bool:
    """Remove a task by id.  Returns True if found and removed."""
    with _manifest_guard():
        tasks = load_tasks()
        filtered = [t for t in tasks if t.id != task_id]
        if len(filtered) == len(tasks):
            return False
        save_tasks(filtered)
    logger.info("Cron task removed: %s", task_id)
    return True


def pause_task(task_id: str) -> bool:
    """Disable a task by id.  Returns True if found."""
    with _manifest_guard():
        tasks = load_tasks()
        for t in tasks:
            if t.id == task_id:
                t.enabled = False
                save_tasks(tasks)
                return True
    return False


def resume_task(task_id: str) -> bool:
    """Re-enable a task by id.  Returns True if found."""
    with _manifest_guard():
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
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


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
    timeout_seconds: int = 600,
) -> str:
    """Run *command* through the agent with optional skill / model /
    provider overrides (4.2.2-4.2.4).  Returns text output or raises."""
    from isaac.agents.agent_loop import build_default_agent
    from isaac.config.settings import get_settings

    settings = get_settings()
    llm = None
    if model or provider:
        from isaac.llm.provider import build_llm_for_profile

        llm = build_llm_for_profile(
            provider or settings.llm.llm_provider,
            model or settings.llm.model_name,
        )

    context = ""
    if skill:
        from isaac.memory.skill_library import SkillLibrary

        code = SkillLibrary(settings.skills_dir).get_code(skill)
        if not code:
            raise ValueError(f"Skill '{skill}' not found in the library.")
        context = f"Reusable helper skill `{skill}`:\n{code}"

    agent = build_default_agent(
        llm=llm,
        max_iterations=settings.graph.max_iterations,
        max_wall_seconds=float(max(int(timeout_seconds), 1)),
    )
    result = agent.run(command, context=context)
    return result.output or "(no output)"


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


def _deliver(task: CronTask, status: str, output: str) -> None:
    """Multi-platform delivery targets per job (roadmap 4.2)."""
    text = f"[cron:{task.name or task.id}] {status}: {output[:800]}"
    for target in task.deliver_to:
        try:
            from isaac.interfaces.delivery import deliver

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
        command = (
            (task.command + "\n\nContext from previous jobs:\n" + context)
            if context
            else task.command
        )

    timeout = max(int(task.timeout_seconds or 180), 1)

    from isaac.security.constitution import review

    decision = review("shell", task.command, context={"cron_task": task.id}, use_llm=False)
    if not decision.allow or decision.requires_approval:
        _append_log(task.id, "blocked", decision.reason)
        return "blocked"

    # --- Phase 4.2: pre-run script / script-only jobs ---------------------
    script_out = ""
    if task.pre_run_script:
        decision = review(
            "shell",
            task.pre_run_script,
            context={"cron_task": task.id},
            use_llm=False,
        )
        if not decision.allow or decision.requires_approval:
            _append_log(task.id, "blocked", decision.reason)
            return "blocked"
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

    parsed = _parse_connector_call(command)
    if parsed is not None:
        connector_name, kwargs = parsed
        try:
            from isaac.skills.connectors.registry import run_connector

            result = run_connector(
                connector_name,
                capability_token=issue_token(connector_name),
                **kwargs,
            )
            status = "error" if result.get("error") else "ok"
            _append_log(task.id, status, json.dumps(result)[:300])
            return status
        except Exception as exc:
            _append_log(task.id, "error", f"connector {connector_name} failed: {exc}")
            return "error"

    # Plain-language jobs go through the agent; the shell connector is only
    # used for an explicit ``shell:`` job.
    try:
        output = run_agent_job(
            command,
            skill=task.skill,
            model=task.model,
            provider=task.provider,
            timeout_seconds=timeout,
        )
        record_run_output(task.id, task.name, output)
        _append_log(task.id, "ok", "agent job completed")
        _deliver(task, "ok", output)
        return "ok"
    except Exception as exc:
        _append_log(task.id, "error", f"agent job failed: {exc}")
        record_run_output(task.id, task.name, f"agent error: {exc}")
        return "error"


def _parse_connector_call(command: str) -> tuple[str, dict[str, Any]] | None:
    """Parse ``connector:key=value ...`` only when the connector exists."""
    head, sep, rest = command.partition(":")
    if not sep:
        return None
    name = head.strip().lower()
    if not _TASK_KEY_RE.fullmatch(name):
        return None
    try:
        from isaac.skills.connectors.registry import get_registry

        if name not in get_registry():
            return None
    except Exception:
        return None
    try:
        tokens = _split_connector_args(rest.strip())
    except ValueError:
        return None
    kwargs: dict[str, Any] = {}
    query: list[str] = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key:
            kwargs[key] = value
        else:
            query.append(token)
    if query:
        kwargs.setdefault("query", " ".join(query))
    if name == "shell" and "command" not in kwargs and "query" in kwargs:
        kwargs["command"] = kwargs.pop("query")
    return name, kwargs


def _split_connector_args(text: str) -> list[str]:
    """Split ``key=value`` arguments while preserving quoted spaces/backslashes."""
    args: list[str] = []
    buf: list[str] = []
    quote = ""
    for char in text:
        if quote:
            if char == quote:
                quote = ""
            else:
                buf.append(char)
        elif char in "\"'":
            quote = char
        elif char.isspace():
            if buf:
                args.append("".join(buf))
                buf = []
        else:
            buf.append(char)
    if quote:
        raise ValueError("unbalanced quote in connector arguments")
    if buf:
        args.append("".join(buf))
    return args


def _is_due(task: CronTask) -> bool:
    """Check if *task* is due based on its cron schedule and last_run.

    Malformed schedules/timestamps disable that task instead of raising
    into the daemon loop (which would skip every later task).
    """
    try:
        from croniter import croniter  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("croniter not installed — cron tasks will not fire.")
        return False

    now = datetime.now(UTC)
    if not task.last_run:
        # Never ran: consider it due immediately
        return True
    try:
        last = datetime.fromisoformat(task.last_run)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        next_run = croniter(task.schedule, last).get_next(datetime)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning(
            "Cron task %s has an invalid schedule/timestamp (%s / %s): %s",
            task.id,
            task.schedule,
            task.last_run,
            exc,
        )
        return False
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
            tasks = load_tasks()
        except CronManifestError as exc:
            logger.error("Cron daemon cannot read its manifest: %s", exc)
            tasks = []

        for task in tasks:
            if _stop_event.is_set():
                break
            try:
                if not task.enabled:
                    continue
                if not _is_due(task):
                    continue
                if not _acquire_lock(task.id):
                    logger.info("Task %s is already running (lock held). Skipping.", task.id)
                    continue
                try:
                    status = _execute_task(task)
                finally:
                    _release_lock(task.id)
                with _manifest_guard():
                    all_tasks = load_tasks()
                    for saved in all_tasks:
                        if saved.id == task.id:
                            saved.last_run = datetime.now(UTC).isoformat()
                            saved.last_status = status
                            break
                    save_tasks(all_tasks)
            except Exception as exc:
                logger.error("Cron task %s failed: %s", task.id, exc, exc_info=True)

        _stop_event.wait(poll_seconds)

    logger.info("Cron daemon loop stopped.")


def _pid_alive(pid: int) -> bool:
    """Check whether *pid* refers to a running process (never signals it).

    ``os.kill(pid, 0)`` is unsafe on Windows: CPython implements it with
    ``TerminateProcess``, so probing a live daemon would kill it.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return int(ctypes.get_last_error()) == 5  # ERROR_ACCESS_DENIED = alive
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _claim_pid_file(path: Path) -> bool:
    """Atomically create the PID file, reclaiming it if its owner is dead."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_pid(path)
            if existing is not None and _pid_alive(existing):
                logger.warning("Cron daemon already running (PID %d).", existing)
                return False
            with suppress(OSError):
                path.unlink()
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        return True
    return False


def _release_pid_file(path: Path) -> None:
    if _read_pid(path) == os.getpid():
        path.unlink(missing_ok=True)


def start_cron_daemon(poll_seconds: int = 30) -> None:
    """Start the cron daemon as a background thread in this process.

    Uses an atomically-created PID file to prevent multiple daemons.
    """
    global _daemon_thread

    if _daemon_thread is not None and _daemon_thread.is_alive():
        logger.warning("Cron daemon thread is already running.")
        return
    if not _claim_pid_file(_pid_path()):
        return

    _stop_event.clear()
    thread = threading.Thread(
        target=_daemon_loop,
        args=(poll_seconds,),
        daemon=True,
        name="isaac-cron",
    )
    try:
        thread.start()
    except Exception:
        _release_pid_file(_pid_path())
        raise
    _daemon_thread = thread
    logger.info("Cron daemon started (PID %d).", os.getpid())


def run_cron_daemon(poll_seconds: int = 30) -> None:
    """Run the cron daemon in the foreground until stopped or signalled."""
    global _daemon_thread

    if not _claim_pid_file(_pid_path()):
        return
    import signal

    def _stop(_signum: int, _frame: Any) -> None:
        _stop_event.set()

    for signame in ("SIGTERM", "SIGINT", "SIGBREAK"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        with suppress(OSError, ValueError):
            signal.signal(sig, _stop)
    _stop_event.clear()
    try:
        _daemon_loop(poll_seconds)
    finally:
        _release_pid_file(_pid_path())
        _daemon_thread = None


def stop_cron_daemon() -> None:
    """Stop the in-process daemon and any daemon recorded in the PID file."""
    global _daemon_thread

    _stop_event.set()
    if _daemon_thread is not None:
        _daemon_thread.join(timeout=5)
        _daemon_thread = None

    pid_path = _pid_path()
    pid = _read_pid(pid_path)
    if pid is not None and pid != os.getpid() and _pid_alive(pid):
        with suppress(OSError):
            os.kill(pid, 15)
    if pid is not None:
        if pid == os.getpid():
            _release_pid_file(pid_path)
        else:
            pid_path.unlink(missing_ok=True)
    logger.info("Cron daemon stopped.")


def is_cron_running() -> bool:
    """Return True if a cron daemon is alive in this or another process."""
    if _daemon_thread is not None and _daemon_thread.is_alive():
        return True
    pid = _read_pid(_pid_path())
    return pid is not None and _pid_alive(pid)

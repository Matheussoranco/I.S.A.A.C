"""Kanban dispatcher — reclaim stale claims and dispatch ready tasks.

Runs in a loop (thread or CLI one-shot) and:

1. **Reclaims stale claims** — running tasks whose ``heartbeat_at`` is older
   than ``stale_seconds`` (default 5 minutes) are reset to ``ready`` so
   another agent can pick them up.
2. **Promotes ready tasks** — calling :meth:`KanbanBoard.list_ready` moves
   dependency-satisfied backlog cards to ``ready``.
3. **Dispatches to agents** — each ready task is claimed for a profile and
   handed to a *spawner* callable.  The default spawner launches an
   ``isaac run --profile <name>`` subprocess with a per-task isolated
   workspace directory under ``~/.isaac/kanban_workspaces/<task_id>/``.

Multi-gateway deployment: every gateway/instance runs its own dispatcher
against the same SQLite board; atomic claim semantics in
:meth:`KanbanBoard.claim_task` prevent double-dispatch.  Gateway agents call
``kanban_heartbeat`` regularly to keep claims alive.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from isaac.background.kanban import KanbanBoard, Task

logger = logging.getLogger(__name__)

DEFAULT_STALE_SECONDS = 300.0  # 5-minute claim timeout


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"


def task_workspace(task_id: str, base: Path | None = None) -> Path:
    """Return (creating) the isolated workspace dir for a task."""
    root = base if base is not None else _isaac_home() / "kanban_workspaces"
    ws = root / task_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def subprocess_spawner(task: Task, *, workspace: Path, profile: str) -> Any:
    """Spawn an ``isaac run`` agent subprocess in the task's workspace.

    Sets ``ISAAC_PROFILE`` so the spawned agent belongs to the owning
    profile, ``ISAAC_KANBAN_TASK`` so it knows which card it is working,
    and runs with ``cwd=workspace`` for per-task file isolation.
    """
    prompt = (
        f"You are working on kanban task [{task.id}] {task.title}.\n"
        f"Description: {task.description or '(none)'}\n"
        "Work inside this directory. Use kanban_heartbeat regularly while "
        "working, then kanban_complete with a summary of the result, or "
        "kanban_block if you are stuck."
    )
    env = dict(os.environ)
    env["ISAAC_PROFILE"] = profile
    env["ISAAC_KANBAN_TASK"] = task.id
    log_path = workspace / "agent.log"
    log_file = open(log_path, "ab")  # noqa: SIM115 — owned by subprocess
    proc = subprocess.Popen(
        [sys.executable, "-m", "isaac", "run", prompt, "--auto-approve"],
        cwd=str(workspace),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    logger.info("Dispatched task %s to profile %s (pid=%s)", task.id, profile, proc.pid)
    return proc


class KanbanDispatcher:
    """Promote, reclaim, and dispatch kanban tasks to profile-owned agents.

    Parameters
    ----------
    board:
        The shared :class:`KanbanBoard` instance.
    profiles:
        Profile names allowed to receive tasks.  ``None`` (default) accepts
        any assignee and falls back to the active profile for unassigned
        tasks.  An explicit list restricts dispatch to those profiles.
    max_concurrent:
        Upper bound of simultaneously running claims this dispatcher
        maintains.
    stale_seconds:
        Claims with no heartbeat for this long are reclaimed.
    spawner:
        Callable ``(task, workspace=Path, profile=str) -> Any`` used to
        launch an agent.  Defaults to :func:`subprocess_spawner`.
    workspace_root:
        Override the per-task workspace root (tests pass a tmp dir).
    """

    def __init__(
        self,
        board: KanbanBoard | None = None,
        *,
        profiles: list[str] | None = None,
        max_concurrent: int = 4,
        stale_seconds: float = DEFAULT_STALE_SECONDS,
        spawner: Callable[..., Any] | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.board = board or KanbanBoard()
        self.board.init_board()
        self.profiles = profiles
        self.max_concurrent = max(1, max_concurrent)
        self.stale_seconds = stale_seconds
        self.spawner = spawner or subprocess_spawner
        self.workspace_root = workspace_root
        self.processes: dict[str, Any] = {}

    # ------------------------------------------------------------------

    def _allowed(self, profile: str) -> bool:
        return self.profiles is None or profile in self.profiles

    def _resolve_profile(self, task: Task) -> str:
        if task.assignee:
            return task.assignee
        try:
            from isaac.config.profiles import get_active_profile

            return get_active_profile()
        except Exception:
            return "default"

    # ------------------------------------------------------------------

    def reclaim_stale(self, now: datetime | None = None) -> list[Task]:
        """Reset running tasks whose heartbeat went stale back to ``ready``.

        Returns the list of reclaimed tasks."""
        now = now or datetime.now(UTC)
        reclaimed: list[Task] = []
        for task in self.board.list_tasks(status="running"):
            raw = task.heartbeat_at or task.claimed_at
            if not raw:
                reclaimed.append(task)
                continue
            try:
                last = datetime.fromisoformat(raw)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
            except ValueError:
                reclaimed.append(task)
                continue
            if (now - last).total_seconds() > self.stale_seconds:
                reclaimed.append(task)
        for task in reclaimed:
            idem = self.board._update(
                task.id,
                status="ready",
                claimed_at="",
                heartbeat_at="",
            )
            self.board.add_comment(
                task.id,
                author="dispatcher",
                body=f"claim reclaimed after {self.stale_seconds:.0f}s without heartbeat",
            )
            self.processes.pop(task.id, None)
            logger.warning("Reclaimed stale claim on task %s (id=%s)", idem.id, task.assignee)
        return reclaimed

    # ------------------------------------------------------------------

    def dispatch_once(self) -> list[str]:
        """One dispatch pass: reclaim stale, promote ready, spawn agents.

        Returns the ids of tasks dispatched during this pass."""
        self.reclaim_stale()
        ready = self.board.list_ready()
        # free slots: running tasks this dispatcher would own minus reaped ones
        running = [t for t in self.board.list_tasks(status="running") if self._allowed(t.assignee)]
        slots = max(0, self.max_concurrent - len(running))
        dispatched: list[str] = []
        for task in ready:
            if slots <= 0:
                break
            if task.assignee and not self._allowed(task.assignee):
                continue  # reserved for a profile we don't serve
            profile = self._resolve_profile(task)
            try:
                claimed = self.board.claim_task(task.id, profile)
            except Exception as exc:  # lost the race to another gateway
                logger.debug("claim race lost on %s: %s", task.id, exc)
                continue
            ws = task_workspace(task.id, self.workspace_root)
            self.board._update(task.id, workspace=str(ws))
            try:
                self.processes[task.id] = self.spawner(claimed, workspace=ws, profile=profile)
            except Exception as exc:
                logger.error("spawn failed for task %s: %s", task.id, exc)
                self.board.block_task(task.id, reason=f"spawn failed: {exc}")
                continue
            dispatched.append(task.id)
            slots -= 1
        return dispatched

    # ------------------------------------------------------------------

    def run_forever(
        self,
        interval: float = 10.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Dispatch loop until ``stop_event`` is set (or forever)."""
        stop = stop_event or threading.Event()
        while not stop.is_set():
            try:
                self.dispatch_once()
            except Exception:
                logger.exception("dispatch pass failed")
            stop.wait(interval)


__all__ = [
    "DEFAULT_STALE_SECONDS",
    "KanbanDispatcher",
    "subprocess_spawner",
    "task_workspace",
]

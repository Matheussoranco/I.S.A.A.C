"""Kanban dispatcher — background worker assignment and task delivery.

The dispatcher runs as a background task that:
1. Periodically reclaims stale claims
2. Finds ready tasks that are not claimed
3. Assigns them to available workers based on profile
4. Delivers tasks to workers via a gateway (HTTP, message queue, etc.)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from isaac.kanban import (
    get_tasks, get_active_claim, reclaim_stale_claims, claim_task,
    get_board, get_columns, Task, Claim,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerProfile:
    """Defines a worker's capabilities and current status."""
    worker_id: str
    profile: str  # e.g., "coder", "researcher", "file_organizer"
    max_concurrent: int = 1
    active_task_count: int = 0
    last_heartbeat: float = 0.0
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_available(self) -> bool:
        return self.active_task_count < self.max_concurrent

    @property
    def is_alive(self, timeout_seconds: float = 300.0) -> bool:
        return (time.time() - self.last_heartbeat) < timeout_seconds


@dataclass
class DispatchConfig:
    """Configuration for the dispatcher."""
    board_name: str = "default"
    stale_threshold_minutes: int = 60
    poll_interval_seconds: float = 30.0
    max_tasks_per_poll: int = 10
    worker_timeout_seconds: float = 300.0
    db_path: Path | None = None


class WorkerRegistry:
    """Registry of available workers and their profiles."""

    def __init__(self):
        self._workers: dict[str, WorkerProfile] = {}
        self._lock = threading.Lock()

    def register(self, worker: WorkerProfile) -> None:
        with self._lock:
            self._workers[worker.worker_id] = worker

    def unregister(self, worker_id: str) -> None:
        with self._lock:
            self._workers.pop(worker_id, None)

    def heartbeat(self, worker_id: str) -> bool:
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                return True
            return False

    def get_available_workers(self, profile: str, timeout_seconds: float = 300.0) -> list[WorkerProfile]:
        with self._lock:
            now = time.time()
            return [
                w for w in self._workers.values()
                if w.profile == profile
                and w.is_available
                and (now - w.last_heartbeat) < timeout_seconds
            ]

    def get_worker(self, worker_id: str) -> WorkerProfile | None:
        with self._lock:
            return self._workers.get(worker_id)

    def increment_task_count(self, worker_id: str) -> bool:
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].active_task_count += 1
                return True
            return False

    def decrement_task_count(self, worker_id: str) -> bool:
        with self._lock:
            if worker_id in self._workers and self._workers[worker_id].active_task_count > 0:
                self._workers[worker_id].active_task_count -= 1
                return True
            return False


# Global worker registry
_worker_registry = WorkerRegistry()


def get_worker_registry() -> WorkerRegistry:
    """Get the global worker registry."""
    return _worker_registry


class TaskGateway:
    """Abstract base for task delivery gateways."""

    def deliver(self, task: Task, worker: WorkerProfile, claim: Claim) -> bool:
        """Deliver a task to a worker. Return True if successful."""
        raise NotImplementedError


class LoggingGateway(TaskGateway):
    """Simple gateway that logs task delivery (for testing/development)."""

    def deliver(self, task: Task, worker: WorkerProfile, claim: Claim) -> bool:
        logger.info(
            "DELIVER: task=%s worker=%s profile=%s claim=%s",
            task.id, worker.worker_id, worker.profile, claim.id
        )
        return True


class HTTPGateway(TaskGateway):
    """HTTP-based gateway that POSTs task to a worker endpoint."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def deliver(self, task: Task, worker: WorkerProfile, claim: Claim) -> bool:
        import requests
        try:
            payload = {
                "task_id": task.id,
                "task_title": task.title,
                "task_description": task.description,
                "worker_id": worker.worker_id,
                "claim_id": claim.id,
                "profile": worker.profile,
            }
            resp = requests.post(
                f"{self.base_url}/tasks/assign",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                logger.info("HTTP delivered task %s to worker %s", task.id, worker.worker_id)
                return True
            else:
                logger.error("HTTP delivery failed: %s %s", resp.status_code, resp.text)
                return False
        except Exception as e:
            logger.error("HTTP delivery error for task %s: %s", task.id, e)
            return False


class Dispatcher:
    """Background dispatcher that assigns tasks to workers."""

    def __init__(
        self,
        config: DispatchConfig,
        gateway: TaskGateway | None = None,
        worker_registry: WorkerRegistry | None = None,
    ):
        self.config = config
        self.gateway = gateway or LoggingGateway()
        self.worker_registry = worker_registry or _worker_registry
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Dispatcher started for board '%s'", self.config.board_name)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Dispatcher stopped")

    def _run_loop(self) -> None:
        from isaac.kanban import init_db
        from isaac.config.settings import get_settings

        db_path = self.config.db_path or get_settings().isaac_home / "kanban.db"
        init_db(db_path)

        while not self._stop_event.is_set():
            try:
                self._dispatch_cycle(db_path)
            except Exception as e:
                logger.exception("Dispatcher cycle error: %s", e)

            # Wait for next poll interval or stop signal
            self._stop_event.wait(self.config.poll_interval_seconds)

    def _dispatch_cycle(self, db_path: Path) -> None:
        # 1. Reclaim stale claims
        reclaimed = reclaim_stale_claims(db_path)
        if reclaimed:
            logger.info("Reclaimed %d stale claims", len(reclaimed))

        # 2. Find ready tasks that are not claimed
        board = get_board(db_path, self.config.board_name)
        if not board:
            logger.warning("Board not found: %s", self.config.board_name)
            return

        # Find the "Ready" column
        columns = get_columns(db_path, board.id)
        ready_col = next((c for c in columns if c.name.lower() == "ready"), None)
        if not ready_col:
            # Fallback to first column after Backlog
            ready_col = next((c for c in columns if c.name.lower() != "backlog"), None)
        if not ready_col:
            return

        ready_tasks = get_tasks(db_path, board.id, ready_col.id)
        unclaimed_ready = [
            t for t in ready_tasks
            if get_active_claim(db_path, t.id) is None
        ][:self.config.max_tasks_per_poll]

        if not unclaimed_ready:
            return

        # 3. Assign tasks to available workers
        # Group tasks by profile (using task tags or metadata)
        for task in unclaimed_ready:
            self._assign_task(db_path, task)

    def _assign_task(self, db_path: Path, task: Task) -> bool:
        """Try to assign a task to an available worker."""
        # Determine required profile from task tags or default to "general"
        profile = "general"
        if task.tag_list:
            # Use first tag as profile hint
            profile = task.tag_list[0]

        workers = self.worker_registry.get_available_workers(
            profile, self.config.worker_timeout_seconds
        )
        if not workers:
            # No workers available for this profile
            return False

        # Pick the worker with the fewest active tasks
        worker = min(workers, key=lambda w: w.active_task_count)

        # Claim the task
        try:
            claim = claim_task(
                db_path, task.id, worker.worker_id, worker.profile,
                self.config.stale_threshold_minutes
            )
        except ValueError:
            # Task already claimed
            return False

        # Update worker task count
        self.worker_registry.increment_task_count(worker.worker_id)

        # Deliver via gateway
        success = self.gateway.deliver(task, worker, claim)
        if not success:
            # Release claim on delivery failure
            from isaac.kanban import release_task_claims
            release_task_claims(db_path, task.id)
            self.worker_registry.decrement_task_count(worker.worker_id)
            return False

        logger.info("Assigned task %s to worker %s (%s)", task.id, worker.worker_id, worker.profile)
        return True

    def complete_task(self, worker_id: str, task_id: str, success: bool = True) -> None:
        """Called by worker when task is complete."""
        self.worker_registry.decrement_task_count(worker_id)
        # Release the claim
        from isaac.kanban import release_task_claims
        from isaac.config.settings import get_settings
        db_path = self.config.db_path or get_settings().isaac_home / "kanban.db"
        release_task_claims(db_path, task_id)
        logger.info("Task %s completed by worker %s (success=%s)", task_id, worker_id, success)


# Convenience functions for global dispatcher

_global_dispatcher: Dispatcher | None = None
_dispatcher_lock = threading.Lock()


def get_dispatcher() -> Dispatcher | None:
    """Get the global dispatcher instance."""
    global _global_dispatcher
    with _dispatcher_lock:
        return _global_dispatcher


def start_dispatcher(config: DispatchConfig | None = None, gateway: TaskGateway | None = None) -> Dispatcher:
    """Start the global dispatcher."""
    global _global_dispatcher
    with _dispatcher_lock:
        if _global_dispatcher is not None:
            return _global_dispatcher
        _global_dispatcher = Dispatcher(config or DispatchConfig(), gateway)
        _global_dispatcher.start()
        return _global_dispatcher


def stop_dispatcher(timeout: float = 10.0) -> None:
    """Stop the global dispatcher."""
    global _global_dispatcher
    with _dispatcher_lock:
        if _global_dispatcher:
            _global_dispatcher.stop(timeout)
            _global_dispatcher = None


def run_dispatcher_once(config: DispatchConfig | None = None) -> int:
    """Run a single dispatch cycle synchronously. Returns number of tasks assigned."""
    from isaac.kanban import init_db
    from isaac.config.settings import get_settings

    cfg = config or DispatchConfig()
    db_path = cfg.db_path or get_settings().isaac_home / "kanban.db"
    init_db(db_path)

    # Reclaim stale claims
    reclaimed = reclaim_stale_claims(db_path)

    board = get_board(db_path, cfg.board_name)
    if not board:
        return 0

    columns = get_columns(db_path, board.id)
    ready_col = next((c for c in columns if c.name.lower() == "ready"), None)
    if not ready_col:
        ready_col = next((c for c in columns if c.name.lower() != "backlog"), None)
    if not ready_col:
        return 0

    ready_tasks = get_tasks(db_path, board.id, ready_col.id)
    unclaimed_ready = [
        t for t in ready_tasks
        if get_active_claim(db_path, t.id) is None
    ][:cfg.max_tasks_per_poll]

    assigned = 0
    registry = get_worker_registry()
    gateway = LoggingGateway()

    for task in unclaimed_ready:
        profile = "general"
        if task.tag_list:
            profile = task.tag_list[0]

        workers = registry.get_available_workers(profile, cfg.worker_timeout_seconds)
        if not workers:
            continue

        worker = min(workers, key=lambda w: w.active_task_count)

        try:
            claim = claim_task(db_path, task.id, worker.worker_id, worker.profile, cfg.stale_threshold_minutes)
        except ValueError:
            continue

        registry.increment_task_count(worker.worker_id)
        success = gateway.deliver(task, worker, claim)
        if not success:
            from isaac.kanban import release_task_claims
            release_task_claims(db_path, task.id)
            registry.decrement_task_count(worker.worker_id)
            continue

        assigned += 1

    return assigned
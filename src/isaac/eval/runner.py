"""Suite runner — execute tasks, score answers, persist the run.

The runner is injectable: ``run_suite`` takes any callable
``(EvalTask) -> TaskAnswer``, so the harness itself is fully testable offline
with a scripted fake, and a real run simply plugs in
:func:`default_runner` (AgentLoop or the specialist Orchestrator).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from isaac.eval.checkers import CheckOutcome, score_answer
from isaac.eval.results import EvalStore
from isaac.eval.suite import EvalTask, suite_hash

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]


@dataclass
class TaskAnswer:
    """What a runner returns for one task."""

    text: str
    stopped_reason: str = "final"


RunnerFn = Callable[[EvalTask], TaskAnswer]


@dataclass
class TaskOutcome:
    """One scored task."""

    task_id: str
    category: str
    passed: bool
    checks: list[CheckOutcome]
    answer: str
    stopped_reason: str
    duration_ms: float


@dataclass
class EvalRunSummary:
    """Aggregate result of one suite run."""

    run_id: str
    suite: str
    suite_hash: str
    model: str
    provider: str
    runner: str
    git_rev: str
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        """category -> (passed, total)."""
        cats: dict[str, tuple[int, int]] = {}
        for o in self.outcomes:
            p, t = cats.get(o.category, (0, 0))
            cats[o.category] = (p + (1 if o.passed else 0), t + 1)
        return cats


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _default_workspace() -> Path:
    from isaac.config.settings import get_settings

    root = get_settings().isaac_home / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_runner(auto_approve: bool = False) -> RunnerFn:
    """A real runner: AgentLoop for ``runner: agent`` tasks, the specialist
    Orchestrator for ``runner: team`` tasks. Requires a configured LLM."""

    def run(task: EvalTask) -> TaskAnswer:
        if task.runner == "team":
            from isaac.specialists import Orchestrator

            result = Orchestrator(auto_approve=auto_approve).run(task.prompt)
            return TaskAnswer(
                text=result.final_output or "",
                stopped_reason="final" if result.success else "error",
            )

        from isaac.agents.agent_loop import build_default_agent

        loop = build_default_agent(
            max_iterations=task.max_iterations,
            max_wall_seconds=task.timeout_seconds,
            auto_approve=auto_approve,
            only=task.tools,
        )
        result = loop.run(task.prompt)
        return TaskAnswer(text=result.output or "", stopped_reason=result.stopped_reason)

    return run


def _seed_files(task: EvalTask, workspace: Path) -> None:
    def _safe_target(rel: str) -> Path | None:
        target = (workspace / rel).resolve()
        try:
            target.relative_to(workspace.resolve())
        except ValueError:
            logger.warning("Task %s: seed file %r escapes workspace; skipped", task.id, rel)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    for rel, content in task.files.items():
        target = _safe_target(rel)
        if target is not None:
            target.write_text(content, encoding="utf-8")
    for rel, src in task.file_paths.items():
        target = _safe_target(rel)
        if target is None:
            continue
        try:
            shutil.copyfile(src, target)
        except OSError as exc:
            logger.warning("Task %s: cannot copy attachment %s: %s", task.id, src, exc)


def run_suite(
    tasks: list[EvalTask],
    runner: RunnerFn | None = None,
    *,
    suite_name: str = "suite",
    workspace: Path | None = None,
    store: EvalStore | None = None,
    model: str = "",
    provider: str = "",
    runner_kind: str = "agent",
    on_event: EventCallback | None = None,
) -> EvalRunSummary:
    """Run every task, score it, and (optionally) persist the run.

    A crashing task is scored as failed with ``stopped_reason="error"`` —
    one bad task never aborts the suite.
    """
    if runner is None:
        runner = default_runner()
    if workspace is None:
        workspace = _default_workspace()
    if not model or not provider:
        try:
            from isaac.config.settings import get_settings

            s = get_settings()
            model = model or s.llm.model_name
            provider = provider or s.llm.llm_provider
        except Exception:
            model, provider = model or "unknown", provider or "unknown"

    def emit(kind: str, **data: object) -> None:
        if on_event is not None:
            try:
                on_event(kind, dict(data))
            except Exception:
                logger.debug("eval on_event raised", exc_info=True)

    run_id = store.new_run_id() if store else f"local{int(time.time())}"
    started_at = time.time()
    summary = EvalRunSummary(
        run_id=run_id,
        suite=suite_name,
        suite_hash=suite_hash(tasks),
        model=model,
        provider=provider,
        runner=runner_kind,
        git_rev=_git_rev(),
    )

    for i, task in enumerate(tasks, 1):
        emit("task_start", n=i, total=len(tasks), task_id=task.id)
        _seed_files(task, workspace)
        t0 = time.monotonic()
        try:
            answer = runner(task)
        except Exception as exc:
            logger.exception("Task %s: runner crashed", task.id)
            answer = TaskAnswer(text=f"(runner crashed: {exc})", stopped_reason="error")
        duration_ms = (time.monotonic() - t0) * 1000
        checks = score_answer(task.checks, answer.text, workspace)
        passed = bool(checks) and all(c.passed for c in checks)
        summary.outcomes.append(
            TaskOutcome(
                task_id=task.id,
                category=task.category,
                passed=passed,
                checks=checks,
                answer=answer.text,
                stopped_reason=answer.stopped_reason,
                duration_ms=duration_ms,
            )
        )
        emit("task_done", task_id=task.id, passed=passed, duration_ms=duration_ms)

    if store is not None:
        store.record_run(
            run_id=run_id,
            suite=suite_name,
            suite_hash=summary.suite_hash,
            model=model,
            provider=provider,
            runner=runner_kind,
            git_rev=summary.git_rev,
            started_at=started_at,
            task_rows=[
                {
                    "task_id": o.task_id,
                    "category": o.category,
                    "passed": o.passed,
                    "stopped_reason": o.stopped_reason,
                    "duration_ms": o.duration_ms,
                    "answer": o.answer,
                    "checks": [
                        {"type": c.type, "passed": c.passed, "detail": c.detail} for c in o.checks
                    ],
                }
                for o in summary.outcomes
            ],
        )
    return summary

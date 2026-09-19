"""Orchestrator — the manager that decomposes a goal and dispatches it.

The :class:`Orchestrator` is the *manager* mini-agent of the I.S.A.A.C.
specialist team.  Given a single high-level *goal* it:

1. **Plans** — asks a manager LLM (or an injected planner) to break the goal
   into an ordered list of :class:`SubTask` s, each addressed to a named
   specialist and optionally depending on earlier subtasks.
2. **Schedules + executes** — runs subtasks whose dependencies are satisfied
   in parallel (via a thread pool), iterating in waves until every subtask is
   done.  Each subtask receives a digest of its dependencies' outputs as extra
   context, so a *coder* can build on what a *researcher* found.
3. **Synthesizes** — folds every subtask output into one cohesive final answer
   with a single LLM call (or, for a lone subtask, returns it verbatim).
4. **Records** the outcome to the :class:`~isaac.meta.learner.MetaLearner` — at
   the orchestration level *and* per specialist — so the team learns which
   specialists actually deliver.

Since 1.5.0 that recorded history is also **read back**: when
``ISAAC_META_SPECIALIST_SELECTION`` is on (or ``use_meta_selection=True`` is
passed), the roster handed to the planner is ordered by each specialist's
Bayesian-smoothed win-rate and annotated with its track record, and an unknown
specialist name resolves to the best-scoring member instead of always falling
through to the generalist.  See :mod:`isaac.meta.specialist_selector`.  The
toggle exists so the ablation in :mod:`isaac.eval.ablation` can measure whether
any of that helps.

The orchestrator is deliberately **decoupled** from the concrete roster: the
specialist factory and the planner are both injectable, the registry is only
imported lazily inside methods, and no LLM or network call happens at import
time.  This keeps the module cheap to import and trivial to test offline.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from isaac.agents.agent_loop import (
    ApprovalCallback,
    StopCallback,
    _active_boundary,
    _BackgroundCall,
    _RunBoundary,
    _RunStopped,
)
from isaac.specialists.base import Specialist, SpecialistResult

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]
SpecialistFactory = Callable[..., Specialist]
Planner = Callable[[str, list[dict], str], "list[SubTask]"]

#: Hard cap on the number of subtasks a plan may contain.
MAX_SUBTASKS = 8


class _InvalidPlan(ValueError):
    pass


# ──────────────────────────────────────────────────────────────────────────
# Structured results
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SubTask:
    """A single planned unit of work addressed to one specialist.

    Attributes:
        id: Stable identifier used to express dependencies (e.g. ``"t1"``).
        description: The natural-language instruction for the specialist.
        specialist: Name of the specialist that should handle this subtask.
        depends_on: Ids of subtasks that must complete before this one runs.
    """

    id: str
    description: str
    specialist: str
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this subtask."""
        return {
            "id": self.id,
            "description": self.description,
            "specialist": self.specialist,
            "depends_on": list(self.depends_on),
        }


@dataclass
class SubTaskResult:
    """Pairs a :class:`SubTask` with the :class:`SpecialistResult` it produced."""

    subtask: SubTask
    result: SpecialistResult

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this subtask result."""
        return {
            "subtask": self.subtask.to_dict(),
            "result": self.result.to_dict(),
        }


@dataclass
class OrchestrationResult:
    """Structured outcome of a full :meth:`Orchestrator.run`."""

    goal: str
    plan: list[SubTask]
    results: list[SubTaskResult]
    final_output: str
    success: bool
    duration_ms: float
    error: str = ""
    stopped_reason: str = "final"

    @property
    def completed(self) -> bool:
        return (
            self.stopped_reason == "final"
            and not self.error
            and bool(self.plan)
            and len(self.results) == len(self.plan)
            and {r.subtask.id for r in self.results} == {s.id for s in self.plan}
            and all(r.result.stopped_reason == "final" for r in self.results)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the orchestration."""
        return {
            "goal": self.goal,
            "plan": [s.to_dict() for s in self.plan],
            "results": [r.to_dict() for r in self.results],
            "final_output": self.final_output,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "stopped_reason": self.stopped_reason,
        }


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────


class Orchestrator:
    """Decompose a goal and dispatch it to specialist mini-agents.

    The orchestrator owns planning, dependency-aware parallel execution, and
    synthesis.  Everything heavy is injectable so the whole pipeline can run
    offline in tests:

    * ``manager_llm`` — the LangChain chat model used for planning and
      synthesis; resolved lazily from :func:`isaac.llm.provider.get_llm`
      (tier ``"strong"``) on first use when not supplied.
    * ``specialist_factory`` — ``(name, **kwargs) -> Specialist``; defaults to
      the lazy registry lookup :func:`isaac.specialists.registry.get_specialist`.
    * ``planner`` — ``(goal, roster, context) -> list[SubTask]``; defaults to
      the built-in LLM planner :meth:`_plan`.
    """

    def __init__(
        self,
        *,
        manager_llm: Any | None = None,
        max_workers: int = 4,
        auto_approve: bool = False,
        on_event: EventCallback | None = None,
        specialist_factory: SpecialistFactory | None = None,
        planner: Planner | None = None,
        use_meta_selection: bool | None = None,
        selector: Any | None = None,
        max_iterations: int = 8,
        timeout_seconds: float = 0,
        max_wall_seconds: float = 0,
        should_stop: StopCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        specialist_max_iterations: int | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            manager_llm: LangChain chat model for planning and synthesis. If
                ``None``, resolved lazily on first use.
            max_workers: Maximum number of subtasks to run concurrently.
            auto_approve: Forwarded to every spawned specialist; auto-approves
                high-risk tool actions when ``True``.
            on_event: Optional ``(kind, data)`` progress callback.
            specialist_factory: Optional factory overriding the registry lookup.
            planner: Optional planner overriding the built-in LLM planner.
            use_meta_selection: Whether to bias specialist selection with
                MetaLearner win-rates. ``None`` (the default) reads
                ``ISAAC_META_SPECIALIST_SELECTION``. Explicitly passing
                ``True``/``False`` is what the ablation harness uses.
            selector: Optional :class:`~isaac.meta.specialist_selector.SpecialistSelector`
                override (tests and the ablation inject an isolated one).
            max_iterations: Hard cap on planning waves (prevents infinite
                dependency loops). Defaults to 8 (== MAX_SUBTASKS).
            timeout_seconds: Per-subtask timeout in seconds (0 = unlimited).
                A timed-out subtask returns a failed result instead of hanging
                the whole wave.
            max_wall_seconds: Total wall-clock budget for the run
                (0 = unlimited). Exhaustion stops scheduling with
                ``stopped_reason="budget_exhausted"`` semantics.
        """
        self._llm = manager_llm
        self.max_workers = max(1, int(max_workers))
        self.auto_approve = auto_approve
        self._on_event = on_event
        self._specialist_factory = specialist_factory or self._default_factory
        self._planner = planner or self._plan
        self._use_meta_selection = use_meta_selection
        self._selector_override = selector
        self.max_iterations = max(1, int(max_iterations))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.max_wall_seconds = max(0.0, float(max_wall_seconds))
        self._should_stop = should_stop
        self._approval_callback = approval_callback
        self.specialist_max_iterations = specialist_max_iterations

    # ------------------------------------------------------------------
    # Lazy dependencies
    # ------------------------------------------------------------------

    @staticmethod
    def _default_factory(name: str, **kwargs: Any) -> Specialist:
        """Default specialist factory — a lazy registry lookup.

        Args:
            name: The specialist's short name.
            **kwargs: Forwarded to the specialist's constructor.

        Returns:
            A freshly constructed specialist.

        Raises:
            KeyError: If no specialist is registered under *name*.
        """
        from isaac.specialists.registry import get_specialist

        return get_specialist(name, **kwargs)

    def _manager(self) -> Any:
        """Lazily resolve and cache the manager LLM."""
        if self._llm is None:
            from isaac.llm.provider import get_llm

            self._llm = get_llm("strong")
        return self._llm

    @property
    def meta_selection(self) -> bool:
        """Whether MetaLearner-guided specialist selection is active."""
        if self._use_meta_selection is not None:
            return bool(self._use_meta_selection)
        try:
            from isaac.meta.specialist_selector import selection_enabled

            return selection_enabled()
        except Exception:  # pragma: no cover - defensive
            return False

    def _selector(self) -> Any:
        """Return the specialist selector (injected override wins)."""
        if self._selector_override is not None:
            return self._selector_override
        from isaac.meta.specialist_selector import get_selector

        return get_selector()

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        """Invoke the event callback defensively (a bad callback never breaks a run)."""
        if self._on_event is None:
            return
        try:
            self._on_event(kind, data)
        except Exception:  # pragma: no cover - defensive
            logger.debug("on_event callback raised for kind=%s", kind, exc_info=True)

    # ------------------------------------------------------------------
    # Public pipeline
    # ------------------------------------------------------------------

    def run(self, goal: str, context: str = "") -> OrchestrationResult:
        """Plan, execute, and synthesize a response to *goal*.

        Args:
            goal: The high-level objective to accomplish.
            context: Optional shared context prepended to every subtask.

        Returns:
            An :class:`OrchestrationResult` capturing the plan, per-subtask
            results, and the synthesized final answer. Top-level exceptions are
            caught and surfaced as a failed result.
        """
        start = time.monotonic()
        boundary = _RunBoundary(
            self.max_wall_seconds, self._should_stop, parent=_active_boundary.get()
        )
        outcome = OrchestrationResult(goal, [], [], "", False, 0)
        try:
            outcome = boundary.call(lambda: self._run(goal, context, boundary, outcome))
        except _RunStopped as exc:
            results = list(outcome.results)
            finished = {r.subtask.id for r in results}
            results.extend(
                SubTaskResult(st, self._timeout_result(st, exc.reason))
                for st in outcome.plan
                if st.id not in finished
            )
            outcome = replace(
                outcome,
                results=results,
                success=False,
                error=exc.reason,
                stopped_reason=exc.reason,
                final_output="\n\n".join(r.result.output for r in results),
            )
        except Exception as exc:
            logger.exception("Orchestration failed for goal=%r", goal)
            outcome = replace(
                outcome,
                success=False,
                error=str(exc),
                stopped_reason="invalid_plan" if isinstance(exc, _InvalidPlan) else "error",
            )
        outcome.duration_ms = round((time.monotonic() - start) * 1000, 1)
        return outcome

    def _run(
        self, goal: str, context: str, boundary: _RunBoundary, outcome: OrchestrationResult
    ) -> OrchestrationResult:
        started = time.monotonic()
        try:
            roster = self._list_roster()
        except Exception:
            logger.exception("Failed to list specialist roster")
            roster = []
        boundary.check()
        roster = self._apply_meta_selection(roster)
        boundary.check()
        outcome.plan = self._planner(goal, roster, context)
        boundary.check()
        self._validate_plan(outcome.plan)
        self._emit("plan", {"plan": [s.to_dict() for s in outcome.plan]})
        outcome.results = self._execute(outcome.plan, context, boundary, outcome.results)
        boundary.check()
        reasons = [
            r.result.stopped_reason for r in outcome.results if r.result.stopped_reason != "final"
        ]
        if reasons:
            outcome.stopped_reason = reasons[0]
            outcome.error = reasons[0]
            outcome.final_output = "\n\n".join(r.result.output for r in outcome.results)
        else:
            outcome.final_output = self._synthesize(goal, outcome.results)
            boundary.check()
            self._emit("synthesis", {"output": outcome.final_output})
        outcome.success = outcome.completed and all(r.result.success for r in outcome.results)
        boundary.check()
        self._emit("final", {"output": outcome.final_output, "success": outcome.success})
        outcome.duration_ms = round((time.monotonic() - started) * 1000, 1)
        self._record(goal, outcome.plan, outcome.results, outcome.success, outcome.duration_ms)
        boundary.check()
        return outcome

    @staticmethod
    def _validate_plan(plan: list[SubTask]) -> None:
        if not isinstance(plan, list) or not plan:
            raise _InvalidPlan("Plan must contain at least one subtask")
        if len(plan) > MAX_SUBTASKS:
            raise _InvalidPlan(f"Plan exceeds {MAX_SUBTASKS} subtasks; nothing was executed")
        ids: set[str] = set()
        for st in plan:
            if (
                not isinstance(st, SubTask)
                or not isinstance(st.id, str)
                or not st.id.strip()
                or not isinstance(st.description, str)
                or not st.description.strip()
                or not isinstance(st.specialist, str)
                or not st.specialist.strip()
                or not isinstance(st.depends_on, list)
                or any(not isinstance(dep, str) for dep in st.depends_on)
            ):
                raise _InvalidPlan("Malformed subtask")
            if st.id in ids:
                raise _InvalidPlan(f"Duplicate subtask id: {st.id}")
            ids.add(st.id)
        for st in plan:
            if st.id in st.depends_on or not set(st.depends_on) <= ids:
                raise _InvalidPlan(f"Invalid dependencies for subtask {st.id}")
        visited: set[str] = set()
        while len(visited) < len(plan):
            ready = {st.id for st in plan if set(st.depends_on) <= visited} - visited
            if not ready:
                raise _InvalidPlan("Dependency cycle in plan")
            visited.update(ready)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    @staticmethod
    def _list_roster() -> list[dict]:
        """Return the roster cards via a lazy registry import."""
        from isaac.specialists.registry import list_specialists

        return list_specialists()

    def _apply_meta_selection(self, roster: list[dict]) -> list[dict]:
        """Order and annotate *roster* with MetaLearner win-rates.

        A no-op when meta-selection is off, when the roster is empty, or when
        no history exists (the selector's sort is stable, so an empty history
        returns the identical ordering). Failures degrade to the raw roster —
        learning never breaks a run.
        """
        if not roster or not self.meta_selection:
            return roster
        try:
            annotated = self._selector().annotate_roster(roster)
        except Exception:  # pragma: no cover - learning is best-effort
            logger.debug("Specialist selector failed; using unranked roster", exc_info=True)
            return roster
        self._emit(
            "specialist_ranking",
            {"order": [str(c.get("name", "")) for c in annotated]},
        )
        return annotated

    def _plan(self, goal: str, roster: list[dict], context: str) -> list[SubTask]:
        """Built-in LLM planner: decompose *goal* into :class:`SubTask` s.

        Asks the manager LLM to emit strict JSON describing the subtasks. Any
        parse failure, an empty roster, or an empty plan falls back to a single
        generalist subtask carrying the whole goal.

        Args:
            goal: The objective to decompose.
            roster: Specialist routing cards from ``list_specialists()``.
            context: Optional shared context (currently advisory).

        Returns:
            A list of subtasks (at most :data:`MAX_SUBTASKS`); never empty.
        """
        fallback = [SubTask(id="t1", description=goal, specialist="generalist")]
        if not roster:
            return fallback

        roster_lines = "\n".join(self._roster_line(c) for c in roster)
        system = (
            "You are the manager of a team of specialist agents. Decompose the "
            "user's goal into the minimum set of subtasks needed, assigning each "
            "to the most suitable specialist. Express ordering with 'depends_on'. "
            "Respond with STRICT JSON only, no prose, no markdown fences."
        )
        if self.meta_selection and any("track_record" in c for c in roster):
            system += (
                " The roster is ordered by measured past success and each entry "
                "shows its track record. Domain fit comes first; use the track "
                "record only to break ties between equally suitable specialists."
            )
        human = (
            f"Available specialists:\n{roster_lines}\n\n"
            f"Goal:\n{goal}\n\n"
            "Return JSON of the form:\n"
            '{"subtasks":[{"id":"t1","description":"...","specialist":"coder",'
            '"depends_on":[]}]}'
        )

        try:
            content = self._invoke(system, human)
            plan = self._parse_plan(content)
        except (_InvalidPlan, _RunStopped):
            raise
        except Exception:  # pragma: no cover - defensive (covers LLM + parse)
            logger.debug("Planner LLM/parse failed; using generalist fallback", exc_info=True)
            return fallback

        return plan or fallback

    @staticmethod
    def _roster_line(card: dict) -> str:
        """Render one roster card for the planner prompt."""
        line = f"- {card.get('name', '?')}: {card.get('domain', '')}".rstrip(": ")
        record = card.get("track_record")
        if record:
            line += f" [track record: {record}]"
        return line

    @staticmethod
    def _parse_plan(content: str) -> list[SubTask]:
        text = (content or "").strip()
        if text.startswith("```"):
            # Drop the opening fence (optionally ```json) and the closing fence.
            text = text.split("\n", 1)[-1] if "\n" in text else ""
            if text.rstrip().endswith("```"):
                text = text.rstrip()[: -len("```")]
            text = text.strip()

        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return []

        raw = data.get("subtasks") if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []

        subtasks: list[SubTask] = []
        if len(raw) > MAX_SUBTASKS:
            raise _InvalidPlan(f"Plan exceeds {MAX_SUBTASKS} subtasks")
        for i, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                raise _InvalidPlan("Malformed subtask")
            subtasks.append(
                SubTask(
                    id=item.get("id", f"t{i}"),
                    description=item.get("description", ""),
                    specialist=item.get("specialist", "generalist"),
                    depends_on=item.get("depends_on", []),
                )
            )
        if subtasks:
            Orchestrator._validate_plan(subtasks)
        return subtasks

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        plan: list[SubTask],
        context: str,
        boundary: _RunBoundary,
        ordered: list[SubTaskResult],
    ) -> list[SubTaskResult]:
        self._validate_plan(plan)
        completed: dict[str, SubTaskResult] = {}
        pending: list[SubTask] = list(plan)
        waves = 0

        while pending:
            boundary.check()
            if waves >= self.max_iterations:
                for st in pending:
                    ordered.append(
                        SubTaskResult(
                            subtask=st,
                            result=self._timeout_result(st, "max_iterations"),
                        )
                    )
                break
            waves += 1
            ready = [st for st in pending if all(dep in completed for dep in st.depends_on)]
            if not ready:
                raise _InvalidPlan("Unresolvable dependencies")

            blocked = [
                st
                for st in ready
                if any(completed[dep].result.stopped_reason != "final" for dep in st.depends_on)
            ]
            for st in blocked:
                res = SubTaskResult(st, self._timeout_result(st, "dependency_failed"))
                completed[st.id] = res
                ordered.append(res)
            runnable = [st for st in ready if st not in blocked]
            wave_results = self._run_wave(runnable, completed, context, boundary, ordered)
            for st, res in wave_results:
                completed[st.id] = res

            ready_ids = {st.id for st in ready}
            pending = [st for st in pending if st.id not in ready_ids]

        return ordered

    def _timeout_result(self, subtask: SubTask, reason: str) -> Any:
        """Build a failed SpecialistResult for skipped/timed-out subtasks."""
        from isaac.specialists.base import SpecialistResult

        return SpecialistResult(
            specialist=subtask.specialist,
            task=subtask.description,
            output=f"Subtask {subtask.id} skipped: {reason}.",
            success=False,
            stopped_reason=reason,
            error=reason,
        )

    def _run_wave(
        self,
        wave: list[SubTask],
        completed: dict[str, SubTaskResult],
        context: str,
        boundary: _RunBoundary,
        ordered: list[SubTaskResult],
    ) -> list[tuple[SubTask, SubTaskResult]]:
        out: list[tuple[SubTask, SubTaskResult]] = []
        pending = list(wave)
        active: list[tuple[SubTask, _RunBoundary, _BackgroundCall]] = []
        try:
            while pending or active:
                boundary.check()
                while pending and len(active) < self.max_workers:
                    st = pending.pop(0)
                    child = _RunBoundary(
                        self.timeout_seconds, parent=boundary, timeout_reason="timeout"
                    )
                    subtask_for_work = st
                    child_for_work = child

                    def _run_worker(
                        _st: SubTask = subtask_for_work,
                        _child: _RunBoundary = child_for_work,
                    ) -> SubTaskResult:
                        return self._run_subtask(_st, completed, context, _child)

                    work = _BackgroundCall(child, _run_worker)
                    active.append((st, child, work))
                for st, child, work in list(active):
                    reason = child.reason()
                    if not reason and not work.done.is_set():
                        continue
                    if reason:
                        work.cancel()
                        res = SubTaskResult(st, self._timeout_result(st, reason))
                        self._emit("subtask_timeout", {"id": st.id, "reason": reason})
                    else:
                        try:
                            res = work.future.result()
                        except _RunStopped as exc:
                            res = SubTaskResult(st, self._timeout_result(st, exc.reason))
                        except Exception as exc:
                            logger.debug("Subtask %s failed: %s", st.id, exc, exc_info=True)
                            res = SubTaskResult(st, self._timeout_result(st, "error"))
                            res.result.error = str(exc)
                    active.remove((st, child, work))
                    out.append((st, res))
                    ordered.append(res)
                if active:
                    time.sleep(0.005)
        finally:
            for _, child, work in active:
                child.stop(boundary.reason() or "cancelled")
                work.cancel()
        return out

    def _run_subtask(
        self,
        subtask: SubTask,
        completed: dict[str, SubTaskResult],
        context: str,
        boundary: _RunBoundary,
    ) -> SubTaskResult:
        """Instantiate the target specialist and run a single subtask.

        Builds the per-subtask context from the shared context plus a digest of
        each completed dependency's output. Falls back to the ``generalist``
        specialist when the requested one is unknown to the factory.

        Args:
            subtask: The subtask to run.
            completed: Map of completed subtask id -> result, for dependencies.
            context: The shared base context.

        Returns:
            The :class:`SubTaskResult` for this subtask.
        """
        ctx = self._build_context(subtask, completed, context)
        self._emit(
            "subtask_start",
            {
                "id": subtask.id,
                "specialist": subtask.specialist,
                "description": subtask.description,
            },
        )

        boundary.check()
        specialist = self._make_specialist(subtask.specialist, boundary)
        boundary.check()
        result = specialist.run(subtask.description, context=ctx)
        boundary.check()

        self._emit(
            "subtask_done",
            {"id": subtask.id, "success": result.success, "output": result.output},
        )
        return SubTaskResult(subtask=subtask, result=result)

    def _make_specialist(self, name: str, boundary: _RunBoundary | None = None) -> Specialist:
        """Instantiate *name*, falling back sensibly when it is unknown.

        With meta-selection on, an unresolvable name falls back to the
        *best-scoring registered* specialist rather than unconditionally to the
        generalist — the one place where accumulated win-rates decide the
        routing outright instead of merely advising the planner.
        """
        kwargs: dict[str, Any] = {
            "auto_approve": self.auto_approve,
            "on_event": self._on_event,
            "approval_callback": self._approval_callback,
            "should_stop": (lambda: bool(boundary.reason())) if boundary else self._should_stop,
            "max_wall_seconds": boundary.remaining() if boundary else self.max_wall_seconds,
        }
        if self.specialist_max_iterations is not None:
            kwargs["max_iterations"] = self.specialist_max_iterations
        try:
            return self._specialist_factory(name, **kwargs)
        except KeyError:
            fallback = self._fallback_specialist_name()
            logger.debug("Unknown specialist %r; falling back to %s", name, fallback)
            try:
                return self._specialist_factory(fallback, **kwargs)
            except KeyError:  # pragma: no cover - generalist is always registered
                return self._specialist_factory("generalist", **kwargs)

    def _fallback_specialist_name(self) -> str:
        """Pick the substitute for an unknown specialist name."""
        if not self.meta_selection:
            return "generalist"
        try:
            from isaac.specialists.registry import specialist_names

            names = specialist_names()
        except Exception:  # pragma: no cover - registry import shouldn't fail
            return "generalist"
        if not names:
            return "generalist"
        try:
            return self._selector().best(names, default="generalist")
        except Exception:  # pragma: no cover - learning is best-effort
            return "generalist"

    @staticmethod
    def _build_context(
        subtask: SubTask,
        completed: dict[str, SubTaskResult],
        context: str,
    ) -> str:
        """Compose the context handed to a subtask's specialist.

        Concatenates the shared base context with a labelled digest of each
        completed dependency's output.
        """
        parts: list[str] = []
        if context:
            parts.append(context.strip())

        digests: list[str] = []
        for dep_id in subtask.depends_on:
            dep = completed.get(dep_id)
            if dep is None:
                continue
            digests.append(f"[{dep.subtask.specialist} · {dep_id}] {dep.result.output}")
        if digests:
            parts.append("Results from prior steps:\n" + "\n\n".join(digests))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(self, goal: str, results: list[SubTaskResult]) -> str:
        """Fold every subtask output into one cohesive final answer.

        A single subtask is returned verbatim. Otherwise one LLM call merges the
        outputs; on failure it falls back to concatenating them under headers.

        Args:
            goal: The original goal, for grounding the synthesis.
            results: The completed subtask results.

        Returns:
            The synthesized final answer (possibly empty if there were no
            results).
        """
        if not results:
            return ""
        if len(results) == 1:
            return results[0].result.output

        sections = "\n\n".join(
            f"## {r.subtask.specialist} — {r.subtask.description}\n{r.result.output}"
            for r in results
        )
        system = (
            "You are the manager of a specialist team. Combine the subtask "
            "results below into one unified, well-structured final answer that "
            "fully addresses the goal. Do not mention the internal subtasks."
        )
        human = f"Goal:\n{goal}\n\nSubtask results:\n{sections}"

        try:
            content = self._invoke(system, human)
            if content and content.strip():
                return content.strip()
        except _RunStopped:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.debug("Synthesis LLM failed; concatenating outputs", exc_info=True)

        return sections

    # ------------------------------------------------------------------
    # LLM + recording helpers
    # ------------------------------------------------------------------

    def _invoke(self, system: str, human: str) -> str:
        """Call the manager LLM with a system+human message pair.

        Args:
            system: The system prompt.
            human: The user prompt.

        Returns:
            The model's ``.content`` as a string.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        boundary = _active_boundary.get()
        if boundary is not None:
            boundary.check()
        response = self._manager().invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        if boundary is not None:
            boundary.check()
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    def _record(
        self,
        goal: str,
        plan: list[SubTask],
        results: list[SubTaskResult],
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record the outcome to the MetaLearner (best-effort).

        Two rows are written per run:

        * one ``orchestration`` row for the run as a whole (as before), and
        * one ``specialist`` row **per subtask**, keyed by the specialist that
          actually ran it.  Those per-specialist rows are the win-rates
          :class:`~isaac.meta.specialist_selector.SpecialistSelector` reads.

        Recording happens regardless of the ``use_meta_selection`` toggle:
        collecting evidence is free, and the ablation needs both arms to build
        the same history so only its *use* differs.
        """
        if not results or any(r.result.verified_success is None for r in results):
            return
        try:
            from isaac.meta.learner import get_learner

            specialists = {r.subtask.specialist for r in results}
            strategy = (
                "team"
                if len(specialists) > 1
                else (next(iter(specialists)) if specialists else "none")
            )
            get_learner().record(
                task_desc=goal,
                task_type="orchestration",
                strategy=strategy,
                success=success,
                duration_ms=duration_ms,
                iterations=len(plan),
            )
        except Exception:  # pragma: no cover - learning is best-effort
            logger.debug("MetaLearner record failed", exc_info=True)

        try:
            selector = self._selector()
            for r in results:
                selector.record(
                    r.result.specialist or r.subtask.specialist,
                    success=bool(r.result.success),
                    task_desc=r.subtask.description,
                    duration_ms=r.result.duration_ms,
                )
        except Exception:  # pragma: no cover - learning is best-effort
            logger.debug("Per-specialist record failed", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────────


def orchestrate(goal: str, *, context: str = "", **kwargs: Any) -> OrchestrationResult:
    """Construct an :class:`Orchestrator` and run *goal* in one call.

    Args:
        goal: The high-level objective.
        context: Optional shared context.
        **kwargs: Forwarded to the :class:`Orchestrator` constructor.

    Returns:
        The :class:`OrchestrationResult`.
    """
    return Orchestrator(**kwargs).run(goal, context=context)

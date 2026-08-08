"""Self-improvement ablation — does any of the learning machinery actually help?

I.S.A.A.C. has carried a MetaLearner, a skill curator, and prompt evolution
since 0.4.0 with **no evidence** that any of it improves outcomes.  Roadmap WS6
asks for "an ablation showing MetaLearner-guided selection >= baseline".  This
module runs that ablation and is deliberately built so a *negative* result is
just as easy to produce and report as a positive one.

Protocol
--------
1. **Warm-up.** Run the task set ``warmup_trials`` times through the specialist
   Orchestrator with recording on, building a per-specialist win/loss history.
   Without this, ON and OFF are identical by construction — the selector is an
   exact no-op on an empty history — and the "ablation" would measure nothing.
2. **Snapshot.** Freeze that history. Every measured trial in *both* arms
   starts from a byte-identical copy of it, in its own SQLite file, so the arms
   cannot contaminate each other and neither drifts ahead of the other.
3. **Measure.** ``trials`` paired trials. Each trial runs the identical task
   set twice — once with ``use_meta_selection=True``, once ``False`` — against
   the same model, in the same workspace, from the same frozen history.

Reporting rules (the point of the exercise)
-------------------------------------------
* Every per-trial accuracy is kept, not just the mean, so variance is visible.
* The paired per-task difference is tested with an exact sign-flip permutation
  test — with a handful of trials on a few dozen tasks, an eyeballed 2-point
  gap is almost always noise, and saying so is the useful output.
* ``AblationReport.verdict`` is computed from the numbers, and "flat" is a
  first-class outcome, not a failure of the run.

The task set is forced through the ``team`` runner: specialist selection is the
thing being ablated, and the single-agent runner never consults it, so leaving
tasks on ``runner: agent`` would dilute any real effect with tasks the
intervention cannot touch.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaac.eval.checkers import score_answer
from isaac.eval.runner import TaskAnswer
from isaac.eval.suite import EvalTask, suite_hash

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]

ARMS = ("on", "off")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    """One pass over the task set in one arm."""

    arm: str
    trial: int
    passed: dict[str, bool] = field(default_factory=dict)
    duration_ms: dict[str, float] = field(default_factory=dict)
    plans: dict[str, list[str]] = field(default_factory=dict)
    """task_id -> the specialists the planner actually dispatched to."""

    @property
    def total(self) -> int:
        return len(self.passed)

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.passed.values() if v)

    @property
    def accuracy(self) -> float:
        return self.n_passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "trial": self.trial,
            "passed": self.passed,
            "n_passed": self.n_passed,
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "duration_ms": self.duration_ms,
            "plans": self.plans,
        }


@dataclass
class AblationReport:
    """Everything measured, in a form that cannot hide a null result."""

    suite: str
    suite_hash: str
    model: str
    provider: str
    git_rev: str
    task_ids: list[str]
    warmup_trials: int
    trials: list[TrialResult] = field(default_factory=list)
    warmup_history: list[dict] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    notes: list[str] = field(default_factory=list)

    # -- slicing -------------------------------------------------------

    def arm(self, arm: str) -> list[TrialResult]:
        return [t for t in self.trials if t.arm == arm]

    def accuracies(self, arm: str) -> list[float]:
        return [t.accuracy for t in self.arm(arm)]

    def mean(self, arm: str) -> float:
        acc = self.accuracies(arm)
        return statistics.fmean(acc) if acc else 0.0

    def stdev(self, arm: str) -> float:
        acc = self.accuracies(arm)
        return statistics.stdev(acc) if len(acc) > 1 else 0.0

    def n_trials(self, arm: str) -> int:
        return len(self.arm(arm))

    def task_pass_counts(self, arm: str) -> dict[str, int]:
        """task_id -> how many trials in *arm* passed it."""
        counts = {tid: 0 for tid in self.task_ids}
        for trial in self.arm(arm):
            for tid, ok in trial.passed.items():
                if ok:
                    counts[tid] = counts.get(tid, 0) + 1
        return counts

    def task_table(self) -> list[dict[str, Any]]:
        """Per-task ON/OFF pass counts, worst-first by ON-minus-OFF."""
        on, off = self.task_pass_counts("on"), self.task_pass_counts("off")
        rows = [
            {
                "task_id": tid,
                "on": on.get(tid, 0),
                "off": off.get(tid, 0),
                "delta": on.get(tid, 0) - off.get(tid, 0),
            }
            for tid in self.task_ids
        ]
        rows.sort(key=lambda r: (r["delta"], r["task_id"]))
        return rows

    # -- statistics ----------------------------------------------------

    @property
    def delta(self) -> float:
        """ON minus OFF, in accuracy points (positive = learning helped)."""
        return self.mean("on") - self.mean("off")

    @staticmethod
    def _sign_flip_p(diffs: list[float], iterations: int, seed: int) -> float:
        """Two-sided sign-flip permutation p-value for paired differences."""
        if not diffs:
            return 1.0
        observed = abs(statistics.fmean(diffs))
        if observed == 0.0:
            return 1.0
        rng = random.Random(seed)
        hits = 0
        for _ in range(iterations):
            flipped = statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)
            if abs(flipped) >= observed - 1e-12:
                hits += 1
        return (hits + 1) / (iterations + 1)

    def permutation_p(self, iterations: int = 20000, seed: int = 0) -> float:
        """Two-sided p-value for the ON/OFF difference, paired **by task**.

        Each task contributes one difference: its ON pass-count minus its OFF
        pass-count across trials.  The null is that the arm label carries no
        information, so each task's difference may independently flip sign.

        Pairing by task rather than by trial is a deliberate power decision.
        With ``t`` trials there are only ``2**t`` distinct trial-level sign
        patterns — at three trials the smallest reachable p-value is ~0.12, so
        a trial-level test *cannot* reach significance no matter how large the
        effect.  Pairing across ``k`` tasks gives ``2**k`` patterns instead.
        """
        on, off = self.task_pass_counts("on"), self.task_pass_counts("off")
        diffs = [float(on.get(t, 0) - off.get(t, 0)) for t in self.task_ids]
        return self._sign_flip_p(diffs, iterations, seed)

    def trial_permutation_p(self, iterations: int = 20000, seed: int = 0) -> float:
        """The same test paired by *trial* — reported for completeness.

        Underpowered at small trial counts (see :meth:`permutation_p`); kept so
        the report shows both rather than appearing to pick the flattering one.
        """
        on, off = self.accuracies("on"), self.accuracies("off")
        n = min(len(on), len(off))
        return self._sign_flip_p([on[i] - off[i] for i in range(n)], iterations, seed)

    @property
    def verdict(self) -> str:
        """``positive`` / ``negative`` / ``flat`` — computed, never chosen."""
        d = self.delta
        # One task flipping on one trial is the smallest change the suite can
        # even represent; anything at or below that is indistinguishable from
        # noise no matter which direction it points.
        resolution = 1.0 / (len(self.task_ids) or 1)
        if abs(d) < resolution or self.permutation_p() > 0.05:
            return "flat"
        return "positive" if d > 0 else "negative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "suite_hash": self.suite_hash,
            "model": self.model,
            "provider": self.provider,
            "git_rev": self.git_rev,
            "task_ids": self.task_ids,
            "n_tasks": len(self.task_ids),
            "warmup_trials": self.warmup_trials,
            "warmup_history": self.warmup_history,
            "trials_per_arm": {a: self.n_trials(a) for a in ARMS},
            "accuracy": {
                a: {
                    "per_trial": [round(x, 4) for x in self.accuracies(a)],
                    "mean": round(self.mean(a), 4),
                    "stdev": round(self.stdev(a), 4),
                }
                for a in ARMS
            },
            "delta": round(self.delta, 4),
            "permutation_p_by_task": round(self.permutation_p(), 4),
            "permutation_p_by_trial": round(self.trial_permutation_p(), 4),
            "verdict": self.verdict,
            "task_table": self.task_table(),
            "trials": [t.to_dict() for t in self.trials],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": round(self.finished_at - self.started_at, 1),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Environment control
# ---------------------------------------------------------------------------


def _point_learner_at(db_path: Path) -> None:
    """Rebind the MetaLearner + selector singletons to *db_path*."""
    from isaac.config.settings import get_settings
    from isaac.meta.learner import reset_learner
    from isaac.meta.specialist_selector import reset_selector

    reset_learner()
    reset_selector()
    get_settings().meta_learner_db_path = str(db_path)


def _specialist_history() -> list[dict]:
    """Snapshot the current per-specialist win/loss table."""
    from isaac.meta.specialist_selector import get_selector

    try:
        return get_selector().summary()
    except Exception:  # pragma: no cover - reporting only
        logger.debug("Could not read specialist history", exc_info=True)
        return []


def _git_rev() -> str:
    from isaac.eval.runner import _git_rev as rev

    return rev()


# ---------------------------------------------------------------------------
# The team runner under test
# ---------------------------------------------------------------------------


def team_runner(
    *,
    use_meta_selection: bool,
    auto_approve: bool = True,
    on_plan: Callable[[str, list[str]], None] | None = None,
) -> Callable[[EvalTask], TaskAnswer]:
    """Return a runner that drives every task through the Orchestrator.

    Args:
        use_meta_selection: The arm — whether the Orchestrator consults
            MetaLearner win-rates when selecting specialists.
        auto_approve: Allow high-risk tools (eval runs are unattended).
        on_plan: Optional ``(task_id, [specialist, ...])`` callback so the
            report can show *what the intervention actually changed*.
    """
    from isaac.specialists import Orchestrator

    def run(task: EvalTask) -> TaskAnswer:
        orch = Orchestrator(auto_approve=auto_approve, use_meta_selection=use_meta_selection)
        result = orch.run(task.prompt)
        if on_plan is not None:
            on_plan(task.id, [s.specialist for s in result.plan])
        return TaskAnswer(
            text=result.final_output or "",
            stopped_reason="final" if result.success else "error",
        )

    return run


#: ``(use_meta_selection, on_plan) -> RunnerFn``.  Injected by tests so the
#: harness itself can be exercised without a model; production passes
#: :func:`team_runner`.
RunnerFactory = Callable[..., Callable[[EvalTask], TaskAnswer]]


def _run_once(
    tasks: Sequence[EvalTask],
    *,
    arm: str,
    trial: int,
    workspace: Path,
    use_meta_selection: bool,
    emit: Callable[..., None],
    runner_factory: RunnerFactory | None = None,
) -> TrialResult:
    """Run the whole task set once and score it."""
    from isaac.eval.runner import _seed_files

    out = TrialResult(arm=arm, trial=trial)
    factory = runner_factory or team_runner
    runner = factory(
        use_meta_selection=use_meta_selection,
        on_plan=lambda tid, specialists: out.plans.__setitem__(tid, specialists),
    )

    for i, task in enumerate(tasks, 1):
        emit("task_start", arm=arm, trial=trial, n=i, total=len(tasks), task_id=task.id)
        _seed_files(task, workspace)
        t0 = time.monotonic()
        try:
            answer = runner(task)
        except Exception as exc:  # a crashing task scores 0, never aborts the arm
            logger.exception("Ablation %s/%d: task %s crashed", arm, trial, task.id)
            answer = TaskAnswer(text=f"(runner crashed: {exc})", stopped_reason="error")
        duration_ms = (time.monotonic() - t0) * 1000
        checks = score_answer(task.checks, answer.text, workspace)
        passed = bool(checks) and all(c.passed for c in checks)
        out.passed[task.id] = passed
        out.duration_ms[task.id] = duration_ms
        emit("task_done", arm=arm, trial=trial, task_id=task.id, passed=passed)

    emit("trial_done", arm=arm, trial=trial, accuracy=out.accuracy)
    return out


# ---------------------------------------------------------------------------
# The ablation
# ---------------------------------------------------------------------------


def run_ablation(
    tasks: Sequence[EvalTask],
    *,
    trials: int = 3,
    warmup_trials: int = 2,
    suite_name: str = "suite",
    workspace: Path | None = None,
    scratch: Path | None = None,
    model: str = "",
    provider: str = "",
    on_event: EventCallback | None = None,
    checkpoint_path: Path | None = None,
    runner_factory: RunnerFactory | None = None,
) -> AblationReport:
    """Run the paired ON/OFF self-improvement ablation.

    Args:
        tasks: The task set. Every task runs through the specialist team.
        trials: Paired measured trials per arm.
        warmup_trials: Passes used to build the MetaLearner history both arms
            start from. ``0`` means both arms start cold, in which case the
            selector is a provable no-op and the ablation can only return
            "flat" — useful as a negative control, useless as a measurement.
        workspace: Directory tasks may write into.
        scratch: Where the per-arm SQLite snapshots live.
        model / provider: Recorded for reproducibility.
        on_event: ``(kind, data)`` progress callback.
        checkpoint_path: If given, the partial report is written here after
            every trial, so an interrupted run is still analysable.
        runner_factory: Override for :func:`team_runner`; lets the harness be
            tested offline against a scripted runner.

    Returns:
        A fully populated :class:`AblationReport`.
    """
    from isaac.config.settings import get_settings

    settings = get_settings()
    if workspace is None:
        workspace = settings.isaac_home / "workspace"
    if scratch is None:
        scratch = settings.isaac_home / "ablation"
    workspace.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    if not model or not provider:
        model = model or settings.llm.model_name
        provider = provider or settings.llm.llm_provider

    original_db = settings.meta_learner_db_path

    def emit(kind: str, **data: object) -> None:
        if on_event is not None:
            try:
                on_event(kind, dict(data))
            except Exception:
                logger.debug("ablation on_event raised", exc_info=True)

    report = AblationReport(
        suite=suite_name,
        suite_hash=suite_hash(list(tasks)),
        model=model,
        provider=provider,
        git_rev=_git_rev(),
        task_ids=[t.id for t in tasks],
        warmup_trials=warmup_trials,
        started_at=time.time(),
    )

    def checkpoint() -> None:
        if checkpoint_path is None:
            return
        try:
            report.finished_at = time.time()
            checkpoint_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # pragma: no cover - checkpointing is best-effort
            logger.debug("Ablation checkpoint failed", exc_info=True)

    try:
        # -- 1. Warm-up: build the shared history -----------------------
        warm_db = scratch / "warmup.db"
        warm_db.unlink(missing_ok=True)
        _point_learner_at(warm_db)

        for w in range(1, warmup_trials + 1):
            emit("warmup_start", trial=w, total=warmup_trials)
            # Warm-up runs with selection ON: the point is to collect
            # outcomes, and a run that also uses them is the realistic
            # deployment the ON arm is meant to represent.
            _run_once(
                tasks,
                arm="warmup",
                trial=w,
                workspace=workspace,
                use_meta_selection=True,
                emit=emit,
                runner_factory=runner_factory,
            )

        report.warmup_history = _specialist_history()
        emit("warmup_done", history=report.warmup_history)
        checkpoint()

        if warmup_trials == 0 or not report.warmup_history:
            report.notes.append(
                "No warm-up history was collected: with an empty MetaLearner the "
                "selector is a provable no-op, so any ON/OFF difference measured "
                "here is pure run-to-run noise."
            )

        # -- 2. Paired measured trials ----------------------------------
        for trial in range(1, trials + 1):
            for arm in ARMS:
                # Both arms start from a byte-identical copy of the frozen
                # history, in their own file — no cross-contamination, no
                # order effect from one arm running first.
                arm_db = scratch / f"{arm}-trial{trial}.db"
                arm_db.unlink(missing_ok=True)
                if warm_db.exists():
                    shutil.copyfile(warm_db, arm_db)
                _point_learner_at(arm_db)

                result = _run_once(
                    tasks,
                    arm=arm,
                    trial=trial,
                    workspace=workspace,
                    use_meta_selection=(arm == "on"),
                    emit=emit,
                    runner_factory=runner_factory,
                )
                report.trials.append(result)
                checkpoint()
    finally:
        settings.meta_learner_db_path = original_db
        from isaac.meta.learner import reset_learner
        from isaac.meta.specialist_selector import reset_selector

        reset_learner()
        reset_selector()

    report.finished_at = time.time()
    checkpoint()
    emit("ablation_done", verdict=report.verdict, delta=report.delta)
    return report


# ---------------------------------------------------------------------------
# Mechanism-level simulation (no LLM)
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Outcome of the deterministic, LLM-free selection simulation."""

    rounds: int
    repeats: int
    on_success_rate: float
    off_success_rate: float
    on_stdev: float
    off_stdev: float
    oracle_rate: float
    random_rate: float

    @property
    def delta(self) -> float:
        return self.on_success_rate - self.off_success_rate

    @property
    def gap_closed(self) -> float:
        """Fraction of the random→oracle headroom that selection recovers."""
        headroom = self.oracle_rate - self.random_rate
        return (self.delta / headroom) if headroom > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "repeats": self.repeats,
            "on_success_rate": round(self.on_success_rate, 4),
            "off_success_rate": round(self.off_success_rate, 4),
            "on_stdev": round(self.on_stdev, 4),
            "off_stdev": round(self.off_stdev, 4),
            "delta": round(self.delta, 4),
            "random_baseline": round(self.random_rate, 4),
            "oracle_ceiling": round(self.oracle_rate, 4),
            "gap_closed": round(self.gap_closed, 4),
        }


def simulate_selection(
    competence: dict[str, float] | None = None,
    *,
    rounds: int = 200,
    repeats: int = 200,
    attention: float = 1.0,
    seed: int = 0,
) -> SimulationResult:
    """Measure the selection *mechanism* in isolation, with no LLM involved.

    Each round a specialist is dispatched and succeeds with its latent
    ``competence``.  The OFF arm picks uniformly at random (the honest model of
    a planner with no evidence to go on).  The ON arm picks the selector's
    top-ranked specialist with probability *attention*, and uniformly
    otherwise — ``attention`` models how much notice the planner LLM takes of
    the ranking it is shown, which on a small local model is well below 1.

    This is a **proxy**: it proves the scoring converges to the better
    specialist given attention, and nothing whatsoever about end-to-end task
    accuracy.  Read it only alongside :func:`run_ablation`.
    """
    from isaac.meta.learner import MetaLearner
    from isaac.meta.specialist_selector import SpecialistSelector

    competence = competence or {
        "coder": 0.9,
        "analyst": 0.7,
        "researcher": 0.5,
        "generalist": 0.3,
    }
    names = list(competence)
    oracle = max(competence.values())
    random_rate = statistics.fmean(competence.values())

    def one_run(use_selection: bool, run_seed: int) -> float:
        rng = random.Random(run_seed)
        tmp = Path(f"{_sim_dir()}/sim-{use_selection}-{run_seed}.db")
        tmp.unlink(missing_ok=True)
        selector = SpecialistSelector(MetaLearner(tmp))
        wins = 0
        for _ in range(rounds):
            if use_selection and rng.random() < attention:
                choice = selector.rank(names)[0]
            else:
                choice = rng.choice(names)
            success = rng.random() < competence[choice]
            wins += int(success)
            selector.record(choice, success=success)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - Windows file locking
            pass
        return wins / rounds

    on = [one_run(True, seed + i) for i in range(repeats)]
    off = [one_run(False, seed + 10_000 + i) for i in range(repeats)]

    return SimulationResult(
        rounds=rounds,
        repeats=repeats,
        on_success_rate=statistics.fmean(on),
        off_success_rate=statistics.fmean(off),
        on_stdev=statistics.stdev(on) if len(on) > 1 else 0.0,
        off_stdev=statistics.stdev(off) if len(off) > 1 else 0.0,
        oracle_rate=oracle,
        random_rate=random_rate,
    )


def _sim_dir() -> Path:
    import tempfile

    d = Path(tempfile.gettempdir()) / "isaac-sim"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(report: AblationReport) -> str:
    """Render an :class:`AblationReport` as a plain-text summary."""
    lines: list[str] = []
    n_on, n_off = report.n_trials("on"), report.n_trials("off")
    lines.append(f"Self-improvement ablation — {report.suite} (hash {report.suite_hash})")
    lines.append(f"  model     : {report.model} ({report.provider})")
    lines.append(f"  git rev   : {report.git_rev or 'unknown'}")
    lines.append(f"  tasks     : {len(report.task_ids)}   runner: team (forced)")
    lines.append(f"  warm-up   : {report.warmup_trials} pass(es)")
    lines.append(f"  trials    : {n_on} ON / {n_off} OFF (paired)")
    lines.append("")
    lines.append(f"  {'arm':<5} {'mean':>8} {'stdev':>8}  per-trial")
    for a in ARMS:
        per = ", ".join(f"{x:.3f}" for x in report.accuracies(a))
        lines.append(f"  {a.upper():<5} {report.mean(a):>8.3f} {report.stdev(a):>8.3f}  [{per}]")
    lines.append("")
    lines.append(f"  delta (ON - OFF) : {report.delta:+.3f} accuracy points")
    lines.append(
        f"  permutation p    : {report.permutation_p():.3f} paired by task "
        f"(by trial: {report.trial_permutation_p():.3f}, underpowered at n={n_on})"
    )
    lines.append(f"  verdict          : {report.verdict.upper()}")

    moved = [r for r in report.task_table() if r["delta"] != 0]
    if moved:
        lines.append("")
        lines.append("  tasks that moved (pass count out of n trials):")
        for r in moved:
            lines.append(f"    {r['task_id']:<22} ON {r['on']}  OFF {r['off']}  ({r['delta']:+d})")
    else:
        lines.append("")
        lines.append("  no task changed its pass count between arms.")

    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"  note: {note}")
    return "\n".join(lines)

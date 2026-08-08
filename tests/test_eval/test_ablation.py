"""Tests for the self-improvement ablation harness (1.5.0, roadmap WS6).

A measurement harness that can quietly flatter its own intervention is worse
than none, so these tests target the properties that keep the result honest:
arms stay isolated, "flat" is reachable, a real effect is still detected, and
the statistics do not manufacture significance out of three trials.

Fully offline — the runner is injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.eval.ablation import (
    AblationReport,
    TrialResult,
    format_report,
    run_ablation,
    simulate_selection,
)
from isaac.eval.runner import TaskAnswer
from isaac.eval.suite import EvalTask


def _tasks(n: int = 4) -> list[EvalTask]:
    return [
        EvalTask(
            id=f"t{i}",
            prompt=f"say ok{i}",
            checks=[{"type": "contains", "value": f"ok{i}"}],
            runner="team",
        )
        for i in range(1, n + 1)
    ]


def _report(on: list[list[bool]], off: list[list[bool]], task_ids: list[str]) -> AblationReport:
    """Build a report straight from pass/fail matrices (no runs involved)."""
    rep = AblationReport(
        suite="s",
        suite_hash="h",
        model="m",
        provider="p",
        git_rev="r",
        task_ids=task_ids,
        warmup_trials=1,
    )
    for arm, matrix in (("on", on), ("off", off)):
        for i, row in enumerate(matrix, 1):
            rep.trials.append(
                TrialResult(arm=arm, trial=i, passed=dict(zip(task_ids, row, strict=True)))
            )
    return rep


class _ScriptedRunner:
    """Injectable runner whose answers depend on the arm, not on a model."""

    def __init__(self, *, on_correct: set[str] | None = None, all_correct: bool = False) -> None:
        self.on_correct = on_correct or set()
        self.all_correct = all_correct
        self.calls: list[tuple[bool, str]] = []

    def __call__(self, *, use_meta_selection: bool, on_plan=None):
        def run(task: EvalTask) -> TaskAnswer:
            self.calls.append((use_meta_selection, task.id))
            if on_plan is not None:
                on_plan(task.id, ["coder"])
            correct = self.all_correct or (use_meta_selection and task.id in self.on_correct)
            return TaskAnswer(text=task.id.replace("t", "ok") if correct else "wrong")

        return run


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_identical_arms_are_flat(self) -> None:
        rows = [[True, True, False, False]] * 3
        rep = _report(rows, rows, ["t1", "t2", "t3", "t4"])

        assert rep.delta == 0.0
        assert rep.permutation_p() == 1.0
        assert rep.verdict == "flat"

    def test_a_one_task_wobble_is_flat_not_positive(self) -> None:
        """The suite cannot resolve a single task flipping once — saying so is
        the whole reason the verdict is computed rather than eyeballed."""
        ids = ["t1", "t2", "t3", "t4"]
        on = [[True, True, True, False], [True, True, False, False], [True, True, False, False]]
        off = [[True, True, False, False]] * 3
        rep = _report(on, off, ids)

        assert rep.delta > 0
        assert rep.verdict == "flat"

    def test_a_large_consistent_gain_is_positive(self) -> None:
        ids = [f"t{i}" for i in range(1, 9)]
        on = [[True] * 8] * 4
        off = [[True, True, False, False, False, False, False, False]] * 4
        rep = _report(on, off, ids)

        assert rep.delta == pytest.approx(0.75)
        assert rep.permutation_p() < 0.05
        assert rep.verdict == "positive"

    def test_a_large_consistent_loss_is_negative(self) -> None:
        ids = [f"t{i}" for i in range(1, 9)]
        on = [[True, True, False, False, False, False, False, False]] * 4
        off = [[True] * 8] * 4
        rep = _report(on, off, ids)

        assert rep.delta < 0
        assert rep.verdict == "negative"

    def test_high_variance_masks_a_small_gap(self) -> None:
        """Noisy arms must not be reported as an effect."""
        ids = [f"t{i}" for i in range(1, 9)]
        on = [[True] * 8, [False] * 8, [True] * 4 + [False] * 4]
        off = [[False] * 8, [True] * 8, [True] * 3 + [False] * 5]
        rep = _report(on, off, ids)

        assert rep.stdev("on") > 0.3
        assert rep.verdict == "flat"


class TestStatistics:
    def test_task_pairing_beats_trial_pairing_for_power(self) -> None:
        """The bug this test locks down: at 3 trials the trial-level test can
        never reach p<0.05, so it must not be the primary statistic."""
        ids = [f"t{i}" for i in range(1, 13)]
        on = [[True] * 12] * 3
        off = [[False] * 12] * 3
        rep = _report(on, off, ids)

        assert rep.permutation_p() < 0.05
        assert rep.trial_permutation_p() > 0.05

    def test_per_trial_accuracies_are_preserved(self) -> None:
        ids = ["t1", "t2"]
        rep = _report([[True, True], [True, False]], [[False, False]] * 2, ids)

        assert rep.accuracies("on") == [1.0, 0.5]
        assert rep.mean("on") == 0.75
        assert rep.stdev("on") > 0

    def test_task_table_flags_what_moved(self) -> None:
        ids = ["t1", "t2"]
        rep = _report([[True, False]] * 2, [[False, False]] * 2, ids)
        table = {r["task_id"]: r for r in rep.task_table()}

        assert table["t1"]["delta"] == 2
        assert table["t2"]["delta"] == 0

    def test_empty_report_does_not_explode(self) -> None:
        rep = AblationReport("s", "h", "m", "p", "r", [], 0)

        assert rep.delta == 0.0
        assert rep.permutation_p() == 1.0
        assert rep.verdict == "flat"


# ---------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------


class TestRunAblation:
    def test_arms_and_trials_are_balanced(self, tmp_path: Path) -> None:
        runner = _ScriptedRunner(all_correct=True)
        rep = run_ablation(
            _tasks(3),
            trials=2,
            warmup_trials=1,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=runner,
        )

        assert rep.n_trials("on") == 2
        assert rep.n_trials("off") == 2
        assert rep.mean("on") == 1.0
        # 3 tasks x (1 warmup + 2 arms x 2 trials)
        assert len(runner.calls) == 15

    def test_each_arm_gets_the_right_flag(self, tmp_path: Path) -> None:
        runner = _ScriptedRunner(all_correct=True)
        run_ablation(
            _tasks(2),
            trials=1,
            warmup_trials=0,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=runner,
        )
        flags = {flag for flag, _ in runner.calls}

        assert flags == {True, False}

    def test_arms_get_separate_history_files(self, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        run_ablation(
            _tasks(2),
            trials=2,
            warmup_trials=1,
            workspace=tmp_path / "ws",
            scratch=scratch,
            runner_factory=_ScriptedRunner(all_correct=True),
        )
        dbs = {p.name for p in scratch.glob("*.db")}

        assert {"on-trial1.db", "off-trial1.db", "on-trial2.db", "off-trial2.db"} <= dbs

    def test_an_arm_only_effect_is_detected(self, tmp_path: Path) -> None:
        """Sanity check the harness can see an effect that is really there."""
        rep = run_ablation(
            _tasks(6),
            trials=3,
            warmup_trials=1,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=_ScriptedRunner(on_correct={f"t{i}" for i in range(1, 7)}),
        )

        assert rep.mean("on") == 1.0
        assert rep.mean("off") == 0.0
        assert rep.verdict == "positive"

    def test_no_warmup_is_annotated_as_a_negative_control(self, tmp_path: Path) -> None:
        rep = run_ablation(
            _tasks(2),
            trials=1,
            warmup_trials=0,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=_ScriptedRunner(all_correct=True),
        )

        assert any("no-op" in n for n in rep.notes)

    def test_a_crashing_task_scores_zero_without_aborting(self, tmp_path: Path) -> None:
        def factory(*, use_meta_selection, on_plan=None):
            def run(task: EvalTask) -> TaskAnswer:
                if task.id == "t2":
                    raise RuntimeError("boom")
                return TaskAnswer(text=task.id.replace("t", "ok"))

            return run

        rep = run_ablation(
            _tasks(3),
            trials=1,
            warmup_trials=0,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=factory,
        )
        trial = rep.arm("on")[0]

        assert trial.passed == {"t1": True, "t2": False, "t3": True}

    def test_checkpoint_is_written_during_the_run(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "partial.json"
        run_ablation(
            _tasks(2),
            trials=1,
            warmup_trials=1,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            checkpoint_path=ckpt,
            runner_factory=_ScriptedRunner(all_correct=True),
        )

        assert ckpt.exists()
        import json

        assert json.loads(ckpt.read_text(encoding="utf-8"))["verdict"] in {
            "flat",
            "positive",
            "negative",
        }

    def test_settings_are_restored_afterwards(self, tmp_path: Path) -> None:
        from isaac.config.settings import get_settings

        before = get_settings().meta_learner_db_path
        run_ablation(
            _tasks(1),
            trials=1,
            warmup_trials=0,
            workspace=tmp_path / "ws",
            scratch=tmp_path / "scratch",
            runner_factory=_ScriptedRunner(all_correct=True),
        )

        assert get_settings().meta_learner_db_path == before


class TestFormatting:
    def test_report_states_n_and_variance(self) -> None:
        ids = ["t1", "t2"]
        text = format_report(_report([[True, True]] * 3, [[True, False]] * 3, ids))

        assert "3 ON / 3 OFF" in text
        assert "stdev" in text
        assert "verdict" in text

    def test_report_names_the_tasks_that_moved(self) -> None:
        text = format_report(_report([[True, False]] * 2, [[False, False]] * 2, ["t1", "t2"]))

        assert "t1" in text
        assert "tasks that moved" in text


# ---------------------------------------------------------------------------
# Mechanism simulation
# ---------------------------------------------------------------------------


class TestSimulation:
    def test_full_attention_recovers_most_of_the_headroom(self) -> None:
        res = simulate_selection(rounds=80, repeats=12, attention=1.0, seed=3)

        assert res.delta > 0
        assert res.gap_closed > 0.5

    def test_zero_attention_is_indistinguishable_from_the_baseline(self) -> None:
        """If the planner ignores the ranking, the mechanism cannot help — the
        simulation must show that rather than crediting it anyway."""
        res = simulate_selection(rounds=80, repeats=12, attention=0.0, seed=3)

        assert abs(res.delta) < 0.1

    def test_result_is_deterministic_for_a_seed(self) -> None:
        a = simulate_selection(rounds=40, repeats=6, seed=11)
        b = simulate_selection(rounds=40, repeats=6, seed=11)

        assert a.to_dict() == b.to_dict()

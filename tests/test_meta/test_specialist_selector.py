"""Tests for MetaLearner-guided specialist selection (1.5.0, roadmap WS6).

The properties that matter for the ablation to be meaningful:

* with no history the selector is an **exact no-op** (otherwise ON vs OFF
  would differ for reasons unrelated to learning);
* a cold specialist is never starved by a specialist with a mediocre record;
* one bad run does not demote a specialist that has a long good record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.meta.learner import MetaLearner
from isaac.meta.specialist_selector import (
    SPECIALIST_TASK_TYPE,
    SpecialistScore,
    SpecialistSelector,
)

ROSTER = ["coder", "researcher", "analyst", "generalist"]


@pytest.fixture()
def learner(tmp_path: Path) -> MetaLearner:
    return MetaLearner(tmp_path / "meta.db")


@pytest.fixture()
def selector(learner: MetaLearner) -> SpecialistSelector:
    return SpecialistSelector(learner)


def _seed(selector: SpecialistSelector, name: str, wins: int, losses: int) -> None:
    for _ in range(wins):
        selector.record(name, success=True)
    for _ in range(losses):
        selector.record(name, success=False)


class TestScoring:
    def test_untried_specialist_scores_the_prior(self, selector: SpecialistSelector) -> None:
        assert selector.scores(["coder"])["coder"].score == pytest.approx(0.7)

    def test_smoothing_pulls_a_single_win_toward_the_prior(
        self, selector: SpecialistSelector
    ) -> None:
        _seed(selector, "coder", wins=1, losses=0)
        score = selector.scores(["coder"])["coder"]

        assert score.raw_win_rate == 1.0  # naive ranking would call this perfect
        assert score.score == pytest.approx((1 + 0.7 * 3) / (1 + 3))
        assert score.score < 1.0
        assert score.is_cold

    def test_a_long_good_record_outranks_a_lucky_single_win(
        self, selector: SpecialistSelector
    ) -> None:
        _seed(selector, "coder", wins=18, losses=2)
        _seed(selector, "analyst", wins=1, losses=0)

        assert selector.rank(["analyst", "coder"]) == ["coder", "analyst"]

    def test_one_bad_run_does_not_demote_a_proven_specialist(
        self, selector: SpecialistSelector
    ) -> None:
        _seed(selector, "coder", wins=20, losses=1)
        before = selector.scores(["coder"])["coder"].score
        _seed(selector, "coder", wins=0, losses=1)
        after = selector.scores(["coder"])["coder"].score

        assert after < before
        assert after > 0.8  # still comfortably the best choice


class TestColdStart:
    def test_cold_specialist_beats_a_mediocre_one(self, selector: SpecialistSelector) -> None:
        """The optimistic prior is what stops exploration from being starved."""
        _seed(selector, "coder", wins=2, losses=5)

        assert selector.rank(["coder", "researcher"]) == ["researcher", "coder"]

    def test_cold_specialist_does_not_beat_a_strong_one(self, selector: SpecialistSelector) -> None:
        _seed(selector, "coder", wins=9, losses=1)

        assert selector.rank(["researcher", "coder"]) == ["coder", "researcher"]


class TestNoOpWithoutHistory:
    def test_rank_is_identity_on_an_empty_history(self, selector: SpecialistSelector) -> None:
        assert selector.rank(ROSTER) == ROSTER
        assert selector.rank(list(reversed(ROSTER))) == list(reversed(ROSTER))

    def test_annotate_roster_preserves_order_without_history(
        self, selector: SpecialistSelector
    ) -> None:
        roster = [{"name": n, "domain": f"{n} stuff"} for n in ROSTER]
        annotated = selector.annotate_roster(roster)

        assert [c["name"] for c in annotated] == ROSTER
        assert all(c["track_record"] == "no track record yet" for c in annotated)

    def test_missing_learner_degrades_to_priors(self) -> None:
        class _Broken:
            def get_best_strategy(self, task_type):
                raise RuntimeError("db is gone")

        selector = SpecialistSelector(_Broken())
        assert selector.rank(ROSTER) == ROSTER


class TestRosterAnnotation:
    def test_roster_is_reordered_and_annotated(self, selector: SpecialistSelector) -> None:
        _seed(selector, "analyst", wins=10, losses=0)
        _seed(selector, "coder", wins=0, losses=6)
        roster = [{"name": n, "domain": ""} for n in ROSTER]

        annotated = selector.annotate_roster(roster)

        assert annotated[0]["name"] == "analyst"
        assert annotated[-1]["name"] == "coder"
        assert annotated[0]["track_record"] == "10/10 succeeded (100%)"
        assert annotated[-1]["track_record"] == "0/6 succeeded (0%)"

    def test_annotation_does_not_mutate_the_input(self, selector: SpecialistSelector) -> None:
        roster = [{"name": "coder", "domain": "code"}]
        selector.annotate_roster(roster)

        assert roster == [{"name": "coder", "domain": "code"}]


class TestRecording:
    def test_outcomes_land_in_the_specialist_bucket(
        self, selector: SpecialistSelector, learner: MetaLearner
    ) -> None:
        selector.record("coder", success=True, task_desc="write a parser")

        rows = learner.get_best_strategy(SPECIALIST_TASK_TYPE)
        assert [r["strategy"] for r in rows] == ["coder"]
        assert rows[0]["wins"] == 1

    def test_orchestration_rows_do_not_contaminate_specialist_scores(
        self, selector: SpecialistSelector, learner: MetaLearner
    ) -> None:
        learner.record(task_desc="g", task_type="orchestration", strategy="coder", success=False)
        learner.record(task_desc="g", task_type="orchestration", strategy="coder", success=False)
        selector.record("coder", success=True)

        assert selector.scores(["coder"])["coder"].losses == 0

    def test_names_are_normalised(self, selector: SpecialistSelector) -> None:
        selector.record("  Coder  ", success=True)
        assert selector.scores(["coder"])["coder"].wins == 1


class TestSpecialistScore:
    def test_dict_view_reports_raw_and_smoothed(self) -> None:
        d = SpecialistScore("coder", wins=3, losses=1).to_dict()

        assert d["runs"] == 4
        assert d["raw_win_rate"] == 0.75
        assert d["score"] != d["raw_win_rate"]
        assert d["cold"] is False

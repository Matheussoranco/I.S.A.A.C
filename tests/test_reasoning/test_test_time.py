"""Tests for test-time compute scaling.

Samplers here are deterministic sequences rather than LLM calls, so the tests
assert on the *policy* — how many samples were drawn, when it stopped, which
answer won — without any model in the loop.
"""

from __future__ import annotations

import time

from isaac.reasoning.test_time import (
    aggregate,
    best_of_n,
    self_consistency,
    solve_hard_step,
)


def _sequence(items: list):
    """A sampler that yields *items* in order, repeating the last forever."""
    state = {"i": 0}

    def sample():
        i = state["i"]
        state["i"] += 1
        return items[min(i, len(items) - 1)]

    return sample


def _counting(items: list) -> tuple:
    """Like :func:`_sequence` but also reports how many draws happened."""
    calls = {"n": 0}
    seq = _sequence(items)

    def sample():
        calls["n"] += 1
        return seq()

    return sample, calls


class TestSelfConsistency:
    def test_majority_wins(self) -> None:
        result = self_consistency(_sequence(["a", "b", "a", "a", "b"]), n=5)
        assert result.answer == "a"
        assert result.agreement == 3 / 5

    def test_unanimous_agreement_is_one(self) -> None:
        assert self_consistency(_sequence(["x"]), n=4).agreement == 1.0

    def test_formatting_differences_do_not_split_the_vote(self) -> None:
        result = self_consistency(_sequence(["Paris", " paris ", "PARIS", "Lyon"]), n=4)
        assert result.answer in {"Paris", " paris ", "PARIS"}
        assert result.agreement == 0.75

    def test_custom_key_controls_bucketing(self) -> None:
        result = self_consistency(
            _sequence([1.01, 1.02, 2.5]), n=3, key=lambda v: str(round(float(v)))
        )
        assert result.agreement == 2 / 3

    def test_min_agreement_rejects_a_weak_winner(self) -> None:
        result = self_consistency(_sequence(["a", "b", "c"]), n=3, min_agreement=0.6)
        assert result.answer is None
        assert result.agreement == 1 / 3

    def test_draws_exactly_n_samples(self) -> None:
        sampler, calls = _counting(["a"])
        self_consistency(sampler, n=7)
        assert calls["n"] == 7

    def test_sampler_exceptions_do_not_abort_the_vote(self) -> None:
        state = {"i": 0}

        def flaky():
            state["i"] += 1
            if state["i"] == 2:
                raise RuntimeError("transient")
            return "ok"

        result = self_consistency(flaky, n=4)
        assert result.answer == "ok"
        assert result.n_sampled == 3

    def test_all_samples_failing_yields_no_answer(self) -> None:
        def broken():
            raise RuntimeError("always")

        assert self_consistency(broken, n=3).answer is None

    def test_dict_answers_are_comparable(self) -> None:
        result = self_consistency(_sequence([{"a": 1}, {"a": 1}, {"b": 2}]), n=3)
        assert result.answer == {"a": 1}


class TestBestOfN:
    def test_exits_on_first_accepted_sample(self) -> None:
        sampler, calls = _counting(["good", "bad", "bad"])
        result = best_of_n(sampler, lambda s: s == "good", n=5)
        assert result.answer == "good"
        assert result.verified is True
        assert result.exited_early is True
        assert calls["n"] == 1, "must not keep sampling after acceptance"

    def test_keeps_sampling_until_one_passes(self) -> None:
        sampler, calls = _counting(["bad", "bad", "good"])
        result = best_of_n(sampler, lambda s: s == "good", n=5)
        assert result.verified is True
        assert calls["n"] == 3

    def test_returns_best_effort_when_nothing_passes(self) -> None:
        result = best_of_n(_sequence(["a", "b"]), lambda s: 0.4 if s == "b" else 0.1, n=2)
        assert result.verified is False
        assert result.answer == "b"
        assert result.score == 0.4

    def test_bool_verifier_is_accepted(self) -> None:
        assert best_of_n(_sequence(["x"]), lambda s: True, n=3).verified is True

    def test_broken_verifier_scores_zero_rather_than_raising(self) -> None:
        def bad(_sample):
            raise ValueError("verifier bug")

        result = best_of_n(_sequence(["a"]), bad, n=2)
        assert result.verified is False

    def test_no_early_exit_flag_when_last_sample_passes(self) -> None:
        result = best_of_n(_sequence(["bad", "good"]), lambda s: s == "good", n=2)
        assert result.verified is True
        assert result.exited_early is False

    def test_budget_stops_sampling(self) -> None:
        def slow():
            time.sleep(0.05)
            return "no"

        result = best_of_n(slow, lambda s: False, n=100, budget_s=0.15)
        assert result.n_sampled < 100


class TestSolveHardStep:
    def test_greedy_sample_short_circuits_when_it_verifies(self) -> None:
        sampler, calls = _counting(["good"])
        result = solve_hard_step(sampler, verifier=lambda s: s == "good", n=5)
        assert result.strategy == "greedy"
        assert result.verified is True
        assert calls["n"] == 1, "an easy step must cost exactly one sample"

    def test_escalates_to_best_of_n_when_greedy_fails(self) -> None:
        sampler, calls = _counting(["bad", "bad", "good"])
        result = solve_hard_step(sampler, verifier=lambda s: s == "good", n=5)
        assert result.verified is True
        assert result.strategy == "best_of_n"
        assert calls["n"] == 3

    def test_falls_back_to_voting_when_the_verifier_never_accepts(self) -> None:
        result = solve_hard_step(_sequence(["a", "a", "a", "b"]), verifier=lambda s: 0.0, n=4)
        assert result.strategy == "best_of_n+vote"
        assert result.answer == "a"

    def test_without_a_verifier_it_votes(self) -> None:
        result = solve_hard_step(_sequence(["a", "b", "a"]), verifier=None, n=3)
        assert result.strategy == "self_consistency"
        assert result.answer == "a"

    def test_greedy_sample_is_counted_in_total_spend(self) -> None:
        result = solve_hard_step(_sequence(["bad", "bad", "good"]), lambda s: s == "good", n=5)
        assert result.n_sampled == 3

    def test_records_elapsed_time(self) -> None:
        result = solve_hard_step(_sequence(["a"]), lambda s: True, n=2)
        assert result.elapsed_s >= 0.0


class TestAggregate:
    def test_empty_batch(self) -> None:
        assert aggregate([])["runs"] == 0

    def test_summarises_a_batch(self) -> None:
        results = [
            solve_hard_step(_sequence(["good"]), lambda s: s == "good", n=3),
            solve_hard_step(_sequence(["bad"]), lambda s: s == "good", n=2),
        ]
        summary = aggregate(results)
        assert summary["runs"] == 2
        assert summary["verified"] == 1
        assert summary["verified_rate"] == 0.5

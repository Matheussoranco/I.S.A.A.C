"""Tests for the tool-call reliability harness.

Scored offline against scripted turns: these check that the *metric* is
computed correctly, which is what the release note's number depends on.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from isaac.eval.toolcall import STUB_TOOLS, SUITE, ToolCallCase, run_suite, stub_tool_schemas


class ReplayLLM:
    """Returns a fixed turn for every case."""

    def __init__(self, turns: list[Any]) -> None:
        self._turns = turns
        self._i = 0

    def bind_tools(self, schemas: Any) -> ReplayLLM:
        return self

    def bind(self, **kwargs: Any) -> ReplayLLM:
        return self

    def invoke(self, messages: list[Any]) -> Any:
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        if isinstance(turn, Exception):
            raise turn
        return turn


def _native(name: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "x"}, "id": "c1", "type": "tool_call"}],
    )


CASES = [ToolCallCase("t1", "do a search", "web_search")]


class TestSuiteDefinition:
    def test_suite_has_twenty_cases(self) -> None:
        assert len(SUITE) == 20

    def test_case_ids_are_unique(self) -> None:
        assert len({c.id for c in SUITE}) == len(SUITE)

    def test_every_expected_tool_exists_in_the_stub_set(self) -> None:
        names = {t["name"] for t in STUB_TOOLS}
        assert all(c.expect_tool in names for c in SUITE)

    def test_every_stub_tool_is_exercised(self) -> None:
        covered = {c.expect_tool for c in SUITE}
        assert covered == {t["name"] for t in STUB_TOOLS}

    def test_schemas_are_openai_shaped(self) -> None:
        schemas = stub_tool_schemas()
        assert all(s["type"] == "function" and "name" in s["function"] for s in schemas)


class TestScoring:
    def test_native_call_scores_native(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([_native("web_search")]))
        assert report.summary()["native"] == 1
        assert report.malformed_rate == 0.0

    def test_fenced_json_scores_repaired(self) -> None:
        turn = AIMessage(
            content='```json\n{"name": "web_search", "arguments": {"query": "x"}}\n```'
        )
        report = run_suite("fake", cases=CASES, mode="repair", llm=ReplayLLM([turn]))
        assert report.summary()["repaired"] == 1
        assert report.malformed_rate == 1.0

    def test_same_output_scores_unrecovered_in_native_mode(self) -> None:
        # The baseline must not be credited with repair it never performed.
        turn = AIMessage(
            content='```json\n{"name": "web_search", "arguments": {"query": "x"}}\n```'
        )
        report = run_suite("fake", cases=CASES, mode="native", llm=ReplayLLM([turn]))
        summary = report.summary()
        assert summary["unrecovered"] == 1
        assert summary["repaired"] == 0
        assert summary["salvageable"] == 1, "but it is recorded as recoverable"

    def test_wrong_tool_is_not_counted_as_malformed(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([_native("code")]))
        summary = report.summary()
        assert summary["wrong_tool"] == 1
        assert summary["malformed"] == 0, "picking the wrong tool is a reasoning error"

    def test_plain_answer_scores_no_attempt(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([AIMessage(content="Paris.")]))
        assert report.summary()["no_attempt"] == 1
        assert report.attempts == 0

    def test_no_attempt_is_excluded_from_the_malformed_denominator(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([AIMessage(content="Paris.")]))
        assert report.malformed_rate == 0.0, "no attempt means nothing to malform"

    def test_provider_error_is_recorded_not_raised(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([RuntimeError("502")]))
        assert report.summary()["error"] == 1

    def test_reflexion_recovers_after_a_corrective_retry(self) -> None:
        broken = AIMessage(content='I will call the web_search tool with {"query": ')
        llm = ReplayLLM([broken, _native("web_search")])
        report = run_suite("fake", cases=CASES, mode="repair", llm=llm)
        assert report.summary()["reflexion"] == 1

    def test_reflexion_can_be_switched_off(self) -> None:
        broken = AIMessage(content='I will call the web_search tool with {"query": ')
        llm = ReplayLLM([broken, _native("web_search")])
        report = run_suite("fake", cases=CASES, mode="repair", reflexion=False, llm=llm)
        assert report.summary()["unrecovered"] == 1

    def test_constrained_envelope_counts_as_native(self) -> None:
        turn = AIMessage(content='{"tool": "web_search", "arguments": {"query": "x"}}')
        report = run_suite("fake", cases=CASES, mode="constrained", llm=ReplayLLM([turn]))
        assert report.summary()["native"] == 1


class TestBeforeAfterDelta:
    """The before/after pair must come from the same model turns."""

    def test_baseline_counts_native_only(self) -> None:
        turn = AIMessage(content='{"name": "web_search", "arguments": {"query": "x"}}')
        report = run_suite("fake", cases=CASES, mode="repair", llm=ReplayLLM([turn]))
        assert report.baseline_usable == 0, "1.3.x would have returned the blob"
        assert report.usable == 1, "1.4.0 executes it"

    def test_native_calls_count_for_both(self) -> None:
        report = run_suite("fake", cases=CASES, llm=ReplayLLM([_native("web_search")]))
        assert report.baseline_usable == report.usable == 1

    def test_rates_are_zero_for_an_empty_suite(self) -> None:
        report = run_suite("fake", cases=[], llm=ReplayLLM([]))
        assert report.malformed_rate == 0.0
        assert report.usable_rate == 0.0
        assert report.baseline_usable_rate == 0.0


class TestReporting:
    @pytest.fixture
    def report(self):
        return run_suite("fake", cases=CASES, llm=ReplayLLM([_native("web_search")]))

    def test_render_includes_the_headline_metric(self, report) -> None:
        assert "MALFORMED RATE" in report.render()

    def test_render_shows_both_sides_of_the_comparison(self, report) -> None:
        text = report.render()
        assert "BEFORE" in text
        assert "AFTER" in text

    def test_constrained_mode_reports_no_baseline(self) -> None:
        # The envelope replaces the native channel rather than recovering from
        # it, so a "BEFORE" figure here would compare the run against itself.
        turn = AIMessage(content='{"tool": "web_search", "arguments": {"query": "x"}}')
        text = run_suite("fake", cases=CASES, mode="constrained", llm=ReplayLLM([turn])).render()
        assert "BEFORE" not in text
        assert "no 1.3.x baseline" in text

    def test_json_round_trips(self, report) -> None:
        import json

        data = json.loads(report.to_json())
        assert data["summary"]["cases"] == 1
        assert len(data["cases"]) == 1

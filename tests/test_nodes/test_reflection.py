"""Tests for the Reflection / Critic node."""

from __future__ import annotations

from unittest.mock import patch

from isaac.core.state import ExecutionResult, PlanStep, make_initial_state
from isaac.nodes.reflection import reflection_node
from tests.conftest import MockLLM


class TestReflectionNode:
    def test_success_produces_skill_candidate(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="print 42", status="active")]
        state["code_buffer"] = "print(42)"
        state["execution_logs"] = [
            ExecutionResult(stdout="42\n", stderr="", exit_code=0, duration_ms=100.0)
        ]

        mock = MockLLM(
            '{"success": true, "summary": "printed 42", '
            '"skill_candidate": {"name": "print_number", "description": "prints a number"}}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert result["skill_candidate"] is not None
        assert result["skill_candidate"].name == "print_number"
        assert result["plan"][0].status == "done"

    def test_failure_appends_error(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="fail", status="active")]
        state["code_buffer"] = "raise ValueError()"
        state["execution_logs"] = [
            ExecutionResult(stdout="", stderr="ValueError", exit_code=1, duration_ms=50.0)
        ]

        mock = MockLLM(
            '{"success": false, "diagnosis": "ValueError raised", '
            '"revised_hypothesis": "handle errors"}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0].node == "reflection"
        assert result["hypothesis"] == "handle errors"

    def test_malformed_json_treated_as_failure(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="x", status="active")]
        state["code_buffer"] = "x"
        state["execution_logs"] = [ExecutionResult()]

        mock = MockLLM("not json!!!")
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert "errors" in result
        assert len(result["errors"]) == 1

"""Tests for the Reflection / Critic node."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import MockLLM

from isaac.core.state import (
    ExecutionResult,
    PlanStep,
    UIAction,
    UIActionResult,
    make_initial_state,
)
from isaac.nodes.reflection import reflection_node


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

    # ------------------------------------------------------------------
    # UI / Computer-Use visual-diff path
    # ------------------------------------------------------------------

    def _make_ui_state(self, llm_response: str):
        """Build a state wired for the computer_use task_mode."""
        state = make_initial_state()
        state["task_mode"] = "computer_use"
        state["plan"] = [
            PlanStep(id="s1", description="click login button", status="active", mode="ui")
        ]
        state["ui_results"] = [
            UIActionResult(
                action=UIAction(type="click", x=100, y=200, description="click login"),
                success=True,
                screenshot_before_b64="before_b64",
                screenshot_after_b64="after_b64",
            )
        ]
        return state, MockLLM(llm_response)

    def test_ui_success_produces_ui_skill_candidate(self) -> None:
        state, mock = self._make_ui_state(
            '{"success": true, "diagnosis": "logged in", '
            '"skill_candidate": {"name": "login_click", "tags": ["ui", "playwright"]}}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert result["plan"][0].status == "done"
        candidate = result.get("skill_candidate")
        assert candidate is not None
        assert candidate.skill_type == "ui"
        assert candidate.name == "login_click"

    def test_ui_failure_appends_error_with_corrective_hint(self) -> None:
        state, mock = self._make_ui_state(
            '{"success": false, "diagnosis": "element not found", '
            '"revised_hypothesis": "try scrolling first", '
            '"corrective_action": "scroll down 200px"}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert result["plan"][0].status == "failed"
        assert "errors" in result
        err = result["errors"][0]
        assert err.node == "reflection"
        assert "scroll down" in err.message
        assert result["hypothesis"] == "try scrolling first"

    def test_ui_no_results_treated_as_failure(self) -> None:
        """If there are no ui_results, fall back to failure without crashing."""
        state = make_initial_state()
        state["task_mode"] = "computer_use"
        state["plan"] = [PlanStep(id="s1", description="open app", status="active", mode="ui")]
        state["ui_results"] = []  # empty

        mock = MockLLM('{"success": true}')  # LLM won't even be called in this path
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = reflection_node(state)

        assert result["plan"][0].status == "failed"
        assert "errors" in result

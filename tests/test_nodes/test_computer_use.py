"""Tests for the ComputerUse node."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from tests.conftest import MockLLM

from isaac.core.state import (
    GUIState,
    PlanStep,
    UIAction,
    UIActionResult,
    make_initial_state,
)
from isaac.nodes.computer_use import (
    _dict_to_ui_action,
    _parse_llm_decision,
    computer_use_node,
    shutdown_ui_executor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_B64 = "aGVsbG8="  # "hello" base64 — non-empty placeholder


def _make_mock_executor(screenshot_b64: str = _DUMMY_B64) -> MagicMock:
    """Build a UIExecutor mock with sensible defaults."""
    executor = MagicMock()
    gui_state = GUIState(screenshot_b64=screenshot_b64)
    executor.get_gui_state.return_value = gui_state
    executor.act.return_value = UIActionResult(
        action=UIAction(type="click", x=100, y=100, description="click"),
        success=True,
        screenshot_before_b64=_DUMMY_B64,
        screenshot_after_b64=_DUMMY_B64,
    )
    return executor


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------


class TestParseDecision:
    def test_valid_done_json(self) -> None:
        result = _parse_llm_decision('{"done": true, "summary": "ok"}')
        assert result["done"] is True
        assert result["summary"] == "ok"

    def test_valid_action_json(self) -> None:
        result = _parse_llm_decision(
            '{"done": false, "action": {"type": "click", "x": 10, "y": 20}}'
        )
        assert result["done"] is False
        assert result["action"]["type"] == "click"

    def test_malformed_falls_back_to_screenshot(self) -> None:
        result = _parse_llm_decision("not json")
        assert result["done"] is False
        assert result["action"]["type"] == "screenshot"

    def test_fenced_json_stripped(self) -> None:
        content = '```json\n{"done": true, "summary": "done"}\n```'
        result = _parse_llm_decision(content)
        assert result["done"] is True


class TestDictToUIAction:
    def test_click_action(self) -> None:
        action = _dict_to_ui_action({"type": "click", "x": 50, "y": 75, "description": "btn"})
        assert action.type == "click"
        assert action.x == 50
        assert action.y == 75

    def test_defaults_applied(self) -> None:
        action = _dict_to_ui_action({})
        assert action.type == "screenshot"
        assert action.scroll_amount == 3


# ---------------------------------------------------------------------------
# Node integration tests (mocked UIExecutor + LLM)
# ---------------------------------------------------------------------------


class TestComputerUseNode:
    def test_success_marks_step_done_and_builds_skill_candidate(self) -> None:
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="open browser", status="active", mode="ui")
        ]

        mock_llm = MockLLM('{"done": true, "summary": "Browser opened."}')
        mock_exec = _make_mock_executor()

        import isaac.nodes.computer_use as cu_mod

        with (
            patch.object(cu_mod, "_get_ui_executor", return_value=mock_exec),
            patch("isaac.llm.provider.get_llm", return_value=mock_llm),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.graph.max_ui_cycles = 5
            result = computer_use_node(state)

        assert result["plan"][0].status == "done"
        assert result["current_phase"] == "computer_use"
        candidate = result.get("skill_candidate")
        assert candidate is not None
        assert candidate.skill_type == "ui"
        assert "ui" in (candidate.tags or [])

    def test_max_cycles_exhaustion_marks_step_failed(self) -> None:
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="open browser", status="active", mode="ui")
        ]

        # LLM always says "not done yet"
        mock_llm = MockLLM(
            '{"done": false, "action": {"type": "click", "x": 10, "y": 10, "description": "try"}}'
        )
        mock_exec = _make_mock_executor()

        import isaac.nodes.computer_use as cu_mod

        with (
            patch.object(cu_mod, "_get_ui_executor", return_value=mock_exec),
            patch("isaac.llm.provider.get_llm", return_value=mock_llm),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.graph.max_ui_cycles = 2
            result = computer_use_node(state)

        assert result["plan"][0].status == "failed"
        assert result["current_phase"] == "computer_use"
        assert "errors" in result
        assert len(result["errors"]) == 1
        assert result["errors"][0].node == "computer_use"
        assert "Max UI cycles" in result["errors"][0].message
        # Cycle counter should equal max_cycles
        assert result["ui_cycle"] == 2

    def test_no_active_step_returns_early(self) -> None:
        state = make_initial_state()
        state["plan"] = [PlanStep(id="s1", description="done step", status="done")]

        import isaac.nodes.computer_use as cu_mod

        with (
            patch.object(cu_mod, "_get_ui_executor", return_value=_make_mock_executor()),
            patch("isaac.llm.provider.get_llm", return_value=MockLLM()),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.graph.max_ui_cycles = 5
            result = computer_use_node(state)

        assert result["current_phase"] == "computer_use"
        assert "errors" not in result

    def test_empty_screenshot_aborts_loop(self) -> None:
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="click button", status="active", mode="ui")
        ]

        # Executor returns empty screenshot → loop should abort immediately
        mock_exec = _make_mock_executor(screenshot_b64="")

        import isaac.nodes.computer_use as cu_mod

        with (
            patch.object(cu_mod, "_get_ui_executor", return_value=mock_exec),
            patch("isaac.llm.provider.get_llm", return_value=MockLLM('{"done": false}')),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.graph.max_ui_cycles = 5
            result = computer_use_node(state)

        # Should mark step failed (max cycles not reached, but loop broken)
        assert result["plan"][0].status == "failed"

    def test_skill_candidate_code_buffer_contains_screenshots(self) -> None:
        """On success the code_buffer JSON must expose before/after screenshots."""
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="fill form", status="active", mode="ui")
        ]

        mock_exec = _make_mock_executor()
        # First call: not done (so it executes an action)
        # Second call: done
        mock_llm_responses = [
            '{"done": false, "action": {"type": "click", "x": 5, "y": 5, "description": "click"}}',
            '{"done": true, "summary": "Form filled."}',
        ]
        call_count = 0

        class SequentialMockLLM:
            def invoke(self, _):
                nonlocal call_count
                resp = mock_llm_responses[min(call_count, len(mock_llm_responses) - 1)]
                call_count += 1
                return type("R", (), {"content": resp})()

        import isaac.nodes.computer_use as cu_mod

        with (
            patch.object(cu_mod, "_get_ui_executor", return_value=mock_exec),
            patch("isaac.llm.provider.get_llm", return_value=SequentialMockLLM()),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.graph.max_ui_cycles = 5
            result = computer_use_node(state)

        assert result["plan"][0].status == "done"
        buf = json.loads(result.get("code_buffer", "{}"))
        # Both screenshot keys should be present (may be empty string if no results yet)
        assert "screenshot_before" in buf
        assert "screenshot_after" in buf


# ---------------------------------------------------------------------------
# shutdown_ui_executor
# ---------------------------------------------------------------------------


class TestShutdownUIExecutor:
    def test_shutdown_idempotent_when_none(self) -> None:
        """Calling shutdown when no executor is running must not raise."""
        import isaac.nodes.computer_use as cu_mod

        original = cu_mod._ui_executor
        cu_mod._ui_executor = None
        try:
            shutdown_ui_executor()  # must not raise
        finally:
            cu_mod._ui_executor = original

    def test_shutdown_calls_stop(self) -> None:
        """shutdown_ui_executor must call stop() on the active executor."""
        import isaac.nodes.computer_use as cu_mod

        mock_exec = MagicMock()
        original = cu_mod._ui_executor
        cu_mod._ui_executor = mock_exec
        try:
            shutdown_ui_executor()
            mock_exec.stop.assert_called_once()
            assert cu_mod._ui_executor is None
        finally:
            cu_mod._ui_executor = original

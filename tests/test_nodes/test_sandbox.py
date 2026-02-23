"""Tests for the Sandbox node (mocked Docker)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaac.core.state import ExecutionResult, make_initial_state
from isaac.nodes.sandbox import sandbox_node


class TestSandboxNode:
    def test_empty_code_buffer(self) -> None:
        state = make_initial_state()
        state["code_buffer"] = ""
        result = sandbox_node(state)
        logs = result["execution_logs"]
        assert len(logs) == 1
        assert logs[0].exit_code == 1

    def test_delegates_to_executor(self) -> None:
        state = make_initial_state()
        state["code_buffer"] = "print('hello')"

        mock_result = ExecutionResult(
            stdout="hello\n", stderr="", exit_code=0, duration_ms=100.0
        )
        mock_executor = MagicMock()
        mock_executor.execute.return_value = mock_result

        with patch("isaac.nodes.sandbox.CodeExecutor", return_value=mock_executor):
            result = sandbox_node(state)

        logs = result["execution_logs"]
        assert len(logs) == 1
        assert logs[0].exit_code == 0
        assert logs[0].stdout == "hello\n"
        mock_executor.close.assert_called_once()

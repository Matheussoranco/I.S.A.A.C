"""ARC candidate code remains inside the Docker execution boundary."""

from unittest.mock import MagicMock, patch

import numpy as np

from isaac.arc.evaluator import ArcPair, ArcTask
from isaac.arc.refinement import _training_accuracy
from isaac.arc.sandbox_runner import run_candidate
from isaac.core.state import ExecutionResult


def test_candidate_predictions_come_from_sandbox_result() -> None:
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        stdout="ISAAC_ARC_RESULT_fixed:[[ [1] ]]\n",
        stderr="",
        exit_code=0,
        duration_ms=1,
    )
    with (
        patch("isaac.arc.sandbox_runner.secrets.token_hex", return_value="fixed"),
        patch("isaac.arc.sandbox_runner.CodeExecutor", return_value=executor),
    ):
        predictions = run_candidate("def solve(grid): return grid", [np.array([[1]])])
    assert predictions is not None
    assert np.array_equal(predictions[0], np.array([[1]]))
    assert "exec(" in executor.execute.call_args.args[0]
    executor.close.assert_called_once()


def test_candidate_fails_closed_without_sandbox() -> None:
    with patch("isaac.arc.sandbox_runner.CodeExecutor", side_effect=RuntimeError("no Docker")):
        assert run_candidate("def solve(grid): return grid", [np.array([[1]])]) is None


def test_fractional_sandbox_payload_is_rejected_without_casting() -> None:
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        stdout="ISAAC_ARC_RESULT_fixed:[[[1.9]]]",
        stderr="",
        exit_code=0,
        duration_ms=1,
    )
    with (
        patch("isaac.arc.sandbox_runner.secrets.token_hex", return_value="fixed"),
        patch("isaac.arc.sandbox_runner.CodeExecutor", return_value=executor),
    ):
        assert run_candidate("def solve(grid): return grid", [np.array([[1]])]) is None


def test_refinement_validation_uses_sandbox_predictions() -> None:
    pair = ArcPair(input=np.array([[0]]), output=np.array([[1]]))
    task = ArcTask(id="safe", train=[pair], test=[])
    with patch("isaac.arc.sandbox_runner.run_candidate", return_value=[pair.output]) as runner:
        accuracy, failures = _training_accuracy("raise RuntimeError('host exec')", task)
    assert accuracy == 1.0
    assert failures == []
    runner.assert_called_once()

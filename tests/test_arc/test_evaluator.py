"""Tests for the ARC-AGI evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac.arc.evaluator import (
    ArcPair,
    ArcTask,
    EvalReport,
    TaskResult,
    build_arc_prompt,
    evaluate,
    load_tasks,
    solve_with_dsl,
)


def _make_rotation_task() -> ArcTask:
    """Create a task where the answer is a 90° clockwise rotation."""
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=-1)
    return ArcTask(
        id="rotation_test",
        train=[ArcPair(input=inp, output=out)],
        test=[ArcPair(input=inp, output=out)],
    )


def _make_flip_task() -> ArcTask:
    """Create a task where the answer is a horizontal flip."""
    inp = np.array([[1, 2, 3], [4, 5, 6]])
    out = np.fliplr(inp)
    return ArcTask(
        id="flip_test",
        train=[ArcPair(input=inp, output=out)],
        test=[ArcPair(input=inp, output=out)],
    )


def _make_unsolvable_task() -> ArcTask:
    """Create a task that DSL search cannot solve."""
    inp = np.array([[1, 2], [3, 4]])
    out = np.array([[99, 98], [97, 96]])
    return ArcTask(
        id="unsolvable",
        train=[ArcPair(input=inp, output=out)],
        test=[ArcPair(input=inp, output=out)],
    )


class TestTaskLoading:
    def test_load_single_task(self, tmp_path: Path) -> None:
        task_data = [{
            "id": "test_001",
            "train": [{"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]}],
            "test": [{"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]}],
        }]
        task_file = tmp_path / "task.json"
        task_file.write_text(json.dumps(task_data), encoding="utf-8")

        tasks = load_tasks(task_file)
        assert len(tasks) == 1
        assert tasks[0].id == "test_001"
        assert len(tasks[0].train) == 1
        assert len(tasks[0].test) == 1
        assert np.array_equal(tasks[0].train[0].input, np.array([[0, 1], [2, 3]]))

    def test_load_single_object(self, tmp_path: Path) -> None:
        """Test loading a single task object (not wrapped in a list)."""
        task_data = {
            "id": "solo",
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        task_file = tmp_path / "solo.json"
        task_file.write_text(json.dumps(task_data), encoding="utf-8")

        tasks = load_tasks(task_file)
        assert len(tasks) == 1
        assert tasks[0].id == "solo"


class TestDSLSolver:
    def test_solves_rotation(self) -> None:
        task = _make_rotation_task()
        result = solve_with_dsl(task)
        assert result.correct is True
        assert result.method in ("dsl_single", "dsl_compose_2")

    def test_solves_flip(self) -> None:
        task = _make_flip_task()
        result = solve_with_dsl(task)
        assert result.correct is True

    def test_unsolvable_returns_false(self) -> None:
        task = _make_unsolvable_task()
        result = solve_with_dsl(task)
        assert result.correct is False
        assert result.method == "dsl_search"

    def test_result_has_timing(self) -> None:
        task = _make_rotation_task()
        result = solve_with_dsl(task)
        assert result.solve_time_ms > 0


class TestBuildArcPrompt:
    def test_prompt_contains_training_data(self) -> None:
        task = _make_rotation_task()
        prompt = build_arc_prompt(task)
        assert "Training Example 1" in prompt
        assert "Test Input 1" in prompt
        assert "solve(grid" in prompt

    def test_prompt_includes_structural_analysis(self) -> None:
        task = _make_rotation_task()
        prompt = build_arc_prompt(task)
        assert "Shape changed" in prompt
        assert "Cells changed" in prompt


class TestEvaluate:
    def test_evaluate_dsl_only(self) -> None:
        tasks = [_make_rotation_task(), _make_flip_task(), _make_unsolvable_task()]
        report = evaluate(tasks, solver="dsl")
        assert isinstance(report, EvalReport)
        assert report.total_tasks == 3
        assert report.correct == 2
        assert len(report.results) == 3
        assert report.total_time_ms > 0

    def test_evaluate_empty(self) -> None:
        report = evaluate([], solver="dsl")
        assert report.total_tasks == 0
        assert report.accuracy == 0.0

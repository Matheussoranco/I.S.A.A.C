"""Tests for the ARC-AGI benchmark adapter (offline — synthetic mini-dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac.eval.arc import arc_runner, load_arc_tasks, score_arc_answer
from isaac.eval.checkers import run_check
from isaac.eval.runner import run_suite
from isaac.eval.suite import load_suite, suite_hash

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_v1.jsonl"

FLIP_H_TASK = {
    # flip_horizontal solves it: every row reversed.
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        {"input": [[5, 0], [0, 5]], "output": [[0, 5], [5, 0]]},
    ],
    "test": [{"input": [[7, 8], [9, 1]], "output": [[8, 7], [1, 9]]}],
}


def _write_split(tmp_path: Path) -> Path:
    split = tmp_path / "evaluation"
    split.mkdir()
    (split / "fliph001.json").write_text(json.dumps(FLIP_H_TASK), encoding="utf-8")
    return split


# ── scoring ──────────────────────────────────────────────────────────────────


def test_score_exact_match_passes() -> None:
    gt = json.dumps([[[8, 7], [1, 9]]])
    passed, detail = score_arc_answer(json.dumps([[[8, 7], [1, 9]]]), gt)
    assert passed is True
    assert "1/1" in detail


def test_score_any_wrong_cell_fails() -> None:
    gt = json.dumps([[[8, 7], [1, 9]]])
    assert score_arc_answer(json.dumps([[[8, 7], [1, 8]]]), gt)[0] is False
    assert score_arc_answer(json.dumps([None]), gt)[0] is False


def test_score_all_pairs_required() -> None:
    gt = json.dumps([[[1]], [[2]]])
    assert score_arc_answer(json.dumps([[[1]], [[2]]]), gt)[0] is True
    assert score_arc_answer(json.dumps([[[1]], [[3]]]), gt)[0] is False
    assert score_arc_answer(json.dumps([[[1]]]), gt)[0] is False  # length mismatch


def test_score_malformed_answer_fails_not_raises() -> None:
    passed, detail = score_arc_answer("I could not solve this", json.dumps([[[1]]]))
    assert passed is False
    assert "not valid JSON" in detail


def test_arc_checker_spec(tmp_path) -> None:
    gt = json.dumps([[[8, 7], [1, 9]]])
    out = run_check({"type": "arc", "value": gt}, json.dumps([[[8, 7], [1, 9]]]), tmp_path)
    assert out.passed is True
    out = run_check({"type": "arc", "value": gt}, "garbage", tmp_path)
    assert out.passed is False


# ── task loading ─────────────────────────────────────────────────────────────


def test_load_arc_tasks(tmp_path) -> None:
    split = _write_split(tmp_path)
    tasks = load_arc_tasks(split, time_budget_s=5.0)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.id == "fliph001"
    assert task.category == "arc-agi-1-evaluation"
    assert task.timeout_seconds == 5.0
    assert task.checks[0]["type"] == "arc"
    assert json.loads(task.checks[0]["value"]) == [[[8, 7], [1, 9]]]


def test_load_missing_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_arc_tasks(tmp_path / "nope")


def test_load_skips_taskfile_without_test_pairs(tmp_path) -> None:
    split = _write_split(tmp_path)
    (split / "broken.json").write_text(json.dumps({"train": [], "test": []}), encoding="utf-8")
    tasks = load_arc_tasks(split)
    assert [t.id for t in tasks] == ["fliph001"]


# ── end-to-end: symbolic runner through run_suite ────────────────────────────


def test_symbolic_runner_solves_flip_task(tmp_path) -> None:
    split = _write_split(tmp_path)
    tasks = load_arc_tasks(split, time_budget_s=10.0)
    summary = run_suite(
        tasks,
        arc_runner(split),
        suite_name="arc-mini",
        workspace=tmp_path / "ws",
        model="arc-synthesis (symbolic, no LLM)",
        provider="none",
    )
    assert summary.total == 1
    assert summary.passed == 1
    assert summary.outcomes[0].checks[0].type == "arc"


# ── the golden suite must be unaffected by adapter changes ───────────────────


def test_golden_suite_hash_unchanged() -> None:
    tasks = load_suite(GOLDEN)
    assert suite_hash(tasks) == "da9b7c08c5bd342a"

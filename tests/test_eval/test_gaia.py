"""Tests for the GAIA benchmark adapter (offline — synthetic mini-dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac.eval.checkers import run_check
from isaac.eval.gaia import extract_final_answer, load_gaia_tasks, question_scorer
from isaac.eval.runner import TaskAnswer, run_suite
from isaac.eval.suite import load_suite, suite_hash

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_v1.jsonl"


# ── official scorer semantics ───────────────────────────────────────────────


def test_scorer_numbers_strip_units_and_commas() -> None:
    assert question_scorer("42", "42") is True
    assert question_scorer("$1,234.5", "1234.5") is True
    assert question_scorer("17%", "17") is True
    assert question_scorer("41", "42") is False
    assert question_scorer("not a number", "42") is False


def test_scorer_strings_normalize_space_case_punct() -> None:
    assert question_scorer("Right Whale", "right whale") is True
    assert question_scorer("  rightwhale ", "Right Whale") is True
    assert question_scorer("right-whale.", "right whale") is True
    assert question_scorer("blue whale", "right whale") is False


def test_scorer_lists_elementwise() -> None:
    assert question_scorer("milk, eggs, 3", "milk,eggs,3") is True
    assert question_scorer("milk; eggs; 3", "milk,eggs,3") is True
    assert question_scorer("milk, eggs", "milk,eggs,3") is False  # length mismatch
    assert question_scorer("milk, eggs, 4", "milk,eggs,3") is False


def test_final_answer_extraction_takes_last_marker() -> None:
    text = "thinking... FINAL ANSWER: draft\nmore thoughts\nFINAL ANSWER: right whale"
    assert extract_final_answer(text) == "right whale"
    assert extract_final_answer("FINAL ANSWER: [42]") == "42"
    assert extract_final_answer("no marker, just 42") == "no marker, just 42"


def test_gaia_checker_spec(tmp_path) -> None:
    out = run_check(
        {"type": "gaia", "value": "right whale"},
        "I looked it up.\nFINAL ANSWER: Right Whale",
        tmp_path,
    )
    assert out.passed is True
    assert "right whale" in out.detail.lower()


# ── loader ───────────────────────────────────────────────────────────────────


def _write_gaia_split(tmp_path: Path) -> Path:
    split = tmp_path / "validation"
    split.mkdir()
    (split / "data.xlsx").write_bytes(b"\x50\x4b\x03\x04fakexlsx")
    rows = [
        {
            "task_id": "t-l1-plain",
            "Question": "What is the capital of France?",
            "Level": 1,
            "Final answer": "Paris",
            "file_name": "",
        },
        {
            "task_id": "t-l1-file",
            "Question": "Sum the amounts in the attached sheet.",
            "Level": 1,
            "Final answer": "100",
            "file_name": "data.xlsx",
        },
        {
            "task_id": "t-l2",
            "Question": "A level-2 question.",
            "Level": 2,
            "Final answer": "x",
            "file_name": "",
        },
        {
            "task_id": "t-hidden",
            "Question": "Test-split question.",
            "Level": 1,
            "Final answer": "?",
            "file_name": "",
        },
    ]
    (split / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return split


def test_load_gaia_filters_level_and_hidden_answers(tmp_path) -> None:
    split = _write_gaia_split(tmp_path)
    tasks = load_gaia_tasks(split, level=1)
    assert [t.id for t in tasks] == ["t-l1-plain", "t-l1-file"]
    plain = tasks[0]
    assert "capital of France" in plain.prompt
    assert "FINAL ANSWER" in plain.prompt  # the official answer contract
    assert plain.checks == [{"type": "gaia", "value": "Paris"}]
    assert plain.category == "gaia-l1"

    with_file = tasks[1]
    assert with_file.file_paths == {"gaia/data.xlsx": str(split / "data.xlsx")}
    assert "gaia/data.xlsx" in with_file.prompt

    assert [t.id for t in load_gaia_tasks(split, level=2)] == ["t-l2"]


def test_load_gaia_missing_metadata_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="metadata.jsonl"):
        load_gaia_tasks(tmp_path)


# ── binary attachment seeding through the runner ─────────────────────────────


def test_runner_copies_binary_attachments(tmp_path) -> None:
    split = _write_gaia_split(tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    task = [t for t in load_gaia_tasks(split, level=1) if t.file_paths][0]

    seen: dict[str, bytes] = {}

    def runner(t) -> TaskAnswer:
        seen["bytes"] = (ws / "gaia" / "data.xlsx").read_bytes()
        return TaskAnswer(text="FINAL ANSWER: 100")

    summary = run_suite([task], runner, workspace=ws, model="m", provider="p")
    assert summary.passed == 1
    assert seen["bytes"].startswith(b"\x50\x4b")  # binary copied intact


# ── hash stability guarantee ─────────────────────────────────────────────────


def test_golden_suite_hash_is_unchanged_by_file_paths_field() -> None:
    # The published golden_v1 result cites this hash; the file_paths extension
    # must not silently invalidate it.
    assert suite_hash(load_suite(GOLDEN)) == "da9b7c08c5bd342a"

"""Tests for the programmatic answer checkers."""

from __future__ import annotations

from isaac.eval.checkers import run_check, score_answer


def test_contains_and_case(tmp_path) -> None:
    assert run_check({"type": "contains", "value": "Hello"}, "well hello there", tmp_path).passed
    assert not run_check(
        {"type": "contains", "value": "Hello", "case_sensitive": True}, "hello", tmp_path
    ).passed
    assert run_check({"type": "not_contains", "value": "secret"}, "all public", tmp_path).passed


def test_any_of_all_of(tmp_path) -> None:
    answer = "We compared SQLite and PostgreSQL."
    assert run_check({"type": "any_of", "values": ["mysql", "sqlite"]}, answer, tmp_path).passed
    assert run_check({"type": "all_of", "values": ["sqlite", "postgres"]}, answer, tmp_path).passed
    assert not run_check(
        {"type": "all_of", "values": ["sqlite", "oracle"]}, answer, tmp_path
    ).passed


def test_regex_and_min_length(tmp_path) -> None:
    assert run_check({"type": "regex", "pattern": r"\bAu\b"}, "Gold is Au.", tmp_path).passed
    assert not run_check({"type": "regex", "pattern": r"^\d+$"}, "abc", tmp_path).passed
    assert run_check({"type": "min_length", "value": 5}, "123456", tmp_path).passed


def test_numeric_with_tolerance_and_comma_decimals(tmp_path) -> None:
    assert run_check({"type": "numeric", "value": 408}, "the answer is 408.", tmp_path).passed
    assert run_check(
        {"type": "numeric", "value": 27.78, "tolerance": 0.01}, "= 27,78 m/s", tmp_path
    ).passed
    assert not run_check({"type": "numeric", "value": 408}, "around 410", tmp_path).passed


def test_file_checkers(tmp_path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "report.md").write_text("Budget approved on August 1", encoding="utf-8")
    assert run_check({"type": "file_exists", "path": "out/report.md"}, "", tmp_path).passed
    assert run_check(
        {"type": "file_contains", "path": "out/report.md", "value": "budget"}, "", tmp_path
    ).passed
    assert run_check(
        {"type": "file_regex", "path": "out/report.md", "pattern": r"august\s*1"}, "", tmp_path
    ).passed
    assert not run_check({"type": "file_exists", "path": "missing.md"}, "", tmp_path).passed


def test_file_checker_blocks_workspace_escape(tmp_path) -> None:
    outside = tmp_path.parent / "leak.txt"
    outside.write_text("secret", encoding="utf-8")
    res = run_check({"type": "file_exists", "path": "../leak.txt"}, "", tmp_path)
    assert res.passed is False


def test_unknown_and_malformed_specs_fail_without_raising(tmp_path) -> None:
    assert not run_check({"type": "telepathy"}, "answer", tmp_path).passed
    assert not run_check({"type": "regex", "pattern": "("}, "answer", tmp_path).passed
    assert not run_check({"type": "numeric"}, "answer", tmp_path).passed


def test_score_answer_requires_all_checks(tmp_path) -> None:
    checks = [
        {"type": "contains", "value": "alpha"},
        {"type": "contains", "value": "beta"},
    ]
    outcomes = score_answer(checks, "alpha only", tmp_path)
    assert [o.passed for o in outcomes] == [True, False]

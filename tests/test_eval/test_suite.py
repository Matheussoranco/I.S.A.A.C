"""Tests for suite loading and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.eval.suite import load_suite, suite_hash

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden_v1.jsonl"


def _write_suite(tmp_path: Path, lines: list[str], name: str = "suite.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_load_suite_parses_fields(tmp_path) -> None:
    p = _write_suite(
        tmp_path,
        [
            "# a comment",
            '{"id": "t1", "prompt": "say hi", "category": "text", '
            '"checks": [{"type": "contains", "value": "hi"}], '
            '"tools": ["code"], "files": {"a.txt": "x"}, "max_iterations": 5}',
        ],
    )
    tasks = load_suite(p)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.id == "t1"
    assert t.category == "text"
    assert t.tools == ["code"]
    assert t.files == {"a.txt": "x"}
    assert t.max_iterations == 5
    assert t.runner == "agent"


def test_load_suite_rejects_duplicates_and_missing_fields(tmp_path) -> None:
    dup = _write_suite(
        tmp_path,
        [
            '{"id": "t1", "prompt": "a", "checks": [{"type": "min_length", "value": 1}]}',
            '{"id": "t1", "prompt": "b", "checks": [{"type": "min_length", "value": 1}]}',
        ],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(dup)

    missing = _write_suite(tmp_path, ['{"id": "t2", "prompt": "no checks"}'], "missing.jsonl")
    with pytest.raises(ValueError, match="required"):
        load_suite(missing)


def test_suite_hash_is_order_independent_and_content_sensitive(tmp_path) -> None:
    a = '{"id": "t1", "prompt": "p1", "checks": [{"type": "min_length", "value": 1}]}'
    b = '{"id": "t2", "prompt": "p2", "checks": [{"type": "min_length", "value": 1}]}'
    h1 = suite_hash(load_suite(_write_suite(tmp_path, [a, b])))
    h2 = suite_hash(load_suite(_write_suite(tmp_path, [b, a])))
    assert h1 == h2
    b_changed = b.replace("p2", "p2 changed")
    h3 = suite_hash(load_suite(_write_suite(tmp_path, [a, b_changed])))
    assert h3 != h1


def test_golden_suite_loads_and_meets_roadmap_bar() -> None:
    tasks = load_suite(GOLDEN)
    assert len(tasks) >= 30, "ROADMAP-1.0 WS1 requires >= 30 golden tasks"
    categories = {t.category for t in tasks}
    assert {"reasoning", "coding", "research", "file-org", "safety"} <= categories
    for t in tasks:
        assert t.checks, f"{t.id} has no checks"

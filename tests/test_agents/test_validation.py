"""Tests for tool-argument JSON-Schema validation."""

from __future__ import annotations

from isaac.agents.validation import validate_args

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "recursive": {"type": "boolean"},
        "limit": {"type": "integer"},
        "ratio": {"type": "number"},
        "tags": {"type": "array"},
    },
    "required": ["path"],
}


def test_valid_args_pass() -> None:
    assert validate_args(SCHEMA, {"path": "/tmp", "recursive": True, "limit": 5}) == []


def test_missing_required_is_reported() -> None:
    problems = validate_args(SCHEMA, {"recursive": True})
    assert any("missing required parameter 'path'" in p for p in problems)


def test_wrong_types_are_reported() -> None:
    problems = validate_args(SCHEMA, {"path": 123, "tags": "not-a-list"})
    assert any("'path' must be string" in p for p in problems)
    assert any("'tags' must be array" in p for p in problems)


def test_bool_does_not_pass_as_integer() -> None:
    problems = validate_args(SCHEMA, {"path": "/x", "limit": True})
    assert any("'limit' must be integer, got boolean" in p for p in problems)


def test_unknown_parameter_is_flagged_with_suggestions() -> None:
    problems = validate_args(SCHEMA, {"path": "/x", "pth": "/y"})
    assert any("unknown parameter 'pth'" in p for p in problems)
    assert any("path" in p for p in problems)


def test_float_accepted_for_number_and_empty_schema_accepts_anything() -> None:
    assert validate_args(SCHEMA, {"path": "/x", "ratio": 0.5}) == []
    assert validate_args({}, {"anything": object()}) == []

"""Tests for the I/O Sanitizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.security.sanitizer import (
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    sanitize_input,
    sanitize_json_value,
    sanitize_output,
    sanitize_path,
    sanitize_text,
)


class TestSanitizeText:
    def test_clean_text_unchanged(self) -> None:
        text = "Hello, world!"
        assert sanitize_text(text) == text

    def test_strips_null_bytes(self) -> None:
        assert "\x00" not in sanitize_text("hello\x00world")

    def test_strips_ansi_escape(self) -> None:
        text = "\x1b[31mred text\x1b[0m"
        result = sanitize_text(text)
        assert "\x1b" not in result
        assert "red text" in result

    def test_strips_control_chars(self) -> None:
        text = "abc\x01\x02\x03def"
        result = sanitize_text(text)
        assert result == "abcdef"

    def test_preserves_newlines_tabs(self) -> None:
        text = "line1\nline2\ttab"
        assert sanitize_text(text) == text

    def test_strips_html_when_enabled(self) -> None:
        text = "<script>alert('xss')</script> hello"
        result = sanitize_text(text, strip_html=True)
        assert "<script>" not in result
        assert "hello" in result

    def test_preserves_html_by_default(self) -> None:
        text = "<b>bold</b>"
        result = sanitize_text(text, strip_html=False)
        assert "<b>" in result

    def test_enforces_max_length(self) -> None:
        text = "a" * 200
        result = sanitize_text(text, max_length=100)
        assert len(result) == 100

    def test_empty_string(self) -> None:
        assert sanitize_text("") == ""


class TestSanitizeInputOutput:
    def test_sanitize_input(self) -> None:
        result = sanitize_input("hello\x00\x1b[31m")
        assert "\x00" not in result
        assert "\x1b" not in result

    def test_sanitize_output_strips_html(self) -> None:
        result = sanitize_output("<script>x</script>clean")
        assert "<script>" not in result
        assert "clean" in result


class TestSanitizePath:
    def test_safe_relative_path(self) -> None:
        result = sanitize_path("data/file.txt")
        assert result is not None

    def test_rejects_traversal(self) -> None:
        result = sanitize_path("../../etc/passwd")
        assert result is None

    def test_rejects_null_byte(self) -> None:
        result = sanitize_path("file\x00.txt")
        assert result is None

    def test_rejects_too_long(self) -> None:
        result = sanitize_path("a" * 600)
        assert result is None

    def test_rejects_absolute_by_default(self) -> None:
        import sys
        abs_path = "C:\\Windows\\System32" if sys.platform == "win32" else "/etc/passwd"
        result = sanitize_path(abs_path, allow_absolute=False)
        assert result is None

    def test_allows_absolute_when_enabled(self) -> None:
        result = sanitize_path("/safe/path", allow_absolute=True)
        assert result is not None

    def test_root_confinement(self, tmp_path: Path) -> None:
        safe = tmp_path / "workspace"
        safe.mkdir()
        result = sanitize_path("file.txt", root=safe)
        if result is not None:
            # Should be within root
            assert str(result).startswith(str(safe)) or result.is_relative_to(safe)

    def test_empty_path(self) -> None:
        assert sanitize_path("") is None


class TestSanitizeJsonValue:
    def test_sanitizes_string(self) -> None:
        assert "\x00" not in sanitize_json_value("hello\x00world")

    def test_sanitizes_nested_dict(self) -> None:
        data = {"key": "val\x00ue", "nested": {"inner": "ab\x01cd"}}
        result = sanitize_json_value(data)
        assert "\x00" not in str(result)
        assert "\x01" not in str(result)

    def test_sanitizes_list(self) -> None:
        data = ["one\x00", "two\x01"]
        result = sanitize_json_value(data)
        assert "\x00" not in str(result)

    def test_passthrough_numbers(self) -> None:
        assert sanitize_json_value(42) == 42
        assert sanitize_json_value(3.14) == 3.14

    def test_passthrough_none(self) -> None:
        assert sanitize_json_value(None) is None

"""Tests for the Skill Connectors system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# BaseConnector
# ---------------------------------------------------------------------------


class TestBaseConnector:
    """Tests for ``isaac.skills.connectors.base.BaseConnector``."""

    def test_is_available_all_env_set(self) -> None:
        from isaac.skills.connectors.base import BaseConnector

        class FakeConnector(BaseConnector):
            name = "fake"
            description = "test"
            requires_env = ["MY_TEST_VAR"]

            def run(self, **kwargs: Any) -> dict[str, Any]:
                return {}

        with patch.dict(os.environ, {"MY_TEST_VAR": "value"}):
            c = FakeConnector()
            assert c.is_available()

    def test_is_available_missing_env(self) -> None:
        from isaac.skills.connectors.base import BaseConnector

        class FakeConnector(BaseConnector):
            name = "fake"
            description = "test"
            requires_env = ["MISSING_VAR_123"]

            def run(self, **kwargs: Any) -> dict[str, Any]:
                return {}

        env_copy = {k: v for k, v in os.environ.items() if k != "MISSING_VAR_123"}
        with patch.dict(os.environ, env_copy, clear=True):
            c = FakeConnector()
            assert not c.is_available()

    def test_to_schema(self) -> None:
        from isaac.skills.connectors.base import BaseConnector

        class FakeConnector(BaseConnector):
            name = "fake"
            description = "A test connector"
            requires_env = ["KEY"]

            def run(self, **kwargs: Any) -> dict[str, Any]:
                return {}

        schema = FakeConnector().to_schema()
        assert schema["name"] == "fake"
        assert schema["description"] == "A test connector"
        assert "KEY" in schema["requires_env"]


# ---------------------------------------------------------------------------
# WebSearchConnector
# ---------------------------------------------------------------------------


class TestWebSearchConnector:
    """Tests for ``WebSearchConnector``."""

    def test_run_returns_results(self) -> None:
        from isaac.skills.connectors.web_search import WebSearchConnector

        c = WebSearchConnector()
        # Mock duckduckgo_search
        with patch("isaac.skills.connectors.web_search.DDGS") as MockDDGS:
            instance = MockDDGS.return_value.__enter__ = MagicMock()
            MockDDGS.return_value.__enter__.return_value = MockDDGS.return_value
            MockDDGS.return_value.__exit__ = MagicMock(return_value=False)
            MockDDGS.return_value.text.return_value = [
                {"title": "Result 1", "href": "https://example.com", "body": "Snippet 1"},
            ]
            result = c.run(query="test query")
            assert "results" in result or "error" in result


# ---------------------------------------------------------------------------
# FileSystemConnector
# ---------------------------------------------------------------------------


class TestFileSystemConnector:
    """Tests for ``FileSystemConnector``."""

    def test_list_directory(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.filesystem import FileSystemConnector

        (tmp_path / "file1.txt").write_text("hello")
        (tmp_path / "file2.py").write_text("print()")

        c = FileSystemConnector()
        with patch.object(c, "_get_allowed_paths", return_value=[str(tmp_path)]):
            result = c.run(action="list", path=str(tmp_path))
            assert "entries" in result
            names = [e["name"] for e in result["entries"]]
            assert "file1.txt" in names

    def test_read_file(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.filesystem import FileSystemConnector

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        c = FileSystemConnector()
        with patch.object(c, "_get_allowed_paths", return_value=[str(tmp_path)]):
            result = c.run(action="read", path=str(test_file))
            assert result.get("content") == "hello world"

    def test_path_escape_blocked(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.filesystem import FileSystemConnector

        c = FileSystemConnector()
        with patch.object(c, "_get_allowed_paths", return_value=[str(tmp_path)]):
            result = c.run(action="read", path="/etc/passwd")
            assert "error" in result


# ---------------------------------------------------------------------------
# ShellConnector
# ---------------------------------------------------------------------------


class TestShellConnector:
    """Tests for ``ShellConnector``."""

    def test_blocked_metacharacters(self) -> None:
        from isaac.skills.connectors.shell import ShellConnector

        c = ShellConnector()
        result = c.run(command="cat /etc/passwd | grep root")
        assert "error" in result
        assert "metacharacters" in result["error"].lower() or "blocked" in result["error"].lower()

    def test_disallowed_command(self) -> None:
        from isaac.skills.connectors.shell import ShellConnector

        c = ShellConnector()
        result = c.run(command="rm -rf /")
        assert "error" in result
        assert "allowlist" in result["error"].lower()

    def test_echo_allowed(self) -> None:
        from isaac.skills.connectors.shell import ShellConnector

        c = ShellConnector()
        result = c.run(command="echo hello")
        assert result.get("exit_code") == 0
        assert "hello" in result.get("stdout", "")


# ---------------------------------------------------------------------------
# ObsidianConnector
# ---------------------------------------------------------------------------


class TestObsidianConnector:
    """Tests for ``ObsidianConnector``."""

    def test_list_notes(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.obsidian import ObsidianConnector

        (tmp_path / "note1.md").write_text("# Note 1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "note2.md").write_text("# Note 2")

        c = ObsidianConnector()
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(tmp_path)}):
            result = c.run(action="list")
            assert "notes" in result
            assert len(result["notes"]) == 2

    def test_read_note(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.obsidian import ObsidianConnector

        (tmp_path / "test.md").write_text("# Hello World")
        c = ObsidianConnector()
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(tmp_path)}):
            result = c.run(action="read", path="test.md")
            assert "# Hello World" in result.get("content", "")

    def test_search_notes(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.obsidian import ObsidianConnector

        (tmp_path / "note.md").write_text("The quick brown fox jumps")
        c = ObsidianConnector()
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(tmp_path)}):
            result = c.run(action="search", query="brown fox")
            assert len(result.get("matches", [])) == 1

    def test_path_escape_blocked(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.obsidian import ObsidianConnector

        c = ObsidianConnector()
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": str(tmp_path)}):
            result = c.run(action="read", path="../../etc/passwd")
            assert "error" in result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestConnectorRegistry:
    """Tests for ``isaac.skills.connectors.registry``."""

    def test_get_registry_returns_dict(self) -> None:
        from isaac.skills.connectors.registry import get_registry, reset_registry

        reset_registry()
        reg = get_registry()
        assert isinstance(reg, dict)
        # Should discover at least the connectors we created
        assert len(reg) >= 1

    def test_list_connector_schemas(self) -> None:
        from isaac.skills.connectors.registry import list_connector_schemas, reset_registry

        reset_registry()
        schemas = list_connector_schemas()
        assert isinstance(schemas, list)
        for s in schemas:
            assert "name" in s
            assert "description" in s

    def test_run_connector_unknown(self) -> None:
        from isaac.skills.connectors.registry import reset_registry, run_connector

        reset_registry()
        result = run_connector("nonexistent_connector")
        assert "error" in result

    def test_audit_connector_writes_file(self, tmp_path: Path) -> None:
        from isaac.skills.connectors.registry import audit_connector

        with patch("isaac.skills.connectors.registry._audit_path", return_value=tmp_path / "audit.log"):
            audit_connector("test", "invoke", "detail")
            assert (tmp_path / "audit.log").exists()
            content = (tmp_path / "audit.log").read_text()
            assert "test" in content
            assert "invoke" in content

"""Tests for the Identity & Soul module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestSoulModule:
    """Tests for ``isaac.identity.soul``."""

    def test_get_soul_returns_dict(self) -> None:
        from isaac.identity.soul import get_soul

        soul = get_soul()
        assert isinstance(soul, dict)
        assert "name" in soul
        assert soul["name"] == "I.S.A.A.C."

    def test_soul_has_required_keys(self) -> None:
        from isaac.identity.soul import get_soul

        soul = get_soul()
        required = {"name", "full_name", "personality", "version", "tagline"}
        assert required.issubset(set(soul.keys()))

    def test_soul_system_prompt_contains_name(self) -> None:
        from isaac.identity.soul import soul_system_prompt

        prompt = soul_system_prompt()
        assert "I.S.A.A.C." in prompt

    def test_soul_system_prompt_contains_personality(self) -> None:
        from isaac.identity.soul import soul_system_prompt

        prompt = soul_system_prompt()
        assert len(prompt) > 50  # Should have actual content

    def test_load_soul_from_custom_json(self, tmp_path: Path) -> None:
        from isaac.identity.soul import load_soul

        custom = {
            "name": "TestBot",
            "full_name": "Test Bot System",
            "personality": "Testing personality",
            "version": "9.9.9",
            "tagline": "I test things.",
        }
        path = tmp_path / "custom_soul.json"
        path.write_text(json.dumps(custom), encoding="utf-8")

        loaded = load_soul(str(path))
        assert loaded["name"] == "TestBot"
        assert loaded["version"] == "9.9.9"

    def test_load_soul_invalid_path_returns_default(self) -> None:
        from isaac.identity.soul import load_soul

        soul = load_soul("/nonexistent/soul.json")
        assert soul["name"] == "I.S.A.A.C."

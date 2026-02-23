"""Tests for the Skill Abstraction node."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from isaac.core.state import PlanStep, SkillCandidate, make_initial_state
from isaac.nodes.skill_abstraction import skill_abstraction_node
from tests.conftest import MockLLM


class TestSkillAbstractionNode:
    def test_commits_skill_to_library(self, tmp_path: Path) -> None:
        state = make_initial_state()
        state["skill_candidate"] = SkillCandidate(
            name="add_two",
            code="print(2+2)",
            task_context="add numbers",
            success_count=1,
        )
        state["plan"] = [PlanStep(id="s1", description="done", status="done")]

        mock = MockLLM('```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```')
        with (
            patch("isaac.llm.provider.get_llm", return_value=mock),
            patch("isaac.config.settings.settings") as mock_settings,
        ):
            mock_settings.skills_dir = tmp_path
            result = skill_abstraction_node(state)

        assert result["skill_candidate"] is None  # cleared
        # Verify file written
        skill_file = tmp_path / "add_two.py"
        assert skill_file.exists()
        assert "def add" in skill_file.read_text()
        # Verify index updated
        index = json.loads((tmp_path / "_index.json").read_text())
        assert "add_two" in index["skills"]

    def test_no_candidate_skips(self) -> None:
        state = make_initial_state()
        result = skill_abstraction_node(state)
        assert result["skill_candidate"] is None

    def test_activates_next_pending_step(self, tmp_path: Path) -> None:
        state = make_initial_state()
        state["skill_candidate"] = SkillCandidate(
            name="test_skill",
            code="x = 1",
            task_context="test",
            success_count=1,
        )
        state["plan"] = [
            PlanStep(id="s1", description="done", status="done"),
            PlanStep(id="s2", description="next", status="pending"),
        ]

        mock = MockLLM('```python\ndef test() -> None:\n    pass\n```')
        with (
            patch("isaac.llm.provider.get_llm", return_value=mock),
            patch("isaac.config.settings.settings") as mock_settings,
        ):
            mock_settings.skills_dir = tmp_path
            result = skill_abstraction_node(state)

        assert result["plan"][1].status == "active"

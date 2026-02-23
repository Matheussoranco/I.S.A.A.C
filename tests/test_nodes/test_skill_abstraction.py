"""Tests for the Skill Abstraction node."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from tests.conftest import MockLLM

from isaac.core.state import PlanStep, SkillCandidate, make_initial_state
from isaac.nodes.skill_abstraction import skill_abstraction_node


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

    # ------------------------------------------------------------------
    # UI / Playwright macro path
    # ------------------------------------------------------------------

    def test_ui_skill_generates_playwright_function(self, tmp_path: Path) -> None:
        """skill_type='ui' candidates get converted to a Playwright function."""
        action_trace = [
            {"type": "click", "x": 100, "y": 200, "description": "click login btn"},
            {"type": "type", "text": "admin", "description": "type username"},
        ]
        candidate_code = json.dumps(
            {
                "actions": action_trace,
                "screenshot_before": "before_b64",
                "screenshot_after": "after_b64",
            }
        )

        state = make_initial_state()
        state["skill_candidate"] = SkillCandidate(
            name="login_flow",
            code=candidate_code,
            task_context="log in to application",
            success_count=1,
            skill_type="ui",
            tags=["ui"],
        )
        state["plan"] = [PlanStep(id="s1", description="done", status="done")]

        playwright_code = (
            "```python\n"
            "async def login_flow(page, username: str = 'admin') -> None:\n"
            "    await page.click('input[type=submit]')\n"
            "    await page.fill('input[name=username]', username)\n"
            "```"
        )
        mock = MockLLM(playwright_code)

        with (
            patch("isaac.llm.provider.get_llm", return_value=mock),
            patch("isaac.config.settings.settings") as mock_settings,
        ):
            mock_settings.skills_dir = tmp_path
            result = skill_abstraction_node(state)

        assert result["skill_candidate"] is None
        skill_file = tmp_path / "login_flow.py"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "async def login_flow" in content

    def test_ui_skill_tags_include_playwright(self, tmp_path: Path) -> None:
        """Auto-adds 'playwright' tag if not already present on UI skills."""
        state = make_initial_state()
        candidate = SkillCandidate(
            name="scroll_skill",
            code=json.dumps({"actions": [], "screenshot_before": "", "screenshot_after": ""}),
            task_context="scroll page",
            success_count=1,
            skill_type="ui",
            tags=["ui"],
        )
        state["skill_candidate"] = candidate
        state["plan"] = []

        mock = MockLLM(
            "```python\nasync def scroll_skill(page) -> None:\n"
            "    await page.mouse.wheel(0,500)\n```"
        )
        with (
            patch("isaac.llm.provider.get_llm", return_value=mock),
            patch("isaac.config.settings.settings") as ms,
        ):
            ms.skills_dir = tmp_path
            skill_abstraction_node(state)

        # The candidate object is mutated in-place before commit
        assert "playwright" in candidate.tags

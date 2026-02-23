"""Tests for transition/routing logic."""

from __future__ import annotations

from unittest.mock import patch

from isaac.core.state import ErrorEntry, PlanStep, SkillCandidate, make_initial_state
from isaac.core.transitions import after_reflection, after_skill_abstraction


class TestAfterReflection:
    def test_success_routes_to_skill_abstraction(self) -> None:
        state = make_initial_state()
        state["skill_candidate"] = SkillCandidate(name="x", code="pass", success_count=1)
        state["iteration"] = 1

        result = after_reflection(state)
        assert result == "skill_abstraction"

    def test_failure_within_retries_routes_to_planner(self) -> None:
        state = make_initial_state()
        state["skill_candidate"] = None
        state["errors"] = [
            ErrorEntry(node="reflection", message="fail", attempt=1),
        ]
        state["iteration"] = 1

        with patch("isaac.config.settings.settings") as cfg:
            cfg.graph.max_retries = 3
            cfg.graph.max_iterations = 10
            result = after_reflection(state)

        assert result == "planner"

    def test_failure_at_max_retries_routes_to_end(self) -> None:
        state = make_initial_state()
        state["skill_candidate"] = None
        state["errors"] = [
            ErrorEntry(node="reflection", message="fail", attempt=i)
            for i in range(3)
        ]
        state["iteration"] = 3

        with patch("isaac.config.settings.settings") as cfg:
            cfg.graph.max_retries = 3
            cfg.graph.max_iterations = 10
            result = after_reflection(state)

        assert result == "__end__"

    def test_iteration_cap_terminates(self) -> None:
        state = make_initial_state()
        state["skill_candidate"] = SkillCandidate(name="x", code="pass", success_count=1)
        state["iteration"] = 10

        with patch("isaac.config.settings.settings") as cfg:
            cfg.graph.max_retries = 3
            cfg.graph.max_iterations = 10
            result = after_reflection(state)

        assert result == "__end__"


class TestAfterSkillAbstraction:
    def test_pending_steps_routes_to_planner(self) -> None:
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="done", status="done"),
            PlanStep(id="s2", description="pending", status="pending"),
        ]
        state["iteration"] = 1

        with patch("isaac.config.settings.settings") as cfg:
            cfg.graph.max_iterations = 10
            result = after_skill_abstraction(state)

        assert result == "planner"

    def test_all_done_routes_to_end(self) -> None:
        state = make_initial_state()
        state["plan"] = [
            PlanStep(id="s1", description="done", status="done"),
        ]
        state["iteration"] = 1

        with patch("isaac.config.settings.settings") as cfg:
            cfg.graph.max_iterations = 10
            result = after_skill_abstraction(state)

        assert result == "__end__"

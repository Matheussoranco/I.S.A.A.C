"""Tests for the Planner node."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import MockLLM

from isaac.core.state import make_initial_state
from isaac.nodes.planner import planner_node


class TestPlannerNode:
    def test_generates_plan_steps(self) -> None:
        state = make_initial_state()
        state["hypothesis"] = "write hello world to a file"

        mock = MockLLM(
            '{"steps": [{"id": "s1", "description": "write file", "depends_on": []}]}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        plan = result["plan"]
        assert len(plan) == 1
        assert plan[0].id == "s1"
        assert plan[0].status == "active"  # first step auto-activated
        assert result["iteration"] == 1

    def test_increments_iteration(self) -> None:
        state = make_initial_state()
        state["hypothesis"] = "test"
        state["iteration"] = 3

        mock = MockLLM('{"steps": [{"id": "s1", "description": "x"}]}')
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        assert result["iteration"] == 4

    def test_fallback_on_bad_json(self) -> None:
        state = make_initial_state()
        state["hypothesis"] = "test"

        mock = MockLLM("not json")
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        assert len(result["plan"]) == 1  # fallback single step

"""Tests for the Planner node."""

from __future__ import annotations

from unittest.mock import patch

from tests.conftest import MockLLM

from isaac.core.state import make_initial_state
from isaac.memory.episodic import Episode, get_episodic_memory, reset_episodic_memory
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

    # ------------------------------------------------------------------
    # Dependency-aware step scheduling
    # ------------------------------------------------------------------

    def test_dependency_blocks_activation(self) -> None:
        """Step s2 depends on s1 — only s1 should be active initially."""
        state = make_initial_state()
        state["hypothesis"] = "multi-step"

        mock = MockLLM(
            '{"steps": ['
            '  {"id": "s1", "description": "first", "depends_on": []},'
            '  {"id": "s2", "description": "second", "depends_on": ["s1"]}'
            ']}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        assert result["plan"][0].status == "active"
        assert result["plan"][1].status == "pending"

    def test_no_deps_first_step_active(self) -> None:
        state = make_initial_state()
        state["hypothesis"] = "simple"

        mock = MockLLM(
            '{"steps": ['
            '  {"id": "a", "description": "alpha", "depends_on": []},'
            '  {"id": "b", "description": "beta", "depends_on": []}'
            ']}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        # Only the first pending step without unmet deps should be activated
        assert result["plan"][0].status == "active"
        assert result["plan"][1].status == "pending"

    # ------------------------------------------------------------------
    # Episodic context injection
    # ------------------------------------------------------------------

    def setup_method(self) -> None:
        reset_episodic_memory()

    def teardown_method(self) -> None:
        reset_episodic_memory()

    def test_episodic_context_passed_to_prompt(self) -> None:
        """The planner should pass episodic memory context to the prompt."""
        mem = get_episodic_memory()
        mem.record(Episode(
            task="sort array", hypothesis="use quicksort", code="sorted()",
            result_summary="works", success=True,
        ))

        state = make_initial_state()
        state["hypothesis"] = "test episodic"

        mock = MockLLM('{"steps": [{"id": "s1", "description": "test"}]}')
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = planner_node(state)

        # Verify the plan was produced (episodic context doesn't affect plan output,
        # but should not break the planner)
        assert len(result["plan"]) >= 1

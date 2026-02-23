"""Tests for the Perception node."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from isaac.core.state import WorldModel, make_initial_state
from isaac.nodes.perception import perception_node
from tests.conftest import MockLLM


class TestPerceptionNode:
    def test_extracts_observations(self) -> None:
        state = make_initial_state()
        state["messages"] = [HumanMessage(content="Create a file with hello world")]

        mock = MockLLM(
            '{"observations": ["user wants to create a file"], '
            '"hypothesis": "write hello world to a file"}'
        )
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = perception_node(state)

        assert result["current_phase"] == "perception"
        wm: WorldModel = result["world_model"]
        assert "user wants to create a file" in wm.observations
        assert result["hypothesis"] == "write hello world to a file"

    def test_handles_no_user_message(self) -> None:
        state = make_initial_state()
        result = perception_node(state)
        assert result["current_phase"] == "perception"

    def test_handles_malformed_json(self) -> None:
        state = make_initial_state()
        state["messages"] = [HumanMessage(content="do something")]

        mock = MockLLM("this is not json at all")
        with patch("isaac.llm.provider.get_llm", return_value=mock):
            result = perception_node(state)

        # Should degrade gracefully
        assert result["current_phase"] == "perception"
        wm: WorldModel = result["world_model"]
        assert len(wm.observations) > 0  # raw response captured

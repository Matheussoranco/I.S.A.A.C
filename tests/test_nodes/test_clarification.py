"""Tests for the clarification node and its graph wiring."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from isaac.core.transitions import (
    NODE_CLARIFICATION,
    NODE_DIRECT_RESPONSE,
    NODE_EXPLORER,
    after_clarification,
    after_perception,
)
from isaac.nodes.clarification import (
    _ambiguity_score,
    clarification_node,
    needs_clarification,
)


def _state(query: str, **kwargs) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "perception_confidence": kwargs.pop("perception_confidence", 1.0),
        **kwargs,
    }


def test_ambiguity_score_short_vague() -> None:
    assert _ambiguity_score("fix it") > 0.6
    assert _ambiguity_score("Refactor the AuthMiddleware class to use JWT") < 0.4


def test_after_perception_routes_direct_to_fast_path() -> None:
    assert after_perception({"task_mode": "direct"}) == NODE_DIRECT_RESPONSE


def test_after_perception_routes_through_clarification() -> None:
    assert after_perception({"task_mode": "code"}) == NODE_CLARIFICATION


def test_after_clarification_routes_to_end_when_question_asked() -> None:
    assert after_clarification({"needs_clarification": True}) == "__end__"


def test_after_clarification_continues_when_unambiguous() -> None:
    assert after_clarification({"needs_clarification": False}) == NODE_EXPLORER


def test_clarification_node_passes_through_when_unambiguous() -> None:
    state = _state(
        "Refactor the AuthMiddleware class in src/auth.py to use JWT tokens "
        "instead of cookies, and update the corresponding tests.",
        perception_confidence=0.95,
    )
    # Stub MoE margin to be wide so the routing signal doesn't trigger
    with patch("isaac.nodes.clarification._moe_margin", return_value=0.9):
        update = clarification_node(state)
    assert update["needs_clarification"] is False
    assert "messages" not in update


def test_clarification_node_emits_question_when_ambiguous() -> None:
    state = _state("fix it", perception_confidence=0.3)
    with (
        patch("isaac.nodes.clarification._moe_margin", return_value=0.0),
        patch(
            "isaac.nodes.clarification._formulate_question",
            return_value="What part of the codebase do you want me to fix?",
        ),
    ):
        update = clarification_node(state)
    assert update["needs_clarification"] is True
    assert update["current_phase"] == "clarification"
    [msg] = update["messages"]
    assert isinstance(msg, AIMessage)
    assert "fix" in msg.content.lower()


def test_needs_clarification_threshold() -> None:
    high_amb = _state("fix it", perception_confidence=0.3)
    low_amb = _state(
        "Add an index on the users.email column in the postgres schema "
        "and run the migration script.",
        perception_confidence=0.95,
    )
    with patch("isaac.nodes.clarification._moe_margin", return_value=0.0):
        assert needs_clarification(high_amb) is True
    with patch("isaac.nodes.clarification._moe_margin", return_value=0.9):
        assert needs_clarification(low_amb) is False

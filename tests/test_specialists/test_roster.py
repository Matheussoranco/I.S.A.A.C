"""Offline tests for the built-in specialist roster.

These tests never call ``.run()``, never touch an LLM, and never hit the
network — they only inspect class attributes and the composed system prompt.
"""

from __future__ import annotations

import pytest

from isaac.specialists.base import Specialist
from isaac.specialists.roster import (
    ROSTER,
    AnalystSpecialist,
    CoderSpecialist,
    CriticSpecialist,
    DesignerSpecialist,
    FileOrganizerSpecialist,
    GeneralistSpecialist,
    OperatorSpecialist,
    PlannerSpecialist,
    ResearcherSpecialist,
)

EXPECTED_CLASSES = [
    CoderSpecialist,
    FileOrganizerSpecialist,
    ResearcherSpecialist,
    DesignerSpecialist,
    OperatorSpecialist,
    AnalystSpecialist,
    CriticSpecialist,
    PlannerSpecialist,
    GeneralistSpecialist,
]

CARD_KEYS = {"name", "title", "domain", "description", "tools", "max_risk"}


def _registered_tool_names() -> set[str]:
    """Return every real tool name from the global registry."""
    from isaac.tools import register_all_tools

    registry = register_all_tools()
    return set(registry.list_names())


def test_roster_has_all_nine() -> None:
    assert len(ROSTER) == 9
    assert ROSTER == EXPECTED_CLASSES


def test_all_are_specialist_subclasses() -> None:
    for cls in ROSTER:
        assert issubclass(cls, Specialist)


def test_names_are_unique_and_non_empty() -> None:
    names = [cls.name for cls in ROSTER]
    for name in names:
        assert isinstance(name, str) and name.strip(), f"bad name: {name!r}"
    assert len(names) == len(set(names)), f"duplicate names in {names}"


@pytest.mark.parametrize("cls", EXPECTED_CLASSES)
def test_card_keys(cls: type[Specialist]) -> None:
    card = cls().card()
    assert set(card.keys()) == CARD_KEYS
    assert card["name"] == cls.name


@pytest.mark.parametrize("cls", EXPECTED_CLASSES)
def test_tool_names_are_real_tools(cls: type[Specialist]) -> None:
    valid = _registered_tool_names()
    names = cls().tool_names
    if names is None:
        return  # None = all tools; nothing to validate
    for tool_name in names:
        assert tool_name in valid, f"{cls.__name__} references unknown tool {tool_name!r}"


@pytest.mark.parametrize("cls", EXPECTED_CLASSES)
def test_system_prompt_mentions_title(cls: type[Specialist]) -> None:
    prompt = cls().system_prompt()
    assert isinstance(prompt, str) and prompt.strip()
    assert cls.title in prompt


def test_planner_has_no_tools() -> None:
    assert PlannerSpecialist().tool_names == []


def test_generalist_has_all_tools() -> None:
    assert GeneralistSpecialist().tool_names is None

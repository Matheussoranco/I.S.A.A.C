"""Tests for the specialist registry."""

from __future__ import annotations

import pytest

from isaac.specialists.registry import (
    all_specialists,
    get_specialist,
    get_specialist_class,
    list_specialists,
    specialist_names,
)
from isaac.specialists.roster import CoderSpecialist, OperatorSpecialist

_EXPECTED = [
    "coder",
    "file_organizer",
    "researcher",
    "designer",
    "operator",
    "analyst",
    "critic",
    "planner",
    "generalist",
]


def test_get_specialist_returns_instance() -> None:
    assert isinstance(get_specialist("coder"), CoderSpecialist)


def test_lookup_is_case_insensitive() -> None:
    assert isinstance(get_specialist("CODER"), CoderSpecialist)
    assert get_specialist_class("Operator") is OperatorSpecialist


def test_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_specialist("does_not_exist")


def test_specialist_names_contains_all() -> None:
    names = specialist_names()
    for expected in _EXPECTED:
        assert expected in names


def test_list_specialists_returns_cards() -> None:
    cards = list_specialists()
    assert len(cards) == 9
    assert all("name" in c and "domain" in c for c in cards)


def test_kwargs_forwarded_to_constructor() -> None:
    sp = get_specialist("operator", auto_approve=True)
    assert sp.auto_approve is True


def test_all_specialists_instantiates_every_one() -> None:
    everyone = all_specialists()
    assert set(everyone) == set(_EXPECTED)
    assert all(hasattr(sp, "run") for sp in everyone.values())

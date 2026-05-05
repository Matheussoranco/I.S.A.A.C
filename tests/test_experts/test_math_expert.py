"""Tests for the symbolic MathExpert."""

from __future__ import annotations

import pytest

sympy = pytest.importorskip("sympy")

from isaac.experts.math import MathExpert


@pytest.fixture()
def expert() -> MathExpert:
    return MathExpert()


def test_arithmetic(expert: MathExpert) -> None:
    response = expert.answer("2 + 2 * 3")
    assert response.is_useful()
    assert "8" in response.answer


def test_solve_quadratic(expert: MathExpert) -> None:
    response = expert.answer("solve x^2 - 4 = 0")
    assert response.is_useful()
    assert "2" in response.answer and "-2" in response.answer


def test_integrate(expert: MathExpert) -> None:
    response = expert.answer("integrate sin(x) dx")
    assert response.is_useful()
    # SymPy returns -cos(x)
    assert "cos" in response.answer.lower()


def test_differentiate(expert: MathExpert) -> None:
    response = expert.answer("derivative of x**3")
    assert response.is_useful()
    assert "3" in response.answer


def test_can_handle_score(expert: MathExpert) -> None:
    assert expert.can_handle("solve x = 1") > 0.5
    assert expert.can_handle("hello there") < 0.3

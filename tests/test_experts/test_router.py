"""Tests for the Knowledge Experts router and registry."""

from __future__ import annotations

from typing import Any

import pytest

from isaac.experts.base import Expert, ExpertResponse
from isaac.experts.registry import ExpertRegistry
from isaac.experts.router import HybridRouter


class _FakeMath(Expert):
    name = "fake_math"
    domains = ("math",)
    description = "fake math"
    cost = 0.1

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        return 0.9 if "math" in query.lower() else 0.0

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        return ExpertResponse(expert=self.name, answer="math:" + query, confidence=0.9)


class _FakeLanguage(Expert):
    name = "language"
    domains = ("general",)
    description = "fake language"
    cost = 1.0

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        return 0.4

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        return ExpertResponse(expert=self.name, answer="lang:" + query, confidence=0.5)


@pytest.fixture()
def registry() -> ExpertRegistry:
    reg = ExpertRegistry()
    reg.register(_FakeLanguage())
    reg.register(_FakeMath())
    return reg


def test_registry_register_and_get(registry: ExpertRegistry) -> None:
    assert registry.get("fake_math") is not None
    assert "language" in registry.names()


def test_router_picks_specialist(registry: ExpertRegistry) -> None:
    router = HybridRouter(registry, winrate_weight=0.0, cost_penalty=0.0)
    result = router.route("solve this math problem")
    assert result.selection.primary == "fake_math"


def test_router_falls_back_to_language(registry: ExpertRegistry) -> None:
    router = HybridRouter(registry, winrate_weight=0.0, cost_penalty=0.0)
    result = router.route("hello there")
    assert result.selection.primary == "language"


def test_router_returns_top_k(registry: ExpertRegistry) -> None:
    router = HybridRouter(registry, winrate_weight=0.0, cost_penalty=0.0)
    result = router.route("solve this math problem", top_k=2)
    names = [name for name, _ in result.selection.candidates]
    assert names[0] == "fake_math"


def test_expert_answer_records_timing() -> None:
    expert = _FakeMath()
    response = expert.answer("math go")
    assert response.is_useful()
    assert response.elapsed_ms >= 0.0

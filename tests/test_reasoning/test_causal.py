"""Tests for the causal reasoner."""

from __future__ import annotations

import random

import pytest

from isaac.reasoning.causal import (
    CausalReasoner,
    counterfactual,
    learn_structure,
    predict,
)


def _generate_data(n: int = 400, seed: int = 0) -> list[dict[str, int]]:
    """Ground truth: A → B → C (and A → C is independent given B).

    Each variable is binary 0/1.
    """
    rng = random.Random(seed)
    data: list[dict[str, int]] = []
    for _ in range(n):
        a = rng.randint(0, 1)
        b = a if rng.random() < 0.85 else 1 - a
        c = b if rng.random() < 0.8 else 1 - b
        data.append({"A": a, "B": b, "C": c})
    return data


def test_structure_learning_recovers_chain() -> None:
    data = _generate_data()
    graph = learn_structure(data, variable_order=["A", "B", "C"])
    assert ("A", "B") in graph.edges
    assert ("B", "C") in graph.edges


def test_predict_marginals() -> None:
    data = _generate_data()
    graph = learn_structure(data, variable_order=["A", "B", "C"])
    posterior = predict(graph, "C", evidence={"A": 1})
    assert sum(posterior.values()) == pytest.approx(1.0)
    # Prob(C=1 | A=1) should be > prob(C=0 | A=1) given the chain
    assert posterior.get(1, 0) > posterior.get(0, 0)


def test_counterfactual_with_intervention() -> None:
    data = _generate_data()
    graph = learn_structure(data, variable_order=["A", "B", "C"])
    cf = counterfactual(
        graph,
        factual={"A": 0, "B": 0, "C": 0},
        intervention={"B": 1},
        target="C",
    )
    # Intervening on B=1 should raise P(C=1) compared to factual
    assert cf.get(1, 0) > 0.5


def test_reasoner_facade() -> None:
    reasoner = CausalReasoner()
    reasoner.add_observations(_generate_data())
    reasoner.learn(variable_order=["A", "B", "C"])
    assert len(reasoner.graph.edges) >= 2

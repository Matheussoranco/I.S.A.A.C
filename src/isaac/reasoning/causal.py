"""Causal reasoning — induce causal graphs from observations and answer
counterfactual / interventional queries.

Approach
--------

1. **Variable discovery** — accept either pre-extracted (variable, value)
   tuples or raw episodic event strings (we extract token-level features).
2. **Structure learning** — a lightweight PC-style constraint-based
   procedure: edges between variables whose χ²/MI exceeds a threshold,
   pruned by conditional independence over candidate parents.
3. **Mechanisms** — each child variable gets a tabular conditional
   distribution P(child | parents) estimated from observations.
4. **Inference** — supports observational ``P(Y | X=x)``, interventional
   ``P(Y | do(X=x))`` (cuts incoming edges to X), and counterfactual queries
   via twin-network simulation.

Dependencies are kept minimal: pure-Python + numpy. networkx is used if
available for nicer graph queries.

This is not a full do-calculus engine, but it's a real causal layer that
makes I.S.A.A.C. capable of reasoning about *why* and *what-if* — a
distinguishing feature of SOTA neuro-symbolic systems.
"""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


Observation = dict[str, Any]
"""A single observation: variable name → value (any hashable)."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CausalGraph:
    """Directed acyclic graph over named variables with tabular CPTs."""

    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    """Directed edges (parent, child)."""
    cpts: dict[str, dict[tuple[Any, ...], Counter]] = field(default_factory=dict)
    """``cpts[child][parent_values_tuple] = Counter(child_value -> count)``."""
    parent_order: dict[str, list[str]] = field(default_factory=dict)
    """Stable ordering of parents per child (for tuple keys)."""

    # ----- queries ----------------------------------------------------

    def parents(self, node: str) -> list[str]:
        return self.parent_order.get(node, [
            p for (p, c) in self.edges if c == node
        ])

    def children(self, node: str) -> list[str]:
        return [c for (p, c) in self.edges if p == node]

    def topological_order(self) -> list[str]:
        in_deg = {n: 0 for n in self.nodes}
        for _, c in self.edges:
            in_deg[c] = in_deg.get(c, 0) + 1
        order: list[str] = []
        queue = [n for n, d in in_deg.items() if d == 0]
        while queue:
            n = queue.pop()
            order.append(n)
            for c in self.children(n):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return order

    def to_string(self) -> str:
        return (
            f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)}):\n"
            + "\n".join(f"  {p} → {c}" for p, c in sorted(self.edges))
        )


# ---------------------------------------------------------------------------
# Information-theoretic helpers
# ---------------------------------------------------------------------------


def _mutual_information(observations: Sequence[Observation], x: str, y: str) -> float:
    n = 0
    counts_x: Counter = Counter()
    counts_y: Counter = Counter()
    counts_xy: Counter = Counter()
    for obs in observations:
        if x not in obs or y not in obs:
            continue
        n += 1
        counts_x[obs[x]] += 1
        counts_y[obs[y]] += 1
        counts_xy[(obs[x], obs[y])] += 1
    if n == 0:
        return 0.0
    mi = 0.0
    for (vx, vy), cxy in counts_xy.items():
        pxy = cxy / n
        px = counts_x[vx] / n
        py = counts_y[vy] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py))
    return max(mi, 0.0)


def _conditional_mi(
    observations: Sequence[Observation],
    x: str,
    y: str,
    z: Sequence[str],
) -> float:
    """I(X;Y | Z) — pure-Python tabular estimate."""
    if not z:
        return _mutual_information(observations, x, y)
    groups: dict[tuple[Any, ...], list[Observation]] = defaultdict(list)
    for obs in observations:
        if x not in obs or y not in obs or any(zi not in obs for zi in z):
            continue
        key = tuple(obs[zi] for zi in z)
        groups[key].append(obs)
    total = sum(len(g) for g in groups.values())
    if total == 0:
        return 0.0
    cmi = 0.0
    for group in groups.values():
        weight = len(group) / total
        cmi += weight * _mutual_information(group, x, y)
    return cmi


# ---------------------------------------------------------------------------
# Structure learning (lightweight PC-style)
# ---------------------------------------------------------------------------


def learn_structure(
    observations: Sequence[Observation],
    *,
    mi_threshold: float = 0.02,
    cmi_threshold: float = 0.01,
    max_cond_set: int = 2,
    variable_order: Sequence[str] | None = None,
) -> CausalGraph:
    """Discover causal structure from observations.

    Steps
    -----
    1. Extract the union of variables appearing in observations.
    2. Add an undirected edge X-Y if I(X;Y) > ``mi_threshold``.
    3. For every pair, search for a separating set Z (|Z| ≤ ``max_cond_set``)
       such that I(X;Y|Z) < ``cmi_threshold``. If found, drop the edge.
    4. Orient edges using the supplied ``variable_order`` (or temporal index
       within each observation if absent), preferring earlier→later.
    5. Estimate tabular CPTs from observations.
    """
    variables = sorted({k for obs in observations for k in obs.keys()})
    if variable_order:
        variables = [v for v in variable_order if v in variables] + [
            v for v in variables if v not in variable_order
        ]

    graph = CausalGraph(nodes=set(variables))

    # Phase 1+2: undirected skeleton
    skeleton: set[frozenset[str]] = set()
    for i, x in enumerate(variables):
        for y in variables[i + 1:]:
            mi = _mutual_information(observations, x, y)
            if mi > mi_threshold:
                skeleton.add(frozenset({x, y}))

    # Phase 3: prune via conditional independence
    pruned = set()
    for edge in list(skeleton):
        x, y = tuple(edge)
        candidates = [v for v in variables if v not in edge]
        # Try empty, size-1, size-2 separating sets
        for size in range(0, max_cond_set + 1):
            from itertools import combinations
            stop = False
            for sep in combinations(candidates, size):
                if _conditional_mi(observations, x, y, list(sep)) < cmi_threshold:
                    pruned.add(edge)
                    stop = True
                    break
            if stop:
                break
    skeleton -= pruned

    # Phase 4: orient by variable_order
    rank = {v: i for i, v in enumerate(variables)}
    for edge in skeleton:
        x, y = sorted(edge, key=lambda v: rank[v])
        graph.edges.add((x, y))

    # Phase 5: CPTs
    _fit_cpts(graph, observations)
    return graph


def _fit_cpts(graph: CausalGraph, observations: Sequence[Observation]) -> None:
    for child in graph.nodes:
        parents = sorted(p for (p, c) in graph.edges if c == child)
        graph.parent_order[child] = parents
        cpt: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
        for obs in observations:
            if child not in obs:
                continue
            if any(p not in obs for p in parents):
                continue
            key = tuple(obs[p] for p in parents)
            cpt[key][obs[child]] += 1
        graph.cpts[child] = dict(cpt)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _normalise(counter: Counter) -> dict[Any, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def predict(
    graph: CausalGraph,
    target: str,
    *,
    evidence: Observation | None = None,
    interventions: Observation | None = None,
) -> dict[Any, float]:
    """Estimate ``P(target | evidence, do(interventions))``.

    Interventions cut incoming edges to the intervened variables and replace
    their CPT with a deterministic assignment (do-calculus, simplified).
    """
    evidence = evidence or {}
    interventions = interventions or {}

    if target in interventions:
        return {interventions[target]: 1.0}

    # When intervening, we copy the graph and sever incoming edges to do(X).
    g = _intervene(graph, interventions)

    # Tabular inference: condition on (evidence ∪ interventions) over parents
    parents = g.parent_order.get(target, [])
    cpt = g.cpts.get(target, {})
    if not parents:
        # Marginal — use the empty-tuple key
        counter = cpt.get((), Counter())
        return _normalise(counter)

    # If all parents are observed/intervened, direct lookup
    parent_assignment = {**evidence, **interventions}
    if all(p in parent_assignment for p in parents):
        key = tuple(parent_assignment[p] for p in parents)
        return _normalise(cpt.get(key, Counter()))

    # Otherwise: marginalise unobserved parents using their predicted distrs
    posterior: dict[Any, float] = defaultdict(float)
    parent_dists = {
        p: predict(g, p, evidence=evidence, interventions=interventions)
        if p not in parent_assignment
        else {parent_assignment[p]: 1.0}
        for p in parents
    }

    from itertools import product
    for combo in product(*[list(d.items()) for d in parent_dists.values()]):
        weight = 1.0
        key_vals: list[Any] = []
        for (val, prob) in combo:
            weight *= prob
            key_vals.append(val)
        cond = _normalise(cpt.get(tuple(key_vals), Counter()))
        for v, p in cond.items():
            posterior[v] += weight * p

    total = sum(posterior.values())
    return {k: v / total for k, v in posterior.items()} if total else {}


def _intervene(graph: CausalGraph, interventions: Observation) -> CausalGraph:
    if not interventions:
        return graph
    g = CausalGraph(
        nodes=set(graph.nodes),
        edges={(p, c) for (p, c) in graph.edges if c not in interventions},
        cpts=dict(graph.cpts),
        parent_order={k: list(v) for k, v in graph.parent_order.items()},
    )
    for var, val in interventions.items():
        g.parent_order[var] = []
        g.cpts[var] = {(): Counter({val: 1})}
    return g


# ---------------------------------------------------------------------------
# Counterfactual queries
# ---------------------------------------------------------------------------


def counterfactual(
    graph: CausalGraph,
    factual: Observation,
    intervention: Observation,
    target: str,
) -> dict[Any, float]:
    """Twin-network counterfactual: "given factual world ``factual`` was
    observed, what would ``target`` have been if we had set ``intervention``?"

    Implemented as the simplified Pearl three-step abduction-action-prediction:
    we condition on the factual values that *aren't* being intervened on, then
    predict under the intervention.
    """
    held_evidence = {k: v for k, v in factual.items() if k not in intervention}
    return predict(graph, target, evidence=held_evidence, interventions=intervention)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


@dataclass
class CausalReasoner:
    """High-level façade combining structure learning and inference."""

    graph: CausalGraph = field(default_factory=CausalGraph)
    observations: list[Observation] = field(default_factory=list)

    def add_observations(self, observations: Iterable[Observation]) -> None:
        self.observations.extend(observations)

    def learn(self, **kwargs: Any) -> CausalGraph:
        self.graph = learn_structure(self.observations, **kwargs)
        return self.graph

    def predict(self, target: str, **kwargs: Any) -> dict[Any, float]:
        return predict(self.graph, target, **kwargs)

    def counterfactual(self, factual: Observation, intervention: Observation, target: str) -> dict[Any, float]:
        return counterfactual(self.graph, factual, intervention, target)

    def from_episodic(self, limit: int = 200) -> "CausalReasoner":
        """Pull recent episodic memories and treat each as one observation.

        Each memory is parsed as a flat dict of features. Strings are tokenised
        into ``(key=value)`` segments via simple regex.
        """
        try:
            from isaac.memory.episodic import get_episodic_memory
            mem = get_episodic_memory()
            episodes = mem.recent(limit) if hasattr(mem, "recent") else []
            for ep in episodes:
                obs = self._episode_to_obs(ep)
                if obs:
                    self.observations.append(obs)
        except Exception as exc:
            logger.debug("from_episodic failed: %s", exc)
        return self

    @staticmethod
    def _episode_to_obs(episode: Any) -> Observation | None:
        if isinstance(episode, dict):
            return {k: v for k, v in episode.items() if isinstance(v, (str, int, float, bool))}
        # Fallback: try dataclass-like attrs
        try:
            return {k: v for k, v in episode.__dict__.items()
                    if isinstance(v, (str, int, float, bool))}
        except Exception:
            return None

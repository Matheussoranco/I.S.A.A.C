"""Graph-of-Thought (GoT) Planner — DAG-based plan decomposition and execution.

Extends the sequential :class:`PlanStep` list into a proper DAG where:
- Steps can have multiple dependencies (fan-in).
- Steps can have multiple dependents (fan-out).
- Parallel-ready steps are activated simultaneously.
- Critical-path analysis guides resource allocation.

The GoT planner wraps the LLM planner output into a ``PlanDAG`` and
exposes helper queries used by the graph builder and transitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from isaac.core.state import PlanStep

logger = logging.getLogger(__name__)


@dataclass
class PlanDAG:
    """DAG representation of a multi-step plan.

    Wraps a list of :class:`PlanStep` into a NetworkX DiGraph for
    topological ordering, parallelism detection, and critical-path
    analysis.
    """

    steps: list[PlanStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._graph = nx.DiGraph()
        self._build()

    def _build(self) -> None:
        """Populate the internal graph from the step list."""
        self._graph.clear()
        step_map = {s.id: s for s in self.steps}

        for step in self.steps:
            self._graph.add_node(
                step.id,
                step=step,
                description=step.description,
                mode=step.mode,
                status=step.status,
            )

        for step in self.steps:
            for dep_id in step.depends_on:
                if dep_id in step_map:
                    self._graph.add_edge(dep_id, step.id)

    def refresh(self) -> None:
        """Re-sync internal graph after step status mutations."""
        for step in self.steps:
            if step.id in self._graph:
                self._graph.nodes[step.id]["status"] = step.status

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def ready_steps(self) -> list[PlanStep]:
        """Return all steps whose dependencies are satisfied and status is pending."""
        ready: list[PlanStep] = []
        for step in self.steps:
            if step.status != "pending":
                continue
            deps_done = all(
                self._graph.nodes.get(d, {}).get("status") == "done"
                for d in step.depends_on
            ) if step.depends_on else True
            if deps_done:
                ready.append(step)
        return ready

    def activate_ready(self) -> list[PlanStep]:
        """Mark all ready steps as ``"active"`` and return them."""
        ready = self.ready_steps()
        for step in ready:
            step.status = "active"
            if step.id in self._graph:
                self._graph.nodes[step.id]["status"] = "active"
        return ready

    def topological_order(self) -> list[str]:
        """Return step IDs in topological order (respecting dependencies)."""
        try:
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXUnfeasible:
            logger.error("PlanDAG contains a cycle — returning flat order.")
            return [s.id for s in self.steps]

    def critical_path(self) -> list[str]:
        """Return the longest path through the DAG (critical path).

        Uses unweighted longest path via DAG longest path algorithm.
        """
        try:
            return nx.dag_longest_path(self._graph)
        except nx.NetworkXUnfeasible:
            return [s.id for s in self.steps]

    def parallelism_level(self) -> int:
        """Maximum number of steps that can execute concurrently (max anti-chain width)."""
        ready = self.ready_steps()
        return max(len(ready), 1)

    def is_complete(self) -> bool:
        """All steps are done or failed."""
        return all(s.status in ("done", "failed") for s in self.steps)

    def pending_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "pending")

    def active_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "active")

    def get_step(self, step_id: str) -> PlanStep | None:
        """Lookup a step by id."""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def dependents(self, step_id: str) -> list[str]:
        """Return IDs of steps that depend on ``step_id``."""
        if step_id in self._graph:
            return list(self._graph.successors(step_id))
        return []

    def to_context_string(self) -> str:
        """Serialise the plan DAG into a compact text block for LLM prompts."""
        lines = [f"Plan DAG ({len(self.steps)} steps):"]
        for step_id in self.topological_order():
            step = self.get_step(step_id)
            if step:
                deps = f" (depends: {', '.join(step.depends_on)})" if step.depends_on else ""
                lines.append(f"  [{step.status}] {step.id}: {step.description}{deps}")
        cp = self.critical_path()
        if cp:
            lines.append(f"  Critical path: {' → '.join(cp)}")
        return "\n".join(lines)


def build_plan_dag(steps: list[PlanStep]) -> PlanDAG:
    """Convenience factory for creating a PlanDAG from step list."""
    return PlanDAG(steps=steps)

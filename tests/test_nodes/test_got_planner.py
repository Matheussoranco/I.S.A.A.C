"""Tests for the Graph-of-Thought PlanDAG."""

from __future__ import annotations

import pytest

from isaac.core.state import PlanStep
from isaac.nodes.got_planner import PlanDAG, build_plan_dag


@pytest.fixture()
def linear_plan() -> list[PlanStep]:
    return [
        PlanStep(id="s1", description="Step 1", status="pending"),
        PlanStep(id="s2", description="Step 2", status="pending", depends_on=["s1"]),
        PlanStep(id="s3", description="Step 3", status="pending", depends_on=["s2"]),
    ]


@pytest.fixture()
def parallel_plan() -> list[PlanStep]:
    return [
        PlanStep(id="s1", description="Step 1", status="pending"),
        PlanStep(id="s2", description="Step 2", status="pending"),
        PlanStep(id="s3", description="Step 3", status="pending", depends_on=["s1", "s2"]),
    ]


class TestPlanDAG:
    def test_build_linear(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        assert len(dag.steps) == 3

    def test_ready_steps_linear(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        ready = dag.ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "s1"

    def test_ready_steps_parallel(self, parallel_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=parallel_plan)
        ready = dag.ready_steps()
        assert len(ready) == 2
        ids = {s.id for s in ready}
        assert ids == {"s1", "s2"}

    def test_activate_ready(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        activated = dag.activate_ready()
        assert len(activated) == 1
        assert activated[0].status == "active"

    def test_topological_order(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        order = dag.topological_order()
        assert order == ["s1", "s2", "s3"]

    def test_critical_path(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        cp = dag.critical_path()
        assert cp == ["s1", "s2", "s3"]

    def test_parallelism_level(self, parallel_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=parallel_plan)
        assert dag.parallelism_level() == 2

    def test_is_complete_false(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        assert not dag.is_complete()

    def test_is_complete_true(self) -> None:
        steps = [
            PlanStep(id="s1", description="done", status="done"),
            PlanStep(id="s2", description="failed", status="failed"),
        ]
        dag = PlanDAG(steps=steps)
        assert dag.is_complete()

    def test_pending_count(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        assert dag.pending_count() == 3

    def test_active_count(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        dag.activate_ready()
        assert dag.active_count() == 1

    def test_get_step(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        s = dag.get_step("s2")
        assert s is not None
        assert s.description == "Step 2"

    def test_get_step_not_found(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        assert dag.get_step("missing") is None

    def test_dependents(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        deps = dag.dependents("s1")
        assert "s2" in deps

    def test_to_context_string(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        ctx = dag.to_context_string()
        assert "Plan DAG" in ctx
        assert "s1" in ctx
        assert "Critical path" in ctx

    def test_refresh(self, linear_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=linear_plan)
        linear_plan[0].status = "done"
        dag.refresh()
        ready_now = dag.ready_steps()
        assert any(s.id == "s2" for s in ready_now)

    def test_build_plan_dag_factory(self, linear_plan: list[PlanStep]) -> None:
        dag = build_plan_dag(linear_plan)
        assert isinstance(dag, PlanDAG)
        assert len(dag.steps) == 3

    def test_fan_in(self, parallel_plan: list[PlanStep]) -> None:
        dag = PlanDAG(steps=parallel_plan)
        # s3 depends on s1 and s2 — not ready until both done
        parallel_plan[0].status = "done"
        dag.refresh()
        ready = dag.ready_steps()
        assert not any(s.id == "s3" for s in ready)  # s2 still pending

        parallel_plan[1].status = "done"
        dag.refresh()
        ready = dag.ready_steps()
        assert any(s.id == "s3" for s in ready)

"""Offline tests for the specialist Orchestrator.

Everything heavy is injected (planner, specialist factory, manager LLM) so the
pipeline runs without touching the registry, a real LLM, or the network.
"""

from __future__ import annotations

import pytest

from isaac.specialists.base import SpecialistResult
from isaac.specialists.orchestrator import Orchestrator, SubTask, orchestrate


class _StubSpecialist:
    """A minimal specialist whose run() echoes the task and flags dependency context."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.last_context = ""

    def run(self, task: str, context: str = "") -> SpecialistResult:
        self.last_context = context
        return SpecialistResult(
            specialist=self.name,
            task=task,
            output=f"did:{task} ctx_has_dep={'did:' in context}",
            success=True,
        )


class _StubLLM:
    """A fake LangChain chat model whose invoke() returns fixed content."""

    def __init__(self, content: str = "SYNTH") -> None:
        self._content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        class _Resp:
            content = self._content

        return _Resp()


def _factory(record: dict | None = None):
    def factory(name: str, **kwargs):
        stub = _StubSpecialist(name)
        if record is not None:
            record[name] = stub
        return stub

    return factory


def _two_task_planner(goal: str, roster: list[dict], context: str) -> list[SubTask]:
    return [
        SubTask(id="t1", description="research it", specialist="researcher"),
        SubTask(id="t2", description="code it", specialist="coder", depends_on=["t1"]),
    ]


def test_run_two_tasks_and_synthesize() -> None:
    orch = Orchestrator(
        planner=_two_task_planner,
        specialist_factory=_factory(),
        manager_llm=_StubLLM("FINAL ANSWER"),
    )
    res = orch.run("build X")
    assert res.success is True
    assert len(res.results) == 2
    assert res.final_output == "FINAL ANSWER"


def test_dependency_ordering_passes_context() -> None:
    orch = Orchestrator(
        planner=_two_task_planner,
        specialist_factory=_factory(),
        manager_llm=_StubLLM(),
    )
    res = orch.run("build X")
    coder = next(r for r in res.results if r.subtask.id == "t2")
    # The coder subtask depends on t1, so its context must carry t1's output.
    assert "ctx_has_dep=True" in coder.result.output


def test_unknown_specialist_falls_back_to_generalist() -> None:
    def planner(goal, roster, context):
        return [SubTask(id="t1", description="do it", specialist="nonexistent")]

    def factory(name, **kwargs):
        if name == "nonexistent":
            raise KeyError(name)
        return _StubSpecialist(name)

    res = Orchestrator(planner=planner, specialist_factory=factory, manager_llm=_StubLLM()).run("g")
    assert res.success is True
    assert res.results[0].result.specialist == "generalist"


def test_default_planner_falls_back_on_bad_json() -> None:
    orch = Orchestrator(specialist_factory=_factory(), manager_llm=_StubLLM("not json at all"))
    plan = orch._plan("goal", [{"name": "coder", "domain": "code"}], "")
    assert len(plan) == 1
    assert plan[0].specialist == "generalist"


def test_default_planner_empty_roster() -> None:
    orch = Orchestrator(manager_llm=_StubLLM("not json"))
    plan = orch._plan("goal", [], "")
    assert len(plan) == 1
    assert plan[0].specialist == "generalist"


def test_events_emitted() -> None:
    kinds: list[str] = []
    orch = Orchestrator(
        planner=_two_task_planner,
        specialist_factory=_factory(),
        manager_llm=_StubLLM(),
        on_event=lambda k, d: kinds.append(k),
    )
    orch.run("build X")
    assert "plan" in kinds
    assert "final" in kinds


def test_orchestrate_convenience() -> None:
    res = orchestrate(
        "ship it",
        planner=_two_task_planner,
        specialist_factory=_factory(),
        manager_llm=_StubLLM("DONE"),
    )
    assert res.final_output == "DONE"
    assert res.success is True


def test_single_task_returns_output_verbatim() -> None:
    def planner(goal, roster, context):
        return [SubTask(id="t1", description=goal, specialist="generalist")]

    res = Orchestrator(planner=planner, specialist_factory=_factory(), manager_llm=_StubLLM()).run(
        "just one thing"
    )
    # A lone subtask is returned verbatim (no synthesis call needed).
    assert res.final_output.startswith("did:just one thing")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

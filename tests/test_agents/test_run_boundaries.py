from __future__ import annotations

import asyncio
import json
import threading
import time
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from isaac.agents.agent_loop import AgentLoop
from isaac.eval.runner import TaskAnswer, default_runner, execute_task
from isaac.eval.suite import EvalTask
from isaac.specialists.base import Specialist, SpecialistResult
from isaac.specialists.orchestrator import MAX_SUBTASKS, Orchestrator, SubTask
from isaac.tools.base import IsaacTool, ToolResult


class ScriptLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def bind_tools(self, schemas):
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        response = next(self.responses)
        return response() if callable(response) else response


class Gate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    def wait(self):
        self.entered.set()
        try:
            assert self.release.wait(3)
        finally:
            self.exited.set()


class Action(IsaacTool):
    name = "action"
    description = "Record an action"

    def __init__(self, effect=None, risk=1):
        self.calls = []
        self.effect = effect
        self.risk_level = risk
        self.closed = threading.Event()

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.effect:
            self.effect()
        return ToolResult(success=True, output="done")

    async def aclose(self):
        self.closed.set()


def calls(*names):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {}, "id": str(i)} for i, name in enumerate(names)],
    )


def success(task, name="generalist"):
    return SpecialistResult(name, task, f"done:{task}", True)


def factory(name, **kwargs):
    return SimpleNamespace(run=lambda task, context="": success(task, name))


@pytest.fixture(autouse=True)
def isolated_team(monkeypatch):
    monkeypatch.setattr(
        Orchestrator, "_list_roster", staticmethod(lambda: [{"name": "generalist"}])
    )
    monkeypatch.setattr(Orchestrator, "_record", lambda *args: None)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_attachments_are_multimodal_content_blocks(asynchronous):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}
    llm = ScriptLLM(AIMessage(content="an image"))
    agent = AgentLoop([], llm=llm)
    if asynchronous:
        result = await agent.arun("describe", context="context", attachments=[image])
    else:
        result = agent.run("describe", context="context", attachments=[image])
    message = llm.messages[0][1]
    assert isinstance(message, HumanMessage)
    assert message.content == [
        {"type": "text", "text": "describe\n\n<context>\ncontext\n</context>"},
        image,
    ]
    message.content[1]["image_url"]["url"] = "changed"
    assert image["image_url"]["url"].startswith("data:")
    assert result.completed


def test_text_only_message_remains_a_string():
    llm = ScriptLLM(AIMessage(content="done"))
    AgentLoop([], llm=llm).run("text")
    assert llm.messages[0][1].content == "text"


@pytest.mark.parametrize("cancel", [False, True])
def test_blocked_agent_model_returns_promptly_and_late_response_cannot_run_tools(cancel):
    gate = Gate()
    action = Action()

    def response():
        gate.wait()
        return calls("action")

    llm = ScriptLLM(response)
    agent = AgentLoop(
        [action],
        llm=llm,
        max_wall_seconds=0 if cancel else 0.12,
        should_stop=gate.entered.is_set if cancel else None,
    )
    started = time.monotonic()
    try:
        result = agent.run("act")
        assert time.monotonic() - started < 0.8
        assert gate.entered.is_set()
        assert not gate.exited.is_set()
        assert result.stopped_reason == ("cancelled" if cancel else "budget_exhausted")
    finally:
        gate.release.set()
    assert gate.exited.wait(1)
    assert action.closed.wait(1)
    assert action.calls == []
    assert len(llm.messages) == 1


def test_blocking_tool_is_bounded_and_cannot_dispatch_later_tools():
    gate = Gate()
    slow = Action(gate.wait)
    later = Action()
    later.name = "later"
    llm = ScriptLLM(calls("action", "later"), AIMessage(content="done"))
    started = time.monotonic()
    try:
        result = AgentLoop([slow, later], llm=llm, max_wall_seconds=0.15).run("act")
        assert time.monotonic() - started < 0.8
        assert gate.entered.is_set()
        assert result.stopped_reason == "budget_exhausted"
        assert not gate.exited.is_set()
    finally:
        gate.release.set()
    assert slow.closed.wait(1)
    assert later.calls == []
    assert len(llm.messages) == 1


def test_cancellation_between_tool_calls_is_latched():
    stop = threading.Event()
    first = Action(stop.set)
    later = Action()
    later.name = "later"
    llm = ScriptLLM(calls("action", "later"))
    result = AgentLoop([first, later], llm=llm, should_stop=stop.is_set).run("act")
    assert result.stopped_reason == "cancelled"
    assert len(first.calls) == 1
    assert later.calls == []


def test_cancellation_in_tool_event_prevents_approval_and_execution():
    stop = threading.Event()
    approvals = []
    action = Action(risk=5)

    def on_event(kind, data):
        if kind == "tool_call":
            stop.set()

    result = AgentLoop(
        [action],
        llm=ScriptLLM(calls("action")),
        should_stop=stop.is_set,
        on_event=on_event,
        approval_callback=lambda *args: approvals.append(args) or True,
    ).run("act")
    assert result.stopped_reason == "cancelled"
    assert approvals == []
    assert action.calls == []


def test_approval_that_cancels_cannot_authorize_a_later_tool():
    stop = threading.Event()
    action = Action(risk=5)

    def approve(*args):
        stop.set()
        return True

    result = AgentLoop(
        [action],
        llm=ScriptLLM(calls("action")),
        should_stop=stop.is_set,
        approval_callback=approve,
    ).run("act")
    assert result.stopped_reason == "cancelled"
    assert action.calls == []


async def test_async_cancellation_during_approval_prevents_late_execution():
    gate = Gate()
    action = Action(risk=5)

    def approve(*args):
        gate.wait()
        return True

    agent = AgentLoop([action], llm=ScriptLLM(calls("action")), approval_callback=approve)
    task = asyncio.create_task(agent.arun("act"))
    try:
        async with asyncio.timeout(1) if hasattr(asyncio, "timeout") else _timeout():
            while not gate.entered.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        gate.release.set()
    assert gate.exited.wait(1)
    assert action.closed.wait(1)
    assert action.calls == []


class _timeout:
    async def __aenter__(self):
        self.task = asyncio.current_task()
        self.handle = asyncio.get_running_loop().call_later(1, self.task.cancel)

    async def __aexit__(self, *args):
        self.handle.cancel()


def test_cancellation_during_retry_backoff_prevents_second_model_call():
    stop = threading.Event()

    def fail():
        stop.set()
        raise RuntimeError("retryable")

    llm = ScriptLLM(fail, AIMessage(content="must not happen"))
    result = AgentLoop([], llm=llm, should_stop=stop.is_set).run("act")
    assert result.stopped_reason == "cancelled"
    assert len(llm.messages) == 1


@pytest.mark.parametrize(
    "plan",
    [
        [SubTask("a", "one", "x"), SubTask("a", "two", "x")],
        [SubTask("a", "one", "x", ["missing"])],
        [SubTask("a", "one", "x", ["a"])],
        [SubTask("a", "one", "x", ["b"]), SubTask("b", "two", "x", ["a"])],
        [SubTask("a", "one", "x", "a")],
        [SubTask(str(i), "one", "x") for i in range(MAX_SUBTASKS + 1)],
        [],
    ],
)
def test_invalid_plan_is_rejected_before_any_dispatch(plan):
    created = []
    result = Orchestrator(
        planner=lambda *args: plan,
        specialist_factory=lambda *args, **kwargs: created.append(args),
    ).run("goal")
    assert result.stopped_reason == "invalid_plan"
    assert result.error
    assert not result.completed
    assert not result.success
    assert created == []


@pytest.mark.parametrize("malformation", ["oversize", "cycle", "dependency_type"])
def test_manager_plan_validation_does_not_fallback_or_truncate(malformation):
    raw = [SubTask("a", "one", "x").to_dict()]
    if malformation == "oversize":
        raw *= MAX_SUBTASKS + 1
    elif malformation == "cycle":
        raw[0]["depends_on"] = ["a"]
    else:
        raw[0]["depends_on"] = "a"
    created = []
    manager = ScriptLLM(AIMessage(content=json.dumps({"subtasks": raw})))
    result = Orchestrator(
        manager_llm=manager,
        specialist_factory=lambda *args, **kwargs: created.append(args),
    ).run("goal")
    assert result.stopped_reason == "invalid_plan"
    assert created == []


def test_wave_limit_does_not_silently_truncate_independent_plan():
    plan = [SubTask(str(i), str(i), "x") for i in range(3)]
    result = Orchestrator(
        planner=lambda *args: plan,
        specialist_factory=factory,
        manager_llm=ScriptLLM(AIMessage(content="synthesized")),
        max_iterations=1,
    ).run("goal")
    assert len(result.plan) == len(result.results) == 3
    assert result.completed


def test_wave_limit_explicitly_marks_all_unexecuted_dependencies():
    plan = [SubTask("a", "one", "x"), SubTask("b", "two", "x", ["a"])]
    result = Orchestrator(
        planner=lambda *args: plan,
        specialist_factory=factory,
        max_iterations=1,
        manager_llm=ScriptLLM(),
    ).run("goal")
    assert len(result.plan) == len(result.results) == 2
    assert result.stopped_reason == "max_iterations"
    assert result.results[1].result.stopped_reason == "max_iterations"
    assert not result.completed


@pytest.mark.parametrize("stage", ["planner", "manager_planning", "synthesis"])
@pytest.mark.parametrize("cancel", [False, True])
def test_orchestration_budget_and_cancellation_include_manager_stages(stage, cancel):
    gate = Gate()
    dispatched = []
    plan = [SubTask("a", "one", "x"), SubTask("b", "two", "x")]

    def planner(*args):
        if stage == "planner":
            gate.wait()
        return plan

    def manager_response():
        gate.wait()
        return AIMessage(content=json.dumps({"subtasks": [st.to_dict() for st in plan]}))

    def make(name, **kwargs):
        dispatched.append(name)
        return factory(name, **kwargs)

    team = Orchestrator(
        planner=None if stage == "manager_planning" else planner,
        specialist_factory=make,
        manager_llm=ScriptLLM(manager_response),
        max_wall_seconds=0 if cancel else 0.15,
        should_stop=gate.entered.is_set if cancel else None,
    )
    started = time.monotonic()
    try:
        result = team.run("goal")
        assert time.monotonic() - started < 0.8
        assert gate.entered.is_set()
        assert not gate.exited.is_set()
        assert result.stopped_reason == ("cancelled" if cancel else "budget_exhausted")
        assert not result.completed
        assert not result.success
        assert len(dispatched) == (2 if stage == "synthesis" else 0)
    finally:
        gate.release.set()
    assert gate.exited.wait(1)


def test_remaining_budget_is_passed_after_planning_and_factory_time(monkeypatch):
    captured = []
    built = []

    def approval(*args):
        return True

    def planner(*args):
        time.sleep(0.06)
        return [SubTask("a", "one", "x")]

    def make(name, **kwargs):
        captured.append(kwargs)
        time.sleep(0.04)
        return Specialist(llm=ScriptLLM(AIMessage(content="done")), persona="", **kwargs)

    def build(**kwargs):
        built.append(kwargs)
        kwargs.pop("only")
        return AgentLoop([], **kwargs)

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", build)
    result = Orchestrator(
        planner=planner,
        specialist_factory=make,
        max_wall_seconds=1,
        approval_callback=approval,
        specialist_max_iterations=3,
    ).run("goal")
    assert result.completed
    assert 0 < built[0]["max_wall_seconds"] < captured[0]["max_wall_seconds"] < 0.97
    assert captured[0]["approval_callback"] is built[0]["approval_callback"] is approval
    assert built[0]["max_iterations"] == 3
    assert callable(built[0]["should_stop"])


@pytest.mark.parametrize("approved", [False, True])
def test_real_approval_callback_reaches_specialist_tool(monkeypatch, approved):
    action = Action(risk=5)
    approvals = []

    def approve(*args):
        approvals.append(args)
        return approved

    def build(**kwargs):
        kwargs.pop("only")
        return AgentLoop([action], **kwargs)

    def make(name, **kwargs):
        if name == "missing":
            raise KeyError(name)
        return Specialist(
            llm=ScriptLLM(calls("action"), AIMessage(content="done")),
            persona="",
            **kwargs,
        )

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", build)
    result = Orchestrator(
        planner=lambda *args: [SubTask("a", "act", "missing")],
        specialist_factory=make,
        approval_callback=approve,
        use_meta_selection=False,
    ).run("goal")
    assert result.completed
    assert approvals == [("action", {}, 5)]
    assert len(action.calls) == int(approved)


def test_parallel_subtask_timeouts_are_measured_from_dispatch_and_block_dependencies():
    gates = {name: Gate() for name in ("a", "b")}
    dispatched = []

    def make(name, **kwargs):
        def run(task, context=""):
            dispatched.append(task)
            gates[task].wait()
            return success(task)

        return SimpleNamespace(run=run)

    plan = [SubTask("a", "a", "x"), SubTask("b", "b", "x"), SubTask("c", "c", "x", ["a"])]
    started = time.monotonic()
    try:
        result = Orchestrator(
            planner=lambda *args: plan,
            specialist_factory=make,
            timeout_seconds=0.15,
            max_workers=2,
        ).run("goal")
        assert time.monotonic() - started < 0.27
        assert all(g.entered.is_set() and not g.exited.is_set() for g in gates.values())
        assert {r.subtask.id: r.result.stopped_reason for r in result.results} == {
            "a": "timeout",
            "b": "timeout",
            "c": "dependency_failed",
        }
        assert sorted(dispatched) == ["a", "b"]
    finally:
        for gate in gates.values():
            gate.release.set()
    assert all(g.exited.wait(1) for g in gates.values())


def test_queued_specialist_gets_its_own_timeout_at_dispatch():
    seen = []

    def make(name, **kwargs):
        def run(task, context=""):
            seen.append((task, kwargs["max_wall_seconds"]))
            time.sleep(0.06)
            return success(task)

        return SimpleNamespace(run=run)

    plan = [SubTask("a", "a", "x"), SubTask("b", "b", "x")]
    result = Orchestrator(
        planner=lambda *args: plan,
        specialist_factory=make,
        manager_llm=ScriptLLM(AIMessage(content="done")),
        timeout_seconds=0.1,
        max_workers=1,
    ).run("goal")
    assert result.completed
    assert [task for task, _ in seen] == ["a", "b"]
    assert all(0.07 < remaining <= 0.101 for _, remaining in seen)


def test_team_cancellation_propagates_through_real_specialist_to_late_model(monkeypatch):
    gate = Gate()
    action = Action()

    def response():
        gate.wait()
        return calls("action")

    def build(**kwargs):
        kwargs.pop("only")
        return AgentLoop([action], **kwargs)

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", build)
    try:
        result = Orchestrator(
            planner=lambda *args: [SubTask("a", "act", "x")],
            specialist_factory=lambda name, **kwargs: Specialist(
                llm=ScriptLLM(response),
                persona="",
                **kwargs,
            ),
            should_stop=gate.entered.is_set,
        ).run("goal")
        assert result.stopped_reason == "cancelled"
    finally:
        gate.release.set()
    assert action.closed.wait(1)
    assert action.calls == []


def test_contextvars_survive_team_and_agent_workers(monkeypatch):
    scope = ContextVar("test_scope", default="missing")
    observed = []
    action = Action(lambda: observed.append(scope.get()))

    def build(**kwargs):
        kwargs.pop("only")
        return AgentLoop([action], **kwargs)

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", build)
    token = scope.set("workspace")
    try:
        result = Orchestrator(
            planner=lambda *args: [SubTask("a", "act", "x")],
            specialist_factory=lambda name, **kwargs: Specialist(
                llm=ScriptLLM(calls("action"), AIMessage(content="done")),
                persona="",
                **kwargs,
            ),
        ).run("goal")
    finally:
        scope.reset(token)
    assert result.completed
    assert observed == ["workspace"]


@pytest.mark.parametrize("runner_kind", ["agent", "team"])
def test_eval_forwards_budgets_cancellation_and_approval(monkeypatch, runner_kind):
    captured = []
    stop = threading.Event()

    def approval(*args):
        return True

    def build(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            run=lambda *args, **kwargs: SimpleNamespace(
                output="partial",
                final_output="partial",
                stopped_reason="budget_exhausted",
            )
        )

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", build)
    monkeypatch.setattr("isaac.specialists.Orchestrator", build)
    task = EvalTask("one", "act", runner=runner_kind, max_iterations=2, timeout_seconds=0.5)
    answer = default_runner(should_stop=stop.is_set, approval_callback=approval)(task)
    assert answer.stopped_reason == "budget_exhausted"
    assert captured[0]["max_wall_seconds"] == 0.5
    assert captured[0]["max_iterations"] == 2
    assert captured[0]["should_stop"] == stop.is_set
    assert captured[0]["approval_callback"] is approval
    if runner_kind == "team":
        assert captured[0]["specialist_max_iterations"] == 2


def test_eval_bounds_injected_runner_and_does_not_accept_late_success(tmp_path):
    gate = Gate()

    def run(task):
        gate.wait()
        return TaskAnswer("success")

    started = time.monotonic()
    try:
        answer, _, _ = execute_task(EvalTask("one", "act", timeout_seconds=0.1), run, tmp_path)
        assert time.monotonic() - started < 0.8
        assert answer.stopped_reason == "budget_exhausted"
        assert gate.entered.is_set() and not gate.exited.is_set()
    finally:
        gate.release.set()
    assert gate.exited.wait(1)

"""Tests for the Phase 1.3 (Hermes-mirror) loop guards in AgentLoop."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from isaac.agents.agent_loop import AgentLoop
from isaac.tools.base import IsaacTool, ToolResult
from tests.test_agents.test_agent_loop import FakeLLM, _tool_call


class StaticTool(IsaacTool):
    """Always succeeds with the same output — never makes progress."""

    name = "static"
    description = "Returns a constant string."
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
    }

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, output="constant")


class DistinctTool(IsaacTool):
    """Succeeds with a fresh output every call — always makes progress."""

    name = "distinct"
    description = "Returns a counter."
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"idx": {"type": "integer"}},
    }

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, output=f"result-{self.calls}")


class TestNoProgressGuard:
    def test_no_progress_streak_triggers_on_static_output(self) -> None:
        # Every call uses fresh args (so the repeat-bounce never fires) but the
        # tool always returns the identical output -> no iteration makes
        # progress -> bail at the no-progress limit (default 5).
        tool = StaticTool()
        llm = FakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("static", {"key": f"k{i}"}, cid=f"c{i}")],
                )
                for i in range(20)
            ]
        )
        result = AgentLoop([tool], llm=llm, max_iterations=20).run("grind")

        assert result.stopped_reason == "no_progress"
        assert result.no_progress_streak >= 5
        assert result.iterations == 6  # first iteration made progress, then 5 stale
        assert tool.calls == 6
        assert result.success is False

    def test_fresh_output_resets_the_streak(self) -> None:
        tool = DistinctTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("distinct", {"idx": i}, cid=f"c{i}")])
                for i in range(3)
            ]
            + [AIMessage(content="all done")]
        )
        result = AgentLoop([tool], llm=llm, max_iterations=10).run("work")

        assert result.stopped_reason == "final"
        assert result.no_progress_streak == 0
        assert tool.calls == 3


class TestRepeatCallBounce:
    def test_third_identical_call_is_steered_not_executed(self) -> None:
        tool = StaticTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("static", {"key": "k"}, cid=f"c{i}")])
                for i in range(3)
            ]
            + [AIMessage(content="giving up, final answer")]
        )
        result = AgentLoop([tool], llm=llm, max_iterations=20).run("repeat")

        assert result.stopped_reason == "final"
        # Only the first two identical calls executed; the third was bounced.
        assert tool.calls == 2
        assert result.tool_call_count == 2
        # The steering observation was fed back to the model.
        steering = [
            m
            for m in result.messages
            if "[guard] repeated call to static" in str(getattr(m, "content", ""))
        ]
        assert len(steering) == 1
        assert "change strategy or finish" in str(steering[0].content)


class TestToolCallBudget:
    def test_max_tool_calls_stops_run(self) -> None:
        tool = DistinctTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("distinct", {"idx": i}, cid=f"c{i}")])
                for i in range(20)
            ]
        )
        result = AgentLoop([tool], llm=llm, max_iterations=50, max_tool_calls=4).run("spend")

        assert result.stopped_reason == "tool_call_budget"
        assert result.tool_call_count == 4
        assert len(result.tool_calls) == 4
        assert tool.calls == 4
        assert "budget" in result.output

    def test_zero_tool_call_budget_disables_guard(self) -> None:
        tool = DistinctTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("distinct", {"idx": i}, cid=f"c{i}")])
                for i in range(3)
            ]
            + [AIMessage(content="done")]
        )
        result = AgentLoop([tool], llm=llm, max_iterations=10, max_tool_calls=0).run("ok")

        assert result.stopped_reason == "final"
        assert tool.calls == 3


class TestIdleIterations:
    def test_generous_limits_preserve_existing_behaviour(self) -> None:
        # With default (generous) limits a normal run is untouched.
        tool = DistinctTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("distinct", {})]),
                AIMessage(content="finished"),
            ]
        )
        result = AgentLoop([tool], llm=llm).run("simple")

        assert result.stopped_reason == "final"
        assert result.idle_streak == 0
        assert result.tool_call_count == 1

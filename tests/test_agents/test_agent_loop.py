"""Tests for the AgentLoop tool-use engine."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from isaac.agents.agent_loop import AgentLoop, build_default_agent
from isaac.tools.base import IsaacTool, ToolResult


class EchoTool(IsaacTool):
    name = "echo"
    description = "Echo back the text."
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(success=True, output=f"ECHO:{kwargs.get('text', '')}")

    async def aclose(self) -> None:
        self.closed = True


class BoomTool(IsaacTool):
    name = "boom"
    description = "Always raises."
    risk_level = 1
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("kaboom")


class DangerTool(IsaacTool):
    name = "danger"
    description = "High-risk tool."
    risk_level = 5
    requires_approval = True
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.executed = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.executed = True
        return ToolResult(success=True, output="executed danger")


class FakeLLM:
    """Returns scripted AIMessages in sequence; records bound schemas."""

    def __init__(self, scripted: list[AIMessage]) -> None:
        self._scripted = scripted
        self._i = 0
        self.bound_schemas: Any = None

    def bind_tools(self, schemas: Any) -> FakeLLM:
        self.bound_schemas = schemas
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        msg = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return msg


def _tool_call(name: str, args: dict[str, Any], cid: str = "c1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


class TestAgentLoop:
    def test_executes_tool_then_returns_final(self) -> None:
        echo = EchoTool()
        llm = FakeLLM(
            [
                AIMessage(content="calling", tool_calls=[_tool_call("echo", {"text": "hi"})]),
                AIMessage(content="All done — echoed hi."),
            ]
        )
        loop = AgentLoop([echo], llm=llm)
        result = loop.run("please echo hi")

        assert result.stopped_reason == "final"
        assert result.success is True
        assert "hi" in result.output
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].output == "ECHO:hi"
        assert echo.calls == [{"text": "hi"}]

    def test_binds_function_schema(self) -> None:
        echo = EchoTool()
        llm = FakeLLM([AIMessage(content="done")])
        AgentLoop([echo], llm=llm).run("noop")
        assert llm.bound_schemas is not None
        fn = llm.bound_schemas[0]["function"]
        assert fn["name"] == "echo"
        assert fn["parameters"]["required"] == ["text"]

    def test_closes_tools_after_run(self) -> None:
        echo = EchoTool()
        llm = FakeLLM([AIMessage(content="immediate answer")])
        AgentLoop([echo], llm=llm).run("noop")
        assert echo.closed is True

    def test_blocks_high_risk_tool_by_default(self) -> None:
        danger = DangerTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("danger", {})]),
                AIMessage(content="cannot, blocked"),
            ]
        )
        result = AgentLoop([danger], llm=llm).run("do danger")
        assert danger.executed is False
        assert result.tool_calls[0].success is False
        assert "BLOCKED" in result.tool_calls[0].output

    def test_auto_approve_runs_high_risk_tool(self) -> None:
        danger = DangerTool()
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("danger", {})]),
                AIMessage(content="done"),
            ]
        )
        result = AgentLoop([danger], llm=llm, auto_approve=True).run("do danger")
        assert danger.executed is True
        assert result.tool_calls[0].success is True

    def test_tool_exception_is_captured(self) -> None:
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("boom", {})]),
                AIMessage(content="handled"),
            ]
        )
        result = AgentLoop([BoomTool()], llm=llm).run("trigger boom")
        assert result.tool_calls[0].success is False
        assert "kaboom" in result.tool_calls[0].output
        assert result.stopped_reason == "final"

    def test_unknown_tool_reported(self) -> None:
        llm = FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call("ghost", {})]),
                AIMessage(content="ok"),
            ]
        )
        result = AgentLoop([EchoTool()], llm=llm).run("call ghost")
        assert result.tool_calls[0].success is False
        assert "No such tool" in result.tool_calls[0].output

    def test_max_iterations_guard(self) -> None:
        # LLM always asks for another tool call -> never produces a final answer.
        llm = FakeLLM([AIMessage(content="", tool_calls=[_tool_call("echo", {"text": "x"})])])
        result = AgentLoop([EchoTool()], llm=llm, max_iterations=3).run("loop forever")
        assert result.stopped_reason == "max_iterations"
        assert result.iterations == 3
        assert len(result.tool_calls) == 3

    def test_handles_block_style_content(self) -> None:
        # Anthropic-style content blocks instead of a plain string.
        llm = FakeLLM([AIMessage(content=[{"type": "text", "text": "final via blocks"}])])
        result = AgentLoop([EchoTool()], llm=llm).run("noop")
        assert result.output == "final via blocks"


class TestBuildDefaultAgent:
    def test_builds_with_all_tools(self) -> None:
        agent = build_default_agent(llm=FakeLLM([AIMessage(content="x")]))
        assert "browser" in agent._tools
        assert "web_search" in agent._tools
        assert "code" in agent._tools

    def test_restricts_tools_with_only(self) -> None:
        agent = build_default_agent(llm=FakeLLM([AIMessage(content="x")]), only=["web_search"])
        assert set(agent._tools) == {"web_search"}

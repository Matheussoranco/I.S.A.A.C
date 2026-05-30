"""MCP tool surface — schema/handler invariants and agent-tool dispatch."""

from __future__ import annotations

from typing import Any

from isaac.agents.agent_loop import AgentRunResult, ToolCallRecord
from isaac.mcp import tools as mcp_tools


def test_every_schema_has_a_handler() -> None:
    names = {t["name"] for t in mcp_tools.TOOL_SCHEMAS}
    assert names == set(mcp_tools._HANDLERS)


def test_agent_tool_is_exposed() -> None:
    names = {t["name"] for t in mcp_tools.TOOL_SCHEMAS}
    assert "isaac_agent" in names


def test_handle_agent_routes_to_loop(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeLoop:
        def run(self, task: str, context: str = "") -> AgentRunResult:
            captured["task"] = task
            return AgentRunResult(
                output="final answer",
                iterations=2,
                tool_calls=[ToolCallRecord("web_search", {"query": "x"}, "out", True, 1.0)],
                stopped_reason="final",
            )

    def fake_build(**kwargs: Any) -> FakeLoop:
        captured["kwargs"] = kwargs
        return FakeLoop()

    monkeypatch.setattr("isaac.agents.agent_loop.build_default_agent", fake_build)

    result = mcp_tools.call_tool("isaac_agent", {"task": "do it", "tools": ["web_search"]})

    assert result["output"] == "final answer"
    assert result["success"] is True
    assert result["iterations"] == 2
    assert result["tool_calls"][0]["name"] == "web_search"
    assert captured["task"] == "do it"
    assert captured["kwargs"]["only"] == ["web_search"]


def test_spawn_subagent_agentic_flag(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    class FakeSubAgent:
        def __init__(self, role: str = "coder") -> None:
            calls["role"] = role

        def run(self, subtask: str, context: str = "") -> dict[str, Any]:
            calls["mode"] = "run"
            return {"result": "oneshot"}

        def run_agentic(self, subtask: str, context: str = "") -> dict[str, Any]:
            calls["mode"] = "agentic"
            return {"result": "agentic"}

    monkeypatch.setattr("isaac.agents.claude_subagent.ClaudeSubAgent", FakeSubAgent)

    out = mcp_tools.call_tool(
        "isaac_spawn_subagent", {"subtask": "research X", "role": "researcher", "agentic": True}
    )
    assert calls["mode"] == "agentic"
    assert calls["role"] == "researcher"
    assert out["result"] == "agentic"

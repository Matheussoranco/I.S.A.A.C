"""MCP tool surface — schema/handler invariants and agent-tool dispatch."""

from __future__ import annotations

import json
from typing import Any

from isaac.agents.agent_loop import AgentRunResult, ToolCallRecord
from isaac.mcp import tools as mcp_tools


def test_every_schema_has_a_handler() -> None:
    names = {t["name"] for t in mcp_tools.TOOL_SCHEMAS}
    assert names == set(mcp_tools._HANDLERS)


def test_agent_tool_is_exposed() -> None:
    names = {t["name"] for t in mcp_tools.TOOL_SCHEMAS}
    assert "isaac_agent" in names


class TestHandlersActuallyCall:
    """Invoke the local-only handlers for real.

    Regression: five of the nine tools called an API that does not exist —
    ``SemanticMemory.search``/``query_triples``, ``WebSearchConnector.search``,
    ``SkillLibrary(str(...))`` whose ``__init__`` calls ``Path.mkdir``,
    ``MemoryManager.recall(top_k=...)`` (the kwarg is ``k``), and
    ``sandbox.executor.SandboxExecutor`` (the class is ``CodeExecutor``).
    Every call raised AttributeError / TypeError / ImportError.  Asserting only
    that a schema has a handler could not catch that — the handler has to be
    *run*, and its output has to survive ``json.dumps`` to cross JSON-RPC.
    """

    def test_skill_search_runs(self) -> None:
        out = mcp_tools.call_tool("isaac_skill_search", {"query": "anything"})
        json.dumps(out)
        assert "skills" in out

    def test_memory_search_runs(self) -> None:
        out = mcp_tools.call_tool("isaac_memory_search", {"query": "anything"})
        json.dumps(out), "RecallResult must be flattened, not returned raw"
        assert "combined_context" in out["results"]

    def test_code_execute_imports_and_degrades_cleanly(self) -> None:
        # Without a Docker engine this must still return the error envelope
        # rather than raising ImportError on the executor class name.
        out = mcp_tools.call_tool("isaac_code_execute", {"code": "print(1)"})
        json.dumps(out)
        assert set(out) == {"exit_code", "stdout", "stderr", "duration_ms"}

    def test_knowledge_query_free_text_runs(self) -> None:
        out = mcp_tools.call_tool("isaac_knowledge_query", {"query": "anything"})
        json.dumps(out), "MCP responses must be JSON-serialisable"
        assert isinstance(out["results"], list)

    def test_knowledge_query_triples_runs(self) -> None:
        out = mcp_tools.call_tool(
            "isaac_knowledge_query", {"subject": "nobody", "predicate": "nothing"}
        )
        json.dumps(out)
        assert out["results"] == []

    def test_web_search_delegates_to_connector_run(self, monkeypatch: Any) -> None:
        from isaac.skills.connectors import web_search as ws

        monkeypatch.setattr(
            ws.WebSearchConnector,
            "run",
            lambda self, **kw: {"results": [{"title": kw["query"], "url": "u"}]},
        )
        out = mcp_tools.call_tool("isaac_web_search", {"query": "cats", "max_results": 3})
        json.dumps(out)
        assert out["results"][0]["title"] == "cats"

    def test_web_search_propagates_connector_error(self, monkeypatch: Any) -> None:
        from isaac.skills.connectors import web_search as ws

        monkeypatch.setattr(
            ws.WebSearchConnector, "run", lambda self, **kw: {"error": "boom", "results": []}
        )
        out = mcp_tools.call_tool("isaac_web_search", {"query": "cats"})
        assert out["error"] == "boom"
        assert out["results"] == []


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

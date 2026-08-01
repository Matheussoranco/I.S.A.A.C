"""MCP tool definitions — JSON schemas and handler dispatch for Isaac's MCP server."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (MCP format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "isaac_ask",
        "description": (
            "Send a task or question to the I.S.A.A.C. cognitive agent. "
            "Isaac will plan, synthesise code if needed, execute in a Docker sandbox, "
            "and return a structured result. Use for: coding tasks, research, analysis, "
            "file operations, web search, computer-use automation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task or question for Isaac."},
                "context": {
                    "type": "string",
                    "description": "Optional extra context (e.g. relevant code, file contents).",
                    "default": "",
                },
                "mode": {
                    "type": "string",
                    "enum": ["code", "ui", "hybrid", "auto"],
                    "description": "Execution mode. 'auto' lets Isaac decide.",
                    "default": "auto",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "isaac_agent",
        "description": (
            "Run I.S.A.A.C.'s autonomous tool-use agent on a task. The agent is given a "
            "real toolbox — a persistent web browser, web search, a Python runner, and a "
            "sandboxed file workspace — and iterates (call tool -> observe -> decide) until "
            "it produces a final answer. Use this for multi-step work that needs live web "
            "browsing, code execution, or file output rather than a single answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for the agent."},
                "context": {
                    "type": "string",
                    "description": "Optional extra context.",
                    "default": "",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Max tool-use rounds (default 12).",
                    "default": 12,
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of tool names to restrict the agent to "
                        "(e.g. ['browser','web_search']). Defaults to all built-in tools."
                    ),
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "isaac_memory_search",
        "description": "Search I.S.A.A.C.'s five-layer memory system for relevant past knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "layers": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "long_term", "procedural", "all"],
                    },
                    "description": "Memory layers to search.",
                    "default": ["all"],
                },
                "top_k": {"type": "integer", "description": "Max results per layer.", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "isaac_skill_search",
        "description": "Search I.S.A.A.C.'s versioned skill library for reusable code patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Description of the skill to find."},
                "top_k": {"type": "integer", "description": "Number of results.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "isaac_code_execute",
        "description": (
            "Execute Python code in I.S.A.A.C.'s secure Docker sandbox (no network, "
            "256 MB RAM, 30 s timeout, seccomp profile). Returns stdout, stderr, exit code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (max 120).",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "isaac_web_search",
        "description": "Perform a web search via DuckDuckGo and return structured results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "isaac_knowledge_query",
        "description": "Query I.S.A.A.C.'s semantic knowledge graph (NetworkX + SQLite).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Subject entity (optional)."},
                "predicate": {
                    "type": "string",
                    "description": "Relationship predicate (optional).",
                },
                "query": {"type": "string", "description": "Free-text query against the KG."},
            },
        },
    },
    {
        "name": "isaac_spawn_subagent",
        "description": (
            "Spawn a Claude sub-agent to handle a specific subtask in parallel. "
            "Returns the sub-agent result. Useful for decomposing complex tasks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": "The specific subtask for the sub-agent.",
                },
                "context": {
                    "type": "string",
                    "description": "Context to pass to the sub-agent.",
                    "default": "",
                },
                "role": {
                    "type": "string",
                    "enum": ["researcher", "coder", "analyst", "planner", "critic"],
                    "description": "Specialization role for the sub-agent.",
                    "default": "coder",
                },
                "agentic": {
                    "type": "boolean",
                    "description": (
                        "When true, the sub-agent runs as a full tool-use loop with "
                        "role-appropriate tools (web search, browser, code, files) instead "
                        "of a single LLM call."
                    ),
                    "default": False,
                },
            },
            "required": ["subtask"],
        },
    },
    {
        "name": "isaac_meta_stats",
        "description": (
            "Get I.S.A.A.C.'s self-improvement statistics: "
            "task success rates, strategy rankings, error patterns."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Filter by task type (optional)."},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------


def _handle_ask(args: dict[str, Any]) -> dict[str, Any]:
    """Run a full Isaac cognitive cycle and return the result."""
    import uuid

    from langchain_core.messages import HumanMessage

    from isaac.core.graph import build_graph
    from isaac.core.state import make_initial_state

    task = args["task"]
    context = args.get("context", "")
    mode = args.get("mode", "auto")

    full_task = task if not context else f"{task}\n\nContext:\n{context}"

    state = make_initial_state()
    state["session_id"] = f"mcp-{uuid.uuid4()}"
    state["messages"] = [HumanMessage(content=full_task)]

    if mode != "auto":
        state["task_mode"] = mode  # type: ignore[literal-required]

    compiled = build_graph()
    result: dict[str, Any] = {}
    for event in compiled.stream(dict(state)):
        for _node, node_output in event.items():
            if isinstance(node_output, dict):
                result.update(node_output)

    from langchain_core.messages import AIMessage

    response_text = ""
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            response_text = msg.content
            break

    logs = result.get("execution_logs", [])
    exec_summary = {}
    if logs:
        last = logs[-1]
        exec_summary = {
            "exit_code": last.exit_code,
            "stdout": last.stdout[:2000],
            "stderr": last.stderr[:500],
            "duration_ms": last.duration_ms,
        }

    return {
        "response": response_text,
        "phase": result.get("current_phase", ""),
        "mode": result.get("task_mode", "code"),
        "execution": exec_summary,
        "errors": [{"node": e.node, "message": e.message} for e in result.get("errors", [])],
    }


def _handle_memory_search(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.memory.manager import MemoryManager

    query = args["query"]
    top_k = args.get("top_k", 5)

    # recall() spells the limit ``k``, and returns a RecallResult dataclass
    # that has to be flattened before it can go back over JSON-RPC.
    mgr = MemoryManager()
    recalled = mgr.recall(query, k=top_k)
    return {"query": query, "results": asdict(recalled)}


def _handle_skill_search(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.config.settings import settings
    from isaac.memory.skill_library import SkillLibrary

    query = args["query"]
    top_k = args.get("top_k", 3)

    # SkillLibrary calls .mkdir() on this: it must stay a Path, not str().
    lib = SkillLibrary(Path(settings.skills_dir))
    skills = lib.search(query, top_k=top_k)
    return {"query": query, "skills": skills}


def _handle_code_execute(args: dict[str, Any]) -> dict[str, Any]:
    # The class is CodeExecutor and execute() is synchronous; the timeout lives
    # on the (frozen) SecurityPolicy rather than on the call.
    from isaac.sandbox.executor import CodeExecutor
    from isaac.sandbox.security import default_policy

    code = args["code"]
    timeout = min(args.get("timeout", 30), 120)

    try:
        executor = CodeExecutor(policy=replace(default_policy(), timeout_seconds=int(timeout)))
        result = executor.execute(code)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout[:3000],
            "stderr": result.stderr[:1000],
            "duration_ms": result.duration_ms,
        }
    except Exception as exc:
        return {"exit_code": -1, "stdout": "", "stderr": str(exc), "duration_ms": 0.0}


def _handle_web_search(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.skills.connectors.web_search import WebSearchConnector

    query = args["query"]
    max_results = args.get("max_results", 5)

    # BaseConnector exposes run(**kwargs); there is no .search().
    connector = WebSearchConnector()
    payload = connector.run(query=query, max_results=max_results)
    out: dict[str, Any] = {"query": query, "results": payload.get("results", [])}
    if payload.get("error"):
        out["error"] = payload["error"]
    return out


def _handle_knowledge_query(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.memory.semantic import SemanticMemory

    subject = args.get("subject", "")
    predicate = args.get("predicate", "")
    query = args.get("query", "")

    mem = SemanticMemory()
    # The real API is search_similar_facts / query_facts, and the latter reads
    # ``None`` — not "" — as "match any", so blank filters must be dropped or
    # every query matches nothing. Facts are dataclasses: serialise them, since
    # the MCP response has to be JSON.
    if query:
        facts = mem.search_similar_facts(query)
    else:
        facts = mem.query_facts(subject=subject or None, predicate=predicate or None)
    return {"results": [f.to_dict() for f in facts]}


def _handle_agent(args: dict[str, Any]) -> dict[str, Any]:
    """Run the autonomous tool-use agent loop and return a structured result."""
    from isaac.agents.agent_loop import build_default_agent

    task = args["task"]
    context = args.get("context", "")
    max_iterations = int(args.get("max_iterations", 12))
    only = args.get("tools") or None

    loop = build_default_agent(max_iterations=max_iterations, only=only)
    result = loop.run(task, context=context)
    return {
        "output": result.output,
        "success": result.success,
        "iterations": result.iterations,
        "stopped_reason": result.stopped_reason,
        "tool_calls": [
            {"name": c.name, "args": c.args, "success": c.success} for c in result.tool_calls
        ],
    }


def _handle_spawn_subagent(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.agents.claude_subagent import ClaudeSubAgent

    subtask = args["subtask"]
    context = args.get("context", "")
    role = args.get("role", "coder")
    agentic = bool(args.get("agentic", False))

    agent = ClaudeSubAgent(role=role)
    if agentic:
        return agent.run_agentic(subtask, context=context)
    return agent.run(subtask, context=context)


def _handle_meta_stats(args: dict[str, Any]) -> dict[str, Any]:
    from isaac.meta.learner import MetaLearner

    task_type = args.get("task_type", "")
    learner = MetaLearner()
    return learner.get_stats(task_type=task_type or None)


_HANDLERS = {
    "isaac_ask": _handle_ask,
    "isaac_agent": _handle_agent,
    "isaac_memory_search": _handle_memory_search,
    "isaac_skill_search": _handle_skill_search,
    "isaac_code_execute": _handle_code_execute,
    "isaac_web_search": _handle_web_search,
    "isaac_knowledge_query": _handle_knowledge_query,
    "isaac_spawn_subagent": _handle_spawn_subagent,
    "isaac_meta_stats": _handle_meta_stats,
}


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name and return a result dict."""
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name!r}")
    try:
        return handler(arguments)
    except Exception as exc:
        logger.exception("Tool %r failed: %s", name, exc)
        return {"error": str(exc)}

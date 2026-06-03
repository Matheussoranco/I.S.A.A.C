"""Claude Sub-Agent — spawn specialised Anthropic API agents for parallel subtasks.

Each sub-agent is a focused Claude instance with a role-specific system prompt.
Results are structured and can be aggregated by the ParallelSynthesis node or
called directly from MCP (isaac_spawn_subagent tool).

Roles
-----
researcher  — web search + knowledge synthesis
coder       — code generation + debugging
analyst     — data analysis + reasoning
planner     — task decomposition + dependency mapping
critic      — code review + error identification

Usage
-----
    agent = ClaudeSubAgent(role="researcher")
    result = agent.run("Find the 3 best Python libraries for PDF parsing", context="...")
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Literal

logger = logging.getLogger(__name__)

Role = Literal["researcher", "coder", "analyst", "planner", "critic"]

_ROLE_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a research specialist. Given a subtask, synthesise precise, "
        "evidence-based information. Prefer concise structured answers with sources where possible."
    ),
    "coder": (
        "You are a senior software engineer. Given a subtask, produce clean, correct, "
        "well-structured Python code. Include a brief explanation of your approach."
    ),
    "analyst": (
        "You are a data and logic analyst. Given a subtask, apply rigorous reasoning, "
        "identify patterns, and produce structured analysis with clear conclusions."
    ),
    "planner": (
        "You are a task decomposition expert. Given a complex goal, break it into "
        "discrete, ordered steps with explicit dependencies. Output as a structured plan."
    ),
    "critic": (
        "You are a code reviewer and error analyst. Given code or a result, identify "
        "bugs, edge cases, security issues, and improvements. Be specific and actionable."
    ),
}

# Tools each role may use when running in agentic (tool-use) mode.  A role with
# an empty list runs as a pure-reasoning sub-agent (no tools).
_ROLE_TOOLS: dict[str, list[str]] = {
    "researcher": ["web_search", "browser", "file_write"],
    "coder": ["code", "file_read", "file_write", "file_list"],
    "analyst": ["code", "web_search", "file_read"],
    "planner": [],
    "critic": ["file_read", "code"],
}


class ClaudeSubAgent:
    """A focused Claude API agent for a specific role."""

    def __init__(
        self, role: Role = "coder", model: str | None = None, tier: str = "strong"
    ) -> None:
        self.role = role
        self._model = model
        self.tier = tier

    def run(self, subtask: str, context: str = "", max_tokens: int = 2048) -> dict[str, Any]:
        """Execute the subtask with a single LLM call and return a structured result.

        Local-first: the chat model is resolved through
        :func:`isaac.llm.provider.get_llm`, which honours the configured local
        backend (Ollama / llama.cpp / OpenAI-compatible) before any cloud
        fallback.  There is **no** hard dependency on the Anthropic SDK — set
        ``ISAAC_LLM_PROVIDER=anthropic`` (with a key) to route to Claude.
        """
        start = time.monotonic()
        system_prompt = _ROLE_PROMPTS.get(self.role, _ROLE_PROMPTS["coder"])

        user_message = subtask
        if context:
            user_message = f"{subtask}\n\n<context>\n{context}\n</context>"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = self._resolve_chat_model()
            # Some providers reject bind kwargs; ignore if so.
            with contextlib.suppress(Exception):  # pragma: no cover
                llm = llm.bind(max_tokens=max_tokens)

            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            )
            content = _message_text(response)
            usage = getattr(response, "usage_metadata", None) or {}
            duration_ms = (time.monotonic() - start) * 1000

            return {
                "role": self.role,
                "subtask": subtask,
                "result": content,
                "model": (
                    getattr(llm, "model", None)
                    or getattr(llm, "model_name", None)
                    or self._model
                    or "local"
                ),
                "input_tokens": int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0,
                "output_tokens": (
                    int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0
                ),
                "duration_ms": round(duration_ms, 1),
                "success": True,
            }
        except Exception as exc:
            logger.exception("ClaudeSubAgent(%s) failed: %s", self.role, exc)
            return {
                "role": self.role,
                "subtask": subtask,
                "result": "",
                "error": str(exc),
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
                "success": False,
            }

    def run_agentic(
        self,
        subtask: str,
        context: str = "",
        max_iterations: int = 8,
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Execute the subtask as a full tool-use agent for this role.

        Unlike :meth:`run` (a single LLM call), this gives the sub-agent the
        role-appropriate tools (web search, browser, code, files) and lets it
        iterate — search, browse, run code, write files — until it finishes.
        This is what makes a "researcher" actually research and a "coder"
        actually run and verify code.

        Returns a structured result dict mirroring :meth:`run`.
        """
        start = time.monotonic()
        from isaac.agents.agent_loop import build_default_agent

        role_tools = tools if tools is not None else _ROLE_TOOLS.get(self.role, [])
        base_prompt = _ROLE_PROMPTS.get(self.role, _ROLE_PROMPTS["coder"])
        system_prompt = (
            f"{base_prompt}\n\nYou have real tools available; use them to gather "
            "evidence and take actions instead of guessing. When finished, reply "
            "with a clear, self-contained final answer."
        )
        try:
            llm = self._resolve_chat_model()
            agent = build_default_agent(
                llm=llm,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                only=role_tools or [],
            )
            result = agent.run(subtask, context=context)
            return {
                "role": self.role,
                "subtask": subtask,
                "result": result.output,
                "iterations": result.iterations,
                "tool_calls": [{"name": c.name, "success": c.success} for c in result.tool_calls],
                "stopped_reason": result.stopped_reason,
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
                "success": result.success,
            }
        except Exception as exc:
            logger.exception("ClaudeSubAgent(%s) agentic run failed: %s", self.role, exc)
            return {
                "role": self.role,
                "subtask": subtask,
                "result": "",
                "error": str(exc),
                "duration_ms": round((time.monotonic() - start) * 1000, 1),
                "success": False,
            }

    def _resolve_chat_model(self) -> Any:
        """Return a LangChain chat model (provider-agnostic, local-first)."""
        from isaac.llm.provider import get_llm

        return get_llm(self.tier)  # type: ignore[arg-type]


def _message_text(message: Any) -> str:
    """Extract plain text from an AIMessage whose content may be str or blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


class ParallelSubAgentPool:
    """Run multiple sub-agents concurrently using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run_all(self, tasks: list[dict[str, Any]], agentic: bool = False) -> list[dict[str, Any]]:
        """Execute a list of {subtask, role, context} dicts in parallel.

        Parameters
        ----------
        tasks:
            List of task dicts with keys: ``subtask`` (required), ``role`` (optional),
            ``context`` (optional).
        agentic:
            When True, each sub-agent runs as a full tool-use loop
            (:meth:`ClaudeSubAgent.run_agentic`) instead of a single LLM call.

        Returns
        -------
        list[dict]
            Results in the same order as ``tasks``.
        """
        import concurrent.futures

        def _run_one(task: dict[str, Any]) -> dict[str, Any]:
            agent = ClaudeSubAgent(role=task.get("role", "coder"))
            if agentic:
                return agent.run_agentic(task["subtask"], context=task.get("context", ""))
            return agent.run(task["subtask"], context=task.get("context", ""))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            return [f.result() for f in futures]

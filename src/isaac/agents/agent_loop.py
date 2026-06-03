"""AgentLoop — a real LLM-driven tool-use loop (the core agentic engine).

This is the capability that turns I.S.A.A.C. from a "plan → synthesise code →
run" pipeline into an autonomous agent in the mould of Claude Code / Claude for
Chrome: the model is given a set of tools with JSON-Schema signatures, it
decides which tool to call and with what arguments, the loop executes the tool
and feeds the observation back, and this repeats until the model produces a
final answer (or a budget is exhausted).

It is provider-agnostic: it relies on LangChain's ``bind_tools`` so it works
with Anthropic, OpenAI and tool-calling Ollama models alike.

Example
-------
    from isaac.agents.agent_loop import build_default_agent

    agent = build_default_agent()
    result = agent.run("Find the latest stable Python version and write it to version.txt")
    print(result.output)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from isaac.tools.base import IsaacTool

logger = logging.getLogger(__name__)

# How much of a tool's output to feed back to the model per call.
_MAX_TOOL_OUTPUT = 12_000

DEFAULT_SYSTEM_PROMPT = (
    "You are I.S.A.A.C., an autonomous problem-solving agent with access to real "
    "tools (a web browser, web search, a Python code runner, and a sandboxed file "
    "workspace).\n\n"
    "Operating rules:\n"
    "1. Break the task into steps and use tools to gather facts and take actions — "
    "do NOT guess or fabricate information you could verify with a tool.\n"
    "2. Call one or more tools per turn. After each observation, decide the next step.\n"
    "3. Prefer the browser for reading specific web pages and web_search for "
    "discovery. Use the code tool for computation, parsing, and file generation.\n"
    "4. When a tool returns an error, read it and adapt (fix arguments, try another "
    "approach) rather than repeating the same call.\n"
    "5. When the task is complete, stop calling tools and reply with a concise, "
    "well-structured final answer that summarises what you did and the result."
)


@dataclass
class ToolCallRecord:
    """One executed tool call and its outcome."""

    name: str
    args: dict[str, Any]
    output: str
    success: bool
    duration_ms: float


@dataclass
class AgentRunResult:
    """Result of an :class:`AgentLoop` run."""

    output: str
    iterations: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stopped_reason: str = "final"  # "final" | "max_iterations" | "error"
    messages: list[Any] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.stopped_reason == "final"


EventCallback = Callable[[str, dict[str, Any]], None]


class AgentLoop:
    """Run an iterative tool-use loop until the model produces a final answer.

    Parameters
    ----------
    tools:
        The tools the model may call.
    llm:
        A LangChain chat model.  If ``None``, the configured ``"strong"`` model
        is used.  Must support ``bind_tools`` for tool calling.
    system_prompt:
        Overrides the default agent system prompt.
    max_iterations:
        Hard cap on model↔tool round-trips (default 12).
    max_risk:
        Tools above this risk level are blocked unless ``auto_approve`` is set.
        Default 3 keeps risk-4/5 actions (send email, delete file, write
        calendar) out of autonomous runs.
    auto_approve:
        When True, run high-risk tools without human approval. Use with care.
    on_event:
        Optional callback ``(kind, data)`` for streaming progress to a UI.
        Kinds: ``iteration``, ``thought``, ``tool_call``, ``tool_result``,
        ``final``, ``error``.
    """

    def __init__(
        self,
        tools: list[IsaacTool],
        llm: Any | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 12,
        max_risk: int = 3,
        auto_approve: bool = False,
        on_event: EventCallback | None = None,
    ) -> None:
        self._tools: dict[str, IsaacTool] = {t.name: t for t in tools}
        self._llm = llm
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max(1, max_iterations)
        self.max_risk = max_risk
        self.auto_approve = auto_approve
        self._on_event = on_event

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit(self, kind: str, **data: Any) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, data)
        except Exception:  # pragma: no cover - UI callbacks must never break the loop
            logger.debug("on_event callback raised for kind=%s", kind, exc_info=True)

    def _resolve_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        from isaac.llm.provider import get_llm

        return get_llm("strong")

    def _bind_tools(self, llm: Any) -> Any:
        if not self._tools:
            return llm
        schemas = [t.to_function_schema() for t in self._tools.values()]
        try:
            return llm.bind_tools(schemas)
        except Exception as exc:
            logger.error("bind_tools failed (%s); running without tools.", exc)
            return llm

    async def _exec_tool(self, name: str, args: dict[str, Any]) -> ToolCallRecord:
        start = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            return ToolCallRecord(
                name, args, f"No such tool: '{name}'. Available: {list(self._tools)}", False, 0.0
            )

        blocked = (tool.requires_approval or tool.risk_level > self.max_risk) and not (
            self.auto_approve
        )
        if blocked:
            return ToolCallRecord(
                name,
                args,
                (
                    f"BLOCKED: tool '{name}' is risk level {tool.risk_level} and requires human "
                    "approval; it was not executed. Continue without it or report that approval "
                    "is needed."
                ),
                False,
                (time.monotonic() - start) * 1000,
            )

        try:
            result = await tool.execute(**args)
        except TypeError as exc:
            return ToolCallRecord(
                name,
                args,
                f"Invalid arguments for '{name}': {exc}",
                False,
                (time.monotonic() - start) * 1000,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Tool '%s' raised", name)
            return ToolCallRecord(
                name,
                args,
                f"Tool '{name}' raised: {exc}",
                False,
                (time.monotonic() - start) * 1000,
            )

        fallback = result.error or result.output or "(no output)"
        output = result.output if result.success else fallback
        return ToolCallRecord(
            name,
            args,
            output or "(empty result)",
            result.success,
            (time.monotonic() - start) * 1000,
        )

    async def _aclose_tools(self) -> None:
        for tool in self._tools.values():
            try:
                await tool.aclose()
            except Exception:  # pragma: no cover - best-effort teardown
                logger.debug("aclose failed for tool %s", tool.name, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def arun(self, task: str, context: str = "") -> AgentRunResult:
        """Run the loop asynchronously and return the result."""
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        llm = self._bind_tools(self._resolve_llm())
        user = task if not context else f"{task}\n\n<context>\n{context}\n</context>"
        messages: list[Any] = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=user),
        ]

        all_calls: list[ToolCallRecord] = []
        final_text = ""
        reason = "max_iterations"
        iterations = 0

        try:
            for i in range(self.max_iterations):
                iterations = i + 1
                self._emit("iteration", n=iterations)
                try:
                    ai = await asyncio.to_thread(llm.invoke, messages)
                except Exception as exc:
                    logger.exception("LLM invocation failed")
                    self._emit("error", message=str(exc))
                    final_text = f"Agent stopped: LLM call failed ({exc})."
                    reason = "error"
                    break

                messages.append(ai)
                tool_calls = list(getattr(ai, "tool_calls", None) or [])
                text = _content_text(ai)

                if not tool_calls:
                    final_text = text
                    reason = "final"
                    self._emit("final", text=final_text)
                    break

                if text:
                    self._emit("thought", text=text)

                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args") or {}
                    call_id = tc.get("id") or name
                    self._emit("tool_call", name=name, args=args)
                    rec = await self._exec_tool(name, args)
                    all_calls.append(rec)
                    self._emit("tool_result", name=name, success=rec.success, output=rec.output)
                    messages.append(
                        ToolMessage(
                            content=rec.output[:_MAX_TOOL_OUTPUT],
                            tool_call_id=call_id,
                            name=name,
                        )
                    )
            else:
                self._emit("error", message="reached max iterations")

            if reason == "max_iterations" and not final_text:
                final_text = (
                    "Reached the iteration limit before finishing. Partial progress was made; "
                    f"{len(all_calls)} tool call(s) were executed."
                )
        finally:
            await self._aclose_tools()

        return AgentRunResult(
            output=final_text,
            iterations=iterations,
            tool_calls=all_calls,
            stopped_reason=reason,
            messages=messages,
        )

    def run(self, task: str, context: str = "") -> AgentRunResult:
        """Run the loop synchronously (safe to call from non-async code)."""
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if not running:
            return asyncio.run(self.arun(task, context))

        # Already inside an event loop — run in a worker thread with its own loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self.arun(task, context))).result()


def _content_text(message: Any) -> str:
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


def build_default_agent(
    *,
    llm: Any | None = None,
    system_prompt: str | None = None,
    max_iterations: int = 12,
    max_risk: int = 3,
    auto_approve: bool = False,
    on_event: EventCallback | None = None,
    only: list[str] | None = None,
) -> AgentLoop:
    """Construct an :class:`AgentLoop` wired with all registered built-in tools.

    Parameters
    ----------
    only:
        If given, restrict the agent to tools whose names are in this list.
    """
    from isaac.tools import register_all_tools

    registry = register_all_tools()
    tools = registry.list_all()
    if only is not None:
        wanted = set(only)
        tools = [t for t in tools if t.name in wanted]
    return AgentLoop(
        tools,
        llm=llm,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        max_risk=max_risk,
        auto_approve=auto_approve,
        on_event=on_event,
    )

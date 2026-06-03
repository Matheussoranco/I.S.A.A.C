"""Specialist — a domain-focused agent built on the :class:`AgentLoop`.

A *Specialist* couples three things into one callable unit:

1. an **identity** — a title, domain, and role-specific system prompt;
2. a **curated toolset** — tool names resolved from the global
   :class:`~isaac.tools.base.ToolRegistry` (``None`` = every tool, ``[]`` = a
   pure-reasoning agent with none, a list = exactly those);
3. a **risk policy** — the maximum tool risk level it may invoke and whether
   high-risk actions are auto-approved.

Running a Specialist drives a full tool-use loop
(:class:`isaac.agents.agent_loop.AgentLoop`), so a *coder* actually writes and
runs code, a *researcher* actually searches and browses, and a *file
organizer* actually moves files.

Specialists are **local-first**: the chat model is resolved through
:func:`isaac.llm.provider.get_llm`, which honours the configured local backend
(Ollama / llama.cpp / OpenAI-compatible) before any cloud fallback.  They are
also **persona-aware**: each prompt is prefixed with the active I.S.A.A.C.
soul/persona so the whole team speaks with one voice.

Subclassing
-----------
Concrete specialists set class attributes and (optionally) override
:meth:`role_instructions`::

    class CoderSpecialist(Specialist):
        name = "coder"
        title = "Software Engineer"
        domain = "writing, running, and debugging code"
        description = "Builds and fixes software; runs and verifies code."
        tools = ["code", "fs_read", "fs_write", "fs_list", "shell"]
        tier = "strong"
        max_risk = 3
        role_prompt = "You are a meticulous senior software engineer. ..."

Anything settable as a class attribute can also be overridden per-instance via
the constructor, e.g. ``CoderSpecialist(max_risk=4, auto_approve=True)``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class SpecialistResult:
    """Structured outcome of a :meth:`Specialist.run`."""

    specialist: str
    task: str
    output: str
    success: bool
    iterations: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "final"
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist": self.specialist,
            "task": self.task,
            "output": self.output,
            "success": self.success,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "stopped_reason": self.stopped_reason,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


_OPERATING_RULES = (
    "\n\nOperating rules:\n"
    "1. Break the task into steps and use your tools to gather facts and take "
    "actions — never fabricate something you could verify with a tool.\n"
    "2. After each tool result, decide the next step; if a tool errors, read it "
    "and adapt rather than repeating the same call.\n"
    "3. Stay within your remit and your allowed tools. If the task needs a "
    "capability you lack, say so clearly in your final answer.\n"
    "4. When finished, stop calling tools and reply with a concise, "
    "self-contained final answer describing what you did and the result."
)


class Specialist:
    """A domain-focused, tool-using agent. Subclass and set class attributes."""

    # ── Identity (override in subclasses) ────────────────────────────────
    name: str = "specialist"
    title: str = "Generalist"
    domain: str = "general problem solving"
    description: str = "A general-purpose problem solver."

    # ── Capability policy ────────────────────────────────────────────────
    #: Tool names to expose. ``None`` = all registered tools; ``[]`` = none.
    tools: list[str] | None = None
    role_prompt: str = ""
    tier: str = "strong"
    default_max_iterations: int = 12
    max_risk: int = 3
    auto_approve: bool = False

    def __init__(
        self,
        *,
        llm: Any | None = None,
        tools: list[str] | None = None,
        tier: str | None = None,
        max_iterations: int | None = None,
        max_risk: int | None = None,
        auto_approve: bool | None = None,
        persona: str | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self._llm = llm
        # ``tools`` left as the sentinel default means "use the class attribute".
        self._tools_override = tools
        self._tools_overridden = tools is not None
        self.tier = tier or self.tier
        self.max_iterations = max_iterations or self.default_max_iterations
        self.max_risk = max_risk if max_risk is not None else self.max_risk
        self.auto_approve = auto_approve if auto_approve is not None else self.auto_approve
        self._persona = persona
        self._on_event = on_event

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str] | None:
        """Resolved tool restriction (``None`` = all tools)."""
        return self._tools_override if self._tools_overridden else self.tools

    def role_instructions(self) -> str:
        """Return the role-specific instruction block.

        Override for dynamic prompts; the default returns :attr:`role_prompt`.
        """
        return self.role_prompt or self.description

    def system_prompt(self) -> str:
        """Compose the full system prompt: persona + role + operating rules."""
        preamble = self._persona if self._persona is not None else _soul_preamble()
        identity = (
            f"You are acting as the **{self.title}** on the I.S.A.A.C. team — "
            f"your specialism is {self.domain}."
        )
        return "\n\n".join(
            part for part in (preamble, identity, self.role_instructions()) if part
        ) + _OPERATING_RULES

    def card(self) -> dict[str, Any]:
        """Return a routing-friendly description card."""
        return {
            "name": self.name,
            "title": self.title,
            "domain": self.domain,
            "description": self.description,
            "tools": self.tool_names,
            "max_risk": self.max_risk,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _resolve_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        from isaac.llm.provider import get_llm

        return get_llm(self.tier)  # type: ignore[arg-type]

    def _build_loop(self) -> Any:
        from isaac.agents.agent_loop import build_default_agent

        return build_default_agent(
            llm=self._resolve_llm(),
            system_prompt=self.system_prompt(),
            max_iterations=self.max_iterations,
            max_risk=self.max_risk,
            auto_approve=self.auto_approve,
            on_event=self._on_event,
            only=self.tool_names,
        )

    def run(self, task: str, context: str = "") -> SpecialistResult:
        """Run the specialist synchronously on *task* and return a result."""
        start = time.monotonic()
        try:
            loop = self._build_loop()
            result = loop.run(task, context=context)
            return SpecialistResult(
                specialist=self.name,
                task=task,
                output=result.output,
                success=result.success,
                iterations=result.iterations,
                tool_calls=[
                    {"name": c.name, "success": c.success} for c in result.tool_calls
                ],
                stopped_reason=result.stopped_reason,
                duration_ms=round((time.monotonic() - start) * 1000, 1),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Specialist(%s) failed", self.name)
            return SpecialistResult(
                specialist=self.name,
                task=task,
                output="",
                success=False,
                stopped_reason="error",
                duration_ms=round((time.monotonic() - start) * 1000, 1),
                error=str(exc),
            )

    async def arun(self, task: str, context: str = "") -> SpecialistResult:
        """Async variant of :meth:`run`."""
        start = time.monotonic()
        try:
            loop = self._build_loop()
            result = await loop.arun(task, context=context)
            return SpecialistResult(
                specialist=self.name,
                task=task,
                output=result.output,
                success=result.success,
                iterations=result.iterations,
                tool_calls=[
                    {"name": c.name, "success": c.success} for c in result.tool_calls
                ],
                stopped_reason=result.stopped_reason,
                duration_ms=round((time.monotonic() - start) * 1000, 1),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Specialist(%s) failed", self.name)
            return SpecialistResult(
                specialist=self.name,
                task=task,
                output="",
                success=False,
                stopped_reason="error",
                duration_ms=round((time.monotonic() - start) * 1000, 1),
                error=str(exc),
            )


def _soul_preamble() -> str:
    """Return the active soul/persona system-prompt preamble (best-effort)."""
    try:
        from isaac.identity.soul import soul_system_prompt

        return soul_system_prompt()
    except Exception:  # pragma: no cover - identity layer is optional
        return ""

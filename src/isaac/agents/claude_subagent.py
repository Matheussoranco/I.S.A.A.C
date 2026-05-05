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


class ClaudeSubAgent:
    """A focused Claude API agent for a specific role."""

    def __init__(self, role: Role = "coder", model: str | None = None) -> None:
        self.role = role
        self._model = model

    def _get_client_and_model(self) -> tuple[Any, str]:
        try:
            import anthropic
            from isaac.config.settings import settings
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
            model = self._model or "claude-sonnet-4-6"
            return client, model
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    def run(self, subtask: str, context: str = "", max_tokens: int = 2048) -> dict[str, Any]:
        """Execute the subtask and return a structured result dict."""
        start = time.monotonic()
        system_prompt = _ROLE_PROMPTS.get(self.role, _ROLE_PROMPTS["coder"])

        user_message = subtask
        if context:
            user_message = f"{subtask}\n\n<context>\n{context}\n</context>"

        try:
            client, model = self._get_client_and_model()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            content = response.content[0].text if response.content else ""
            duration_ms = (time.monotonic() - start) * 1000

            return {
                "role": self.role,
                "subtask": subtask,
                "result": content,
                "model": model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
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


class ParallelSubAgentPool:
    """Run multiple sub-agents concurrently using ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run_all(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute a list of {subtask, role, context} dicts in parallel.

        Parameters
        ----------
        tasks:
            List of task dicts with keys: ``subtask`` (required), ``role`` (optional),
            ``context`` (optional).

        Returns
        -------
        list[dict]
            Results in the same order as ``tasks``.
        """
        import concurrent.futures

        def _run_one(task: dict[str, Any]) -> dict[str, Any]:
            agent = ClaudeSubAgent(role=task.get("role", "coder"))
            return agent.run(task["subtask"], context=task.get("context", ""))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            return [f.result() for f in futures]

"""Refinement Loop — iterative improvement via Reflection ↔ Synthesis feedback.

When Reflection detects a failure it can invoke a tighter inner loop
that re-synthesises code with corrective hints from the error diagnosis,
re-executes, and re-reflects — all within a single graph step.

This avoids the overhead of a full Planner re-plan for simple bugs
(off-by-one, missing import, wrong variable name, etc.) and only
escalates back to the Planner when the refinement budget is exhausted.

Integration
-----------
The Reflection node calls ``attempt_refinement(state, diagnosis)``
before falling back to the Planner retry edge.  If refinement succeeds,
the reflection returns a success update directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from isaac.core.state import (
    ExecutionResult,
    IsaacState,
    PlanStep,
    SkillCandidate,
)

logger = logging.getLogger(__name__)

# Maximum inner-loop iterations before giving up and handing to Planner.
_MAX_REFINEMENT_ATTEMPTS = 3


def attempt_refinement(
    state: IsaacState,
    diagnosis: str,
    *,
    max_attempts: int = _MAX_REFINEMENT_ATTEMPTS,
) -> dict[str, Any] | None:
    """Try to fix the code via a tight Synthesis → Sandbox inner loop.

    Parameters
    ----------
    state:
        Current graph state.
    diagnosis:
        Error diagnosis from the Reflection node.
    max_attempts:
        Number of refinement attempts before giving up.

    Returns
    -------
    dict or None
        A partial state update if refinement succeeded, ``None`` if it
        failed and the caller should fall through to the Planner retry.
    """
    try:
        from isaac.llm.router import get_router, TaskComplexity
        from langchain_core.messages import SystemMessage, HumanMessage

        router = get_router()
    except ImportError:
        logger.debug("Refinement loop: LLM router not available.")
        return None

    code = state.get("code_buffer", "")
    plan: list[PlanStep] = state.get("plan", [])
    active_step = _get_active_step(plan)
    step_desc = active_step.description if active_step else "unknown"

    for attempt in range(1, max_attempts + 1):
        logger.info("Refinement attempt %d/%d for step '%s'.", attempt, max_attempts, step_desc)

        # ── Re-synthesise with corrective hints ──────────────────────
        llm = router.route(TaskComplexity.MODERATE)
        messages = [
            SystemMessage(content=_REFINEMENT_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"## Step\n{step_desc}\n\n"
                f"## Previous Code\n```python\n{code}\n```\n\n"
                f"## Error Diagnosis\n{diagnosis}\n\n"
                f"## Instruction\nFix the code. Output ONLY the corrected Python code."
            )),
        ]

        try:
            response = llm.invoke(messages)
            new_code = _extract_code(response.content)
        except Exception as exc:
            logger.error("Refinement synthesis failed: %s", exc)
            continue

        if not new_code:
            continue

        # ── Re-execute in sandbox ────────────────────────────────────
        exec_result = _run_in_sandbox(new_code)

        if exec_result.exit_code == 0:
            # Success — build update dict
            logger.info("Refinement succeeded on attempt %d.", attempt)
            if active_step:
                active_step.status = "done"

            return {
                "code_buffer": new_code,
                "execution_logs": [exec_result],
                "plan": plan,
                "skill_candidate": SkillCandidate(
                    name=f"refined_{step_desc[:30]}",
                    code=new_code,
                    task_context=step_desc,
                    success_count=1,
                ),
                "current_phase": "reflection",
            }

        # Update diagnosis for next attempt
        diagnosis = (
            f"Exit code: {exec_result.exit_code}\n"
            f"Stderr: {exec_result.stderr[:500]}\n"
            f"Stdout: {exec_result.stdout[:300]}"
        )
        code = new_code

    logger.info("Refinement exhausted (%d attempts) — escalating to Planner.", max_attempts)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REFINEMENT_SYSTEM_PROMPT = (
    "You are an expert debugger. Given previous Python code and an error "
    "diagnosis, fix the bugs and produce corrected code. Output ONLY the "
    "complete corrected Python code — no markdown fences, no explanation. "
    "Keep changes minimal."
)


def _extract_code(content: Any) -> str:
    """Extract Python code from LLM output, stripping fences."""
    if isinstance(content, list):
        content = "\n".join(str(c) for c in content)
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step.status == "active":
            return step
    return None


def _run_in_sandbox(code: str) -> ExecutionResult:
    """Execute code in the Docker sandbox and return the result."""
    try:
        from isaac.sandbox.executor import CodeExecutor

        executor = CodeExecutor()
        try:
            result = executor.execute(code)
        finally:
            executor.close()
        return result
    except Exception as exc:
        logger.error("Refinement sandbox error: %s", exc)
        return ExecutionResult(
            stdout="",
            stderr=str(exc),
            exit_code=1,
        )

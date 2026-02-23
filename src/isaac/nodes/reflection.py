"""Reflection / Critic Node — analyses execution logs and decides next action.

On **failure**: updates ``hypothesis``, appends to ``errors``, routes back
to the Planner.

On **success**: marks the active step as ``"done"``, populates
``skill_candidate``, routes to Skill Abstraction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from isaac.core.state import (
    ErrorEntry,
    ExecutionResult,
    IsaacState,
    PlanStep,
    SkillCandidate,
)
from isaac.llm.prompts import reflection_prompt

logger = logging.getLogger(__name__)


def _latest_log(state: IsaacState) -> ExecutionResult:
    """Return the most recent execution log, or a sentinel."""
    logs = state.get("execution_logs", [])
    if logs:
        return logs[-1]
    return ExecutionResult(stdout="", stderr="No execution logs.", exit_code=-1)


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step.status == "active":
            return step
    return None


def reflection_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Reflection / Critic.

    Calls the LLM to analyse the latest sandbox output and determine
    success or failure.
    """
    from isaac.llm.provider import get_llm  # noqa: PLC0415

    llm = get_llm()

    log = _latest_log(state)
    code = state.get("code_buffer", "")
    plan: list[PlanStep] = state.get("plan", [])
    errors = list(state.get("errors", []))
    active_step = _get_active_step(plan)
    step_desc = active_step.description if active_step else "unknown"

    prompt = reflection_prompt(
        code=code,
        stdout=log.stdout,
        stderr=log.stderr,
        exit_code=log.exit_code,
        step_description=step_desc,
    )
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse structured JSON
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.error("Reflection: failed to parse LLM JSON — treating as failure.")
        parsed = {
            "success": False,
            "diagnosis": f"Unparseable reflection output: {content[:300]}",
            "revised_hypothesis": state.get("hypothesis", ""),
        }

    updates: dict[str, Any] = {"current_phase": "reflection"}

    if parsed.get("success", False):
        # Mark active step done
        if active_step:
            active_step.status = "done"
        updates["plan"] = plan

        # Build skill candidate
        candidate_info = parsed.get("skill_candidate", {})
        updates["skill_candidate"] = SkillCandidate(
            name=candidate_info.get("name", "unnamed_skill"),
            code=code,
            task_context=step_desc,
            success_count=1,
        )

        logger.info("Reflection: step SUCCEEDED — skill candidate proposed.")
    else:
        # Failure path
        attempt = len([e for e in errors if e.node == "reflection"]) + 1
        new_error = ErrorEntry(
            node="reflection",
            message=parsed.get("diagnosis", "Unknown failure"),
            traceback=log.stderr[:2000] if log.stderr else None,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            attempt=attempt,
        )
        updates["errors"] = [new_error]
        updates["hypothesis"] = parsed.get(
            "revised_hypothesis", state.get("hypothesis", "")
        )

        if active_step:
            active_step.status = "failed"
        updates["plan"] = plan

        logger.info(
            "Reflection: step FAILED (attempt %d) — hypothesis revised.",
            attempt,
        )

    return updates

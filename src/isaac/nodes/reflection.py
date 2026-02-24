"""Reflection / Critic Node — analyses execution logs and decides next action.

On **failure**: updates ``hypothesis``, appends to ``errors``, routes back
to the Planner.

On **success**: marks the active step as ``"done"``, populates
``skill_candidate``, routes to Skill Abstraction.

For Computer-Use / UI tasks the node performs a *visual diff*: it compares
the before/after screenshots of the last ``UIActionResult`` to decide whether
the GUI action actually achieved its goal.
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
    UIActionResult,
)
from isaac.llm.prompts import reflection_prompt, reflection_ui_prompt
from isaac.memory.episodic import Episode, get_episodic_memory

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


def _parse_reflection_json(content: str, fallback_hypothesis: str) -> dict:
    """Parse LLM JSON with graceful fallback."""
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.error("Reflection: failed to parse LLM JSON — treating as failure.")
        return {
            "success": False,
            "diagnosis": f"Unparseable reflection output: {content[:300]}",
            "revised_hypothesis": fallback_hypothesis,
        }


# ---------------------------------------------------------------------------
# Visual-diff path (Computer-Use / UI tasks)
# ---------------------------------------------------------------------------

def _reflect_ui(
    state: IsaacState,
    llm: Any,
    plan: list[PlanStep],
    active_step: PlanStep | None,
    errors: list[ErrorEntry],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Reflection path for `task_mode == "computer_use"`.

    Compares the before/after screenshots of the last UIActionResult to
    determine whether the GUI action succeeded.
    """
    ui_results: list[UIActionResult] = state.get("ui_results", [])
    step_desc = active_step.description if active_step else "unknown"

    if not ui_results:
        logger.warning("Reflection (UI): no ui_results found — treating as failure.")
        parsed = {
            "success": False,
            "diagnosis": "No UI action results available for visual diff.",
            "revised_hypothesis": state.get("hypothesis", ""),
        }
    else:
        latest: UIActionResult = ui_results[-1]
        prompt = reflection_ui_prompt(
            step_description=step_desc,
            action=latest.action,
            screenshot_before_b64=latest.screenshot_before_b64 or "",
            screenshot_after_b64=latest.screenshot_after_b64 or "",
            error=latest.error,
        )
        response = llm.invoke(prompt)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        parsed = _parse_reflection_json(content, state.get("hypothesis", ""))

    episodic = get_episodic_memory()

    if parsed.get("success", False):
        if active_step:
            active_step.status = "done"
        updates["plan"] = plan

        candidate_info = parsed.get("skill_candidate", {})
        updates["skill_candidate"] = SkillCandidate(
            name=candidate_info.get("name", "ui_skill"),
            code=state.get("code_buffer", ""),   # JSON-encoded UIAction trace
            task_context=step_desc,
            success_count=1,
            skill_type="ui",
            tags=candidate_info.get("tags", ["ui"]),
        )
        episodic.record(Episode(
            task=step_desc,
            hypothesis=state.get("hypothesis", ""),
            code=state.get("code_buffer", ""),
            result_summary=parsed.get("summary", "UI step succeeded."),
            success=True,
            node="reflection_ui",
            iteration=state.get("iteration", 0),
        ))
        logger.info("Reflection (UI): step SUCCEEDED — UI skill candidate proposed.")
    else:
        attempt = len([e for e in errors if e.node == "reflection"]) + 1
        corrective = parsed.get("corrective_action", "")
        message = parsed.get("diagnosis", "Visual diff indicated failure.")
        if corrective:
            message = f"{message}  Corrective hint: {corrective}"
        new_error = ErrorEntry(
            node="reflection",
            message=message,
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
        episodic.record(Episode(
            task=step_desc,
            hypothesis=state.get("hypothesis", ""),
            code=state.get("code_buffer", ""),
            result_summary=message,
            success=False,
            node="reflection_ui",
            iteration=state.get("iteration", 0),
        ))
        logger.info(
            "Reflection (UI): step FAILED (attempt %d) — hypothesis revised.",
            attempt,
        )

    updates["current_phase"] = "reflection"
    return updates


# ---------------------------------------------------------------------------
# Code-execution path (default)
# ---------------------------------------------------------------------------

def _reflect_code(
    state: IsaacState,
    llm: Any,
    plan: list[PlanStep],
    active_step: PlanStep | None,
    errors: list[ErrorEntry],
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Reflection path for standard code-execution tasks."""
    log = _latest_log(state)
    code = state.get("code_buffer", "")
    step_desc = active_step.description if active_step else "unknown"

    prompt = reflection_prompt(
        code=code,
        stdout=log.stdout,
        stderr=log.stderr,
        exit_code=log.exit_code,
        step_description=step_desc,
    )
    response = llm.invoke(prompt)
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    parsed = _parse_reflection_json(content, state.get("hypothesis", ""))

    episodic = get_episodic_memory()

    if parsed.get("success", False):
        if active_step:
            active_step.status = "done"
        updates["plan"] = plan

        candidate_info = parsed.get("skill_candidate", {})
        updates["skill_candidate"] = SkillCandidate(
            name=candidate_info.get("name", "unnamed_skill"),
            code=code,
            task_context=step_desc,
            success_count=1,
        )
        episodic.record(Episode(
            task=step_desc,
            hypothesis=state.get("hypothesis", ""),
            code=code,
            result_summary=parsed.get("summary", f"exit={log.exit_code}"),
            success=True,
            node="reflection",
            iteration=state.get("iteration", 0),
        ))
        logger.info("Reflection: step SUCCEEDED — skill candidate proposed.")
    else:
        attempt = len([e for e in errors if e.node == "reflection"]) + 1
        diagnosis = parsed.get("diagnosis", "Unknown failure")

        # ── Refinement loop: try tight Synthesis→Sandbox fix first ──
        try:
            from isaac.nodes.refinement import attempt_refinement

            refined = attempt_refinement(state, diagnosis)
            if refined is not None:
                # Refinement succeeded — record episode and return
                episodic.record(Episode(
                    task=step_desc,
                    hypothesis=state.get("hypothesis", ""),
                    code=refined.get("code_buffer", ""),
                    result_summary="Refined successfully after inner loop.",
                    success=True,
                    node="reflection_refinement",
                    iteration=state.get("iteration", 0),
                ))
                logger.info("Reflection: refinement loop SUCCEEDED — skipping Planner re-plan.")
                refined["current_phase"] = "reflection"
                return refined
        except Exception as exc:
            logger.debug("Refinement loop unavailable: %s", exc)

        # ── Fallback: escalate to Planner ──
        new_error = ErrorEntry(
            node="reflection",
            message=diagnosis,
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
        episodic.record(Episode(
            task=step_desc,
            hypothesis=state.get("hypothesis", ""),
            code=code,
            result_summary=f"FAILED: {diagnosis}",
            success=False,
            node="reflection",
            iteration=state.get("iteration", 0),
        ))
        logger.info(
            "Reflection: step FAILED (attempt %d) — hypothesis revised.",
            attempt,
        )

    updates["current_phase"] = "reflection"
    return updates


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def reflection_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Reflection / Critic.

    Dispatches to the visual-diff path for Computer-Use tasks, or the
    standard code-analysis path otherwise.
    """
    from isaac.llm.provider import get_llm

    llm = get_llm("strong")
    plan: list[PlanStep] = state.get("plan", [])
    errors: list[ErrorEntry] = list(state.get("errors", []))
    active_step = _get_active_step(plan)
    updates: dict[str, Any] = {"current_phase": "reflection"}

    task_mode: str = state.get("task_mode", "code")
    if task_mode == "computer_use":
        return _reflect_ui(state, llm, plan, active_step, errors, updates)

    return _reflect_code(state, llm, plan, active_step, errors, updates)

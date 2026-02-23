"""Skill Abstraction Node — generalises successful code into reusable Programs.

Takes the ``skill_candidate`` produced by the Reflection node, asks the LLM
to parameterise and generalise it, then commits it to the Skill Library.

For UI / Computer-Use tasks the node converts a raw ``UIAction`` trace into a
Playwright function that can be replayed and adapted for similar GUI tasks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from isaac.core.state import IsaacState, PlanStep, SkillCandidate, UIAction
from isaac.llm.prompts import skill_abstraction_prompt, skill_abstraction_ui_prompt

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _has_pending_steps(plan: list[PlanStep]) -> bool:
    return any(s.status == "pending" for s in plan)


def _advance_plan(plan: list[PlanStep]) -> None:
    """Activate the first pending step in-place."""
    for step in plan:
        if step.status == "pending":
            step.status = "active"
            return


# ---------------------------------------------------------------------------
# UI / Playwright abstraction
# ---------------------------------------------------------------------------

def _abstract_ui_skill(
    candidate: SkillCandidate,
    state: IsaacState,
    llm: Any,
) -> str:
    """Convert a UIAction trace into a generalised Playwright function.

    ``candidate.code`` is expected to be a JSON string produced by
    ``computer_use_node``::

        {
            "actions": [<UIAction dicts>, ...],
            "screenshot_before": "<base64>",
            "screenshot_after": "<base64>"
        }

    Falls back gracefully if the JSON is malformed.
    """
    raw_code = candidate.code or "{}"
    try:
        payload = json.loads(raw_code)
    except json.JSONDecodeError:
        logger.warning("Skill Abstraction (UI): code_buffer is not valid JSON — using raw string.")
        payload = {}

    action_trace_raw: list[dict] = payload.get("actions", [])
    screenshot_before_b64: str = payload.get("screenshot_before", "")
    screenshot_after_b64: str = payload.get("screenshot_after", "")

    # Convert raw dicts to UIAction objects so the prompt helper can access attributes
    action_trace: list[UIAction] = [
        UIAction(
            type=a.get("type", "screenshot"),
            x=a.get("x"),
            y=a.get("y"),
            text=a.get("text"),
            key=a.get("key"),
            description=a.get("description", ""),
        )
        if isinstance(a, dict) else a
        for a in action_trace_raw
    ]

    # Also try to pull screenshots from code_buffer if not in candidate.code
    if not screenshot_before_b64 or not screenshot_after_b64:
        buf = state.get("code_buffer", "")
        try:
            buf_payload = json.loads(buf)
            screenshot_before_b64 = (
                screenshot_before_b64 or buf_payload.get("screenshot_before", "")
            )
            screenshot_after_b64 = (
                screenshot_after_b64 or buf_payload.get("screenshot_after", "")
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    prompt = skill_abstraction_ui_prompt(
        action_trace=action_trace,
        task_context=candidate.task_context,
        screenshot_before_b64=screenshot_before_b64,
        screenshot_after_b64=screenshot_after_b64,
    )
    response = llm.invoke(prompt)
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )
    return _extract_code(content)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def skill_abstraction_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Skill Abstraction.

    Generalises the ``skill_candidate`` into a library-worthy function and
    commits it.  Then routes to Planner if steps remain, or END otherwise.

    Dispatches to the Playwright-based UI abstraction path when
    ``candidate.skill_type == "ui"``.
    """
    from isaac.config.settings import settings
    from isaac.llm.provider import get_llm
    from isaac.memory.skill_library import SkillLibrary

    llm = get_llm()
    skill_lib = SkillLibrary(settings.skills_dir)

    candidate: SkillCandidate | None = state.get("skill_candidate")
    plan: list[PlanStep] = state.get("plan", [])

    if candidate is None or not candidate.code:
        logger.warning("Skill Abstraction: no valid candidate — skipping.")
        if _has_pending_steps(plan):
            _advance_plan(plan)
        return {"plan": plan, "skill_candidate": None, "current_phase": "skill_abstraction"}

    skill_type = getattr(candidate, "skill_type", "code")

    if skill_type == "ui":
        # ── UI / Playwright path ─────────────────────────────────────────
        generalised_code = _abstract_ui_skill(candidate, state, llm)
        candidate.code = generalised_code
        candidate.success_count += 1
        # Ensure UI tags propagate
        existing_tags: list[str] = list(getattr(candidate, "tags", []) or [])
        for tag in ("ui", "playwright"):
            if tag not in existing_tags:
                existing_tags.append(tag)
        candidate.tags = existing_tags
        skill_lib.commit(candidate)
        logger.info(
            "Skill Abstraction (UI): committed Playwright skill '%s' to library.",
            candidate.name,
        )
    else:
        # ── Code / default path ──────────────────────────────────────────
        prompt = skill_abstraction_prompt(
            concrete_code=candidate.code,
            task_context=candidate.task_context,
        )
        response = llm.invoke(prompt)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        generalised_code = _extract_code(content)
        candidate.code = generalised_code
        candidate.success_count += 1
        skill_lib.commit(candidate)
        logger.info(
            "Skill Abstraction: committed skill '%s' to library.",
            candidate.name,
        )

    # Activate next pending step if any remain
    if _has_pending_steps(plan):
        _advance_plan(plan)

    return {
        "plan": plan,
        "skill_candidate": None,  # Clear the slot
        "current_phase": "skill_abstraction",
    }

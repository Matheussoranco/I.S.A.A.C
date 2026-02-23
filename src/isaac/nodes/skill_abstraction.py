"""Skill Abstraction Node — generalises successful code into reusable Programs.

Takes the ``skill_candidate`` produced by the Reflection node, asks the LLM
to parameterise and generalise it, then commits it to the Skill Library.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from isaac.core.state import IsaacState, PlanStep, SkillCandidate
from isaac.llm.prompts import skill_abstraction_prompt

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _has_pending_steps(plan: list[PlanStep]) -> bool:
    return any(s.status == "pending" for s in plan)


def skill_abstraction_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Skill Abstraction.

    Generalises the ``skill_candidate`` into a library-worthy function and
    commits it.  Then routes to Planner if steps remain, or END otherwise.
    """
    from isaac.config.settings import settings  # noqa: PLC0415
    from isaac.llm.provider import get_llm  # noqa: PLC0415
    from isaac.memory.skill_library import SkillLibrary  # noqa: PLC0415

    llm = get_llm()
    skill_lib = SkillLibrary(settings.skills_dir)

    candidate: SkillCandidate | None = state.get("skill_candidate")
    plan: list[PlanStep] = state.get("plan", [])

    if candidate is None or not candidate.code:
        logger.warning("Skill Abstraction: no valid candidate — skipping.")
        return {"skill_candidate": None, "current_phase": "skill_abstraction"}

    # Ask LLM to generalise
    prompt = skill_abstraction_prompt(
        concrete_code=candidate.code,
        task_context=candidate.task_context,
    )
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    generalised_code = _extract_code(content)

    # Update candidate with generalised code
    candidate.code = generalised_code
    candidate.success_count += 1

    # Commit to library
    skill_lib.commit(candidate)

    logger.info("Skill Abstraction: committed skill '%s' to library.", candidate.name)

    # Activate next pending step if any remain
    if _has_pending_steps(plan):
        for step in plan:
            if step.status == "pending":
                step.status = "active"
                break

    return {
        "plan": plan,
        "skill_candidate": None,  # Clear the slot
        "current_phase": "skill_abstraction",
    }

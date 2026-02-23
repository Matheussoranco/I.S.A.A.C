"""Synthesis Node — generates executable Python code from the active plan step.

Uses the CodeAgent paradigm: the LLM produces pure Python (no JSON tool
calls) that will be injected into the Docker sandbox.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from isaac.core.state import IsaacState, PlanStep, WorldModel
from isaac.llm.prompts import synthesis_prompt

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Extract the first Python code block from a fenced markdown response."""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: treat the entire response as code if no fence found
    return text.strip()


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    """Return the first step with ``status == 'active'``."""
    for step in plan:
        if step.status == "active":
            return step
    return None


def synthesis_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Synthesis.

    Reads the active ``PlanStep``, world model, and hypothesis, then asks
    the LLM to produce a self-contained Python script.  The result is placed
    in ``code_buffer``.
    """
    from isaac.config.settings import settings  # noqa: PLC0415
    from isaac.llm.provider import get_llm  # noqa: PLC0415
    from isaac.memory.skill_library import SkillLibrary  # noqa: PLC0415

    llm = get_llm()
    skill_lib = SkillLibrary(settings.skills_dir)

    plan: list[PlanStep] = state.get("plan", [])
    world_model: WorldModel = state.get("world_model", WorldModel())
    hypothesis: str = state.get("hypothesis", "")

    active_step = _get_active_step(plan)
    if active_step is None:
        logger.warning("Synthesis: no active step found in plan.")
        return {
            "code_buffer": "# No active step — nothing to execute.\nprint('NOOP')",
            "current_phase": "synthesis",
        }

    available_skills = skill_lib.list_names()

    prompt = synthesis_prompt(active_step, world_model, hypothesis, available_skills)
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    code = _extract_code(content)

    logger.info(
        "Synthesis: generated %d chars of code for step '%s'.",
        len(code),
        active_step.id,
    )

    return {
        "code_buffer": code,
        "current_phase": "synthesis",
    }

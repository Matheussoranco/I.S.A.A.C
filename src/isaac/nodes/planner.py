"""Planner Node — decomposes the task into a Graph-of-Thought plan.

Reads the current ``world_model``, ``hypothesis``, past ``errors``, and
the Skill Library to produce an ordered list of ``PlanStep`` objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from isaac.core.state import IsaacState, PlanStep, WorldModel
from isaac.llm.prompts import planner_prompt

logger = logging.getLogger(__name__)


def planner_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Planner.

    Generates or refines a multi-step plan.  Increments ``iteration`` on
    every invocation to prevent infinite loops.
    """
    from isaac.config.settings import settings
    from isaac.llm.provider import get_llm
    from isaac.memory.episodic import get_episodic_memory
    from isaac.memory.skill_library import SkillLibrary

    llm = get_llm("fast")
    skill_lib = SkillLibrary(settings.skills_dir)
    episodic = get_episodic_memory()

    world_model: WorldModel = state.get("world_model", WorldModel())
    hypothesis: str = state.get("hypothesis", "")
    errors = state.get("errors", [])
    iteration: int = state.get("iteration", 0) + 1

    available_skills = skill_lib.list_names()
    episodic_context = episodic.summarise_recent(5)

    # Call LLM
    prompt = planner_prompt(
        world_model, hypothesis, errors, available_skills, episodic_context,
    )
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse steps
    steps: list[PlanStep] = []
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        raw_steps = parsed.get("steps", [])
        for raw in raw_steps:
            steps.append(
                PlanStep(
                    id=raw["id"],
                    description=raw["description"],
                    mode=raw.get("mode", "code"),
                    status="pending",
                    depends_on=raw.get("depends_on", []),
                )
            )
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Planner: failed to parse LLM plan: %s", exc)
        # Fallback: single generic step
        task_mode = state.get("task_mode", "code")
        fallback_mode = "ui" if task_mode == "computer_use" else "code"
        steps = [
            PlanStep(
                id="s1",
                description=f"Execute hypothesis directly: {hypothesis[:200]}",
                mode=fallback_mode,
                status="pending",
            )
        ]

    # Mark the first dependency-satisfied pending step as active
    for step in steps:
        if step.status == "pending":
            # Check if all dependencies are satisfied (done)
            deps_ok = all(
                any(s.id == dep and s.status == "done" for s in steps)
                for dep in step.depends_on
            ) if step.depends_on else True
            if deps_ok:
                step.status = "active"
                break

    logger.info("Planner: %d steps generated, iteration=%d", len(steps), iteration)

    return {
        "plan": steps,
        "iteration": iteration,
        "current_phase": "planner",
    }

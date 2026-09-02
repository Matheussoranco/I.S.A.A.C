"""Planner Node — decomposes the task into a Graph-of-Thought plan.

Reads the current ``world_model``, ``hypothesis``, past ``errors``, and
the Skill Library to produce an ordered list of ``PlanStep`` objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

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

    # Sync WorldModel into the knowledge graph so Planner/Reflection can query it
    try:
        from isaac.memory.manager import get_memory_manager

        get_memory_manager().sync_kg_from_world_model(world_model)
    except Exception:
        logger.debug("Planner: KG sync failed — continuing.", exc_info=True)

    # Preserve steps already marked 'done' from previous rounds — the
    # LLM only needs to (re-)plan the remaining work.
    existing_plan: list[PlanStep] = state.get("plan", [])
    completed_steps: list[PlanStep] = [s for s in existing_plan if s.status == "done"]
    completed_descriptions = [s.description for s in completed_steps]

    available_skills = skill_lib.list_names()
    session_id: str = state.get("session_id", "")
    episodic_context = episodic.summarise_recent(5, session_id=session_id)

    # Call LLM
    prompt = planner_prompt(
        world_model,
        hypothesis,
        errors,
        available_skills,
        episodic_context,
        completed_descriptions=completed_descriptions,
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
            raw_mode: object = raw.get("mode", "code")
            mode: Literal["code", "ui", "hybrid"]
            if raw_mode in ("code", "ui", "hybrid"):
                mode = cast(Literal["code", "ui", "hybrid"], raw_mode)
            else:
                mode = "code"
            steps.append(
                PlanStep(
                    id=raw["id"],
                    description=raw["description"],
                    mode=mode,
                    status="pending",
                    depends_on=raw.get("depends_on", []),
                )
            )
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Planner: failed to parse LLM plan: %s", exc)
        # Fallback: single generic step
        task_mode = state.get("task_mode", "code")
        fallback_mode: Literal["code", "ui"] = "ui" if task_mode == "computer_use" else "code"
        steps = [
            PlanStep(
                id="s1",
                description=f"Execute hypothesis directly: {hypothesis[:200]}",
                mode=fallback_mode,
                status="pending",
            )
        ]

    # Merge completed (preserved) steps with newly generated pending steps.
    # completed_steps keep their 'done' status; new steps start as 'pending'.
    all_steps = completed_steps + steps

    # Activate the complete ready frontier only when parallel execution is
    # enabled.  The sequential path must consume one active step at a time.
    from isaac.nodes.got_planner import PlanDAG

    dag = PlanDAG(steps=all_steps)
    activation_limit = None if settings.parallel_synthesis_enabled else 1
    activated = dag.activate_ready(limit=activation_limit)

    # Expose DAG context string in the state for Synthesis/Reflection to use
    dag_context = dag.to_context_string()
    critical_path = dag.critical_path()

    logger.info(
        "Planner: %d total steps (%d preserved done, %d new), "
        "%d activated now, critical_path=%s, iteration=%d",
        len(all_steps),
        len(completed_steps),
        len(steps),
        len(activated),
        critical_path,
        iteration,
    )

    return {
        "plan": all_steps,
        "iteration": iteration,
        "current_phase": "planner",
        "dag_context": dag_context,
    }

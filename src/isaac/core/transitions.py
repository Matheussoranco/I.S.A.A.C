"""Conditional edge functions — routing logic between graph nodes.

Each function inspects the current ``IsaacState`` and returns the name of
the next node to execute.  These are registered as conditional edges in
:func:`isaac.core.graph.build_graph`.
"""

from __future__ import annotations

import logging

from isaac.core.state import IsaacState, PlanStep

logger = logging.getLogger(__name__)

# Sentinel node names (must match keys used in graph.py)
NODE_PLANNER = "planner"
NODE_SKILL_ABSTRACTION = "skill_abstraction"
END = "__end__"


def _count_reflection_errors(state: IsaacState) -> int:
    """Total number of errors recorded by the Reflection node."""
    return len([e for e in state.get("errors", []) if e.node == "reflection"])


def _has_pending_steps(plan: list[PlanStep]) -> bool:
    return any(s.status == "pending" for s in plan)


# ---------------------------------------------------------------------------
# Edge: Reflection → {Skill Abstraction | Planner | END}
# ---------------------------------------------------------------------------


def after_reflection(state: IsaacState) -> str:
    """Decide what happens after the Reflection node.

    * **Success** (``skill_candidate`` is set) → ``skill_abstraction``
    * **Failure below retry cap** → ``planner`` (re-plan with revised hypothesis)
    * **Failure at/above retry cap** → ``__end__`` (terminate with error report)
    """
    from isaac.config.settings import settings  # noqa: PLC0415

    max_retries = settings.graph.max_retries
    max_iterations = settings.graph.max_iterations
    iteration = state.get("iteration", 0)

    # Hard iteration cap (safety net)
    if iteration >= max_iterations:
        logger.warning("Transition: iteration cap (%d) reached — terminating.", max_iterations)
        return END

    candidate = state.get("skill_candidate")
    if candidate is not None and candidate.code:
        logger.info("Transition: Reflection → Skill Abstraction (success).")
        return NODE_SKILL_ABSTRACTION

    n_errors = _count_reflection_errors(state)
    if n_errors >= max_retries:
        logger.warning(
            "Transition: %d errors >= max_retries (%d) — terminating.",
            n_errors,
            max_retries,
        )
        return END

    logger.info("Transition: Reflection → Planner (retry, errors=%d).", n_errors)
    return NODE_PLANNER


# ---------------------------------------------------------------------------
# Edge: Skill Abstraction → {Planner | END}
# ---------------------------------------------------------------------------


def after_skill_abstraction(state: IsaacState) -> str:
    """Decide what happens after Skill Abstraction.

    * **More pending steps** → ``planner``
    * **All steps done** → ``__end__``
    """
    from isaac.config.settings import settings  # noqa: PLC0415

    max_iterations = settings.graph.max_iterations
    iteration = state.get("iteration", 0)

    if iteration >= max_iterations:
        logger.warning("Transition: iteration cap reached after skill abstraction — terminating.")
        return END

    plan = state.get("plan", [])
    if _has_pending_steps(plan):
        logger.info("Transition: Skill Abstraction → Planner (more steps pending).")
        return NODE_PLANNER

    logger.info("Transition: Skill Abstraction → END (plan complete).")
    return END

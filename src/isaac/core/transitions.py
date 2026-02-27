"""Conditional edge functions — routing logic between graph nodes.

Each function inspects the current ``IsaacState`` and returns the name of
the next node to execute.  These are registered as conditional edges in
:func:`isaac.core.graph.build_graph`.

Computer-Use addition
---------------------
* ``after_synthesis`` — routes to ``computer_use`` (UI mode) or ``sandbox`` (code/hybrid).
"""

from __future__ import annotations

import logging

from isaac.core.state import IsaacState, PlanStep

logger = logging.getLogger(__name__)

# Sentinel node names (must match keys used in graph.py)
NODE_PLANNER = "planner"
NODE_SKILL_ABSTRACTION = "skill_abstraction"
NODE_COMPUTER_USE = "computer_use"
NODE_SANDBOX = "sandbox"
NODE_DIRECT_RESPONSE = "direct_response"
NODE_EXPLORER = "explorer"
NODE_PERCEPTION = "perception"
END = "__end__"


def _count_reflection_errors(state: IsaacState) -> int:
    """Total number of errors recorded by the Reflection node."""
    return len([e for e in state.get("errors", []) if e.node == "reflection"])


def _count_computer_use_errors(state: IsaacState) -> int:
    """Total number of errors recorded by the ComputerUse node."""
    return len([e for e in state.get("errors", []) if e.node == "computer_use"])


def _has_pending_steps(plan: list[PlanStep]) -> bool:
    return any(s.status == "pending" for s in plan)


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step.status == "active":
            return step
    return None


# ---------------------------------------------------------------------------
# Edge: Guard → {Perception | END}
# ---------------------------------------------------------------------------


def after_guard(state: IsaacState) -> str:
    """Route after the Guard node.

    * ``guard_blocked=True`` → ``__end__``  (input flagged as injection attack)
    * otherwise              → ``perception`` (normal path)
    """
    if state.get("guard_blocked", False):
        logger.warning("Transition: Guard → END (prompt injection detected — input blocked).")
        return END
    logger.info("Transition: Guard → Perception.")
    return NODE_PERCEPTION


# ---------------------------------------------------------------------------
# Edge: Perception → {DirectResponse | Explorer}
# ---------------------------------------------------------------------------


def after_perception(state: IsaacState) -> str:
    """Route after Perception based on task_mode.

    * ``task_mode == 'direct'`` → ``direct_response`` (fast single-call path)
    * anything else             → ``explorer`` (full pipeline)
    """
    task_mode = state.get("task_mode", "code")
    if task_mode == "direct":
        logger.info("Transition: Perception → DirectResponse (task_mode=direct).")
        return NODE_DIRECT_RESPONSE
    logger.info("Transition: Perception → Explorer (task_mode=%s).", task_mode)
    return NODE_EXPLORER


# ---------------------------------------------------------------------------
# Edge: Synthesis → {ComputerUse | Sandbox}
# ---------------------------------------------------------------------------


def after_synthesis(state: IsaacState) -> str:
    """Route after Synthesis based on the active step's execution mode.

    * ``mode == 'ui'`` → ``computer_use``  (xdotool loop)
    * ``mode == 'hybrid'`` → ``sandbox``    (Playwright script in UI container)
    * ``mode == 'code'`` → ``sandbox``     (pure Python in code container)
    """
    plan: list[PlanStep] = state.get("plan", [])
    active = _get_active_step(plan)
    if active and active.mode == "ui":
        logger.info("Transition: Synthesis → ComputerUse (mode=ui).")
        return NODE_COMPUTER_USE
    logger.info("Transition: Synthesis → Sandbox (mode=%s).", active.mode if active else "none")
    return NODE_SANDBOX


# ---------------------------------------------------------------------------
# Edge: Reflection → {Skill Abstraction | Planner | END}
# ---------------------------------------------------------------------------


def after_reflection(state: IsaacState) -> str:
    """Decide what happens after the Reflection node.

    * **Success** (``skill_candidate`` is set) → ``skill_abstraction``
    * **Failure below retry cap** → ``planner`` (re-plan with revised hypothesis)
    * **Failure at/above retry cap** → ``__end__`` (terminate with error report)
    """
    from isaac.config.settings import settings

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

    n_errors = _count_reflection_errors(state) + _count_computer_use_errors(state)
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
    from isaac.config.settings import settings

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


# ---------------------------------------------------------------------------

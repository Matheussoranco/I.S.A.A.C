"""Meta-Learner Node — post-task self-improvement recording and adaptive feedback.

Runs after SkillAbstraction (or after Reflection on terminal success/failure).
Records the task outcome to MetaLearner and optionally emits a reflection summary
that is injected into the next planning cycle.

What it does
------------
1. Records outcome (success/failure, strategy, duration, errors) → SQLite.
2. Queries best strategies for this task type → updates world_model.resources.
3. Detects recurring failure patterns → emits targeted plan amendment.
4. On Nth session, triggers failure analysis and writes a self-critique note.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from isaac.core.state import IsaacState

logger = logging.getLogger(__name__)


def meta_learner_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: record outcome + update adaptive strategy ranking."""
    from isaac.meta.learner import get_learner

    learner = get_learner()

    # Determine success from current state
    errors = state.get("errors", [])
    logs = state.get("execution_logs", [])
    last_log = logs[-1] if logs else None
    success = (last_log is not None and last_log.exit_code == 0) if last_log else (len(errors) == 0)

    # Extract strategy from world_model resources
    wm = state.get("world_model")
    resources = wm.resources if wm else {}
    strategy = resources.get("_last_strategy", "unknown")
    task_type = resources.get("_task_type", "general")
    session_id = state.get("session_id", "")

    # Collect error info
    error_type = ""
    error_msg = ""
    if errors:
        last_err = errors[-1]
        error_type = last_err.node
        error_msg = last_err.message

    # Duration from last execution log
    duration_ms = last_log.duration_ms if last_log else 0.0

    # Extract token usage if available
    input_tokens = int(resources.get("_input_tokens", 0))
    output_tokens = int(resources.get("_output_tokens", 0))

    # Get user task from messages
    messages = state.get("messages", [])
    task_desc = ""
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "human":
            task_desc = str(msg.content)[:300]
            break

    learner.record(
        task_desc=task_desc,
        task_type=task_type,
        strategy=strategy,
        success=success,
        error_type=error_type,
        error_msg=error_msg,
        iterations=state.get("iteration", 0),
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        session_id=session_id,
    )

    # Query best strategies and surface into world_model for next planning cycle
    best = learner.get_best_strategy(task_type)
    if best and wm:
        wm.resources["_recommended_strategies"] = [s["strategy"] for s in best[:3]]
        wm.resources["_strategy_win_rates"] = {s["strategy"]: s["win_rate"] for s in best[:5]}

    logger.debug(
        "MetaLearner: task_type=%s strategy=%s success=%s",
        task_type, strategy, success,
    )

    return {
        "world_model": wm,
        "current_phase": "meta_learning",
    }

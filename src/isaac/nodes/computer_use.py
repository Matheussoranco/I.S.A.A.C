"""ComputerUse Node — screenshot → vision LLM → UIAction execution loop.

This node drives the virtual-desktop automation cycle:

1. Acquire a screenshot from the UI sandbox container.
2. Ask the vision LLM whether the active ``PlanStep`` is complete.
3. If not complete, execute the suggested ``UIAction`` via ``UIExecutor``.
4. Record the ``UIActionResult`` (before/after screenshots).
5. Repeat until the LLM declares success OR ``max_ui_cycles`` is reached.

The container lifecycle is managed externally via the ``_UI_EXECUTOR``
module-level singleton (started lazily, kept alive across cycles, torn down
by the ``build_and_run`` REPL on exit).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from isaac.core.state import (
    GUIState,
    IsaacState,
    PlanStep,
    UIAction,
    UIActionResult,
    WorldModel,
)
from isaac.llm.prompts import computer_use_prompt

logger = logging.getLogger(__name__)

# Module-level UI executor singleton (lazy init)
_ui_executor: Any = None


def _get_ui_executor() -> Any:
    """Return the module-level ``UIExecutor``, starting it if needed."""
    global _ui_executor
    if _ui_executor is None:
        from isaac.sandbox.ui_executor import UIExecutor

        _ui_executor = UIExecutor()
        _ui_executor.start()
        logger.info("ComputerUse: UIExecutor started.")
    return _ui_executor


def shutdown_ui_executor() -> None:
    """Public hook — call from REPL teardown to stop the UI container."""
    global _ui_executor
    if _ui_executor is not None:
        _ui_executor.stop()
        _ui_executor = None
        logger.info("ComputerUse: UIExecutor stopped.")


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    for step in plan:
        if step.status == "active":
            return step
    return None


def _parse_llm_decision(content: str) -> dict[str, Any]:
    """Parse the vision LLM's done/action JSON response."""
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        logger.warning("ComputerUse: could not parse LLM JSON: %s", content[:300])
        return {"done": False, "action": {"type": "screenshot"}}


def _dict_to_ui_action(d: dict[str, Any]) -> UIAction:
    """Convert a JSON dict from the LLM into a ``UIAction`` dataclass."""
    return UIAction(
        type=d.get("type", "screenshot"),
        x=d.get("x"),
        y=d.get("y"),
        target_x=d.get("target_x"),
        target_y=d.get("target_y"),
        text=d.get("text"),
        key=d.get("key"),
        scroll_direction=d.get("scroll_direction"),
        scroll_amount=int(d.get("scroll_amount", 3)),
        duration_ms=int(d.get("duration_ms", 0)),
        description=d.get("description", ""),
    )


def computer_use_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: ComputerUse.

    Drives the screenshot → vision LLM → UIAction → execute loop for the
    active ``PlanStep`` with ``mode == 'ui'``.

    On success:
        - Marks the active step as ``'done'``
        - Sets ``skill_candidate`` with the complete UIAction trace

    On max-cycle exhaustion:
        - Marks the step as ``'failed'``
        - Appends an ``ErrorEntry``
    """
    from datetime import datetime, timezone

    from isaac.config.settings import settings
    from isaac.core.state import ErrorEntry, SkillCandidate
    from isaac.llm.provider import get_llm

    llm = get_llm()
    max_cycles: int = settings.graph.max_ui_cycles
    plan: list[PlanStep] = state.get("plan", [])
    world_model: WorldModel = state.get("world_model", WorldModel())
    current_ui_cycle: int = state.get("ui_cycle", 0)
    accumulated_results: list[UIActionResult] = list(state.get("ui_results", []))

    active_step = _get_active_step(plan)
    if active_step is None:
        logger.warning("ComputerUse: no active step found — skipping.")
        return {"current_phase": "computer_use"}

    executor = _get_ui_executor()

    # -- Screenshot → LLM → Action loop ------------------------------------
    new_results: list[UIActionResult] = []
    step_done = False
    done_summary = ""

    while current_ui_cycle < max_cycles:
        current_ui_cycle += 1

        # Capture current state
        gui_state: GUIState = executor.get_gui_state()
        screenshot_b64 = gui_state.screenshot_b64

        if not screenshot_b64:
            logger.error("ComputerUse: empty screenshot — aborting cycle.")
            break

        # Ask vision LLM: done or next action?
        pending = state.get("ui_actions", [])
        prompt = computer_use_prompt(
            step_description=active_step.description,
            pending_actions=pending,
            screenshot_b64=screenshot_b64,
            gui_state=gui_state,
            ui_cycle=current_ui_cycle,
        )
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        decision = _parse_llm_decision(content)

        if decision.get("done", False):
            step_done = True
            done_summary = decision.get("summary", "Step completed.")
            logger.info(
                "ComputerUse: step '%s' done after %d cycles. %s",
                active_step.id,
                current_ui_cycle,
                done_summary,
            )
            break

        # Execute the suggested action
        action_dict = decision.get("action", {"type": "screenshot"})
        action = _dict_to_ui_action(action_dict)
        result = executor.act(action)
        new_results.append(result)

        # Refresh GUI state in world model
        updated_gui = executor.get_gui_state()
        world_model.gui_state = updated_gui
        if updated_gui.current_url:
            world_model.last_url = updated_gui.current_url

        logger.debug(
            "ComputerUse: cycle %d — action=%s success=%s",
            current_ui_cycle,
            action.type,
            result.success,
        )

    # -- Build output -------------------------------------------------------
    updates: dict[str, Any] = {
        "world_model": world_model,
        "ui_cycle": current_ui_cycle,
        "ui_results": new_results,
        "current_phase": "computer_use",
    }

    all_results = accumulated_results + new_results

    if step_done:
        active_step.status = "done"
        updates["plan"] = plan

        # Build a UIAction trace for SkillAbstraction
        trace = [r.action for r in all_results]
        first_b64 = all_results[0].screenshot_before_b64 if all_results else ""
        last_b64 = all_results[-1].screenshot_after_b64 if all_results else ""

        updates["skill_candidate"] = SkillCandidate(
            name=f"ui_{active_step.id}_{active_step.description[:30].replace(' ', '_').lower()}",
            code=json.dumps(
                [
                    {
                        "type": a.type, "x": a.x, "y": a.y,
                        "text": a.text, "key": a.key,
                        "description": a.description,
                    }
                    for a in trace
                ],
                indent=2,
            ),
            task_context=active_step.description,
            skill_type="ui",
            tags=["ui", "xdotool", active_step.description[:20]],
            success_count=1,
        )
        # Stash before/after for SkillAbstraction to use
        updates["code_buffer"] = json.dumps(
            {"screenshot_before": first_b64, "screenshot_after": last_b64}
        )
    else:
        # Exhausted cycles without success
        active_step.status = "failed"
        updates["plan"] = plan
        updates["errors"] = [
            ErrorEntry(
                node="computer_use",
                message=(
                    f"Max UI cycles ({max_cycles}) reached without completing "
                    f"step '{active_step.id}': {active_step.description}"
                ),
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                attempt=1,
            )
        ]

    return updates

"""Synthesis Node — generates executable actions from the active plan step.

Branches on ``PlanStep.mode``:

* ``"code"``    → existing CodeAgent behaviour (pure Python for Sandbox).
* ``"ui"``      → emits a ``UIAction`` list (JSON) into ``state["ui_actions"]``.
* ``"hybrid"``  → emits a Playwright Python script into ``code_buffer``
                  (executed in the virtual-desktop container via ``UIExecutor``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from isaac.core.state import GUIState, IsaacState, PlanStep, UIAction, WorldModel
from isaac.llm.prompts import (
    synthesis_hybrid_prompt,
    synthesis_prompt,
    synthesis_ui_prompt,
)

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Extract the first Python code block from a fenced markdown response."""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_json_array(text: str) -> list[dict]:
    """Extract the first JSON array from a response string."""
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        # wrapped in {"actions": [...]}
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
    except (json.JSONDecodeError, IndexError):
        pass
    logger.warning("Synthesis(ui): could not parse UIAction array from response.")
    return []


def _get_active_step(plan: list[PlanStep]) -> PlanStep | None:
    """Return the first step with ``status == 'active'``."""
    for step in plan:
        if step.status == "active":
            return step
    return None


def synthesis_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Synthesis.

    Reads the active ``PlanStep``, determines execution mode, and produces
    either a code string (``code_buffer``) or a ``UIAction`` list
    (``ui_actions``) depending on ``PlanStep.mode``.
    """
    from isaac.config.settings import settings
    from isaac.llm.provider import get_llm
    from isaac.memory.skill_library import SkillLibrary

    llm = get_llm()
    skill_lib = SkillLibrary(settings.skills_dir)

    plan: list[PlanStep] = state.get("plan", [])
    world_model: WorldModel = state.get("world_model", WorldModel())
    hypothesis: str = state.get("hypothesis", "")
    available_skills = skill_lib.list_names()

    active_step = _get_active_step(plan)
    if active_step is None:
        logger.warning("Synthesis: no active step found in plan.")
        return {
            "code_buffer": "# No active step — nothing to execute.\nprint('NOOP')",
            "current_phase": "synthesis",
        }

    mode = active_step.mode

    # ── 'code' mode — existing CodeAgent behaviour ─────────────────────────
    if mode == "code":
        prompt = synthesis_prompt(active_step, world_model, hypothesis, available_skills)
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        code = _extract_code(content)
        logger.info(
            "Synthesis(code): generated %d chars for step '%s'.",
            len(code), active_step.id,
        )
        return {"code_buffer": code, "current_phase": "synthesis"}

    # ── 'ui' mode — emit UIAction JSON list ────────────────────────────────
    if mode == "ui":
        gui_state: GUIState = world_model.gui_state or GUIState()
        screenshot_b64 = gui_state.screenshot_b64

        if not screenshot_b64:
            # No screenshot yet — add a "take screenshot first" action
            logger.info("Synthesis(ui): no screenshot available; queuing screenshot action.")
            return {
                "ui_actions": [UIAction(type="screenshot", description="Capture initial screen")],
                "current_phase": "synthesis",
            }

        prompt = synthesis_ui_prompt(active_step, gui_state, screenshot_b64)
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        raw_actions = _extract_json_array(content)

        ui_actions = [
            UIAction(
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
            for d in raw_actions
        ]
        logger.info(
            "Synthesis(ui): generated %d UIActions for step '%s'.",
            len(ui_actions), active_step.id,
        )
        return {"ui_actions": ui_actions, "current_phase": "synthesis"}

    # ── 'hybrid' mode — emit Playwright Python script ──────────────────────
    # mode == "hybrid"
    gui_state = world_model.gui_state or GUIState()
    screenshot_b64 = gui_state.screenshot_b64

    prompt = synthesis_hybrid_prompt(
        active_step, gui_state, screenshot_b64 or "", available_skills
    )
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    code = _extract_code(content)

    logger.info(
        "Synthesis(hybrid): generated %d chars of Playwright script for step '%s'.",
        len(code), active_step.id,
    )
    return {"code_buffer": code, "current_phase": "synthesis"}

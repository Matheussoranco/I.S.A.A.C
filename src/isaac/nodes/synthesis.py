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


def _arc_synthesis(
    state: IsaacState,
    llm: Any,
    world_model: WorldModel,
    hypothesis: str,
) -> dict[str, Any] | None:
    """Try to solve an ARC task using the full neuro-symbolic synthesis engine.

    Returns a state update dict if successful, or None to fall through to
    the generic synthesis path.
    """
    resources = world_model.resources
    if not resources.get("_arc_task"):
        return None

    train_pairs_raw = resources.get("train", [])
    if not train_pairs_raw:
        return None

    try:
        import numpy as np
        from isaac.arc.evaluator import ArcTask, ArcPair
        from isaac.arc.solver import synthesise

        train_pairs = [
            ArcPair(
                input=np.array(p["input"], dtype=int),
                output=np.array(p.get("output", []), dtype=int),
            )
            for p in train_pairs_raw
            if p.get("output")
        ]
        test_pairs_raw = resources.get("test", [])
        test_pairs = [
            ArcPair(
                input=np.array(p["input"], dtype=int),
                output=np.array(p.get("output", p["input"]), dtype=int),
            )
            for p in test_pairs_raw
        ]

        task = ArcTask(
            id=resources.get("task_id", "arc_task"),
            train=train_pairs,
            test=test_pairs,
            description=hypothesis,
        )

        # Use remaining time from world_model constraints or defaults
        result = synthesise(
            task,
            llm=llm,
            time_budget_s=25.0,
            beam_width=30,
            max_depth=3,
            max_refine_iterations=4,
        )

        if isinstance(result.program, str) and result.program not in ("unsolved", ""):
            code = result.program
        elif isinstance(result.program, list) and result.program:
            # DSL program — wrap in executable Python
            from isaac.arc.dsl import apply_program
            import json as _json
            ops_json = _json.dumps(result.program)
            code = (
                f"import numpy as np\n"
                f"import sys, json\n"
                f"sys.path.insert(0, '/app/src')\n"
                f"from isaac.arc.dsl import apply_program\n\n"
                f"test_inputs = {[p.input.tolist() for p in test_pairs]!r}\n"
                f"ops = {ops_json}\n"
                f"for i, inp in enumerate(test_inputs):\n"
                f"    grid = np.array(inp, dtype=int)\n"
                f"    out = apply_program(ops, grid)\n"
                f"    print(f'Test {{i+1}} output:')\n"
                f"    print(out.tolist())\n"
            )
        else:
            code = (
                "import sys\n"
                "print('ARC solver could not find a solution within compute budget.')\n"
                "sys.exit(1)\n"
            )

        train_acc = result.method
        logger.info(
            "ARC synthesis: method=%s, code_len=%d chars",
            result.method, len(code),
        )
        return {"code_buffer": code, "current_phase": "synthesis"}

    except Exception as exc:
        logger.warning("ARC synthesis path failed: %s — falling back to generic.", exc)
        return None


def synthesis_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Synthesis.

    Reads the active ``PlanStep``, determines execution mode, and produces
    either a code string (``code_buffer``) or a ``UIAction`` list
    (``ui_actions``) depending on ``PlanStep.mode``.

    ARC-aware: when ``world_model.resources["_arc_task"]`` is set, routes
    through the full neuro-symbolic synthesis engine (analogy + beam search +
    object-level synthesis + LLM + self-refinement) instead of generic code
    synthesis.
    """
    from isaac.config.settings import settings
    from isaac.llm.provider import get_llm
    from isaac.memory.skill_library import SkillLibrary

    llm = get_llm("strong")
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

    # ── ARC fast-path: full neurosymbolic synthesis engine ─────────────────
    if mode == "code":
        arc_result = _arc_synthesis(state, llm, world_model, hypothesis)
        if arc_result is not None:
            return arc_result

    # ── 'code' mode — generic CodeAgent behaviour ──────────────────────────
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

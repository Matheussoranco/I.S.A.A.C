"""Perception Node — parses user input and syncs the world model.

FAST PATH: ``fast_classify()`` is called first (zero LLM cost).  When it
returns confidence ≥ 0.85 the full LLM roundtrip is skipped entirely —
typical conversational messages are classified in < 1 ms.

FALL-THROUGH: ambiguous inputs still go to the LLM, but the call is capped
at 200 tokens (``max_tokens=200`` / ``num_predict=200``) so the model cannot
generate a novel that we discard anyway.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from isaac.core.state import GUIState, IsaacState, WorldModel
from isaac.llm.prompts import perception_multimodal_prompt, perception_prompt
from isaac.memory.world_model import merge_observations
from isaac.nodes.classifier import classify_hypothesis, fast_classify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_parts(message: HumanMessage) -> tuple[str, str]:
    """Return ``(text_content, screenshot_b64)`` from a HumanMessage.

    Handles both plain-string content and the OpenAI multimodal list format::

        [{"type": "text", "text": "..."},
         {"type": "image_url", "image_url": {"url": "data:...;base64,<b64>"}}]
    """
    content = message.content
    if isinstance(content, str):
        return content, ""

    text_parts: list[str] = []
    screenshot_b64 = ""
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image_url":
            url: str = block.get("image_url", {}).get("url", "")
            if url.startswith("data:image"):
                # strip "data:image/png;base64," prefix
                screenshot_b64 = url.split(",", 1)[-1]

    return " ".join(text_parts).strip(), screenshot_b64


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def perception_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Perception.

    Fast path: uses ``fast_classify()`` (< 1 ms) for obvious inputs.
    Fall-through: invokes LLM with a 200-token cap for ambiguous inputs.
    """
    # -- Extract latest user message ----------------------------------------
    messages = state.get("messages", [])
    user_text = ""
    screenshot_b64 = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text, screenshot_b64 = _extract_user_parts(msg)
            break

    if not user_text and not screenshot_b64:
        logger.warning("Perception: no user message found in state.")
        return {"current_phase": "perception"}

    world_model: WorldModel = state.get("world_model", WorldModel())

    # === FAST PATH: heuristic classifier (zero LLM cost) ===================
    if not screenshot_b64:
        task_mode_hint, confidence = fast_classify(user_text)
        if task_mode_hint is not None and confidence >= 0.85:
            hypothesis = classify_hypothesis(user_text, task_mode_hint)
            logger.debug(
                "Perception (fast): mode=%s conf=%.2f input=%r",
                task_mode_hint, confidence, user_text[:60],
            )
            return {
                "world_model": world_model,
                "hypothesis": hypothesis,
                "task_mode": task_mode_hint,
                "plan": [],
                "iteration": 0,
                "ui_actions": [],
                "ui_results": [],
                "ui_cycle": 0,
                "current_phase": "perception",
            }
    # =======================================================================

    # -- LLM path: only reached for ambiguous or multimodal inputs -----------
    from isaac.llm.provider import get_perception_llm
    llm = get_perception_llm()

    # -- Long-term memory recall (skip for obviously-direct queries) --------
    ltm_context = ""
    try:
        from isaac.memory.long_term import get_long_term_memory
        ltm = get_long_term_memory()
        ltm_context = ltm.to_context_string(user_text, top_k=3)
    except Exception:
        pass

    # -- User profile context -----------------------------------------------
    profile_context = ""
    try:
        from isaac.memory.user_profile import get_user_profile
        profile = get_user_profile()
        profile.record_interaction()
        profile.save()
        profile_context = profile.to_context_string()
    except Exception:
        pass

    # -- Choose prompt based on modality ------------------------------------
    if screenshot_b64:
        prompt = perception_multimodal_prompt(user_text, world_model, screenshot_b64)
    else:
        prompt = perception_prompt(user_text, world_model)

    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # -- Parse structured JSON from response --------------------------------
    observations: list[str] = []
    hypothesis = ""
    task_mode = "code"
    gui_meta: dict[str, Any] = {}
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        observations = parsed.get("observations", [])
        hypothesis = parsed.get("hypothesis", "")
        task_mode = parsed.get("task_mode", "code")
        gui_meta = {k: parsed[k] for k in (
            "active_window_title", "current_url", "screen_width", "screen_height"
        ) if k in parsed}
    except (json.JSONDecodeError, IndexError):
        logger.error("Perception: failed to parse LLM response as JSON.")
        observations = [f"Raw LLM response: {content[:500]}"]
        hypothesis = content[:200]
        if screenshot_b64:
            task_mode = "computer_use"

    # -- Update world model -------------------------------------------------
    updated_world = merge_observations(world_model, observations)

    if screenshot_b64 or task_mode in ("computer_use", "hybrid"):
        existing_gui = updated_world.gui_state or GUIState()
        updated_world.gui_state = GUIState(
            screenshot_b64=screenshot_b64 or existing_gui.screenshot_b64,
            active_window_title=gui_meta.get(
                "active_window_title", existing_gui.active_window_title
            ),
            current_url=gui_meta.get("current_url", existing_gui.current_url),
            screen_width=gui_meta.get("screen_width", existing_gui.screen_width),
            screen_height=gui_meta.get("screen_height", existing_gui.screen_height),
            display=existing_gui.display,
            elements=existing_gui.elements,
            accessibility_tree=existing_gui.accessibility_tree,
            cursor_x=existing_gui.cursor_x,
            cursor_y=existing_gui.cursor_y,
        )

    # -- Enrich hypothesis with context ------------------------------------
    context_parts: list[str] = []
    if ltm_context:
        context_parts.append(ltm_context)
    if profile_context:
        context_parts.append(f"## User profile\n{profile_context}")
    if context_parts and hypothesis:
        hypothesis = hypothesis + "\n\n" + "\n\n".join(context_parts)

    logger.info(
        "Perception (LLM): %d observations, mode=%s",
        len(observations), task_mode,
    )

    return {
        "world_model": updated_world,
        "hypothesis": hypothesis,
        "task_mode": task_mode,
        "plan": [],
        "iteration": 0,
        "ui_actions": [],
        "ui_results": [],
        "ui_cycle": 0,
        "current_phase": "perception",
    }

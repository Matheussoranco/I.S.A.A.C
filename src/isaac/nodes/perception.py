"""Perception Node — parses user input and syncs the world model.

Reads the latest user message, invokes the LLM with the perception prompt,
and produces:
* Updated ``world_model.observations``
* An initial ``hypothesis``
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from isaac.core.state import IsaacState, WorldModel
from isaac.llm.prompts import perception_prompt
from isaac.memory.world_model import merge_observations

logger = logging.getLogger(__name__)


def perception_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Perception.

    Extracts observations and a preliminary hypothesis from the user request,
    updating the ``world_model`` and ``hypothesis`` fields.
    """
    from isaac.llm.provider import get_llm  # noqa: PLC0415

    llm = get_llm()

    # Extract latest user text
    messages = state.get("messages", [])
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not user_text:
        logger.warning("Perception: no user message found in state.")
        return {"current_phase": "perception"}

    world_model: WorldModel = state.get("world_model", WorldModel())

    # Call LLM
    prompt = perception_prompt(user_text, world_model)
    response = llm.invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)

    # Parse structured JSON from response
    observations: list[str] = []
    hypothesis = ""
    try:
        # Strip markdown fences if present
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        observations = parsed.get("observations", [])
        hypothesis = parsed.get("hypothesis", "")
    except (json.JSONDecodeError, IndexError):
        logger.error("Perception: failed to parse LLM response as JSON.")
        observations = [f"Raw LLM response: {content[:500]}"]
        hypothesis = content[:200]

    updated_world = merge_observations(world_model, observations)

    logger.info(
        "Perception: %d observations, hypothesis=%s",
        len(observations),
        hypothesis[:80],
    )

    return {
        "world_model": updated_world,
        "hypothesis": hypothesis,
        "plan": [],  # Reset plan for new perception cycle
        "iteration": 0,
        "current_phase": "perception",
    }

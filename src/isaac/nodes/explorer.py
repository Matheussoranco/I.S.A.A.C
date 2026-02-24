"""Explorer Node — active exploration for ARC-AGI and general tasks.

The Explorer performs systematic experimentation on ARC grids:
- Applies DSL primitives to training inputs and observes results.
- Extracts structural features, object decompositions, and symmetries.
- Records observations in the semantic memory as facts.
- Generates hypotheses for the Planner to refine.

For non-ARC tasks it acts as a research node that probes the environment
using the tool registry (web search, file reads, etc.) to gather context.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isaac.core.state import IsaacState, WorldModel

logger = logging.getLogger(__name__)


def explorer_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: explore the problem space before planning.

    If the task contains ARC grids, run structural analysis and DSL
    experiments.  Otherwise, use tools to gather context.
    """
    world_model: WorldModel = state.get("world_model", WorldModel())
    observations = list(world_model.observations)
    hypothesis = state.get("hypothesis", "")

    # Detect ARC task by checking for grid data
    arc_grids = _extract_arc_grids(state)

    if arc_grids:
        new_obs, new_hyp = _explore_arc(arc_grids)
        observations.extend(new_obs)
        if new_hyp and not hypothesis:
            hypothesis = new_hyp
    else:
        new_obs = _explore_general(state)
        observations.extend(new_obs)

    # Store facts in semantic memory
    _store_exploration_facts(observations)

    # Update world model
    world_model.observations = observations

    return {
        "world_model": world_model,
        "hypothesis": hypothesis,
        "current_phase": "explorer",
    }


# ---------------------------------------------------------------------------
# ARC exploration
# ---------------------------------------------------------------------------


def _extract_arc_grids(state: IsaacState) -> list[dict[str, Any]]:
    """Look for ARC grid data in the world model or messages."""
    grids: list[dict[str, Any]] = []

    world_model = state.get("world_model", WorldModel())
    resources = world_model.resources

    # Check if resources contain training pairs
    train_pairs = resources.get("train", [])
    if isinstance(train_pairs, list):
        for pair in train_pairs:
            if isinstance(pair, dict) and "input" in pair:
                grids.append(pair)

    return grids


def _explore_arc(grids: list[dict[str, Any]]) -> tuple[list[str], str]:
    """Apply structural analysis and DSL experiments to ARC grids."""
    observations: list[str] = []
    hypotheses: list[str] = []

    try:
        from isaac.arc.grid_ops import (
            extract_objects,
            detect_symmetry,
            extract_colours,
        )
        from isaac.arc.dsl import PRIMITIVES, compose
    except ImportError as exc:
        logger.warning("ARC modules not available: %s", exc)
        return observations, ""

    for i, pair in enumerate(grids):
        input_grid = np.array(pair["input"])
        output_grid = np.array(pair.get("output", []))

        prefix = f"[Train {i}]"

        # Grid dimensions
        in_dims = input_grid.shape
        observations.append(f"{prefix} Input: {in_dims[0]}x{in_dims[1]}")
        if output_grid.size > 0:
            out_dims = output_grid.shape
            observations.append(f"{prefix} Output: {out_dims[0]}x{out_dims[1]}")

            if in_dims == out_dims:
                observations.append(f"{prefix} Same dimensions → likely in-place transform.")
            else:
                observations.append(f"{prefix} Different dimensions → resize/crop/pad suspected.")

        # Colour analysis
        in_hist = extract_colours(input_grid)
        observations.append(f"{prefix} Input colours: {dict(in_hist)}")
        if output_grid.size > 0:
            out_hist = extract_colours(output_grid)
            observations.append(f"{prefix} Output colours: {dict(out_hist)}")

            # New colours in output?
            new_colours = set(out_hist.keys()) - set(in_hist.keys())
            if new_colours:
                observations.append(f"{prefix} New colours in output: {new_colours}")

        # Object decomposition
        objects = extract_objects(input_grid)
        observations.append(f"{prefix} Found {len(objects)} objects in input.")
        for obj in objects[:5]:
            observations.append(
                f"  Object {obj.id}: colour={obj.colour}, size={obj.size}, shape={obj.shape}"
            )

        # Symmetry detection
        symmetry = detect_symmetry(input_grid)
        if symmetry:
            observations.append(f"{prefix} Detected symmetry: {symmetry}")

        # DSL primitive testing  — try each primitive and check if it matches output
        if output_grid.size > 0:
            for name, fn in list(PRIMITIVES.items())[:20]:
                try:
                    result = fn(input_grid)
                    if isinstance(result, np.ndarray) and result.shape == output_grid.shape:
                        if np.array_equal(result, output_grid):
                            observations.append(
                                f"{prefix} ✓ Primitive '{name}' produces exact match!"
                            )
                            hypotheses.append(
                                f"Transformation = {name} (single primitive)."
                            )
                except Exception:
                    continue

    # Synthesise hypothesis from observations
    combined_hyp = " ".join(hypotheses) if hypotheses else ""
    if not combined_hyp and observations:
        combined_hyp = "Need LLM inspection — no single primitive matched all pairs."

    return observations, combined_hyp


# ---------------------------------------------------------------------------
# General task exploration
# ---------------------------------------------------------------------------


def _explore_general(state: IsaacState) -> list[str]:
    """Use tools to gather context about a non-ARC task."""
    observations: list[str] = []

    messages = state.get("messages", [])
    if not messages:
        return observations

    # Extract the latest user message for probing
    user_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_text = str(msg.content)
            break

    if not user_text:
        return observations

    # Attempt a web search for background context
    try:
        from isaac.tools.base import get_tool_registry
        import asyncio

        registry = get_tool_registry()
        search_tool = registry.get("web_search")
        if search_tool is not None:
            # Formulate a concise search query from the user text
            query = user_text[:100]
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run,
                            search_tool.execute(query=query, max_results=3),
                        ).result()
                else:
                    result = loop.run_until_complete(
                        search_tool.execute(query=query, max_results=3)
                    )
            except RuntimeError:
                result = asyncio.run(
                    search_tool.execute(query=query, max_results=3)
                )

            if result.success and result.output:
                observations.append(f"Web search results for '{query[:50]}':")
                observations.append(result.output[:1000])
    except Exception as exc:
        logger.debug("General exploration web search failed: %s", exc)

    return observations


# ---------------------------------------------------------------------------
# Semantic memory integration
# ---------------------------------------------------------------------------


def _store_exploration_facts(observations: list[str]) -> None:
    """Store exploration observations as facts in semantic memory."""
    try:
        from isaac.memory.manager import get_memory_manager

        mm = get_memory_manager()
        for obs in observations[:20]:  # limit to avoid flooding
            if len(obs) > 10:  # skip trivial entries
                mm.store_fact(
                    subject="exploration",
                    predicate="observed",
                    object=obs[:200],
                    confidence=0.7,
                )
    except Exception as exc:
        logger.debug("Failed to store exploration facts: %s", exc)

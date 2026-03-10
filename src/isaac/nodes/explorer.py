"""Explorer Node — active exploration for ARC-AGI and general tasks.

For **ARC tasks** the Explorer runs a full prior + analogy analysis pipeline:
  1. Core-knowledge prior analysis (objectness, topology, geometry, counting)
  2. Analogy engine (cross-pair delta extraction → transformation hypotheses)
  3. DSL primitive scan (quick single-primitive match)
  4. Hypothesis ranking and summary for the Planner

For **general tasks** the Explorer uses the tool registry (web search, file
reads) to gather context before planning.

The Explorer deliberately does NOT call the LLM — it is a pure symbolic
pre-processing stage that feeds high-quality structured observations to the
Planner and Synthesis nodes, reducing their LLM token budget and improving
accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isaac.core.state import IsaacState, WorldModel

logger = logging.getLogger(__name__)


def explorer_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: explore the problem space before planning."""
    world_model: WorldModel = state.get("world_model", WorldModel())
    observations = list(world_model.observations)
    hypothesis = state.get("hypothesis", "")

    arc_grids = _extract_arc_grids(state)

    if arc_grids:
        new_obs, new_hyp, analogy_context = _explore_arc(arc_grids)
        observations.extend(new_obs)
        if new_hyp and not hypothesis:
            hypothesis = new_hyp
        # Store analogy context in resources for Synthesis/Planner
        world_model.resources["_analogy_context"] = analogy_context
        world_model.resources["_arc_task"] = True
    else:
        new_obs = _explore_general(state)
        observations.extend(new_obs)

    _store_exploration_facts(observations)
    world_model.observations = observations

    return {
        "world_model": world_model,
        "hypothesis": hypothesis,
        "current_phase": "explorer",
    }


# ---------------------------------------------------------------------------
# ARC exploration — full prior + analogy pipeline
# ---------------------------------------------------------------------------


def _extract_arc_grids(state: IsaacState) -> list[dict[str, Any]]:
    world_model = state.get("world_model", WorldModel())
    resources = world_model.resources
    train_pairs = resources.get("train", [])
    if isinstance(train_pairs, list):
        return [p for p in train_pairs if isinstance(p, dict) and "input" in p]
    return []


def _explore_arc(grids: list[dict[str, Any]]) -> tuple[list[str], str, str]:
    """Run full ARC analysis: priors + analogy engine + DSL scan.

    Returns
    -------
    observations:
        Structured text observations for the world model.
    hypothesis:
        Best hypothesis string for the Planner.
    analogy_context:
        Formatted analogy report (rich string for LLM prompts).
    """
    observations: list[str] = []
    hypothesis = ""
    analogy_context = ""

    try:
        from isaac.arc.grid_ops import analyse_grid, format_grid_for_prompt, grid_diff
        from isaac.arc.dsl import PRIMITIVES
        from isaac.arc.priors import full_prior_analysis, describe_prior_analysis
        from isaac.arc.analogy import run_analogy_engine, format_analogy_for_prompt
    except ImportError as exc:
        logger.warning("ARC modules unavailable: %s", exc)
        return observations, hypothesis, analogy_context

    # ── 1. Core-knowledge prior analysis (per pair) ─────────────────────
    observations.append(f"=== ARC Task Analysis ({len(grids)} training pairs) ===")

    for i, pair in enumerate(grids):
        in_grid = np.array(pair["input"], dtype=int)
        out_grid_raw = pair.get("output", [])
        prefix = f"[Pair {i}]"

        in_analysis = analyse_grid(in_grid)
        observations.append(
            f"{prefix} Input {in_grid.shape[0]}x{in_grid.shape[1]}, "
            f"{in_analysis.n_colours} colours, {len(in_analysis.objects)} objects"
        )

        # Prior analysis on input
        prior = full_prior_analysis(in_grid)
        for obs in describe_prior_analysis(prior)[:8]:
            observations.append(f"  {prefix} {obs}")

        if out_grid_raw:
            out_grid = np.array(out_grid_raw, dtype=int)
            out_analysis = analyse_grid(out_grid)
            observations.append(
                f"{prefix} Output {out_grid.shape[0]}x{out_grid.shape[1]}, "
                f"{out_analysis.n_colours} colours, {len(out_analysis.objects)} objects"
            )

            diff = grid_diff(in_grid, out_grid)
            observations.append(
                f"{prefix} Shape changed: {diff['shape_changed']}, "
                f"Cells changed: {diff['n_changed_cells']}, "
                f"Colours added: {diff['colour_changes']['added']}, "
                f"removed: {diff['colour_changes']['removed']}"
            )

            # Quick single-primitive scan on first pair only
            if i == 0:
                for name, fn in list(PRIMITIVES.items())[:30]:
                    try:
                        result = fn(in_grid)
                        if isinstance(result, np.ndarray) and result.shape == out_grid.shape:
                            if np.array_equal(result, out_grid):
                                observations.append(
                                    f"  {prefix} EXACT MATCH with primitive '{name}'"
                                )
                                hypothesis = f"Single primitive '{name}' solves this task."
                    except Exception:
                        continue

    # ── 2. Analogy engine (cross-pair) ───────────────────────────────────
    try:
        analogy_result = run_analogy_engine(grids)
        analogy_context = format_analogy_for_prompt(analogy_result)

        if analogy_result.consistent_observations:
            observations.append("\n[Analogy] Consistent across ALL pairs:")
            for obs in analogy_result.consistent_observations[:5]:
                observations.append(f"  - {obs}")

        if analogy_result.hypotheses and not hypothesis:
            top_hyp = analogy_result.hypotheses[0]
            hypothesis = (
                f"{top_hyp.name}: {top_hyp.description} "
                f"(confidence: {top_hyp.confidence:.0%})"
            )
            observations.append(
                f"\n[Analogy] Top hypothesis: {hypothesis}"
            )

        if len(analogy_result.hypotheses) > 1:
            observations.append("[Analogy] Other candidates:")
            for h in analogy_result.hypotheses[1:5]:
                observations.append(f"  - [{h.confidence:.0%}] {h.name}: {h.description}")

    except Exception as exc:
        logger.warning("Analogy engine failed: %s", exc)

    # ── 3. Fallback hypothesis ───────────────────────────────────────────
    if not hypothesis:
        hypothesis = (
            "No single primitive matched. Requires composite transformation or "
            "object-level reasoning. Use LLM-guided code synthesis with the "
            "prior analysis and analogy context provided."
        )

    return observations, hypothesis, analogy_context


# ---------------------------------------------------------------------------
# General task exploration
# ---------------------------------------------------------------------------


def _explore_general(state: IsaacState) -> list[str]:
    observations: list[str] = []
    messages = state.get("messages", [])
    if not messages:
        return observations

    user_text = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_text = str(msg.content)
            break

    if not user_text:
        return observations

    try:
        from isaac.tools.base import get_tool_registry
        import asyncio

        registry = get_tool_registry()
        search_tool = registry.get("web_search")
        if search_tool is not None:
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
                result = asyncio.run(search_tool.execute(query=query, max_results=3))

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
    try:
        from isaac.memory.manager import get_memory_manager
        mm = get_memory_manager()
        for obs in observations[:20]:
            if len(obs) > 10:
                mm.store_fact(
                    subject="exploration",
                    predicate="observed",
                    object=obs[:200],
                    confidence=0.7,
                )
    except Exception as exc:
        logger.debug("Failed to store exploration facts: %s", exc)

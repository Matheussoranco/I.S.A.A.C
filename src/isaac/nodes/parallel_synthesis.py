"""Parallel Synthesis Node — spawn multiple Claude sub-agents for concurrent subtask execution.

When a plan contains multiple independent steps (no inter-dependencies), this node
fires all of them concurrently via ``ClaudeSubAgent`` and merges results back into
the state before the Reflection node evaluates them.

LangGraph integration
---------------------
This node is invoked instead of (or before) the single-threaded Synthesis node
when ``world_model.resources["_parallel_eligible"]`` is True and there are
≥2 independent pending steps.

The node uses ``ParallelSubAgentPool`` (ThreadPoolExecutor) rather than
LangGraph's Send API to stay compatible with non-async graphs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage

from isaac.core.state import ExecutionResult, IsaacState, PlanStep

logger = logging.getLogger(__name__)


def _get_independent_steps(plan: list[PlanStep]) -> list[PlanStep]:
    """Return pending steps that have no unfinished dependencies."""
    done_ids = {s.id for s in plan if s.status == "done"}
    return [
        s for s in plan
        if s.status == "pending" and all(dep in done_ids for dep in s.depends_on)
    ]


def _map_role(mode: str, description: str) -> str:
    """Heuristically assign a sub-agent role from step mode and description."""
    desc_lower = description.lower()
    if mode == "ui":
        return "coder"
    if any(kw in desc_lower for kw in ("research", "find", "search", "look up")):
        return "researcher"
    if any(kw in desc_lower for kw in ("analyse", "analyze", "review", "check", "verify")):
        return "analyst"
    if any(kw in desc_lower for kw in ("plan", "decompose", "break down", "outline")):
        return "planner"
    return "coder"


def parallel_synthesis_node(state: IsaacState) -> dict[str, Any]:
    """Spawn parallel sub-agents for independent plan steps."""
    from isaac.agents.claude_subagent import ParallelSubAgentPool

    plan: list[PlanStep] = state.get("plan", [])
    independent = _get_independent_steps(plan)

    if len(independent) < 2:
        # Fall back to sequential synthesis — nothing to parallelize
        logger.debug("ParallelSynthesis: <2 independent steps; skipping to sequential synthesis")
        return {}

    wm = state.get("world_model")
    context_parts = []
    if wm:
        context_parts.extend(wm.observations[-5:])
        for cr in state.get("connector_results", [])[-3:]:
            if isinstance(cr, dict) and "result" in cr:
                context_parts.append(str(cr["result"])[:500])
    context = "\n".join(context_parts)

    tasks = [
        {
            "subtask": step.description,
            "role": _map_role(step.mode, step.description),
            "context": context,
        }
        for step in independent
    ]

    logger.info("ParallelSynthesis: launching %d sub-agents", len(tasks))
    pool = ParallelSubAgentPool(max_workers=min(len(tasks), 4))
    results = pool.run_all(tasks)

    # Mark steps as done/failed based on sub-agent results
    updated_plan = list(plan)
    exec_logs: list[ExecutionResult] = []
    messages = []

    for step, agent_result in zip(independent, results):
        for ps in updated_plan:
            if ps.id == step.id:
                ps.status = "done" if agent_result.get("success") else "failed"
                break

        result_text = agent_result.get("result", "")
        error = agent_result.get("error", "")

        exec_logs.append(ExecutionResult(
            stdout=result_text[:3000],
            stderr=error[:500] if error else "",
            exit_code=0 if agent_result.get("success") else 1,
            duration_ms=agent_result.get("duration_ms", 0.0),
        ))

        summary = f"[{step.id}] ({agent_result.get('role', '?')}) {result_text[:500]}"
        if error:
            summary += f"\n  ERROR: {error}"
        messages.append(AIMessage(content=summary))

    return {
        "plan": updated_plan,
        "execution_logs": exec_logs,
        "messages": messages,
        "current_phase": "parallel_synthesis",
    }

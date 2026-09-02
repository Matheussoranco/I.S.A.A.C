"""Regression tests for the agentic parallel synthesis path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaac.core.state import PlanStep, make_initial_state
from isaac.nodes.parallel_synthesis import parallel_synthesis_node


def test_parallel_synthesis_executes_active_frontier_agentically() -> None:
    state = make_initial_state()
    state["plan"] = [
        PlanStep(id="s1", description="research one", status="active"),
        PlanStep(id="s2", description="review two", status="active"),
        PlanStep(id="s3", description="later", status="pending", depends_on=["s1"]),
    ]
    pool = MagicMock()
    pool.run_all.return_value = [
        {"success": True, "result": "one", "role": "researcher", "duration_ms": 1.0},
        {"success": True, "result": "two", "role": "analyst", "duration_ms": 2.0},
    ]

    with patch("isaac.agents.claude_subagent.ParallelSubAgentPool", return_value=pool):
        result = parallel_synthesis_node(state)

    pool.run_all.assert_called_once()
    assert pool.run_all.call_args.kwargs == {"agentic": True}
    assert [step.status for step in result["plan"]] == ["done", "done", "pending"]
    assert len(result["execution_logs"]) == 2

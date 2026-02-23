"""LangGraph StateGraph builder — wires nodes and conditional edges.

Call :func:`build_graph` to get a compiled graph, or :func:`build_and_run`
to execute a full interactive session.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from isaac.core.state import IsaacState, make_initial_state
from isaac.core.transitions import after_reflection, after_skill_abstraction
from isaac.nodes.perception import perception_node
from isaac.nodes.planner import planner_node
from isaac.nodes.reflection import reflection_node
from isaac.nodes.sandbox import sandbox_node
from isaac.nodes.skill_abstraction import skill_abstraction_node
from isaac.nodes.synthesis import synthesis_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------

_PERCEPTION = "perception"
_PLANNER = "planner"
_SYNTHESIS = "synthesis"
_SANDBOX = "sandbox"
_REFLECTION = "reflection"
_SKILL_ABSTRACTION = "skill_abstraction"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct and compile the I.S.A.A.C. cognitive graph.

    Returns a compiled ``StateGraph`` ready for ``.invoke()`` or
    ``.stream()``.

    Topology::

        START ─► Perception ─► Planner ─► Synthesis ─► Sandbox ─► Reflection
                      ▲                                                │
                      │          ┌──────── Skill Abstraction ◄─────────┤
                      │          │                │                    │
                      │          ▼                ▼                    ▼
                      └──── Planner (retry)      END               END
    """
    graph = StateGraph(IsaacState)

    # Register nodes
    graph.add_node(_PERCEPTION, perception_node)
    graph.add_node(_PLANNER, planner_node)
    graph.add_node(_SYNTHESIS, synthesis_node)
    graph.add_node(_SANDBOX, sandbox_node)
    graph.add_node(_REFLECTION, reflection_node)
    graph.add_node(_SKILL_ABSTRACTION, skill_abstraction_node)

    # Linear edges
    graph.set_entry_point(_PERCEPTION)
    graph.add_edge(_PERCEPTION, _PLANNER)
    graph.add_edge(_PLANNER, _SYNTHESIS)
    graph.add_edge(_SYNTHESIS, _SANDBOX)
    graph.add_edge(_SANDBOX, _REFLECTION)

    # Conditional edges
    graph.add_conditional_edges(
        _REFLECTION,
        after_reflection,
        {
            _SKILL_ABSTRACTION: _SKILL_ABSTRACTION,
            _PLANNER: _PLANNER,
            END: END,
        },
    )
    graph.add_conditional_edges(
        _SKILL_ABSTRACTION,
        after_skill_abstraction,
        {
            _PLANNER: _PLANNER,
            END: END,
        },
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Interactive runner
# ---------------------------------------------------------------------------


def build_and_run() -> int:
    """Entry-point for ``python -m isaac``.

    Launches an interactive REPL that feeds user input into the cognitive
    graph and streams partial state updates.

    Returns
    -------
    int
        Exit code (0 = normal exit).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )

    compiled = build_graph()
    state = make_initial_state()

    print("I.S.A.A.C. — Intelligent System for Autonomous Action and Cognition")
    print("Type your task below.  Press Ctrl+C to exit.\n")

    try:
        while True:
            try:
                user_input = input(">>> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            # Append the user message
            state["messages"] = [HumanMessage(content=user_input)]

            # Run the graph
            try:
                result = compiled.invoke(dict(state))
                # Update local state with graph output
                state.update(result)  # type: ignore[arg-type]

                # Print final response
                msgs = result.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage):
                        print(f"\n[I.S.A.A.C.] {msg.content}\n")

                # Print execution summary
                logs = result.get("execution_logs", [])
                if logs:
                    latest = logs[-1]
                    print(f"  ─ exit_code: {latest.exit_code}")
                    if latest.stdout.strip():
                        print(f"  ─ stdout: {latest.stdout.strip()[:500]}")
                    if latest.stderr.strip():
                        print(f"  ─ stderr: {latest.stderr.strip()[:300]}")

                phase = result.get("current_phase", "")
                print(f"  ─ phase: {phase}\n")

            except Exception as exc:
                logger.exception("Graph execution failed.")
                print(f"\n[ERROR] {exc}\n")

    except KeyboardInterrupt:
        print("\nShutting down.")

    return 0

"""LangGraph StateGraph builder — wires nodes and conditional edges.

Call :func:`build_graph` to get a compiled graph, or :func:`build_and_run`
to execute a full interactive session.

Full topology
-------------
::

    START ─► Guard ─► Perception ─► Explorer ─► Planner ─► Synthesis
                                                               │
                          ┌────────────────────────────────────┴────────────────┐
                          │ mode=ui                              │ mode=code/hybrid
                          ▼                                      ▼
                    ComputerUse                              Sandbox
                          │                                      │
                          └────────────────┬─────────────────────┘
                                           ▼
                                      Reflection
                                           │
              ┌───────────────────────┬────┴────────────────────────┐
              ▼ success               ▼ retry                       ▼ max errors
       SkillAbstraction            Planner                         END
              │
          ┌───┴─────────────┐
          ▼ pending         ▼ complete
        Planner             END

AwaitApproval is inserted dynamically when pending_approvals exist.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from isaac.core.state import IsaacState, make_initial_state
from isaac.core.transitions import (
    NODE_COMPUTER_USE,
    NODE_DIRECT_RESPONSE,
    NODE_EXPLORER,
    NODE_PERCEPTION,
    NODE_SANDBOX,
    after_guard,
    after_perception,
    after_reflection,
    after_skill_abstraction,
    after_synthesis,
)
from isaac.nodes.approval import await_approval_node
from isaac.nodes.computer_use import computer_use_node, shutdown_ui_executor
from isaac.nodes.connector_execution import connector_execution_node
from isaac.nodes.direct_response import direct_response_node
from isaac.nodes.explorer import explorer_node
from isaac.nodes.guard import guard_node
from isaac.nodes.perception import perception_node
from isaac.nodes.planner import planner_node
from isaac.nodes.reflection import reflection_node
from isaac.nodes.sandbox import sandbox_node
from isaac.nodes.skill_abstraction import skill_abstraction_node
from isaac.nodes.synthesis import synthesis_node
from isaac.memory.context_manager import compress_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node name constants
# ---------------------------------------------------------------------------

_GUARD = "guard"
_PERCEPTION = "perception"
_EXPLORER = "explorer"
_PLANNER = "planner"
_SYNTHESIS = "synthesis"
_SANDBOX = "sandbox"
_COMPUTER_USE = "computer_use"
_REFLECTION = "reflection"
_SKILL_ABSTRACTION = "skill_abstraction"
_CONNECTOR_EXEC = "connector_execution"
_DIRECT_RESPONSE = "direct_response"
_AWAIT_APPROVAL = "await_approval"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph() -> Any:
    """Construct and compile the I.S.A.A.C. cognitive graph.

    Returns a compiled ``StateGraph`` ready for ``.invoke()`` or
    ``.stream()``.

    Full topology::

        START ─► Perception ─► Planner ─► Synthesis
                                              │
                    ┌─────────────────────────┴──────────────────────┐
                    │ mode=ui                  │ mode=code/hybrid
                    ▼                          ▼
              ComputerUse                  Sandbox
                    │                          │
                    └───────────────┬──────────┘
                                    ▼
                               Reflection
                                    │
           ┌───────────────────────┬┴────────────────────────┐
           ▼ success               ▼ retry                   ▼ max errors
    SkillAbstraction            Planner                      END
           │
       ┌───┴─────────────┐
       ▼ pending         ▼ complete
     Planner             END
    """
    graph = StateGraph(IsaacState)

    # Register nodes
    graph.add_node(_GUARD, guard_node)
    graph.add_node(_PERCEPTION, perception_node)
    graph.add_node(_EXPLORER, explorer_node)
    graph.add_node(_PLANNER, planner_node)
    graph.add_node(_SYNTHESIS, synthesis_node)
    graph.add_node(_SANDBOX, sandbox_node)
    graph.add_node(_COMPUTER_USE, computer_use_node)
    graph.add_node(_REFLECTION, reflection_node)
    graph.add_node(_SKILL_ABSTRACTION, skill_abstraction_node)
    graph.add_node(_CONNECTOR_EXEC, connector_execution_node)
    graph.add_node(_DIRECT_RESPONSE, direct_response_node)
    graph.add_node(_AWAIT_APPROVAL, await_approval_node)

    # Entry: Guard → {Perception | END} → {DirectResponse | Explorer}
    graph.set_entry_point(_GUARD)
    graph.add_conditional_edges(
        _GUARD,
        after_guard,
        {
            NODE_PERCEPTION: _PERCEPTION,
            END: END,
        },
    )

    # Conditional edge: simple queries go to DirectResponse (fast-path)
    graph.add_conditional_edges(
        _PERCEPTION,
        after_perception,
        {
            NODE_DIRECT_RESPONSE: _DIRECT_RESPONSE,
            NODE_EXPLORER: _EXPLORER,
        },
    )

    # DirectResponse → END (no further processing needed)
    graph.add_edge(_DIRECT_RESPONSE, END)

    # Full pipeline continues: Explorer → Planner → ...
    graph.add_edge(_EXPLORER, _PLANNER)
    graph.add_edge(_PLANNER, _CONNECTOR_EXEC)
    graph.add_edge(_CONNECTOR_EXEC, _SYNTHESIS)

    # Synthesis → ComputerUse OR Sandbox (based on active step mode)
    graph.add_conditional_edges(
        _SYNTHESIS,
        after_synthesis,
        {
            NODE_COMPUTER_USE: _COMPUTER_USE,
            NODE_SANDBOX: _SANDBOX,
        },
    )

    # Both execution paths converge on Reflection
    graph.add_edge(_COMPUTER_USE, _REFLECTION)
    graph.add_edge(_SANDBOX, _REFLECTION)

    # Reflection → SkillAbstraction | Planner | END
    graph.add_conditional_edges(
        _REFLECTION,
        after_reflection,
        {
            _SKILL_ABSTRACTION: _SKILL_ABSTRACTION,
            _PLANNER: _PLANNER,
            END: END,
        },
    )

    # SkillAbstraction → Planner | END
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

    # Force UTF-8 on Windows so Unicode characters work in the terminal
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True,
        )

    # Register tools and start background services
    try:
        from isaac.tools import register_all_tools
        register_all_tools()
    except Exception:
        pass

    stop_scheduler = lambda: None  # noqa: E731
    try:
        from isaac.scheduler.heartbeat import start_scheduler
        from isaac.scheduler.heartbeat import stop_scheduler as _stop_sched
        stop_scheduler = _stop_sched
        start_scheduler()
    except Exception:
        pass

    # Audit system startup
    try:
        from isaac.security.audit import audit
        audit("system", "startup")
    except Exception:
        pass

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

            # Compress history if it's grown too large
            state["messages"] = compress_messages(state.get("messages", []))

            # Run the graph with streaming progress
            try:
                result: dict[str, Any] = {}
                seen_phases: set[str] = set()
                is_direct = False

                for event in compiled.stream(dict(state)):
                    # event is {node_name: state_update}
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            result.update(node_output)

                            phase = node_output.get("current_phase", "")
                            if phase == "direct_response":
                                is_direct = True

                            # Show progress for non-direct paths
                            if not is_direct and phase and phase not in seen_phases:
                                seen_phases.add(phase)
                                _phase_label = phase.replace("_", " ").title()
                                sys.stdout.write(
                                    f"\r  > {_phase_label}..."
                                    f"{'': <40}"
                                )
                                sys.stdout.flush()

                if not is_direct and seen_phases:
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    sys.stdout.flush()

                # Merge result back into state
                state.update(result)  # type: ignore[arg-type]

                # Print final response
                msgs = result.get("messages", [])
                # DirectResponse already streams to stdout — skip reprinting
                if not is_direct:
                    for msg in msgs:
                        if isinstance(msg, AIMessage):
                            print(f"\n\033[1;36m[I.S.A.A.C.]\033[0m {msg.content}\n")

                # Print execution summary
                logs = result.get("execution_logs", [])
                if logs:
                    latest = logs[-1]
                    print(f"  - exit_code: {latest.exit_code}")
                    if latest.stdout.strip():
                        print(f"  - stdout: {latest.stdout.strip()[:500]}")
                    if latest.stderr.strip():
                        print(f"  - stderr: {latest.stderr.strip()[:300]}")

                # Print UI-mode summary
                ui_results = result.get("ui_results", [])
                if ui_results:
                    print(f"  - ui_actions_executed: {len(ui_results)}")
                    print(f"  - ui_cycle: {result.get('ui_cycle', 0)}")
                    last_ui = ui_results[-1]
                    status = "OK" if last_ui.success else "FAIL"
                    desc = last_ui.action.description[:80]
                    print(f"  - last_ui_action: [{status}] {last_ui.action.type} - {desc}")

                mode = result.get("task_mode", "code")
                final_phase = result.get("current_phase", "")
                print(f"  - mode: {mode}  phase: {final_phase}\n")

            except Exception as exc:
                logger.exception("Graph execution failed.")
                print(f"\n[ERROR] {exc}\n")

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        # Tear down the UI container if one was started
        shutdown_ui_executor()
        # Stop background scheduler
        try:
            stop_scheduler()
        except Exception:
            pass
        # Audit system shutdown
        try:
            from isaac.security.audit import audit
            audit("system", "shutdown")
        except Exception:
            pass

    return 0

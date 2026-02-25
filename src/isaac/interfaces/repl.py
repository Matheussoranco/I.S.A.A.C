"""Rich REPL — the new interactive terminal for I.S.A.A.C.

Replaces the plain ``build_and_run()`` REPL in ``graph.py`` with a
beautiful, Cline/Claude-Code-inspired terminal experience featuring:

* Rich panels with Markdown rendering for responses
* Animated spinner during LLM processing
* Streaming token-by-token output for the DirectResponse fast-path
* Node-progress indicators showing which cognitive phase is active
* Syntax-highlighted code blocks in output
* prompt_toolkit-powered input with history & multi-line support
* Slash commands: /help, /clear, /status, /compact, /exit
"""

from __future__ import annotations

import logging
import sys
import time
import threading
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from isaac.core.graph import build_graph
from isaac.core.state import IsaacState, make_initial_state
from isaac.interfaces.terminal_ui import TerminalUI
from isaac.memory.context_manager import compress_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global UI handle so DirectResponse can stream to it
# ---------------------------------------------------------------------------

_active_ui: TerminalUI | None = None


def get_active_ui() -> TerminalUI | None:
    """Return the currently active TerminalUI instance (if running)."""
    return _active_ui


# ---------------------------------------------------------------------------
# prompt_toolkit input (with fallback)
# ---------------------------------------------------------------------------


def _make_prompt_session() -> Any:
    """Create a prompt_toolkit PromptSession with styling, or return None."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style as PStyle

        style = PStyle.from_dict(
            {
                "prompt_bracket": "#00d7ff bold",  # cyan ❯
            }
        )
        import os
        history_path = os.path.expanduser("~/.isaac/history.txt")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        return PromptSession(
            style=style,
            history=FileHistory(history_path),
            multiline=False,
        )
    except Exception:
        return None


def _get_input(session: Any, ui: TerminalUI) -> str:
    """Read user input using prompt_toolkit or plain input()."""
    if session is not None:
        from prompt_toolkit.formatted_text import FormattedText
        tokens = FormattedText(ui.get_prompt_tokens())
        return session.prompt(tokens).strip()
    return input("\u276f ").strip()  # ❯


# ---------------------------------------------------------------------------
# Slash-command handlers
# ---------------------------------------------------------------------------


def _handle_slash_command(cmd: str, ui: TerminalUI, state: IsaacState) -> bool:
    """Handle /commands.  Returns True if the REPL should continue."""
    cmd = cmd.lower().strip()

    if cmd in ("/exit", "/quit"):
        return False

    if cmd == "/help":
        ui.print_help()
        return True

    if cmd == "/clear":
        ui.clear()
        ui.print_banner()
        return True

    if cmd == "/status":
        # Gather live status
        model = "unknown"
        try:
            from isaac.config.settings import settings
            model = settings.llm.model
        except Exception:
            pass
        tools_count = 0
        try:
            from isaac.tools.base import get_tool_registry
            tools_count = len(get_tool_registry().list_all())
        except Exception:
            pass
        memory_ok = False
        try:
            from isaac.memory.manager import get_memory_manager
            get_memory_manager()
            memory_ok = True
        except Exception:
            pass
        scheduler_ok = False
        try:
            from isaac.scheduler.heartbeat import _scheduler
            scheduler_ok = _scheduler is not None and _scheduler.running
        except Exception:
            pass

        ui.print_status(
            model=model,
            tools_count=tools_count,
            memory_ok=memory_ok,
            scheduler_ok=scheduler_ok,
        )
        return True

    if cmd == "/compact":
        ui.print_info("Compact mode toggled (not yet implemented).")
        return True

    ui.print_warning(f"Unknown command: {cmd}.  Type /help for options.")
    return True


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def run_repl() -> int:
    """Launch the interactive REPL with full Rich UI.

    This is the replacement for ``graph.build_and_run()``.

    Returns
    -------
    int
        Exit code (0 = normal).
    """
    global _active_ui

    # -- Setup logging (suppress to avoid clutter during rich output) -------
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )
    # Suppress noisy third-party loggers
    for noisy in ("httpx", "apscheduler", "chromadb", "isaac"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Force UTF-8 on Windows
    if sys.platform == "win32":
        import io as _io
        try:
            sys.stdout = _io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
            )
            sys.stderr = _io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True,
            )
        except Exception:
            pass

    ui = TerminalUI()
    _active_ui = ui

    # -- Register tools & start services -----------------------------------
    try:
        from isaac.tools import register_all_tools
        register_all_tools()
    except Exception:
        pass

    stop_scheduler = lambda: None  # noqa: E731
    try:
        from isaac.scheduler.heartbeat import start_scheduler
        from isaac.scheduler.heartbeat import stop_scheduler as _stop
        start_scheduler()
        stop_scheduler = _stop
    except Exception:
        pass

    try:
        from isaac.security.audit import audit
        audit("system", "startup")
    except Exception:
        pass

    # -- Build graph & state -----------------------------------------------
    compiled = build_graph()
    state: dict[str, Any] = dict(make_initial_state())

    # -- prompt_toolkit session --------------------------------------------
    prompt_session = _make_prompt_session()

    # -- Banner ------------------------------------------------------------
    ui.print_banner()

    # -- REPL loop ----------------------------------------------------------
    try:
        while True:
            try:
                user_input = _get_input(prompt_session, ui)
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                if not _handle_slash_command(user_input, ui, state):
                    break
                continue

            if user_input.lower() in {"exit", "quit"}:
                break

            # -- Append user message ----------------------------------------
            state["messages"] = [HumanMessage(content=user_input)]
            state["messages"] = compress_messages(state.get("messages", []))

            ui.start_thinking()

            # -- Run the graph with streaming --------------------------------
            try:
                result: dict[str, Any] = {}
                seen_phases: set[str] = set()
                is_direct = False

                for event in compiled.stream(dict(state)):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            result.update(node_output)

                            phase = node_output.get("current_phase", "")
                            if phase == "direct_response":
                                is_direct = True

                            # Show progress phases (skip for direct-response)
                            if not is_direct and phase and phase not in seen_phases:
                                seen_phases.add(phase)
                                ui.print_phase(phase)

                # Merge result into state
                state.update(result)

                # -- Render response ----------------------------------------
                if not is_direct:
                    msgs = result.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            ui.print_assistant_response(msg.content)
                else:
                    # DirectResponse already streamed — just print final time
                    elapsed = time.monotonic() - ui._start_time if ui._start_time else 0
                    ui.end_stream(elapsed)

                # -- Execution summary (for sandbox runs) -------------------
                logs = result.get("execution_logs", [])
                if logs:
                    latest = logs[-1]
                    ui.print_execution_summary(
                        exit_code=latest.exit_code,
                        stdout=latest.stdout,
                        stderr=latest.stderr,
                    )

                # -- UI-mode summary ----------------------------------------
                ui_results = result.get("ui_results", [])
                if ui_results:
                    last_ui = ui_results[-1]
                    ui.print_ui_summary(
                        actions_count=len(ui_results),
                        ui_cycle=result.get("ui_cycle", 0),
                        last_action_success=last_ui.success,
                        last_action_type=last_ui.action.type,
                        last_action_desc=last_ui.action.description[:80],
                    )

                # -- Mode badge ---------------------------------------------
                mode = result.get("task_mode", "code")
                final_phase = result.get("current_phase", "")
                ui.print_mode_badge(mode, final_phase)

            except Exception as exc:
                logger.exception("Graph execution failed.")
                ui.print_error(str(exc))

    except KeyboardInterrupt:
        pass
    finally:
        _active_ui = None
        ui.print_goodbye()

        # Shutdown
        try:
            from isaac.nodes.computer_use import shutdown_ui_executor
            shutdown_ui_executor()
        except Exception:
            pass
        try:
            stop_scheduler()
        except Exception:
            pass
        try:
            from isaac.security.audit import audit
            audit("system", "shutdown")
        except Exception:
            pass

    return 0

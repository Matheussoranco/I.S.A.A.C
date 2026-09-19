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
* Multimodal attachment support
* Conversation management
* Live tool call visualization
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from isaac.agents.agent_loop import build_default_agent
from isaac.core.state import make_initial_state
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


def _handle_slash_command(cmd: str, ui: TerminalUI, state: dict[str, Any]) -> bool:
    """Handle /commands.  Returns True if the REPL should continue."""
    raw = cmd.strip()
    cmd = raw.lower().strip()

    if cmd in ("/exit", "/quit"):
        return False

    if cmd == "/help":
        ui.print_help()
        ui.print_info(
            "Extra: /attach <file> (text, image, document or audio), /remind <text @ in 2h>, "
            "/reminders, /persona <slug>, /team <goal>, /speak <text>, /improve"
        )
        return True

    if cmd == "/attach" or cmd.startswith("/attach "):
        parts = raw.split(maxsplit=1)
        if len(parts) < 2:
            ui.print_warning("Usage: /attach <path> [-- question]")
            return True
        rest = parts[1]
        # Split optional question after " -- " or " -p "
        path, _, question = rest.partition(" -- ")
        if not question:
            path, _, question = rest.partition(" -p ")
        path = path.strip().strip('"')
        try:
            from isaac.multimodal.files import MAX_ATTACHMENTS, MAX_TOTAL_BYTES, read_any_file

            info = read_any_file(path)
            if not info["ok"]:
                raise ValueError(info["error"])
            pending = state.setdefault("pending_attachments", [])
            if len(pending) >= MAX_ATTACHMENTS or (
                sum(item["meta"]["size_bytes"] for item in pending) + info["meta"]["size_bytes"]
                > MAX_TOTAL_BYTES
            ):
                raise ValueError("Pending attachments exceed count or size limit")
            if question.strip():
                info["attachments"].insert(0, {"type": "text", "text": question.strip()})
            pending.append(info)
            ui.print_info(f"Attached {Path(path).name} for the next message.")
            # Show attachment preview
            ui.print_attachments(
                [
                    {
                        "name": Path(path).name,
                        "size": info["meta"]["size_bytes"],
                        "type": info["meta"]["suffix"][1:],
                    }
                ]
            )
        except Exception as exc:
            ui.print_warning(f"Attach failed: {exc}")
        return True

    if cmd == "/remind" or cmd.startswith("/remind "):
        text = raw[len("/remind") :].strip()
        if not text:
            ui.print_warning("Usage: /remind <text @ in 2h>")
            return True
        try:
            from isaac.memory.reminders import add_reminder, parse_remind_args

            body, due = parse_remind_args(text)
            rem = add_reminder(body, due)
            ui.print_info(
                f"Saved [{rem.id}] {rem.text}" + (f" (due {rem.due_at})" if rem.due_at else "")
            )
        except Exception as exc:
            ui.print_warning(f"Remind failed: {exc}")
        return True

    if cmd == "/reminders":
        try:
            from isaac.memory.reminders import due_reminders, list_reminders

            items = list_reminders()
            if not items:
                ui.print_info("No reminders.")
            for r in items:
                ui.print_info(f"[{r.id}] {r.text}" + (f" (due {r.due_at})" if r.due_at else ""))
            overdue = due_reminders()
            if overdue:
                ui.print_warning(f"{len(overdue)} due!")
        except Exception as exc:
            ui.print_warning(f"Reminders failed: {exc}")
        return True

    if cmd.startswith("/persona"):
        slug = raw[len("/persona") :].strip()
        try:
            from isaac.identity import persona_builder as pb

            if not slug:
                ui.print_info(
                    f"Active: {pb.active_persona()} | Available: {', '.join(pb.list_personas())}"
                )
            else:
                pb.activate_persona(slug)
                ui.print_info(f"Persona '{slug}' activated.")
        except Exception as exc:
            ui.print_warning(f"Persona failed: {exc}")
        return True

    if cmd.startswith("/team"):
        goal = raw[len("/team") :].strip()
        if not goal:
            ui.print_warning("Usage: /team <goal>")
            return True
        try:
            from isaac.specialists import Orchestrator
            from isaac.tools import register_all_tools

            register_all_tools()
            ui.start_thinking()
            result = Orchestrator(timeout_seconds=600.0, max_wall_seconds=1800.0).run(goal)
            ui.print_assistant_response(result.final_output or "(no output)")
        except Exception as exc:
            ui.print_warning(f"Team failed: {exc}")
        return True

    if cmd.startswith("/speak"):
        text = raw[len("/speak") :].strip() or "Ol\u00e1, eu sou I.S.A.A.C."
        try:
            import tempfile

            from isaac.multimodal.voice.audio_io import play_wav
            from isaac.multimodal.voice.tts import TextToSpeech

            with tempfile.TemporaryDirectory(prefix="isaac-speech-") as directory:
                out = Path(directory) / "speech.wav"
                TextToSpeech().synthesize(text, out_path=out)
                play_wav(str(out))
        except Exception as exc:
            ui.print_warning(f"TTS unavailable: {exc}")
        return True

    if cmd == "/improve":
        try:
            from isaac.improvement.engine import get_engine

            res = get_engine().run_cycle()
            ui.print_info(
                f"Improvement: {len(res.curation_decisions)} decisions, "
                f"pruned {res.pruned_rows}. {res.critique_summary[:500]}"
            )
        except Exception as exc:
            ui.print_warning(f"Improve failed: {exc}")
        return True

    if cmd == "/clear":
        state["messages"] = []
        state["pending_attachments"] = []
        ui.clear()
        ui.print_banner()
        return True

    if cmd == "/status":
        # Gather live status
        model = "unknown"
        try:
            from isaac.config.settings import settings

            model = settings.llm.model_name
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
        msgs = state.get("messages", [])
        if not msgs:
            ui.print_info("Nothing to compact — message history is empty.")
            return True
        try:
            compressed = compress_messages(msgs)
            state["messages"] = compressed
            saved = len(msgs) - len(compressed)
            ui.print_info(
                f"Conversation compacted: {len(msgs)} \u2192 {len(compressed)} messages "
                f"({saved} condensed into context summary)."
            )
        except Exception as exc:
            ui.print_warning(f"Compact failed: {exc}")
        return True

    ui.print_warning(f"Unknown command: {cmd}.  Type /help for options.")
    return True


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def _run_turn(agent: Any, task: str, state: dict[str, Any], ui: TerminalUI) -> Any:
    messages = state.setdefault("messages", [])
    context = "\n\n".join(f"{message.type}: {message.content}" for message in messages)[-40_000:]
    pending = state.get("pending_attachments", [])
    attachments = [block for info in pending for block in info["attachments"]]
    kwargs = {"attachments": attachments} if attachments else {}
    result = agent.run(task, context=context, **kwargs)
    attached_text = "\n".join(b["text"] for b in attachments if b["type"] == "text")
    messages.append(HumanMessage(content=task + ("\n\n" + attached_text if attached_text else "")))
    messages.append(AIMessage(content=result.output))
    state["messages"] = messages[-40:]
    state["pending_attachments"] = []
    return result


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
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
            sys.stderr = _io.TextIOWrapper(
                sys.stderr.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
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

    def approve(name: str, args: dict, risk: int) -> bool:
        if not sys.stdin.isatty():
            return False
        ui.print_warning(f"Allow {name} (risk {risk}) with {args}?")
        return _get_input(prompt_session, ui).lower() in {"y", "yes"}

    agent = build_default_agent(
        auto_approve=False,
        approval_callback=approve,
        stream_callback=ui.stream_token if hasattr(ui, "stream_token") else None,
    )
    state: dict[str, Any] = dict(make_initial_state())

    # -- Background model pre-warm (load weights into VRAM before first query)
    def _prewarm() -> None:
        try:
            from isaac.llm.provider import get_direct_response_llm

            llm = get_direct_response_llm()
            # Minimal prompt — just enough to trigger model load, no output needed
            for _ in llm.stream("hi"):
                break
        except Exception:
            pass

    import threading as _threading

    _threading.Thread(target=_prewarm, daemon=True, name="isaac-prewarm").start()

    # -- prompt_toolkit session --------------------------------------------
    prompt_session = _make_prompt_session()

    # -- Banner ------------------------------------------------------------
    ui.print_banner()

    # -- REPL loop ---------------------------------------------------------
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

            # Sanitize user input before it enters the cognitive graph
            try:
                from isaac.security.sanitizer import sanitize_input

                user_input = sanitize_input(user_input)
            except Exception:
                pass

            # Print user message
            ui.console.print(f"  [isaac.user]\u276f {user_input}[/isaac.user]")

            ui.start_thinking()

            try:
                result = _run_turn(agent, user_input, state, ui)
                ui.print_assistant_response(result.output or "(no output)")
            except Exception as exc:
                logger.exception("Agent execution failed.")
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
        with contextlib.suppress(Exception):
            stop_scheduler()
        try:
            from isaac.security.audit import audit

            audit("system", "shutdown")
        except Exception:
            pass

    return 0

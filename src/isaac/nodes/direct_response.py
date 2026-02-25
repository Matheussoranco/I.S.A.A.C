"""DirectResponse Node — fast-path for simple conversational queries.

When Perception detects that the user's request is a simple question,
greeting, or conversational message that does NOT require code execution,
this node generates a direct LLM response and routes to END — bypassing
Explorer, Planner, Synthesis, Sandbox, and Reflection entirely.

This is the single most impactful optimisation: it reduces a 6+ LLM call
pipeline to a **single** LLM call with streaming output.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from isaac.core.state import IsaacState

logger = logging.getLogger(__name__)


def _build_direct_prompt(user_text: str, hypothesis: str) -> list[Any]:
    """Build a minimal prompt for direct conversational response."""
    from isaac.identity.soul import soul_system_prompt

    soul = ""
    try:
        soul = soul_system_prompt()
    except Exception:
        pass

    system_content = (
        f"{soul}\n\n"
        "You are in DIRECT RESPONSE mode.  The user's message is conversational "
        "and does NOT require code execution, planning, or tool use.  Reply "
        "naturally, concisely, and helpfully.  Be warm but efficient.  "
        "If the user greets you, greet them back and ask how you can help.  "
        "If they ask a knowledge question, answer it directly.  "
        "Keep responses under 200 words unless the topic demands more."
    )
    from langchain_core.messages import HumanMessage

    messages: list[Any] = [SystemMessage(content=system_content)]

    if hypothesis:
        messages.append(SystemMessage(content=f"Context from perception: {hypothesis[:300]}"))

    messages.append(HumanMessage(content=user_text))
    return messages


def direct_response_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: DirectResponse — single-call fast-path.

    Streams the LLM response token-by-token.  When the Rich terminal UI
    is active it streams through that; otherwise falls back to raw stdout.
    """
    from isaac.llm.provider import get_llm

    llm = get_llm("fast")

    # Extract user text
    messages = state.get("messages", [])
    user_text = ""
    from langchain_core.messages import HumanMessage

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            user_text = content if isinstance(content, str) else str(content)
            break

    hypothesis = state.get("hypothesis", "")
    prompt = _build_direct_prompt(user_text, hypothesis)

    # Resolve the active UI (if running inside the Rich REPL)
    ui = None
    try:
        from isaac.interfaces.repl import get_active_ui
        ui = get_active_ui()
    except ImportError:
        pass

    # Try streaming for instant feedback
    full_response = ""
    try:
        if ui is not None:
            ui.start_stream()
        else:
            sys.stdout.write("\n[I.S.A.A.C.] ")
            sys.stdout.flush()

        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                if ui is not None:
                    ui.stream_token(token)
                else:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                full_response += token

        if ui is None:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
        # Note: end_stream is called by the REPL after the graph finishes

    except (AttributeError, TypeError):
        # Fallback: non-streaming
        response = llm.invoke(prompt)
        full_response = response.content if isinstance(response.content, str) else str(response.content)
        if ui is not None:
            ui.start_stream()
            ui.stream_token(full_response)
        else:
            print(f"\n[I.S.A.A.C.] {full_response}\n")

    logger.info("DirectResponse: %d chars (fast-path).", len(full_response))

    return {
        "messages": [AIMessage(content=full_response)],
        "current_phase": "direct_response",
    }

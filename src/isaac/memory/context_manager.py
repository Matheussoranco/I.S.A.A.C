"""Context window management — prevents unbounded message history growth.

Implements a sliding-window strategy with optional summarisation.  When the
message list exceeds the configured cap, older messages are compressed into
a single summary message, preserving key context while freeing token budget
for the active cognitive cycle.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_MAX_MESSAGES = 40
DEFAULT_KEEP_RECENT = 10
DEFAULT_SUMMARY_PREFIX = "[Context Summary] "


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimate — ~4 chars per token heuristic."""
    total_chars = sum(
        len(m.content) if isinstance(m.content, str) else len(str(m.content))
        for m in messages
    )
    return total_chars // 4


def _extract_text(message: BaseMessage) -> str:
    """Get plain text from a message, handling multimodal content."""
    content = message.content
    if isinstance(content, str):
        return content
    # Multimodal list format
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    return " ".join(parts)


def summarise_messages(
    messages: list[BaseMessage],
    llm: Any | None = None,
) -> str:
    """Produce a text summary of a message batch.

    If an LLM is provided, uses it for abstractive summarisation.
    Otherwise, falls back to extractive (key excerpts).
    """
    if llm is not None:
        return _summarise_with_llm(messages, llm)
    return _summarise_extractive(messages)


def _summarise_extractive(messages: list[BaseMessage]) -> str:
    """Extractive fallback — concatenates condensed excerpts."""
    lines: list[str] = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        text = _extract_text(msg)
        truncated = text[:200] + "..." if len(text) > 200 else text
        lines.append(f"[{role}] {truncated}")
    return "\n".join(lines)


def _summarise_with_llm(messages: list[BaseMessage], llm: Any) -> str:
    """Use the LLM to produce an abstractive summary."""
    conversation = _summarise_extractive(messages)
    prompt = [
        SystemMessage(
            content=(
                "You are a context compressor. Summarise the following conversation "
                "into a concise paragraph preserving: (1) the user's original task, "
                "(2) key decisions made, (3) what succeeded and failed, (4) current "
                "hypothesis. Be brief — max 300 words."
            )
        ),
        HumanMessage(content=f"Conversation to summarise:\n\n{conversation}"),
    ]
    try:
        response = llm.invoke(prompt)
        return (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception:
        logger.warning("LLM summarisation failed — using extractive fallback.", exc_info=True)
        return _summarise_extractive(messages)


def compress_messages(
    messages: list[BaseMessage],
    max_messages: int = DEFAULT_MAX_MESSAGES,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    llm: Any | None = None,
) -> list[BaseMessage]:
    """Compress message history when it exceeds *max_messages*.

    Strategy:
    1. If ``len(messages) <= max_messages``, return unchanged.
    2. Otherwise, split into ``old_messages`` and ``recent_messages``.
    3. Summarise the old messages into a single ``SystemMessage``.
    4. Return ``[summary] + recent_messages``.

    Parameters
    ----------
    messages:
        Full message history.
    max_messages:
        Threshold above which compression is triggered.
    keep_recent:
        Number of most-recent messages to preserve verbatim.
    llm:
        Optional LLM for abstractive summarisation.  If ``None``, uses
        extractive fallback.

    Returns
    -------
    list[BaseMessage]
        Compressed message list.
    """
    if len(messages) <= max_messages:
        return messages

    # Preserve system messages at the start
    system_prefix: list[BaseMessage] = []
    non_system: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and not non_system:
            system_prefix.append(msg)
        else:
            non_system.append(msg)

    if len(non_system) <= keep_recent:
        return messages

    old = non_system[:-keep_recent]
    recent = non_system[-keep_recent:]

    summary_text = summarise_messages(old, llm)
    summary_msg = SystemMessage(
        content=f"{DEFAULT_SUMMARY_PREFIX}{summary_text}"
    )

    compressed = system_prefix + [summary_msg] + recent
    logger.info(
        "Context compressed: %d messages → %d (summarised %d old messages).",
        len(messages),
        len(compressed),
        len(old),
    )
    return compressed

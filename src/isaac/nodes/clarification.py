"""Clarification node — active-learning when the agent is genuinely unsure.

Sits between Perception and Planning. Looks at the perception output
(hypothesis confidence, ambiguity flags, MoE expert scores) and either:

* lets the graph proceed (default), or
* emits a *single* concise clarifying question to the user and short-circuits
  the loop, waiting for the next user message.

Why this matters: a SOTA agent must know when to *ask* rather than *guess*.
Asking once at the start prevents long, mistaken trajectories — a common
failure mode of fully-autonomous agents.

The node uses three signals:

1. ``perception_confidence`` (0..1) — set by Perception.
2. ``ambiguity_score`` (0..1) — heuristic: short query, multiple referents,
   missing object/file, vague verbs.
3. ``MoE routing margin`` — if the top-2 experts tie, ambiguity is high.

If any signal is past its threshold the node switches to clarification mode.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)


_VAGUE_VERBS = {
    "do",
    "fix",
    "make",
    "handle",
    "deal",
    "take care",
    "improve",
    "update",
    "thing",
    "stuff",
}
_QUESTION_HINTS = ("what", "which", "where", "when", "how", "why")


def _ambiguity_score(query: str) -> float:
    if not query:
        return 1.0
    q = query.lower().strip()
    score = 0.0
    if len(q.split()) <= 3:
        score += 0.4
    if any(v in q.split() for v in _VAGUE_VERBS):
        score += 0.3
    # Pronouns without referent
    if re.search(r"^(it|this|that|those|these)\b", q):
        score += 0.4
    # Lacks any noun-phrase capitalisation or quoted name
    if not re.search(r"[A-Z][a-z]+|\".+\"|'[^']+'", query):
        score += 0.1
    return min(score, 1.0)


def _moe_margin(state: dict[str, Any]) -> float:
    """Return the margin between top-1 and top-2 expert scores. Smaller →
    more ambiguous routing."""
    try:
        from isaac.experts import get_moe

        moe = get_moe()
        last_user = ""
        msgs = state.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                last_user = str(m.content)
                break
        if not last_user:
            return 1.0
        routing = moe.route(last_user, top_k=2)
        cands = routing.selection.candidates
        if len(cands) < 2:
            return 1.0
        return float(cands[0][1] - cands[1][1])
    except Exception as exc:
        logger.debug("MoE margin failed: %s", exc)
        return 1.0


def needs_clarification(state: dict[str, Any], threshold: float = 0.55) -> bool:
    """Return True iff at least one signal indicates clarification is warranted."""
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_user = str(m.content)
            break

    perception_conf = float(state.get("perception_confidence", 1.0))
    ambiguity = _ambiguity_score(last_user)
    moe_margin = _moe_margin(state)

    score = 0.45 * (1.0 - perception_conf) + 0.35 * ambiguity + 0.20 * (1.0 - moe_margin)
    state["_clarification_score"] = round(score, 3)
    state["_clarification_signals"] = {
        "perception_confidence": perception_conf,
        "ambiguity": ambiguity,
        "moe_margin": moe_margin,
    }
    return score >= threshold


def _formulate_question(state: dict[str, Any]) -> str:
    """Compose a focused clarifying question — a single sentence."""
    last_user = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_user = str(m.content)
            break

    try:
        from langchain_core.messages import SystemMessage

        from isaac.llm.provider import get_llm

        llm = get_llm("fast")
        system = SystemMessage(
            content=(
                "You are I.S.A.A.C.'s clarification assistant. The user's request "
                "is ambiguous. Respond with EXACTLY ONE concise clarifying "
                "question — no preamble, no greeting. Pick the single most "
                "useful piece of missing information."
            )
        )
        resp = llm.invoke([system, HumanMessage(content=last_user)])
        return str(resp.content).strip().split("\n")[0]
    except Exception as exc:
        logger.debug("LLM clarification failed: %s", exc)
        return "Could you clarify what you'd like me to focus on?"


def clarification_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node entry point.

    Returns *delta* updates only (LangGraph merges into state):

    * If clarification is **not** needed — sets ``needs_clarification=False``
      and lets the conditional edge route on to ``explorer``.
    * If clarification **is** needed — appends an ``AIMessage`` with the
      question, sets ``needs_clarification=True``, and the conditional edge
      routes straight to ``__end__``. The user's next reply re-enters the
      graph at ``Guard``, so the loop naturally resumes.
    """
    if not needs_clarification(state):
        return {
            "needs_clarification": False,
            "current_phase": "clarification_skipped",
        }

    question = _formulate_question(state)
    return {
        "messages": [AIMessage(content=question)],
        "needs_clarification": True,
        "current_phase": "clarification",
    }

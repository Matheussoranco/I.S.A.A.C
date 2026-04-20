"""Self-critique — periodic meta-reflection on overall agent performance.

The agent points its strong model at its own performance dataset and
asks: *what is failing, and what should change?*  The output is a
plain-text "improvement note" appended to long-term memory and surfaced
to the user when they run ``isaac improve --report``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CritiqueReport:
    generated_at: float
    summary: str
    weakest_node: str
    weakest_skill: str
    improvement_note: str


CRITIQUE_PROMPT = """You are I.S.A.A.C.'s self-improvement reviewer.

Below is a report of how each cognitive node and skill has performed
over the past N days.  Identify:

1. The single biggest weakness (lowest success rate × highest run count)
2. The single most flaky node (lowest p95 latency reliability)
3. ONE concrete change you would make (be specific — name the node /
   skill / prompt slot)

Reply in this format exactly:

WEAKEST_NODE: <node name>
WEAKEST_SKILL: <skill name or "none">
SUMMARY: <one short paragraph>
ACTION: <one specific recommendation>

--- METRICS ---
{metrics}
"""


def build_critique() -> CritiqueReport:
    """Generate one self-critique cycle and return the result."""
    from isaac.improvement.performance import get_tracker

    tracker = get_tracker()
    node_stats = tracker.node_stats()
    skill_stats = tracker.skill_stats()

    if not node_stats and not skill_stats:
        return CritiqueReport(
            generated_at=time.time(),
            summary="No telemetry yet — run more tasks before improving.",
            weakest_node="",
            weakest_skill="",
            improvement_note="",
        )

    metrics_lines: list[str] = ["NODES:"]
    for n in sorted(node_stats, key=lambda x: x.success_rate)[:8]:
        metrics_lines.append(
            f"  - {n.node}: runs={n.runs}, success={n.success_rate:.2f}, "
            f"avg={n.avg_duration_ms:.0f}ms, p95={n.p95_duration_ms:.0f}ms"
        )
        for err, cnt in n.common_errors:
            metrics_lines.append(f"      └─ ({cnt}x) {err[:140]}")
    metrics_lines.append("SKILLS:")
    for s in sorted(skill_stats, key=lambda x: x.success_rate)[:8]:
        metrics_lines.append(
            f"  - {s.skill_name}: runs={s.runs}, success={s.success_rate:.2f}, "
            f"avg={s.avg_duration_ms:.0f}ms"
        )

    metrics_text = "\n".join(metrics_lines)

    try:
        from isaac.llm.multimodal_router import (
            Complexity,
            Modality,
            get_multimodal_router,
        )

        llm = get_multimodal_router().route(Modality.TEXT, Complexity.STRONG)
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=CRITIQUE_PROMPT.format(metrics=metrics_text))])
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.warning("Self-critique: LLM call failed: %s — falling back to heuristic.", exc)
        text = _heuristic_critique(node_stats, skill_stats)

    weakest_node = ""
    weakest_skill = ""
    summary = ""
    action = ""
    for line in text.splitlines():
        if line.startswith("WEAKEST_NODE:"):
            weakest_node = line.split(":", 1)[1].strip()
        elif line.startswith("WEAKEST_SKILL:"):
            weakest_skill = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("ACTION:"):
            action = line.split(":", 1)[1].strip()

    report = CritiqueReport(
        generated_at=time.time(),
        summary=summary or text[:500],
        weakest_node=weakest_node,
        weakest_skill=weakest_skill,
        improvement_note=action or text,
    )

    # Mirror to long-term memory for the next planning cycle
    try:
        from isaac.memory.long_term import get_long_term_memory
        ltm = get_long_term_memory()
        ltm.store(
            f"[self-critique] {summary}",
            metadata={"kind": "self_critique", "action": action},
        )
    except Exception:
        pass

    return report


def _heuristic_critique(node_stats: list, skill_stats: list) -> str:
    weakest_n = min(node_stats, key=lambda n: n.success_rate, default=None)
    weakest_s = min(skill_stats, key=lambda s: s.success_rate, default=None)
    return (
        f"WEAKEST_NODE: {weakest_n.node if weakest_n else 'none'}\n"
        f"WEAKEST_SKILL: {weakest_s.skill_name if weakest_s else 'none'}\n"
        f"SUMMARY: heuristic mode — LLM unavailable\n"
        f"ACTION: investigate {weakest_n.node if weakest_n else 'planner'} latency and error patterns\n"
    )

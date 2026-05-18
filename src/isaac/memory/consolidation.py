"""Memory consolidation — episodic → semantic promotion ('sleep cycle').

Inspired by complementary-learning-systems theory (McClelland et al. 1995):
short-term episodic traces are reactivated and slowly transferred into
long-term semantic memory. The consolidation pass:

1. Pulls recent episodes from :class:`EpisodicMemory`.
2. Asks the local LLM to extract atomic ``(subject, predicate, object)``
   facts from each episode (with a deterministic JSON schema).
3. Deduplicates and promotes the facts into :class:`SemanticMemory`,
   strengthening the confidence of facts that recur (Hebbian rule:
   ``c ← c + (1−c) · η``) and decaying facts that haven't been seen
   for a long time.
4. Optionally prunes redundant low-confidence facts (LRU-style).

The pass can be triggered manually via :func:`consolidate_now` or scheduled
via the heartbeat scheduler with :func:`schedule_consolidation`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_EXTRACTION_PROMPT = """You are a memory consolidator. Extract durable facts from this episode.

Output JSON with key 'facts': a list of objects, each with:
  - subject (string)
  - predicate (string, snake_case verb or 'is_a'/'has_property'/'depends_on'/'caused_by')
  - object (string)
  - confidence (float in 0..1)

Only extract facts that are likely to remain true outside this episode
(durable knowledge), not transient state. Output JSON only — no commentary.

Episode:
{episode}
"""


@dataclass
class ConsolidationReport:
    episodes_seen: int = 0
    facts_extracted: int = 0
    facts_promoted: int = 0
    facts_strengthened: int = 0
    facts_pruned: int = 0
    elapsed_ms: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)


def consolidate_now(
    *,
    max_episodes: int = 50,
    min_confidence: float = 0.6,
    hebbian_eta: float = 0.15,
    decay_factor: float = 0.98,
    prune_below: float = 0.15,
    use_llm: bool = True,
) -> ConsolidationReport:
    """Run one consolidation pass. Returns a report you can log/inspect."""
    t0 = time.perf_counter()
    report = ConsolidationReport()

    try:
        from isaac.memory.episodic import get_episodic_memory
        from isaac.memory.semantic import get_semantic_memory
    except Exception as exc:
        logger.warning("Memory subsystems unavailable: %s", exc)
        return report

    episodic = get_episodic_memory()
    semantic = get_semantic_memory()

    episodes = episodic.recent(max_episodes) if hasattr(episodic, "recent") else []
    report.episodes_seen = len(episodes)

    extracted: list[dict[str, Any]] = []
    for ep in episodes:
        episode_text = _episode_to_text(ep)
        if not episode_text:
            continue
        if use_llm:
            try:
                extracted.extend(_extract_with_llm(episode_text))
            except Exception as exc:
                logger.debug("LLM extraction failed: %s — using heuristic.", exc)
                extracted.extend(_extract_heuristic(episode_text))
        else:
            extracted.extend(_extract_heuristic(episode_text))

    # Filter, dedupe, promote
    seen: dict[tuple[str, str, str], float] = {}
    for fact in extracted:
        s = str(fact.get("subject", "")).strip()
        p = str(fact.get("predicate", "")).strip()
        o = str(fact.get("object", "")).strip()
        c = float(fact.get("confidence", 0.7))
        if not s or not p or not o or c < min_confidence:
            continue
        key = (s.lower(), p.lower(), o.lower())
        if key in seen:
            seen[key] = max(seen[key], c)
        else:
            seen[key] = c

    report.facts_extracted = len(seen)

    for (s, p, o), c in seen.items():
        existing = _existing_confidence(semantic, s, p, o)
        if existing is None:
            semantic.add_fact(s, p, o, confidence=c, source="consolidation")
            report.facts_promoted += 1
        else:
            new_conf = existing + (1.0 - existing) * hebbian_eta
            semantic.add_fact(s, p, o, confidence=min(1.0, new_conf), source="consolidation")
            report.facts_strengthened += 1
        if len(report.samples) < 5:
            report.samples.append({"subject": s, "predicate": p, "object": o, "confidence": c})

    # Decay + prune long-unused facts (best-effort — depends on backend API)
    pruned = _decay_and_prune(semantic, decay=decay_factor, prune_below=prune_below)
    report.facts_pruned = pruned

    report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Consolidation: episodes=%d extracted=%d promoted=%d strengthened=%d pruned=%d (%.0fms)",
        report.episodes_seen,
        report.facts_extracted,
        report.facts_promoted,
        report.facts_strengthened,
        report.facts_pruned,
        report.elapsed_ms,
    )
    return report


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _episode_to_text(episode: Any) -> str:
    if isinstance(episode, dict):
        parts = [f"{k}: {v}" for k, v in episode.items() if isinstance(v, (str, int, float))]
        return " | ".join(parts)
    parts = []
    for attr in ("task", "hypothesis", "code", "result_summary", "node"):
        val = getattr(episode, attr, None)
        if val:
            parts.append(f"{attr}: {val}")
    return " | ".join(parts)


def _extract_with_llm(episode_text: str) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage

    from isaac.llm.provider import get_llm

    llm = get_llm("fast")
    prompt = _EXTRACTION_PROMPT.format(episode=episode_text[:2000])
    raw = str(llm.invoke([HumanMessage(content=prompt)]).content).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    data = json.loads(raw)
    facts = data.get("facts", [])
    return [f for f in facts if isinstance(f, dict)]


def _extract_heuristic(episode_text: str) -> list[dict[str, Any]]:
    """Regex-only extraction — fallback when no LLM is available."""
    facts: list[dict[str, Any]] = []
    # "X is a Y" / "X is Y"
    for m in re.finditer(
        r"([A-Z][\w\s]{1,40})\s+is\s+(?:a|an)\s+([A-Za-z][\w\s]{1,40})", episode_text
    ):
        facts.append(
            {
                "subject": m.group(1).strip(),
                "predicate": "is_a",
                "object": m.group(2).strip(),
                "confidence": 0.7,
            }
        )
    # "X has Y"
    for m in re.finditer(r"([A-Z][\w\s]{1,40})\s+has\s+([A-Za-z][\w\s]{1,40})", episode_text):
        facts.append(
            {
                "subject": m.group(1).strip(),
                "predicate": "has_property",
                "object": m.group(2).strip(),
                "confidence": 0.65,
            }
        )
    return facts


def _existing_confidence(semantic: Any, s: str, p: str, o: str) -> float | None:
    try:
        graph = getattr(semantic, "_graph", None)
        if graph is None:
            return None
        if not graph.has_edge(s, o):
            return None
        data = graph.get_edge_data(s, o) or {}
        if data.get("predicate") != p:
            return None
        return float(data.get("confidence", 0.5))
    except Exception:
        return None


def _decay_and_prune(semantic: Any, *, decay: float, prune_below: float) -> int:
    """Multiplicatively decay confidence and prune facts below threshold.

    Best-effort — only runs if the SemanticMemory backend exposes its graph.
    """
    pruned = 0
    try:
        graph = getattr(semantic, "_graph", None)
        if graph is None:
            return 0
        edges_to_remove: list[tuple[str, str]] = []
        for u, v, data in graph.edges(data=True):
            new_conf = float(data.get("confidence", 0.5)) * decay
            data["confidence"] = new_conf
            if new_conf < prune_below:
                edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            graph.remove_edge(u, v)
            pruned += 1
        # Persist if SemanticMemory exposes a save method
        save = getattr(semantic, "_persist_all", None) or getattr(semantic, "save", None)
        if callable(save):
            with contextlib.suppress(Exception):
                save()
    except Exception as exc:
        logger.debug("decay_and_prune failed: %s", exc)
    return pruned


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------


def schedule_consolidation(every_seconds: int = 900) -> Any:
    """Register a heartbeat callback that runs ``consolidate_now`` periodically.

    Returns the registered callable handle (or ``None`` if scheduler is
    unavailable). The default cadence (15 min) is light enough to run
    alongside normal interaction.
    """
    try:
        from isaac.scheduler.heartbeat import register_callback
    except Exception as exc:
        logger.info("Heartbeat scheduler not available: %s", exc)
        return None

    def _tick() -> None:
        try:
            consolidate_now()
        except Exception as exc:
            logger.warning("Scheduled consolidation failed: %s", exc)

    register_callback(_tick, interval_seconds=every_seconds, name="memory_consolidation")
    return _tick

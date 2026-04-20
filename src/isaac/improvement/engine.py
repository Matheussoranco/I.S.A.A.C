"""Self-improvement orchestrator.

Runs (manually or on a schedule) a single improvement cycle:

    1. Skill curation — promote / deprecate based on track record.
    2. Self-critique — meta-reflection over the metrics dataset.
    3. Memory consolidation — passes through to MemoryManager.
    4. Telemetry pruning — drop very old metric rows.

All steps are best-effort: a failure in one does not abort the others.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    started_at: float
    finished_at: float
    curation_decisions: list[dict[str, Any]] = field(default_factory=list)
    critique_summary: str = ""
    critique_action: str = ""
    pruned_rows: int = 0
    errors: list[str] = field(default_factory=list)


class ImprovementEngine:
    """Run a single improvement pass across the agent's cognitive infrastructure."""

    def __init__(self) -> None:
        from isaac.improvement.skill_curation import SkillCurator
        self._curator = SkillCurator()

    def run_cycle(self) -> ImprovementResult:
        from isaac.improvement.performance import get_tracker

        result = ImprovementResult(started_at=time.time(), finished_at=0.0)

        # 1. Skill curation
        try:
            decisions = self._curator.curate_all()
            result.curation_decisions = [asdict(d) for d in decisions]
            promoted = sum(1 for d in decisions if d.action == "promote")
            deprecated = sum(1 for d in decisions if d.action == "deprecate")
            logger.info("Improvement: curated skills — promoted=%d, deprecated=%d", promoted, deprecated)
        except Exception as exc:
            logger.exception("Improvement: skill curation failed.")
            result.errors.append(f"curation: {exc}")

        # 2. Self-critique
        try:
            from isaac.improvement.self_critique import build_critique
            report = build_critique()
            result.critique_summary = report.summary
            result.critique_action = report.improvement_note
        except Exception as exc:
            logger.exception("Improvement: self-critique failed.")
            result.errors.append(f"critique: {exc}")

        # 3. Telemetry pruning (90-day window)
        try:
            result.pruned_rows = get_tracker().prune(older_than_days=90)
        except Exception as exc:
            logger.exception("Improvement: prune failed.")
            result.errors.append(f"prune: {exc}")

        # 4. Memory consolidation hand-off
        try:
            from isaac.memory.manager import get_memory_manager
            mm = get_memory_manager()
            if hasattr(mm, "consolidate"):
                mm.consolidate()
        except Exception as exc:
            logger.exception("Improvement: memory consolidation failed.")
            result.errors.append(f"consolidation: {exc}")

        result.finished_at = time.time()
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: ImprovementEngine | None = None


def get_engine() -> ImprovementEngine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = ImprovementEngine()
    return _engine


def run_improvement_cycle() -> ImprovementResult:
    """Top-level convenience — run one cycle and return the result."""
    return get_engine().run_cycle()

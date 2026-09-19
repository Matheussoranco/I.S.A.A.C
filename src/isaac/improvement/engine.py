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
from threading import Lock
from typing import Any

from isaac.improvement.skill_curation import SkillCurator

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
    refinement_status: str = "not_requested"
    improvement_verified: bool = False


class ImprovementEngine:
    _active: bool = False
    _cycle_lock = Lock()
    _curator: SkillCurator

    def __init__(self) -> None:
        self._curator = SkillCurator()

    def run_cycle(self, *, recursive: bool = False, _depth: int = 0) -> ImprovementResult:
        if not ImprovementEngine._cycle_lock.acquire(blocking=False):
            result = ImprovementResult(started_at=time.time(), finished_at=time.time())
            result.errors.append("re-entrancy blocked: another cycle is running")
            return result
        try:
            if ImprovementEngine._active:
                result = ImprovementResult(started_at=time.time(), finished_at=time.time())
                result.errors.append("re-entrancy blocked: another cycle is running")
                return result
            ImprovementEngine._active = True
            try:
                return self._run_cycle_inner(recursive=recursive, _depth=_depth)
            finally:
                ImprovementEngine._active = False
        finally:
            ImprovementEngine._cycle_lock.release()

    def _run_cycle_inner(self, *, recursive: bool = False, _depth: int = 0) -> ImprovementResult:
        result = ImprovementResult(started_at=time.time(), finished_at=0.0)
        if recursive:
            result.refinement_status = "unsupported"
            result.errors.append(
                "recursive refinement unavailable: no bounded before/after evaluator is "
                "configured; ran one maintenance cycle without verified improvement"
            )

        # 1. Skill curation
        try:
            decisions = self._curator.curate_all()
            result.curation_decisions = [asdict(d) for d in decisions]
            promoted = sum(1 for d in decisions if d.action == "promote")
            deprecated = sum(1 for d in decisions if d.action == "deprecate")
            logger.info(
                "Improvement: curated skills — promoted=%d, deprecated=%d", promoted, deprecated
            )
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
            from isaac.improvement.performance import get_tracker

            result.pruned_rows = get_tracker().prune(older_than_days=90)
        except Exception as exc:
            logger.exception("Improvement: prune failed.")
            result.errors.append(f"prune: {exc}")

        # 4. Memory consolidation hand-off
        try:
            from isaac.memory.manager import get_memory_manager

            mm = get_memory_manager()
            mm.consolidate()
        except Exception as exc:
            logger.exception("Improvement: memory consolidation failed.")
            result.errors.append(f"consolidation: {exc}")

        try:
            from isaac.security.audit import audit

            audit(
                "system",
                "improvement-cycle",
                details={
                    "decisions": len(result.curation_decisions),
                    "pruned": result.pruned_rows,
                    "errors": len(result.errors),
                    "recursive": recursive,
                    "refinement_status": result.refinement_status,
                    "improvement_verified": result.improvement_verified,
                },
            )
        except Exception as exc:
            result.errors.append(f"audit: {exc}")

        result.finished_at = time.time()
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: ImprovementEngine | None = None
_engine_lock = Lock()


def get_engine() -> ImprovementEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ImprovementEngine()
        return _engine


def run_improvement_cycle() -> ImprovementResult:
    """Top-level convenience — run one cycle and return the result."""
    return get_engine().run_cycle()

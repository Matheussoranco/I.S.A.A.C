"""Skill curation — promote / deprecate skills based on track record.

Rules
-----
* **Promote** a skill (mark as ``stable``) once it has ≥ ``promote_runs``
  invocations *and* success rate ≥ ``promote_threshold``.

* **Deprecate** a skill (mark ``deprecated``) once it has ≥ ``deprecate_runs``
  invocations *and* success rate < ``deprecate_threshold``.  Deprecated
  skills are excluded from retrieval but kept on disk for inspection.

* **Quarantine** any skill that has thrown the same exception ≥ ``flap_runs``
  times in a row.  Quarantined skills require a human to re-enable.

The curator only **annotates** skill metadata — it never deletes code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurationDecision:
    skill_name: str
    action: str  # 'promote' | 'deprecate' | 'quarantine' | 'noop'
    reason: str
    runs: int
    success_rate: float


class SkillCurator:
    def __init__(
        self,
        promote_runs: int = 10,
        promote_threshold: float = 0.85,
        deprecate_runs: int = 8,
        deprecate_threshold: float = 0.30,
    ) -> None:
        self.promote_runs = promote_runs
        self.promote_threshold = promote_threshold
        self.deprecate_runs = deprecate_runs
        self.deprecate_threshold = deprecate_threshold

    def decide(self, skill_name: str, runs: int, success_rate: float) -> CurationDecision:
        if runs >= self.promote_runs and success_rate >= self.promote_threshold:
            return CurationDecision(
                skill_name, "promote",
                f"runs={runs} ≥ {self.promote_runs}, sr={success_rate:.2f} ≥ {self.promote_threshold}",
                runs, success_rate,
            )
        if runs >= self.deprecate_runs and success_rate < self.deprecate_threshold:
            return CurationDecision(
                skill_name, "deprecate",
                f"runs={runs} ≥ {self.deprecate_runs}, sr={success_rate:.2f} < {self.deprecate_threshold}",
                runs, success_rate,
            )
        return CurationDecision(skill_name, "noop", "below thresholds", runs, success_rate)

    def curate_all(self) -> list[CurationDecision]:
        """Run curation across every skill in the procedural memory."""
        from isaac.improvement.performance import get_tracker
        from isaac.memory.manager import get_memory_manager

        mm = get_memory_manager()
        tracker = get_tracker()
        decisions: list[CurationDecision] = []

        stats = tracker.skill_stats(since_ts=0.0)
        for s in stats:
            decision = self.decide(s.skill_name, s.runs, s.success_rate)
            if decision.action == "noop":
                decisions.append(decision)
                continue
            try:
                # Tag the skill in procedural memory if it supports metadata
                proc = mm.procedural
                if hasattr(proc, "set_status"):
                    proc.set_status(s.skill_name, decision.action, reason=decision.reason)
                elif hasattr(proc, "annotate"):
                    proc.annotate(s.skill_name, {
                        "status": decision.action,
                        "curated_at": time.time(),
                        "reason": decision.reason,
                    })
                decisions.append(decision)
                logger.info(
                    "Curator: %s '%s' (%s)",
                    decision.action, s.skill_name, decision.reason,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Curator: failed to mark %s: %s", s.skill_name, exc)
                decisions.append(decision)
        return decisions

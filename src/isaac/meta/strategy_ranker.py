"""Strategy Ranker — dynamically orders synthesis strategies based on MetaLearner data.

The ARC solver and generic synthesis node both try strategies in a fixed order.
This module lets MetaLearner data re-rank those strategies per task type so that
historically successful approaches are tried first (test-time compute allocation).

Usage
-----
    from isaac.meta.strategy_ranker import rank_strategies

    ordered = rank_strategies("arc", ["analogy", "beam", "llm", "refinement"])
    # → e.g. ["analogy", "llm", "beam", "refinement"] if analogy+llm have best history
"""

from __future__ import annotations

import logging
from typing import Sequence

from isaac.meta.learner import get_learner

logger = logging.getLogger(__name__)

# Default ordering when no history exists
_DEFAULTS: dict[str, list[str]] = {
    "arc": ["analogy", "beam", "object", "llm", "refinement"],
    "code": ["skill_retrieval", "direct", "llm", "refinement"],
    "general": ["direct", "llm", "skill_retrieval"],
}


def rank_strategies(task_type: str, candidates: Sequence[str]) -> list[str]:
    """Return ``candidates`` sorted by historical win-rate for ``task_type``.

    Falls back to the original order when no data is available.
    """
    learner = get_learner()
    ranked = learner.get_best_strategy(task_type)

    if not ranked:
        return list(candidates)

    score_map = {r["strategy"]: r["win_rate"] for r in ranked}
    return sorted(candidates, key=lambda s: score_map.get(s, 0.0), reverse=True)


def default_order(task_type: str) -> list[str]:
    """Return the hardcoded default strategy order for a given task type."""
    return _DEFAULTS.get(task_type, _DEFAULTS["general"]).copy()

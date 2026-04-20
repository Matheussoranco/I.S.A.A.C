"""I.S.A.A.C. self-improvement subsystem.

Modules
-------
* :pymod:`isaac.improvement.performance` — per-node / per-skill metrics store
* :pymod:`isaac.improvement.skill_curation` — promote / deprecate skills
* :pymod:`isaac.improvement.prompt_evolution` — A/B prompt variants with elo
* :pymod:`isaac.improvement.self_critique` — periodic meta-reflection
* :pymod:`isaac.improvement.engine` — orchestrator + scheduler hook

Run ``python -m isaac improve`` to trigger an immediate cycle, or enable
the background scheduler with ``ISAAC_IMPROVEMENT_ENABLED=true``.
"""

from __future__ import annotations

from isaac.improvement.engine import (
    ImprovementEngine,
    get_engine,
    run_improvement_cycle,
)
from isaac.improvement.performance import (
    PerformanceTracker,
    get_tracker,
)

__all__ = [
    "ImprovementEngine",
    "PerformanceTracker",
    "get_engine",
    "get_tracker",
    "run_improvement_cycle",
]

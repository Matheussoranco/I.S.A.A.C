"""ArcExpert — routes ARC-AGI grid tasks through the 5-strategy solver.

Detects ARC-style grid tasks in the query / context and dispatches to
:func:`isaac.arc.solver.synthesise`. Returns the program (DSL ops) and the
predicted output grid.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertNotApplicable, ExpertResponse

logger = logging.getLogger(__name__)


class ArcExpert(Expert):
    """ARC-AGI task expert (neuro-symbolic synthesis)."""

    name: ClassVar[str] = "arc"
    domains: ClassVar[tuple[str, ...]] = ("arc", "grid", "abstraction")
    description: ClassVar[str] = (
        "ARC-AGI grid synthesis (analogy + beam + object + LLM + refinement)."
    )
    cost: ClassVar[float] = 4.0

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        ctx = context or {}
        if ctx.get("arc_task") or (ctx.get("world_model_resources") or {}).get("_arc_task"):
            return 0.99
        if ctx.get("train_pairs") and ctx.get("test_inputs"):
            return 0.95
        q = query.lower()
        if "arc-agi" in q or "arc agi" in q or "abstraction and reasoning" in q:
            return 0.7
        return 0.0

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        try:
            import numpy as np

            from isaac.arc.evaluator import ArcPair, ArcTask
            from isaac.arc.solver import synthesise
        except ImportError as exc:
            raise ExpertNotApplicable(f"arc solver unavailable: {exc}") from exc

        # Accept either an already-constructed ArcTask, a dict, or raw pairs
        arc_task = context.get("arc_task")
        if isinstance(arc_task, ArcTask):
            task = arc_task
        elif isinstance(arc_task, dict):
            task = self._dict_to_task(arc_task)
        elif context.get("train_pairs"):
            task = ArcTask(
                id=str(context.get("task_id", "expert")),
                train=[
                    ArcPair(np.asarray(p["input"]), np.asarray(p["output"]))
                    for p in context["train_pairs"]
                ],
                test=[
                    ArcPair(np.asarray(p["input"]), np.asarray(p.get("output", p["input"])))
                    for p in (context.get("test_pairs") or [])
                ],
            )
        else:
            raise ExpertNotApplicable("ARC task payload missing (need arc_task / train_pairs)")

        result = synthesise(task)
        # TaskResult is a dataclass — use attribute access defensively
        program = getattr(result, "program", None)
        method = getattr(result, "method", "unknown")
        correct = getattr(result, "correct", False)
        confidence = 0.95 if correct else 0.4

        return ExpertResponse(
            expert=self.name,
            answer=(f"ARC solver finished — method={method}, correct={correct}, program={program}"),
            confidence=confidence,
            evidence=[f"method={method}", f"correct={correct}"],
            artifacts={
                "program": program,
                "predicted": getattr(result, "predicted", None),
                "task_id": getattr(task, "id", ""),
            },
        )

    @staticmethod
    def _dict_to_task(arc_dict: dict[str, Any]) -> Any:
        import numpy as np

        from isaac.arc.evaluator import ArcPair, ArcTask

        return ArcTask(
            id=str(arc_dict.get("id", "expert")),
            train=[
                ArcPair(np.asarray(p["input"]), np.asarray(p["output"]))
                for p in arc_dict.get("train", [])
            ],
            test=[
                ArcPair(np.asarray(p["input"]), np.asarray(p.get("output", p["input"])))
                for p in arc_dict.get("test", [])
            ],
            description=arc_dict.get("description", ""),
        )

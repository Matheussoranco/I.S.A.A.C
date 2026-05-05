"""VisionExpert — grid / image perception.

Currently focused on ARC-style grids (numpy arrays of integer colours), but
extensible to images via the multimodal pipeline. Reports object counts,
symmetries, and dominant colours — the "core knowledge priors" Chollet
identifies as essential for visual abstraction.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertNotApplicable, ExpertResponse

logger = logging.getLogger(__name__)


class VisionExpert(Expert):
    name: ClassVar[str] = "vision"
    domains: ClassVar[tuple[str, ...]] = ("vision", "perception", "grid")
    description: ClassVar[str] = "Grid/image perception — objects, symmetry, colours."
    cost: ClassVar[float] = 0.2

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        ctx = context or {}
        if ctx.get("grid") is not None or ctx.get("image_path"):
            return 0.85
        q = query.lower()
        if any(s in q for s in ("describe the grid", "how many objects",
                                "is it symmetric", "count colours", "count colors")):
            return 0.7
        return 0.0

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        grid = context.get("grid")
        if grid is None:
            raise ExpertNotApplicable("no grid in context")

        try:
            import numpy as np
            from isaac.arc.grid_ops import extract_objects, detect_symmetry
        except ImportError as exc:
            raise ExpertNotApplicable(str(exc)) from exc

        if not isinstance(grid, np.ndarray):
            grid = np.asarray(grid)

        objects = extract_objects(grid)
        symmetry = detect_symmetry(grid)
        unique, counts = np.unique(grid, return_counts=True)
        colour_dist = sorted(
            zip(unique.tolist(), counts.tolist()),
            key=lambda x: -x[1],
        )

        text = (
            f"Grid {grid.shape}: {len(objects)} objects, "
            f"symmetry={symmetry}, colours={colour_dist[:5]}"
        )

        return ExpertResponse(
            expert=self.name,
            answer=text,
            confidence=0.9,
            evidence=[
                f"object_count={len(objects)}",
                f"symmetry={symmetry}",
            ],
            artifacts={
                "objects": [{"colour": int(o.colour), "size": int(o.size)} for o in objects[:20]],
                "symmetry": symmetry,
                "colour_distribution": colour_dist,
            },
        )

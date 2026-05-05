"""Expert base class and shared protocols.

An *Expert* is any module that can answer queries within a domain. Every
expert implements three things:

1. ``can_handle(query) -> float`` — a confidence score in ``[0, 1]`` for the
   given query, used by the router to rank experts. Cheap to compute.
2. ``answer(query, context) -> ExpertResponse`` — produce an answer.
3. ``name`` and ``domains`` — metadata.

The router calls ``can_handle`` on every registered expert (this should be
fast: regex / keyword / cheap classifier — *not* an LLM call). Top-K experts
are then asked to ``answer``, and their responses merged.

Experts are registered in :class:`isaac.experts.registry.ExpertRegistry`.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar


class ExpertNotApplicable(RuntimeError):
    """Raised by an expert that decides at answer-time it cannot handle the query."""


@dataclass
class ExpertResponse:
    """A single expert's answer to a query."""

    expert: str
    """Name of the expert that produced the answer."""
    answer: str
    """The textual answer."""
    confidence: float = 0.5
    """Self-reported confidence in ``[0, 1]``."""
    evidence: list[str] = field(default_factory=list)
    """Supporting facts / citations / proof steps."""
    artifacts: dict[str, Any] = field(default_factory=dict)
    """Structured artifacts (code, sympy expressions, KG subgraphs, etc.)."""
    elapsed_ms: float = 0.0
    """Wall-clock time the expert took to respond."""
    error: str = ""
    """Non-empty iff the expert failed; ``answer`` may still hold a fallback."""

    def is_useful(self) -> bool:
        return not self.error and bool(self.answer.strip())


@dataclass
class ExpertSelection:
    """The router's routing decision."""

    primary: str
    """Top-ranked expert name."""
    candidates: list[tuple[str, float]] = field(default_factory=list)
    """All experts considered, ranked by confidence."""
    rationale: str = ""
    """Short human-readable explanation of why ``primary`` was selected."""


class Expert(abc.ABC):
    """Abstract base class for all knowledge experts."""

    name: ClassVar[str] = "expert"
    """Stable, lowercase identifier (used in routing & metrics)."""

    domains: ClassVar[tuple[str, ...]] = ()
    """Coarse domain tags (``"math"``, ``"code"``, ``"kg"``, …)."""

    description: ClassVar[str] = ""
    """One-line description for the router prompt."""

    cost: ClassVar[float] = 1.0
    """Relative cost (latency) hint — local symbolic = 0.1, local LLM = 1.0,
    remote LLM = 5.0. Used by the router as a tiebreaker."""

    # -- Required overrides -------------------------------------------------

    @abc.abstractmethod
    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        """Return confidence in ``[0, 1]`` that this expert can answer.

        Must be cheap — never call an LLM here. Use regex, keywords, or
        a small classifier.
        """

    @abc.abstractmethod
    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        """Produce an answer (raise :class:`ExpertNotApplicable` if you can't)."""

    # -- Public answer wrapper with timing ----------------------------------

    def answer(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ExpertResponse:
        """Public entry point — adds timing and error handling."""
        ctx = context or {}
        t0 = time.perf_counter()
        try:
            resp = self._answer(query, ctx)
            resp.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp.expert = resp.expert or self.name
            return resp
        except ExpertNotApplicable as exc:
            return ExpertResponse(
                expert=self.name,
                answer="",
                confidence=0.0,
                error=f"not_applicable: {exc}",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return ExpertResponse(
                expert=self.name,
                answer="",
                confidence=0.0,
                error=str(exc),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # -- Convenience --------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Expert {self.name} domains={self.domains}>"

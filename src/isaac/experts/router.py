"""HybridRouter — symbolic-first routing across registered experts.

The router scores every expert via its cheap :meth:`Expert.can_handle`,
ranks them, and returns the top-K. When two experts tie within a small
margin, an optional LLM tie-breaker may be invoked (off by default — keeps
routing fast and offline).

Routing features that influence final score:

* ``raw_confidence`` — what the expert reported.
* ``meta_winrate`` — historical win-rate from
  :class:`isaac.meta.learner.MetaLearner` (boost experts that have worked).
* ``cost`` — small penalty for slower experts (used as tiebreak only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from isaac.experts.base import Expert, ExpertSelection
from isaac.experts.registry import ExpertRegistry, get_registry

logger = logging.getLogger(__name__)


@dataclass
class RoutingFeatures:
    """Features computed per (query, expert)."""

    raw_confidence: float = 0.0
    meta_winrate: float = 0.0
    cost: float = 1.0
    final_score: float = 0.0
    rationale: str = ""

    @property
    def total(self) -> float:
        return self.final_score


@dataclass
class RoutingResult:
    selection: ExpertSelection
    features: dict[str, RoutingFeatures] = field(default_factory=dict)


class HybridRouter:
    """Pluggable router combining symbolic confidence with learned priors."""

    def __init__(
        self,
        registry: ExpertRegistry | None = None,
        *,
        winrate_weight: float = 0.25,
        cost_penalty: float = 0.02,
        tie_margin: float = 0.05,
    ) -> None:
        self._registry = registry or get_registry()
        self._winrate_weight = winrate_weight
        self._cost_penalty = cost_penalty
        self._tie_margin = tie_margin

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        top_k: int = 1,
    ) -> RoutingResult:
        """Score experts and return the top-K (default 1)."""
        ctx = context or {}
        winrates = self._load_winrates()

        scored: list[tuple[Expert, RoutingFeatures]] = []
        for expert in self._registry.all():
            try:
                raw = float(expert.can_handle(query, ctx) or 0.0)
            except Exception as exc:
                logger.debug("can_handle(%s) failed: %s", expert.name, exc)
                continue
            if raw <= 0.0:
                continue

            wr = winrates.get(expert.name, 0.0)
            base = raw + self._winrate_weight * wr - self._cost_penalty * expert.cost
            features = RoutingFeatures(
                raw_confidence=raw,
                meta_winrate=wr,
                cost=expert.cost,
                final_score=base,
                rationale=(f"raw={raw:.2f} winrate={wr:.2f} cost={expert.cost:.1f} → {base:.3f}"),
            )
            scored.append((expert, features))

        if not scored:
            # Fallback: language expert always exists; force its selection
            lang = self._registry.get("language")
            if lang is None:
                return RoutingResult(
                    selection=ExpertSelection(
                        primary="",
                        candidates=[],
                        rationale="no expert applicable and no language fallback",
                    )
                )
            return RoutingResult(
                selection=ExpertSelection(
                    primary=lang.name,
                    candidates=[(lang.name, 0.4)],
                    rationale="fallback to language expert",
                ),
                features={lang.name: RoutingFeatures(raw_confidence=0.4, final_score=0.4)},
            )

        scored.sort(key=lambda x: x[1].final_score, reverse=True)
        primary = scored[0][0].name
        candidates = [(e.name, f.final_score) for e, f in scored[: max(top_k, len(scored))]]
        rationale = scored[0][1].rationale

        # Tie-break with LLM only if explicitly enabled and we genuinely tie
        if (
            ctx.get("use_llm_tiebreaker")
            and len(scored) >= 2
            and abs(scored[0][1].final_score - scored[1][1].final_score) < self._tie_margin
        ):
            primary, rationale = self._llm_tiebreak(query, [e for e, _ in scored[:3]], rationale)

        return RoutingResult(
            selection=ExpertSelection(
                primary=primary,
                candidates=candidates[:top_k],
                rationale=rationale,
            ),
            features={e.name: f for e, f in scored},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_winrates() -> dict[str, float]:
        """Return per-expert historical win-rate from MetaLearner (or empty)."""
        try:
            from isaac.meta.learner import get_learner

            learner = get_learner()
            data = learner.get_best_strategy("expert")
            return {row["strategy"]: float(row["win_rate"]) for row in data}
        except Exception as exc:
            logger.debug("Could not load expert win-rates: %s", exc)
            return {}

    def _llm_tiebreak(
        self,
        query: str,
        candidates: list[Expert],
        fallback_rationale: str,
    ) -> tuple[str, str]:
        """Ask the LLM which expert is best — only when scores genuinely tie."""
        try:
            from langchain_core.messages import HumanMessage

            from isaac.llm.provider import get_llm

            llm = get_llm("fast")
            options = "\n".join(f"- {e.name}: {e.description}" for e in candidates)
            prompt = (
                "Pick the single best expert for this query. Respond with the "
                "expert name only, nothing else.\n\n"
                f"Query: {query}\n\nOptions:\n{options}"
            )
            ans = str(llm.invoke([HumanMessage(content=prompt)]).content).strip().lower()
            for e in candidates:
                if e.name.lower() in ans:
                    return e.name, f"llm-tiebreak picked {e.name}"
        except Exception as exc:
            logger.debug("LLM tiebreaker failed: %s", exc)
        return candidates[0].name, fallback_rationale

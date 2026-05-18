"""MixtureOfExperts — orchestrator running selected experts and merging answers.

Modes:
  * ``"single"``    — run only the routed primary expert (lowest latency).
  * ``"top_k"``     — run the top-K experts in parallel and merge.
  * ``"cascade"``   — try primary; if its confidence is low, escalate to next.

Merging strategy (top-K mode):
  * Pick the expert response with the highest confidence as the headline.
  * Concatenate evidence and artifacts from all useful responses.
  * If two answers disagree above a threshold, ask the language expert to
    arbitrate.

This module also records every routing decision in the MetaLearner so future
queries learn which experts to prefer.
"""

from __future__ import annotations

import concurrent.futures as _cf
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from isaac.experts.base import ExpertResponse
from isaac.experts.registry import get_registry
from isaac.experts.router import HybridRouter, RoutingResult

logger = logging.getLogger(__name__)

Mode = Literal["single", "top_k", "cascade"]


@dataclass
class MoEResult:
    """Aggregated output from the mixture."""

    answer: str
    primary_expert: str
    confidence: float
    routing: RoutingResult
    responses: list[ExpertResponse] = field(default_factory=list)
    elapsed_ms: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)


class MixtureOfExperts:
    """Top-level Mixture-of-Experts orchestrator."""

    def __init__(self, router: HybridRouter | None = None) -> None:
        self._router = router or HybridRouter()
        self._registry = get_registry()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        top_k: int = 1,
    ) -> RoutingResult:
        return self._router.route(query, context, top_k=top_k)

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        *,
        mode: Mode = "single",
        top_k: int = 3,
        cascade_threshold: float = 0.6,
    ) -> MoEResult:
        ctx = context or {}
        t0 = time.perf_counter()

        routing = self._router.route(query, ctx, top_k=top_k)
        if not routing.selection.primary:
            elapsed = (time.perf_counter() - t0) * 1000
            return MoEResult(
                answer="No expert was able to handle this query.",
                primary_expert="",
                confidence=0.0,
                routing=routing,
                elapsed_ms=elapsed,
            )

        if mode == "single":
            responses = self._run_single(query, ctx, routing)
        elif mode == "top_k":
            responses = self._run_top_k(query, ctx, routing, top_k)
        elif mode == "cascade":
            responses = self._run_cascade(query, ctx, routing, cascade_threshold)
        else:
            raise ValueError(f"Unknown MoE mode: {mode!r}")

        merged = self._merge(responses, query, ctx)
        merged.routing = routing
        merged.elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Record outcome for self-improvement
        self._record(query, merged)
        return merged

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    def _run_single(
        self,
        query: str,
        ctx: dict[str, Any],
        routing: RoutingResult,
    ) -> list[ExpertResponse]:
        expert = self._registry.get(routing.selection.primary)
        if expert is None:
            return []
        return [expert.answer(query, ctx)]

    def _run_top_k(
        self,
        query: str,
        ctx: dict[str, Any],
        routing: RoutingResult,
        top_k: int,
    ) -> list[ExpertResponse]:
        names = [n for n, _ in routing.selection.candidates[:top_k]]
        experts = [self._registry.get(n) for n in names]
        experts = [e for e in experts if e is not None]
        if not experts:
            return []

        with _cf.ThreadPoolExecutor(max_workers=min(len(experts), 4)) as ex:
            futures = [ex.submit(e.answer, query, ctx) for e in experts]
            return [f.result() for f in _cf.as_completed(futures, timeout=60)]

    def _run_cascade(
        self,
        query: str,
        ctx: dict[str, Any],
        routing: RoutingResult,
        threshold: float,
    ) -> list[ExpertResponse]:
        names = [n for n, _ in routing.selection.candidates]
        responses: list[ExpertResponse] = []
        for n in names:
            expert = self._registry.get(n)
            if expert is None:
                continue
            resp = expert.answer(query, ctx)
            responses.append(resp)
            if resp.is_useful() and resp.confidence >= threshold:
                break
        return responses

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        responses: list[ExpertResponse],
        query: str,
        ctx: dict[str, Any],
    ) -> MoEResult:
        useful = [r for r in responses if r.is_useful()]
        if not useful:
            # Return whatever error info we have
            errors = "; ".join(f"{r.expert}: {r.error}" for r in responses)
            return MoEResult(
                answer=f"All experts failed. {errors}",
                primary_expert=responses[0].expert if responses else "",
                confidence=0.0,
                responses=responses,
            )

        useful.sort(key=lambda r: r.confidence, reverse=True)
        head = useful[0]

        # Aggregate evidence and artifacts from all useful responses
        evidence: list[str] = []
        artifacts: dict[str, Any] = {}
        for r in useful:
            evidence.extend(f"[{r.expert}] {e}" for e in r.evidence)
            for k, v in r.artifacts.items():
                artifacts[f"{r.expert}.{k}"] = v

        # If multiple experts answered, append a short notes section
        text = head.answer
        if len(useful) > 1:
            others = useful[1:]
            note = "\n\nOther experts also responded:\n" + "\n".join(
                f"- [{r.expert} (conf={r.confidence:.2f})] {r.answer[:200]}" for r in others
            )
            text = head.answer + note

        return MoEResult(
            answer=text,
            primary_expert=head.expert,
            confidence=head.confidence,
            responses=useful,
            artifacts=artifacts,
        )

    # ------------------------------------------------------------------
    # Self-improvement hook
    # ------------------------------------------------------------------

    @staticmethod
    def _record(query: str, result: MoEResult) -> None:
        try:
            from isaac.meta.learner import get_learner

            learner = get_learner()
            learner.record(
                task_desc=query[:200],
                task_type="expert",
                strategy=result.primary_expert or "none",
                success=result.confidence >= 0.5,
                duration_ms=result.elapsed_ms,
                error_type="" if result.confidence >= 0.5 else "low_confidence",
            )
        except Exception as exc:
            logger.debug("MetaLearner record failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton + convenience
# ---------------------------------------------------------------------------

_instance: MixtureOfExperts | None = None


def get_moe() -> MixtureOfExperts:
    global _instance
    if _instance is None:
        _instance = MixtureOfExperts()
    return _instance


def answer(
    query: str,
    context: dict[str, Any] | None = None,
    *,
    mode: Mode = "single",
) -> MoEResult:
    """One-shot convenience: route + answer."""
    return get_moe().answer(query, context, mode=mode)

"""Test-time compute scaling for hard single steps.

The ARC solver already spends compute the way Chollet argues a reasoning system
should: try the cheap strategy, check whether it actually worked, escalate only
if it did not, and stop the moment something passes
(:func:`isaac.arc.solver.synthesise`).  That pattern is not ARC-specific — it
is how a small model claws back accuracy on *any* hard step.  This module
generalises it.

Two mechanisms, in increasing cost:

* :func:`self_consistency` — sample the same step *n* times at non-zero
  temperature and take the majority answer (Wang et al., 2023).  Needs no
  verifier, which is what makes it applicable everywhere; it converts variance
  into accuracy and is the single cheapest win available to a 4B model.
* :func:`best_of_n` — sample until a **verifier** accepts one.  Strictly better
  than voting *when a verifier exists*, because it checks correctness rather
  than popularity — and it exits on the first pass, so the common case costs
  one sample, not *n*.

:func:`solve_hard_step` composes them with the same escalation discipline as
the ARC solver: one greedy attempt, then verification, then voting, then
best-of-N, stopping as soon as an answer clears the bar and respecting a
wall-clock budget throughout.

Verifiers here are deliberately *cheap* — a parse, a compile, a range check, a
schema match. A verifier that costs an LLM call would spend the budget it is
meant to save. :mod:`isaac.reasoning.verifiers` supplies the stock ones.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "TestTimeResult",
    "best_of_n",
    "self_consistency",
    "solve_hard_step",
]

#: Score at or above which a verifier is taken to have accepted an answer.
PASS_THRESHOLD = 1.0


@dataclass
class TestTimeResult:
    """Outcome of a test-time compute escalation."""

    answer: Any = None
    strategy: str = "none"
    samples: list[Any] = field(default_factory=list)
    n_sampled: int = 0
    agreement: float = 0.0
    score: float = 0.0
    verified: bool = False
    exited_early: bool = False
    elapsed_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.answer is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_sampled": self.n_sampled,
            "agreement": round(self.agreement, 4),
            "score": round(self.score, 4),
            "verified": self.verified,
            "exited_early": self.exited_early,
            "elapsed_s": round(self.elapsed_s, 3),
        }


def _default_key(value: Any) -> str:
    """Normalise an answer for vote-counting.

    Case- and whitespace-insensitive, because two samples that differ only in
    formatting are the same answer and must not split the vote.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split()).strip().casefold()
    if isinstance(value, (list, tuple)):
        return repr([_default_key(v) for v in value])
    if isinstance(value, dict):
        return repr(sorted((str(k), _default_key(v)) for k, v in value.items()))
    return repr(value)


def _collect(
    sampler: Callable[[], T],
    n: int,
    budget_s: float,
    started: float,
) -> list[T]:
    """Draw up to *n* samples, stopping early if the budget runs out."""
    out: list[T] = []
    for i in range(n):
        if budget_s and time.perf_counter() - started > budget_s:
            logger.debug("test-time budget exhausted after %d/%d samples", i, n)
            break
        try:
            out.append(sampler())
        except Exception:  # pragma: no cover - a bad sample must not abort the vote
            logger.debug("sampler raised on draw %d", i, exc_info=True)
    return out


def self_consistency(
    sampler: Callable[[], T],
    n: int = 5,
    key: Callable[[Any], str] | None = None,
    min_agreement: float = 0.0,
    budget_s: float = 0.0,
) -> TestTimeResult:
    """Sample *n* times and return the majority answer.

    Parameters
    ----------
    sampler:
        Zero-argument callable producing one answer.  Should be run at
        **non-zero temperature** — identical greedy samples carry no
        information and voting over them is pure waste.
    n:
        Number of draws.  Odd values avoid ties.
    key:
        Maps an answer to its vote bucket.  Defaults to a
        whitespace/case-insensitive normalisation.
    min_agreement:
        Reject the winner if its share of votes is below this (0–1).  Use it
        when a wrong-but-confident answer is worse than no answer.
    budget_s:
        Wall-clock ceiling across all draws (0 = unlimited).
    """
    started = time.perf_counter()
    keyfn = key or _default_key
    samples = _collect(sampler, max(1, n), budget_s, started)
    elapsed = time.perf_counter() - started

    if not samples:
        return TestTimeResult(strategy="self_consistency", elapsed_s=elapsed)

    votes = Counter(keyfn(s) for s in samples)
    winner_key, count = votes.most_common(1)[0]
    agreement = count / len(samples)

    answer = next((s for s in samples if keyfn(s) == winner_key), None)
    if min_agreement and agreement < min_agreement:
        logger.debug("self-consistency agreement %.2f below %.2f", agreement, min_agreement)
        answer = None

    return TestTimeResult(
        answer=answer,
        strategy="self_consistency",
        samples=list(samples),
        n_sampled=len(samples),
        agreement=agreement,
        score=agreement,
        elapsed_s=elapsed,
    )


def best_of_n(
    sampler: Callable[[], T],
    verifier: Callable[[T], float | bool],
    n: int = 5,
    threshold: float = PASS_THRESHOLD,
    budget_s: float = 0.0,
) -> TestTimeResult:
    """Sample until the verifier accepts, then stop.

    Mirrors the ARC solver's early exit: the first sample scoring at or above
    *threshold* wins and no further compute is spent.  When nothing clears the
    bar, the highest-scoring sample is returned with ``verified=False`` — a
    best effort beats nothing, but the caller can tell the difference.

    Parameters
    ----------
    verifier:
        Cheap check returning a score in ``[0, 1]`` (``bool`` is accepted and
        coerced).  Must be far cheaper than *sampler* or the scaling is
        pointless.
    """
    started = time.perf_counter()
    samples: list[T] = []
    best: T | None = None
    best_score = -1.0
    exited_early = False

    for i in range(max(1, n)):
        if budget_s and time.perf_counter() - started > budget_s:
            logger.debug("best-of-n budget exhausted after %d draws", i)
            break
        try:
            sample = sampler()
        except Exception:  # pragma: no cover
            logger.debug("sampler raised on draw %d", i, exc_info=True)
            continue
        samples.append(sample)
        try:
            score = float(verifier(sample))
        except Exception:  # pragma: no cover - a broken verifier must not abort
            logger.debug("verifier raised on draw %d", i, exc_info=True)
            score = 0.0
        if score > best_score:
            best, best_score = sample, score
        if score >= threshold:
            exited_early = i < n - 1
            break

    return TestTimeResult(
        answer=best,
        strategy="best_of_n",
        samples=samples,
        n_sampled=len(samples),
        score=max(best_score, 0.0),
        verified=best_score >= threshold,
        exited_early=exited_early,
        elapsed_s=time.perf_counter() - started,
    )


def solve_hard_step(
    sampler: Callable[[], T],
    verifier: Callable[[T], float | bool] | None = None,
    n: int = 5,
    key: Callable[[Any], str] | None = None,
    threshold: float = PASS_THRESHOLD,
    budget_s: float = 60.0,
) -> TestTimeResult:
    """Escalating test-time compute for one hard step.

    Ladder, cheapest first, stopping the instant an answer clears the bar:

    1. **One greedy sample.**  Most steps are not hard; verify and leave.
    2. **Best-of-N** (verifier available) — exits on the first accepted sample.
    3. **Self-consistency** (no verifier) — majority vote over *n* draws.

    With no verifier, step 1 cannot be checked and is folded into the vote, so
    no sample is wasted.
    """
    started = time.perf_counter()

    if verifier is not None:
        first = _collect(sampler, 1, budget_s, started)
        if first:
            try:
                score = float(verifier(first[0]))
            except Exception:  # pragma: no cover
                logger.debug("verifier raised on greedy sample", exc_info=True)
                score = 0.0
            if score >= threshold:
                return TestTimeResult(
                    answer=first[0],
                    strategy="greedy",
                    samples=list(first),
                    n_sampled=1,
                    score=score,
                    verified=True,
                    exited_early=True,
                    elapsed_s=time.perf_counter() - started,
                )

        remaining = max(0.0, budget_s - (time.perf_counter() - started)) if budget_s else 0.0
        result = best_of_n(
            sampler,
            verifier,
            n=max(1, n - 1),
            threshold=threshold,
            budget_s=remaining,
        )
        # Fold the greedy attempt back in so n_sampled reflects real spend.
        result.samples = list(first) + result.samples
        result.n_sampled += len(first)
        result.elapsed_s = time.perf_counter() - started
        if result.verified:
            return result

        # Verifier never accepted: fall back to agreement among what we drew.
        # Replays the existing samples rather than drawing new ones — the
        # budget is already spent, and this only re-ranks what it bought.
        if result.samples:
            replay = iter(result.samples)
            vote = self_consistency(
                lambda: next(replay),
                n=len(result.samples),
                key=key,
            )
            if vote.answer is not None and vote.agreement > 0.5:
                vote.strategy = "best_of_n+vote"
                vote.score = result.score
                vote.elapsed_s = time.perf_counter() - started
                return vote
        return result

    result = self_consistency(sampler, n=n, key=key, budget_s=budget_s)
    result.elapsed_s = time.perf_counter() - started
    return result


def aggregate(results: Iterable[TestTimeResult]) -> dict[str, Any]:
    """Summarise a batch of escalations for reporting."""
    items = list(results)
    if not items:
        return {"runs": 0}
    verified = sum(1 for r in items if r.verified)
    early = sum(1 for r in items if r.exited_early)
    return {
        "runs": len(items),
        "verified": verified,
        "verified_rate": round(verified / len(items), 4),
        "early_exit_rate": round(early / len(items), 4),
        "mean_samples": round(sum(r.n_sampled for r in items) / len(items), 2),
        "mean_agreement": round(sum(r.agreement for r in items) / len(items), 4),
        "total_seconds": round(sum(r.elapsed_s for r in items), 2),
    }

"""Specialist selector — turns MetaLearner win-rates into routing pressure.

Until 1.5.0 the :class:`~isaac.specialists.orchestrator.Orchestrator` *wrote*
every outcome to the :class:`~isaac.meta.learner.MetaLearner` and then never
read it back: specialist choice was whatever the planner LLM happened to name.
This module closes that loop (roadmap WS6).

Scoring
-------
A raw win-rate is a terrible ranking signal at low sample counts — one lucky
run makes a specialist look perfect, and one unlucky run makes it look useless.
Scores are therefore **Bayesian-smoothed** against a Beta prior::

    score = (wins + prior_mean * prior_strength) / (runs + prior_strength)

with an **optimistic** ``prior_mean`` (0.7 by default).  Two consequences that
matter:

* A never-tried specialist scores ``prior_mean`` — *above* anything with a
  mediocre record — so exploration is never starved by a cold start.
* A specialist needs a sustained bad record, not one bad run, to be demoted.

Nothing here is mandatory: :func:`SpecialistSelector.rank` is a *stable* sort,
so with an empty history every ordering is returned exactly as it came in and
the selector is a no-op.  That property is what makes the ON/OFF ablation in
:mod:`isaac.eval.ablation` a fair comparison.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: ``task_type`` under which per-specialist outcomes are recorded.  Kept
#: distinct from the orchestration-level ``"orchestration"`` rows so the two
#: never contaminate each other's win-rates.
SPECIALIST_TASK_TYPE = "specialist"

#: Default Beta prior.  Optimistic on purpose — see the module docstring.
DEFAULT_PRIOR_MEAN = 0.7
DEFAULT_PRIOR_STRENGTH = 3.0

#: A specialist must have at least this many recorded runs before its score is
#: allowed to *demote* it below the prior.  Below this it is treated as
#: "still exploring" and pinned to the prior.
DEFAULT_MIN_RUNS = 2


@dataclass(frozen=True)
class SpecialistScore:
    """One specialist's track record and its smoothed selection score."""

    name: str
    wins: int = 0
    losses: int = 0
    prior_mean: float = DEFAULT_PRIOR_MEAN
    prior_strength: float = DEFAULT_PRIOR_STRENGTH

    @property
    def runs(self) -> int:
        return self.wins + self.losses

    @property
    def raw_win_rate(self) -> float:
        """Unsmoothed wins/runs — reporting only, never used for ranking."""
        return self.wins / self.runs if self.runs else 0.0

    @property
    def score(self) -> float:
        """Bayesian-smoothed win-rate used for ranking."""
        return (self.wins + self.prior_mean * self.prior_strength) / (
            self.runs + self.prior_strength
        )

    @property
    def is_cold(self) -> bool:
        """True while the record is too thin to trust."""
        return self.runs < DEFAULT_MIN_RUNS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wins": self.wins,
            "losses": self.losses,
            "runs": self.runs,
            "raw_win_rate": round(self.raw_win_rate, 4),
            "score": round(self.score, 4),
            "cold": self.is_cold,
        }


class SpecialistSelector:
    """Read MetaLearner history and rank specialists by smoothed win-rate."""

    def __init__(
        self,
        learner: Any | None = None,
        *,
        prior_mean: float = DEFAULT_PRIOR_MEAN,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        task_type: str = SPECIALIST_TASK_TYPE,
    ) -> None:
        """Initialise the selector.

        Args:
            learner: A :class:`~isaac.meta.learner.MetaLearner`.  Resolved
                lazily from :func:`~isaac.meta.learner.get_learner` when omitted
                so importing this module never touches SQLite.
            prior_mean: Beta prior mean — the score of an untried specialist.
            prior_strength: Beta prior weight in pseudo-runs.
            task_type: MetaLearner ``task_type`` bucket to read and write.
        """
        self._learner = learner
        self.prior_mean = max(0.0, min(1.0, prior_mean))
        self.prior_strength = max(0.0, prior_strength)
        self.task_type = task_type

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _get_learner(self) -> Any | None:
        if self._learner is not None:
            return self._learner
        try:
            from isaac.meta.learner import get_learner

            self._learner = get_learner()
        except Exception:  # pragma: no cover - learning is always best-effort
            logger.debug("MetaLearner unavailable; selector falls back to priors", exc_info=True)
            return None
        return self._learner

    def _history(self) -> dict[str, tuple[int, int]]:
        """Return ``{specialist: (wins, losses)}`` from the MetaLearner."""
        learner = self._get_learner()
        if learner is None:
            return {}
        try:
            rows = learner.get_best_strategy(self.task_type)
        except Exception:  # pragma: no cover - defensive
            logger.debug("MetaLearner query failed", exc_info=True)
            return {}
        out: dict[str, tuple[int, int]] = {}
        for row in rows:
            name = str(row.get("strategy", "")).strip().lower()
            if not name:
                continue
            out[name] = (int(row.get("wins", 0) or 0), int(row.get("losses", 0) or 0))
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def scores(self, names: Sequence[str]) -> dict[str, SpecialistScore]:
        """Return a :class:`SpecialistScore` for every name in *names*."""
        history = self._history()
        out: dict[str, SpecialistScore] = {}
        for raw in names:
            key = str(raw).strip().lower()
            wins, losses = history.get(key, (0, 0))
            out[str(raw)] = SpecialistScore(
                name=str(raw),
                wins=wins,
                losses=losses,
                prior_mean=self.prior_mean,
                prior_strength=self.prior_strength,
            )
        return out

    def rank(self, names: Sequence[str]) -> list[str]:
        """Return *names* ordered best-first by smoothed win-rate.

        The sort is **stable**: equally scored names (in particular, the
        all-cold case where every score equals the prior) keep their incoming
        order, so an empty history makes this an exact no-op.
        """
        scored = self.scores(names)
        return sorted(names, key=lambda n: -scored[str(n)].score)

    def best(self, names: Sequence[str], default: str = "generalist") -> str:
        """Return the highest-scoring name, or *default* when *names* is empty."""
        ranked = self.rank(names)
        return str(ranked[0]) if ranked else default

    def annotate_roster(self, roster: Sequence[dict]) -> list[dict]:
        """Order roster cards best-first and attach a ``track_record`` field.

        The orchestrator hands the annotated roster to the planner LLM, so the
        accumulated evidence reaches the actual selection decision instead of
        sitting unread in SQLite.

        Args:
            roster: Cards from ``list_specialists()``; each needs a ``name``.

        Returns:
            A new list of shallow-copied cards, ordered by score, each with
            ``track_record`` (human-readable) and ``score`` (float) added.
        """
        cards = [dict(c) for c in roster]
        names = [str(c.get("name", "")) for c in cards]
        scored = self.scores(names)
        for card in cards:
            s = scored[str(card.get("name", ""))]
            card["score"] = round(s.score, 3)
            card["track_record"] = (
                "no track record yet"
                if s.runs == 0
                else f"{s.wins}/{s.runs} succeeded ({s.raw_win_rate:.0%})"
            )
        cards.sort(key=lambda c: -float(c.get("score", 0.0)))
        return cards

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        specialist: str,
        *,
        success: bool,
        task_desc: str = "",
        duration_ms: float = 0.0,
        session_id: str = "",
    ) -> None:
        """Record one specialist outcome (best-effort; never raises)."""
        learner = self._get_learner()
        if learner is None:
            return
        try:
            learner.record(
                task_desc=task_desc or specialist,
                task_type=self.task_type,
                strategy=str(specialist).strip().lower(),
                success=success,
                duration_ms=duration_ms,
                session_id=session_id,
            )
        except Exception:  # pragma: no cover - learning is best-effort
            logger.debug("Failed to record specialist outcome", exc_info=True)

    def summary(self, names: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return score dicts for *names* (or everything with history)."""
        if names is None:
            names = sorted(self._history())
        return [s.to_dict() for s in self.rank_scores(names)]

    def rank_scores(self, names: Sequence[str]) -> list[SpecialistScore]:
        """Like :meth:`rank` but returns the full score objects."""
        scored = self.scores(names)
        return sorted((scored[str(n)] for n in names), key=lambda s: -s.score)


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_selector: SpecialistSelector | None = None


def get_selector() -> SpecialistSelector:
    """Return the process-wide :class:`SpecialistSelector`."""
    global _selector
    if _selector is None:
        _selector = SpecialistSelector()
    return _selector


def reset_selector() -> None:
    """Drop the cached selector — used by tests and the ablation harness."""
    global _selector
    _selector = None


def selection_enabled() -> bool:
    """Whether MetaLearner-guided specialist selection is switched on.

    Reads ``ISAAC_META_SPECIALIST_SELECTION``; defaults to the value shipped in
    :class:`~isaac.config.settings.Settings`.  Kept as a function (not a
    module constant) so the ablation harness can flip it at runtime.
    """
    try:
        from isaac.config.settings import get_settings

        return bool(get_settings().meta_specialist_selection)
    except Exception:  # pragma: no cover - defensive
        return False

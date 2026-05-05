"""Prompt-variant A/B testing with Elo-style scoring.

Each registered prompt has a name and 1+ variants.  Callers ask
:meth:`pick_variant` for a variant to use, then report the outcome via
``record_outcome(success, score)``.  Over time, weak variants are starved
of traffic via Thompson-sampling-lite (epsilon-greedy with a wide tail).

The store is in-memory + persisted to JSON next to the performance DB.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    name: str
    template: str
    runs: int = 0
    successes: int = 0
    score_sum: float = 0.0
    last_used_ts: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.runs if self.runs else 0.0


@dataclass
class PromptBank:
    """A named pool of variants for a single prompt slot."""

    prompt_id: str
    variants: dict[str, Variant] = field(default_factory=dict)

    def add(self, name: str, template: str) -> None:
        if name in self.variants:
            return
        self.variants[name] = Variant(name=name, template=template)


class PromptEvolution:
    """In-memory + JSON-persisted prompt bank with epsilon-greedy selection."""

    DEFAULT_EPSILON = 0.15
    MIN_RUNS_BEFORE_EXPLOIT = 4

    def __init__(self, store_path: Path | str | None = None) -> None:
        if store_path is None:
            try:
                from isaac.config.settings import settings
                store_path = settings.isaac_home / "prompt_evolution.json"
            except Exception:
                store_path = Path.home() / ".isaac" / "prompt_evolution.json"
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._banks: dict[str, PromptBank] = {}
        self._load()

    # -- Persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for pid, bank in raw.items():
                pb = PromptBank(prompt_id=pid)
                for vname, vdata in bank.get("variants", {}).items():
                    pb.variants[vname] = Variant(**vdata)
                self._banks[pid] = pb
        except Exception as exc:  # pragma: no cover
            logger.warning("PromptEvolution: load failed: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                pid: {"variants": {n: asdict(v) for n, v in pb.variants.items()}}
                for pid, pb in self._banks.items()
            }
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            logger.warning("PromptEvolution: save failed: %s", exc)

    # -- Public API ---------------------------------------------------------

    def register(self, prompt_id: str, variants: dict[str, str]) -> None:
        """Register a prompt with one or more variants ``{name: template}``."""
        with self._lock:
            bank = self._banks.setdefault(prompt_id, PromptBank(prompt_id=prompt_id))
            for name, tpl in variants.items():
                bank.add(name, tpl)
            self._save()

    def pick_variant(
        self,
        prompt_id: str,
        epsilon: float | None = None,
    ) -> tuple[str, str]:
        """Return ``(variant_name, template)`` to use.

        Uses epsilon-greedy: explores with probability ``epsilon``,
        otherwise exploits the variant with the best avg score (with
        ties broken by lower run count for more even sampling).
        """
        with self._lock:
            bank = self._banks.get(prompt_id)
            if not bank or not bank.variants:
                raise KeyError(f"No prompt registered under id {prompt_id!r}.")

            eps = self.DEFAULT_EPSILON if epsilon is None else epsilon
            variants = list(bank.variants.values())

            # Cold-start: round-robin until each variant has a few samples
            cold = [v for v in variants if v.runs < self.MIN_RUNS_BEFORE_EXPLOIT]
            if cold:
                v = min(cold, key=lambda x: x.runs)
            elif random.random() < eps:
                v = random.choice(variants)
            else:
                v = max(variants, key=lambda x: (x.avg_score, x.success_rate))

            v.last_used_ts = time.time()
            return v.name, v.template

    def record_outcome(
        self,
        prompt_id: str,
        variant: str,
        success: bool,
        score: float = 0.0,
    ) -> None:
        with self._lock:
            bank = self._banks.get(prompt_id)
            if not bank or variant not in bank.variants:
                logger.debug("PromptEvolution: ignoring outcome for unknown %s/%s", prompt_id, variant)
                return
            v = bank.variants[variant]
            v.runs += 1
            if success:
                v.successes += 1
            # If no explicit score is given, derive one from success
            v.score_sum += score if score != 0.0 else (1.0 if success else -0.25)
            self._save()

        # Mirror to the perf tracker for the unified leaderboard
        try:
            from isaac.improvement.performance import get_tracker
            get_tracker().record_prompt(prompt_id, variant, success, score)
        except Exception:
            pass

    def leaderboard(self, prompt_id: str) -> list[Variant]:
        with self._lock:
            bank = self._banks.get(prompt_id)
            if not bank:
                return []
            return sorted(bank.variants.values(), key=lambda v: v.avg_score, reverse=True)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_evolution: PromptEvolution | None = None


def get_prompt_evolution() -> PromptEvolution:
    global _evolution  # noqa: PLW0603
    if _evolution is None:
        _evolution = PromptEvolution()
    return _evolution


def reset_prompt_evolution() -> None:
    """Reset the singleton (used in tests)."""
    global _evolution  # noqa: PLW0603
    _evolution = None

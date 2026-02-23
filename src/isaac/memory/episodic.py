"""Episodic memory — short/mid-term storage for conversation and task traces.

Records what happened (success or failure) so the Planner and Reflection
nodes can learn from recent experience within a session.

Backed by an in-memory list for now; may graduate to a persistent store
(SQLite, Redis) when session continuity is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single trace of one cognitive cycle."""

    task: str
    hypothesis: str
    code: str
    result_summary: str
    success: bool
    node: str = ""
    """Which node produced the result (e.g. 'reflection', 'computer_use')."""
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Append-only episodic store scoped to one session.

    Parameters
    ----------
    max_episodes:
        Rolling window — oldest episodes are evicted when the cap is hit.
    """

    def __init__(self, max_episodes: int = 200) -> None:
        self._episodes: list[Episode] = []
        self._max = max_episodes

    # -- write --

    def record(self, episode: Episode) -> None:
        """Persist an episode, evicting the oldest if at capacity."""
        self._episodes.append(episode)
        if len(self._episodes) > self._max:
            self._episodes = self._episodes[-self._max :]
        logger.debug(
            "Episodic memory: recorded %s episode '%s' (total=%d)",
            "successful" if episode.success else "failed",
            episode.task[:60],
            len(self._episodes),
        )

    # -- read --

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the *n* most recent episodes."""
        return self._episodes[-n:]

    def recent_failures(self, n: int = 5) -> list[Episode]:
        """Return the *n* most recent failed episodes."""
        failures = [ep for ep in self._episodes if not ep.success]
        return failures[-n:]

    def recent_successes(self, n: int = 5) -> list[Episode]:
        """Return the *n* most recent successful episodes."""
        successes = [ep for ep in self._episodes if ep.success]
        return successes[-n:]

    def search(self, keyword: str) -> list[Episode]:
        """Naïve keyword search across task descriptions and hypotheses."""
        kw = keyword.lower()
        return [
            ep
            for ep in self._episodes
            if kw in ep.task.lower() or kw in ep.hypothesis.lower()
        ]

    def summarise_recent(self, n: int = 5) -> str:
        """Return a concise text summary of recent episodes for prompt injection."""
        episodes = self.recent(n)
        if not episodes:
            return "No prior episodes."
        lines: list[str] = []
        for i, ep in enumerate(episodes, 1):
            status = "SUCCESS" if ep.success else "FAILURE"
            lines.append(
                f"  {i}. [{status}] {ep.task[:80]} — {ep.result_summary[:120]}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Discard all stored episodes."""
        self._episodes.clear()

    @property
    def size(self) -> int:
        """Number of stored episodes."""
        return len(self._episodes)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_episodic_memory: EpisodicMemory | None = None


def get_episodic_memory() -> EpisodicMemory:
    """Return the session-scoped episodic memory singleton."""
    global _episodic_memory  # noqa: PLW0603
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory


def reset_episodic_memory() -> None:
    """Reset the singleton (used in tests)."""
    global _episodic_memory  # noqa: PLW0603
    _episodic_memory = None

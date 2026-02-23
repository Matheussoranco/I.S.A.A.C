"""Episodic memory — short/mid-term storage for conversation and task traces.

Backed by an in-memory list for now; may graduate to a persistent store
(SQLite, Redis) when session continuity is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Episode:
    """A single trace of one cognitive cycle."""

    task: str
    hypothesis: str
    code: str
    result_summary: str
    success: bool
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

    # -- read --

    def recent(self, n: int = 10) -> list[Episode]:
        """Return the *n* most recent episodes."""
        return self._episodes[-n:]

    def search(self, keyword: str) -> list[Episode]:
        """Naïve keyword search across task descriptions and hypotheses."""
        kw = keyword.lower()
        return [
            ep
            for ep in self._episodes
            if kw in ep.task.lower() or kw in ep.hypothesis.lower()
        ]

    @property
    def size(self) -> int:
        return len(self._episodes)

    def clear(self) -> None:
        self._episodes.clear()

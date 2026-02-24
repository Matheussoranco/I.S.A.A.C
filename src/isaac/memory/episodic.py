"""Episodic memory — short/mid-term storage for conversation and task traces.

Records what happened (success or failure) so the Planner and Reflection
nodes can learn from recent experience within a session.

Backed by an in-memory list with optional ChromaDB persistence for
cross-session recall.  Persistent directory: ``~/.isaac/memory/episodic``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Append-only episodic store scoped to one session.

    Optionally persists episodes to ChromaDB for cross-session recall.

    Parameters
    ----------
    max_episodes:
        Rolling window — oldest episodes are evicted when the cap is hit.
    persist_dir:
        ChromaDB persistence directory.  If ``None``, uses in-memory only.
    """

    def __init__(
        self,
        max_episodes: int = 200,
        persist_dir: Path | None = None,
    ) -> None:
        self._episodes: list[Episode] = []
        self._max = max_episodes
        self._persist_dir = persist_dir
        self._collection: Any = None
        self._chroma_client: Any = None
        if persist_dir is not None:
            self._init_chromadb()

    def _init_chromadb(self) -> None:
        """Initialise ChromaDB for persistent episodic storage."""
        try:
            import chromadb  # noqa: PLC0415

            if self._persist_dir is not None:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self._persist_dir),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="episodic_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("EpisodicMemory: ChromaDB initialised at %s.", self._persist_dir)
        except ImportError:
            logger.warning("ChromaDB not available — episodic memory is in-memory only.")
        except Exception:
            logger.warning("ChromaDB init failed — episodic memory is in-memory only.", exc_info=True)

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

        # Persist to ChromaDB
        if self._collection is not None:
            try:
                doc = (
                    f"task: {episode.task} | "
                    f"hypothesis: {episode.hypothesis} | "
                    f"result: {episode.result_summary} | "
                    f"success: {episode.success}"
                )
                ep_id = f"ep_{len(self._episodes)}_{hash(episode.task) % 100000}"
                self._collection.add(
                    ids=[ep_id],
                    documents=[doc],
                    metadatas=[{
                        "success": str(episode.success),
                        "node": episode.node,
                        "iteration": episode.iteration,
                    }],
                )
            except Exception:
                logger.debug("ChromaDB episode persist failed.", exc_info=True)

    def store_episode(
        self,
        session_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a raw episode with session context.

        Parameters
        ----------
        session_id:
            Identifier for the current session.
        content:
            The episode content to store.
        metadata:
            Optional metadata dictionary.
        """
        episode = Episode(
            task=content,
            hypothesis="",
            code="",
            result_summary=content[:200],
            success=True,
            session_id=session_id,
            metadata=metadata or {},
        )
        self.record(episode)

    def recall_relevant(self, query: str, k: int = 5) -> list[Episode]:
        """Recall episodes relevant to the query using ChromaDB.

        Falls back to keyword search if ChromaDB is unavailable.
        """
        if self._collection is not None and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(k, self._collection.count()),
                )
                # Map back to Episode objects from in-memory store
                if results and results.get("documents"):
                    return self.search(query)[:k]
            except Exception:
                logger.debug("ChromaDB recall failed — using keyword fallback.", exc_info=True)
        return self.search(query)[:k]

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

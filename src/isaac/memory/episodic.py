"""Episodic memory — short/mid-term storage for conversation and task traces.

Records what happened (success or failure) so the Planner and Reflection
nodes can learn from recent experience within a session.

Backed by an in-memory list with optional ChromaDB persistence for
cross-session recall.  Persistent directory: ``~/.isaac/memory/episodic``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
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
            import chromadb

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
            logger.warning(
                "ChromaDB init failed — episodic memory is in-memory only.", exc_info=True
            )

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
                now = time.time()
                doc = (
                    f"task: {episode.task} | "
                    f"hypothesis: {episode.hypothesis} | "
                    f"result: {episode.result_summary} | "
                    f"success: {episode.success}"
                )
                payload = json.dumps(
                    {
                        "task": episode.task,
                        "hypothesis": episode.hypothesis,
                        "code": episode.code,
                        "result_summary": episode.result_summary,
                        "success": episode.success,
                        "node": episode.node,
                        "iteration": episode.iteration,
                        "session_id": episode.session_id,
                        "metadata": episode.metadata,
                        "recorded_at": now,
                    },
                    default=str,
                )
                self._collection.add(
                    ids=[f"ep_{uuid.uuid4().hex}"],
                    documents=[doc],
                    metadatas=[
                        {
                            "success": str(episode.success),
                            "node": episode.node,
                            "iteration": episode.iteration,
                            "session_id": episode.session_id,
                            "recorded_at": now,
                            "payload": payload,
                        }
                    ],
                )
                if self._collection.count() > self._max * 2:
                    self._prune_persistent()
            except Exception:
                logger.debug("ChromaDB episode persist failed.", exc_info=True)

    def _prune_persistent(self) -> None:
        """Drop the oldest persisted episodes once the collection overgrows."""
        if self._collection is None:
            return
        data = self._collection.get(include=["metadatas"])
        rows = list(zip(data.get("ids") or [], data.get("metadatas") or [], strict=False))
        excess = len(rows) - self._max
        if excess <= 0:
            return
        rows.sort(key=lambda row: float((row[1] or {}).get("recorded_at", 0.0)))
        stale = [ep_id for ep_id, _ in rows[:excess]]
        if stale:
            self._collection.delete(ids=stale)

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
        k = max(1, k)
        if self._collection is not None and self._collection.count() > 0:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(k, self._collection.count()),
                )
                metadatas = (results or {}).get("metadatas") or []
                episodes = [
                    episode
                    for meta in (metadatas[0] if metadatas else [])
                    if isinstance(meta, dict)
                    and (episode := self._decode(meta.get("payload"))) is not None
                ]
                if episodes:
                    return episodes[:k]
            except Exception:
                logger.debug("ChromaDB recall failed — using keyword fallback.", exc_info=True)
        return self.search(query)[:k]

    @staticmethod
    def _decode(payload: Any) -> Episode | None:
        if not isinstance(payload, str) or not payload:
            return None
        try:
            data = json.loads(payload)
            return Episode(
                task=str(data.get("task", "")),
                hypothesis=str(data.get("hypothesis", "")),
                code=str(data.get("code", "")),
                result_summary=str(data.get("result_summary", "")),
                success=bool(data.get("success", False)),
                node=str(data.get("node", "")),
                iteration=int(data.get("iteration", 0)),
                session_id=str(data.get("session_id", "")),
                metadata=dict(data.get("metadata") or {}),
            )
        except (TypeError, ValueError):
            logger.debug("Ignoring malformed persisted episode.", exc_info=True)
            return None

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
        return [ep for ep in self._episodes if kw in ep.task.lower() or kw in ep.hypothesis.lower()]

    def summarise_recent(self, n: int = 5, session_id: str = "") -> str:
        """Return a concise text summary of recent episodes for prompt injection.

        Parameters
        ----------
        n:
            Maximum number of episodes to include.
        session_id:
            When provided, only episodes belonging to this session are considered.
            This prevents cross-session context bleed in concurrent server deployments.
        """
        if session_id:
            pool = [ep for ep in self._episodes if ep.session_id == session_id]
        else:
            pool = self._episodes
        return self._format(pool[-n:])

    def recall_context(self, query: str, k: int = 5, session_id: str = "") -> str:
        """Summarise query-relevant episodes, falling back to recent history."""
        relevant = self.recall_relevant(query, k)
        return self._format(relevant) if relevant else self.summarise_recent(k, session_id)

    @staticmethod
    def _format(episodes: list[Episode]) -> str:
        if not episodes:
            return "No prior episodes."
        lines: list[str] = []
        for i, ep in enumerate(episodes, 1):
            status = "SUCCESS" if ep.success else "FAILURE"
            lines.append(f"  {i}. [{status}] {ep.task[:80]} — {ep.result_summary[:120]}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Discard all stored episodes, including persisted ones."""
        self._episodes.clear()
        if self._collection is not None:
            try:
                ids = self._collection.get(include=[]).get("ids") or []
                if ids:
                    self._collection.delete(ids=ids)
            except Exception:
                logger.debug("ChromaDB clear failed.", exc_info=True)

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
    global _episodic_memory
    if _episodic_memory is None:
        from isaac.config.settings import get_settings

        persist_dir = get_settings().isaac_home / "memory" / "episodic"
        _episodic_memory = EpisodicMemory(persist_dir=persist_dir)
    return _episodic_memory


def reset_episodic_memory() -> None:
    """Reset the singleton (used in tests)."""
    global _episodic_memory
    _episodic_memory = None

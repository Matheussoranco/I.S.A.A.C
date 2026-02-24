"""Memory Manager — unified interface over episodic, semantic, and procedural memory.

Orchestrates the three memory layers, deciding what to store where and
providing a unified ``recall()`` interface used by Perception and Reflection
nodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Unified recall result combining all memory layers."""

    episodic_context: str = ""
    """Recent experience summaries from episodic memory."""
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    """Relevant facts from the knowledge graph."""
    relevant_skills: list[str] = field(default_factory=list)
    """Skill names matching the query."""
    combined_context: str = ""
    """Formatted string ready for LLM prompt injection."""


class MemoryManager:
    """Unified memory orchestrator across all three layers.

    Provides a single ``recall()`` method that queries all memory layers
    and returns a combined context for LLM prompts.

    Parameters
    ----------
    isaac_home:
        Root directory for Isaac persistent data.
    skills_dir:
        Directory for the skill library.
    """

    def __init__(
        self,
        isaac_home: Path | None = None,
        skills_dir: Path | None = None,
    ) -> None:
        from isaac.config.settings import settings

        self._isaac_home = isaac_home or settings.isaac_home
        self._skills_dir = skills_dir or settings.skills_dir

        # Lazy-init memory layers
        self._episodic: Any = None
        self._semantic: Any = None
        self._procedural: Any = None

    # -- Lazy access --------------------------------------------------------

    @property
    def episodic(self) -> Any:
        """Episodic memory (session-scoped experiences)."""
        if self._episodic is None:
            from isaac.memory.episodic import get_episodic_memory
            self._episodic = get_episodic_memory()
        return self._episodic

    @property
    def semantic(self) -> Any:
        """Semantic memory (knowledge graph)."""
        if self._semantic is None:
            from isaac.memory.semantic import SemanticMemory
            db_path = self._isaac_home / "memory" / "semantic.db"
            self._semantic = SemanticMemory(db_path=db_path)
        return self._semantic

    @property
    def procedural(self) -> Any:
        """Procedural memory (versioned skill library)."""
        if self._procedural is None:
            from isaac.memory.procedural import ProceduralMemory
            self._procedural = ProceduralMemory(skills_dir=self._skills_dir)
        return self._procedural

    # -- Unified recall -----------------------------------------------------

    def recall(self, query: str, k: int = 5) -> RecallResult:
        """Query all memory layers and return unified context.

        Parameters
        ----------
        query:
            Natural language query to search across all layers.
        k:
            Maximum number of results per layer.

        Returns
        -------
        RecallResult
            Combined context from all memory layers.
        """
        result = RecallResult()

        # Episodic: recent experience
        try:
            result.episodic_context = self.episodic.summarise_recent(k)
        except Exception:
            logger.warning("MemoryManager: episodic recall failed.", exc_info=True)
            result.episodic_context = "No episodic context available."

        # Semantic: knowledge graph facts
        try:
            words = query.lower().split()
            all_facts: list[dict[str, Any]] = []
            for word in words[:5]:  # Query first 5 words as entities
                facts = self.semantic.query_facts(subject=word)
                all_facts.extend(f.to_dict() for f in facts[:k])
                facts_obj = self.semantic.query_facts(object=word)
                all_facts.extend(f.to_dict() for f in facts_obj[:k])
            # Deduplicate
            seen: set[str] = set()
            unique_facts: list[dict[str, Any]] = []
            for f in all_facts:
                key = f"{f['subject']}_{f['predicate']}_{f['object']}"
                if key not in seen:
                    seen.add(key)
                    unique_facts.append(f)
            result.semantic_facts = unique_facts[:k]
        except Exception:
            logger.warning("MemoryManager: semantic recall failed.", exc_info=True)

        # Procedural: relevant skills
        try:
            result.relevant_skills = self.procedural.search(query, top_k=k)
        except Exception:
            logger.warning("MemoryManager: procedural recall failed.", exc_info=True)

        # Build combined context string
        parts: list[str] = []
        if result.episodic_context and result.episodic_context != "No prior episodes.":
            parts.append(f"## Recent Experience\n{result.episodic_context}")
        if result.semantic_facts:
            facts_str = "\n".join(
                f"  ({f['subject']}) --[{f['predicate']}]--> ({f['object']})"
                for f in result.semantic_facts
            )
            parts.append(f"## Known Facts\n{facts_str}")
        if result.relevant_skills:
            parts.append(f"## Relevant Skills\n  {', '.join(result.relevant_skills)}")

        result.combined_context = "\n\n".join(parts) if parts else "No prior memory context."
        return result

    # -- Storage helpers ----------------------------------------------------

    def store_episode(
        self,
        task: str,
        hypothesis: str,
        code: str,
        result_summary: str,
        success: bool,
        node: str = "",
        iteration: int = 0,
    ) -> None:
        """Store an episode in episodic memory."""
        from isaac.memory.episodic import Episode
        self.episodic.record(Episode(
            task=task,
            hypothesis=hypothesis,
            code=code,
            result_summary=result_summary,
            success=success,
            node=node,
            iteration=iteration,
        ))

    def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """Store a fact in semantic memory."""
        self.semantic.add_fact(subject, predicate, object, confidence, source)

    def close(self) -> None:
        """Close all memory layer connections."""
        if self._semantic is not None:
            self._semantic.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """Return the session-scoped memory manager singleton."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = MemoryManager()
    return _manager


def reset_memory_manager() -> None:
    """Reset the singleton (used in tests)."""
    global _manager  # noqa: PLW0603
    if _manager is not None:
        _manager.close()
    _manager = None

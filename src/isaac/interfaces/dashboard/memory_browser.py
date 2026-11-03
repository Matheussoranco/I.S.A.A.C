"""Interface to search and inspect Isaac memory layers.

Queries Episodic, Semantic, and Procedural memory.
"""

from __future__ import annotations

from typing import Any

from isaac.memory.manager import get_memory_manager


def search_memory(query: str) -> dict[str, Any]:
    """Perform a unified recall across all memory layers."""
    result = get_memory_manager().recall(query)
    return {
        "episodic": result.episodic_context,
        "semantic": result.semantic_facts,
        "procedural": result.relevant_skills,
        "combined": result.combined_context,
    }


def inspect_semantic_graph() -> dict:
    """Return a snapshot of the semantic knowledge graph."""
    return get_memory_manager().semantic.graph_snapshot()

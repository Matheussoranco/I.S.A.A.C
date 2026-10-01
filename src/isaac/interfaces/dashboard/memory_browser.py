"""Interface to search and inspect Isaac memory layers.
Queries Episodic, Semantic, and Procedural memory.
"""
from __future__ import annotations
from typing import Any
from fastapi import HTTPException
from isaac.memory.manager import MemoryManager
from isaac.config.settings import settings
from pathlib import Path

# Instance shared for the API session
_manager = MemoryManager(isaac_home=settings.isaac_home, skills_dir=settings.skills_dir)

def search_memory(query: str) -> dict[str, Any]:
    """Perform a unified recall across all memory layers."""
    try:
        result = _manager.recall(query)
        return {
            "episodic": result.episodic_context,
            "semantic": result.semantic_facts,
            "procedural": result.relevant_skills,
            "combined": result.combined_context
        }
    except Exception as e:
        raise HTTPException(500, f"Memory search failed: {e}")

def inspect_semantic_graph() -> dict:
    """Returns a snapshot of the semantic knowledge graph."""
    try:
        # Access internal semantic layer
        sem = _manager._semantic
        if not sem:
            raise HTTPException(503, "Semantic memory not initialized")
        
        # Return edges as a list of triples
        edges = []
        for u, v, data in sem._graph.edges(data=True):
            edges.append({"subject": u, "object": v, "predicate": data.get("predicate", "related")})
        return {"nodes": list(sem._graph.nodes), "edges": edges}
    except Exception as e:
        raise HTTPException(500, f"Graph inspection failed: {e}")

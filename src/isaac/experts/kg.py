"""KGExpert — answers entity/relation queries from the WorldModelKG.

The agent's working knowledge graph (built incrementally by Perception and
Explorer) is queried for nodes, neighbours, and shortest paths. When a query
mentions an entity that exists in the KG, this expert returns structured
facts; otherwise it abstains.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from isaac.experts.base import Expert, ExpertNotApplicable, ExpertResponse

logger = logging.getLogger(__name__)


class KGExpert(Expert):
    """Symbolic KG-query expert."""

    name: ClassVar[str] = "kg"
    domains: ClassVar[tuple[str, ...]] = ("kg", "knowledge_graph", "facts")
    description: ClassVar[str] = "Queries the WorldModelKG for known entities and relations."
    cost: ClassVar[float] = 0.1

    def can_handle(self, query: str, context: dict[str, Any] | None = None) -> float:
        try:
            from isaac.memory.world_model_kg import get_world_model_kg

            kg = get_world_model_kg()
        except Exception:
            return 0.0
        if kg.node_count == 0:
            return 0.0

        q = query.lower()
        # Match against known node labels
        labels = [str(d.get("label", "")).lower() for _, d in kg._graph.nodes(data=True)]  # type: ignore[attr-defined]
        for label in labels:
            if label and label in q:
                return 0.8
        # Relational phrasing
        if any(
            p in q
            for p in (
                " related to ",
                " connected to ",
                " between ",
                " who knows ",
                " path from ",
                "neighbours of",
                "neighbors of",
            )
        ):
            return 0.6
        return 0.0

    def _answer(self, query: str, context: dict[str, Any]) -> ExpertResponse:
        from isaac.memory.world_model_kg import get_world_model_kg

        kg = get_world_model_kg()
        if kg.node_count == 0:
            raise ExpertNotApplicable("KG is empty")

        q = query.lower()
        evidence: list[str] = []

        # Find mentioned entities
        mentioned: list[str] = []
        for node_id, data in kg._graph.nodes(data=True):  # type: ignore[attr-defined]
            label = str(data.get("label", "")).lower()
            if label and label in q:
                mentioned.append(node_id)

        if not mentioned:
            raise ExpertNotApplicable("no KG entities mentioned")

        # Path query
        m = re.search(r"path\s+from\s+(\w+)\s+to\s+(\w+)", q)
        if m and len(mentioned) >= 2:
            src, tgt = mentioned[0], mentioned[1]
            path = kg.shortest_path(src, tgt)
            if path:
                return ExpertResponse(
                    expert=self.name,
                    answer="Path: " + " → ".join(path),
                    confidence=0.95,
                    evidence=[f"length={len(path)}"],
                    artifacts={"path": path},
                )

        # Neighbour / facts query
        ent = mentioned[0]
        neighbours = kg.neighbours(ent, direction="out")
        node = kg.get_node(ent)
        facts = []
        if node:
            evidence.append(f"node={ent} kind={node.get('kind', '?')}")
        for n in neighbours[:10]:
            edge_data = kg._graph.get_edge_data(ent, n) or {}  # type: ignore[attr-defined]
            rel = edge_data.get("relation", "related_to")
            facts.append(f"{ent} --{rel}--> {n}")

        if not facts and not node:
            raise ExpertNotApplicable("no facts found")

        answer = (f"Facts about {ent}:\n" + "\n".join(facts)) if facts else f"Known: {node}"
        return ExpertResponse(
            expert=self.name,
            answer=answer,
            confidence=0.85,
            evidence=evidence,
            artifacts={"entity": ent, "neighbours": neighbours, "node": node},
        )

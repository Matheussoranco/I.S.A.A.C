"""World Model Knowledge Graph — NetworkX DiGraph overlay on the WorldModel.

Extends the flat WorldModel dataclass with a rich knowledge graph that
tracks entities, relations, and observations as nodes/edges.  This lets
the Planner perform graph queries (shortest path, subgraph extraction,
community detection) when reasoning about complex tasks.

The KG is built incrementally by Perception and Explorer nodes and
queried by the Planner and Reflection nodes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from isaac.core.state import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    """A node in the world-model knowledge graph."""

    id: str
    label: str
    kind: str = "entity"
    """One of 'entity', 'observation', 'file', 'resource', 'constraint'."""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGEdge:
    """A directed edge in the knowledge graph."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


class WorldModelKG:
    """Knowledge-graph layer on top of the flat :class:`WorldModel`.

    Internally uses a ``networkx.DiGraph``.  Persists to SQLite at
    ``~/.isaac/memory/world_model_kg.db``.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._graph = nx.DiGraph()
        self._db_path: Path | None = None

        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = persist_dir / "world_model_kg.db"
            self._init_db()
            self._load_from_db()

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_node(self, node: KGNode) -> None:
        """Add or update a node."""
        self._graph.add_node(
            node.id,
            label=node.label,
            kind=node.kind,
            **node.properties,
        )
        self._persist_node(node)

    def add_edge(self, edge: KGEdge) -> None:
        """Add or update a directed edge."""
        self._graph.add_edge(
            edge.source,
            edge.target,
            relation=edge.relation,
            weight=edge.weight,
            **edge.properties,
        )
        self._persist_edge(edge)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        if node_id in self._graph:
            self._graph.remove_node(node_id)
            self._delete_node_db(node_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return node attributes or ``None``."""
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None

    def neighbours(self, node_id: str, direction: str = "both") -> list[str]:
        """Return neighbour IDs.

        Parameters
        ----------
        direction:
            ``"out"`` = successors, ``"in"`` = predecessors, ``"both"`` = union.
        """
        if node_id not in self._graph:
            return []
        if direction == "out":
            return list(self._graph.successors(node_id))
        elif direction == "in":
            return list(self._graph.predecessors(node_id))
        else:
            return list(set(self._graph.successors(node_id)) | set(self._graph.predecessors(node_id)))

    def subgraph(self, node_ids: list[str]) -> nx.DiGraph:
        """Return the induced subgraph for the given nodes."""
        return self._graph.subgraph(node_ids).copy()

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Shortest path between two nodes, or empty list if unreachable."""
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_by_kind(self, kind: str) -> list[str]:
        """Return all node IDs of a given kind."""
        return [
            n for n, d in self._graph.nodes(data=True)
            if d.get("kind") == kind
        ]

    def to_context_string(self, max_nodes: int = 50) -> str:
        """Serialise the KG into a compact text block for LLM prompts."""
        lines: list[str] = []
        nodes_list = list(self._graph.nodes(data=True))[:max_nodes]
        for node_id, data in nodes_list:
            label = data.get("label", node_id)
            kind = data.get("kind", "?")
            lines.append(f"  [{kind}] {node_id}: {label}")

        edges_list = list(self._graph.edges(data=True))[:max_nodes * 2]
        for src, tgt, data in edges_list:
            rel = data.get("relation", "->")
            lines.append(f"  {src} --{rel}--> {tgt}")

        return f"WorldModel KG ({self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges):\n" + "\n".join(lines)

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Sync with flat WorldModel
    # ------------------------------------------------------------------

    def sync_from_world_model(self, wm: WorldModel) -> None:
        """Import data from the flat WorldModel into the KG."""
        # Files
        for path, summary in wm.files.items():
            self.add_node(KGNode(id=f"file:{path}", label=path, kind="file", properties={"summary": summary}))

        # Resources
        for key, value in wm.resources.items():
            self.add_node(KGNode(id=f"resource:{key}", label=key, kind="resource", properties={"value": str(value)[:200]}))

        # Constraints
        for i, constraint in enumerate(wm.constraints):
            self.add_node(KGNode(id=f"constraint:{i}", label=constraint, kind="constraint"))

        # Observations
        for i, obs in enumerate(wm.observations[-20:]):
            self.add_node(KGNode(id=f"obs:{i}", label=obs[:150], kind="observation"))

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(str(self._db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT,
                kind TEXT,
                properties TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                relation TEXT,
                weight REAL DEFAULT 1.0,
                properties TEXT,
                PRIMARY KEY (source, target, relation)
            );
        """)
        conn.close()

    def _load_from_db(self) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(str(self._db_path))
        try:
            for row in conn.execute("SELECT id, label, kind, properties FROM nodes"):
                props = json.loads(row[3]) if row[3] else {}
                self._graph.add_node(row[0], label=row[1], kind=row[2], **props)
            for row in conn.execute("SELECT source, target, relation, weight, properties FROM edges"):
                props = json.loads(row[4]) if row[4] else {}
                self._graph.add_edge(row[0], row[1], relation=row[2], weight=row[3], **props)
        finally:
            conn.close()

    def _persist_node(self, node: KGNode) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO nodes (id, label, kind, properties) VALUES (?, ?, ?, ?)",
                (node.id, node.label, node.kind, json.dumps(node.properties)),
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_edge(self, edge: KGEdge) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                "INSERT OR REPLACE INTO edges (source, target, relation, weight, properties) VALUES (?, ?, ?, ?, ?)",
                (edge.source, edge.target, edge.relation, edge.weight, json.dumps(edge.properties)),
            )
            conn.commit()
        finally:
            conn.close()

    def _delete_node_db(self, node_id: str) -> None:
        if self._db_path is None:
            return
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: WorldModelKG | None = None


def get_world_model_kg() -> WorldModelKG:
    """Return the singleton WorldModelKG instance."""
    global _instance
    if _instance is None:
        try:
            from isaac.config.settings import get_settings
            persist_dir = get_settings().isaac_home / "memory"
        except Exception:
            persist_dir = None
        _instance = WorldModelKG(persist_dir=persist_dir)
    return _instance


def reset_world_model_kg() -> None:
    """Reset singleton — for testing."""
    global _instance
    _instance = None

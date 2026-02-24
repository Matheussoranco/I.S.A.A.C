"""Tests for the World Model Knowledge Graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.memory.world_model_kg import KGEdge, KGNode, WorldModelKG
from isaac.core.state import WorldModel


@pytest.fixture()
def kg(tmp_path: Path) -> WorldModelKG:
    return WorldModelKG(persist_dir=tmp_path)


@pytest.fixture()
def kg_no_persist() -> WorldModelKG:
    return WorldModelKG(persist_dir=None)


class TestWorldModelKG:
    """Unit tests for WorldModelKG."""

    def test_add_and_get_node(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="n1", label="Node 1", kind="entity"))
        result = kg.get_node("n1")
        assert result is not None
        assert result["label"] == "Node 1"
        assert result["kind"] == "entity"

    def test_add_and_query_edge(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="a", label="A"))
        kg.add_node(KGNode(id="b", label="B"))
        kg.add_edge(KGEdge(source="a", target="b", relation="depends_on"))
        neighbours = kg.neighbours("a", direction="out")
        assert "b" in neighbours

    def test_remove_node(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="x", label="X"))
        kg.remove_node("x")
        assert kg.get_node("x") is None

    def test_neighbours_both(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="a", label="A"))
        kg.add_node(KGNode(id="b", label="B"))
        kg.add_node(KGNode(id="c", label="C"))
        kg.add_edge(KGEdge(source="a", target="b", relation="r1"))
        kg.add_edge(KGEdge(source="c", target="a", relation="r2"))
        both = kg.neighbours("a", direction="both")
        assert "b" in both
        assert "c" in both

    def test_shortest_path(self, kg: WorldModelKG) -> None:
        for i in range(4):
            kg.add_node(KGNode(id=str(i), label=str(i)))
        kg.add_edge(KGEdge(source="0", target="1", relation="->"))
        kg.add_edge(KGEdge(source="1", target="2", relation="->"))
        kg.add_edge(KGEdge(source="2", target="3", relation="->"))
        path = kg.shortest_path("0", "3")
        assert path == ["0", "1", "2", "3"]

    def test_shortest_path_unreachable(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="a", label="A"))
        kg.add_node(KGNode(id="b", label="B"))
        path = kg.shortest_path("a", "b")
        assert path == []

    def test_find_by_kind(self, kg: WorldModelKG) -> None:
        kg.add_node(KGNode(id="f1", label="file.py", kind="file"))
        kg.add_node(KGNode(id="f2", label="data.csv", kind="file"))
        kg.add_node(KGNode(id="e1", label="Entity", kind="entity"))
        files = kg.find_by_kind("file")
        assert len(files) == 2
        assert "f1" in files
        assert "f2" in files

    def test_persistence(self, tmp_path: Path) -> None:
        kg1 = WorldModelKG(persist_dir=tmp_path)
        kg1.add_node(KGNode(id="p1", label="Persisted"))
        del kg1

        kg2 = WorldModelKG(persist_dir=tmp_path)
        assert kg2.get_node("p1") is not None

    def test_node_count_edge_count(self, kg: WorldModelKG) -> None:
        assert kg.node_count == 0
        kg.add_node(KGNode(id="a", label="A"))
        kg.add_node(KGNode(id="b", label="B"))
        assert kg.node_count == 2
        kg.add_edge(KGEdge(source="a", target="b", relation="r"))
        assert kg.edge_count == 1

    def test_sync_from_world_model(self, kg_no_persist: WorldModelKG) -> None:
        wm = WorldModel(
            files={"main.py": "entry point"},
            resources={"cpu": "4 cores"},
            constraints=["no internet"],
            observations=["test observation"],
        )
        kg_no_persist.sync_from_world_model(wm)
        assert kg_no_persist.node_count > 0
        files = kg_no_persist.find_by_kind("file")
        assert len(files) == 1

    def test_to_context_string(self, kg_no_persist: WorldModelKG) -> None:
        kg_no_persist.add_node(KGNode(id="a", label="Alpha"))
        kg_no_persist.add_node(KGNode(id="b", label="Beta"))
        kg_no_persist.add_edge(KGEdge(source="a", target="b", relation="links"))
        ctx = kg_no_persist.to_context_string()
        assert "WorldModel KG" in ctx
        assert "2 nodes" in ctx

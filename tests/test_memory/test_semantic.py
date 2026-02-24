"""Tests for the Semantic Memory knowledge graph."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from isaac.memory.semantic import Fact, SemanticMemory


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "semantic.db"


@pytest.fixture()
def memory(tmp_db: Path) -> SemanticMemory:
    return SemanticMemory(db_path=tmp_db)


class TestSemanticMemory:
    """Unit tests for the SemanticMemory class."""

    def test_add_and_query_fact(self, memory: SemanticMemory) -> None:
        memory.add_fact("python", "is_a", "programming_language")
        facts = memory.query_facts(subject="python")
        assert len(facts) >= 1
        assert any(f.object == "programming_language" for f in facts)

    def test_query_by_object(self, memory: SemanticMemory) -> None:
        memory.add_fact("cat", "is_a", "animal")
        memory.add_fact("dog", "is_a", "animal")
        facts = memory.query_facts(object="animal")
        assert len(facts) >= 2

    def test_query_by_predicate(self, memory: SemanticMemory) -> None:
        memory.add_fact("A", "depends_on", "B")
        memory.add_fact("C", "depends_on", "D")
        facts = memory.query_facts(predicate="depends_on")
        assert len(facts) >= 2

    def test_confidence_stored(self, memory: SemanticMemory) -> None:
        memory.add_fact("X", "has_property", "Y", confidence=0.75)
        facts = memory.query_facts(subject="X")
        assert facts[0].confidence == pytest.approx(0.75)

    def test_persistence(self, tmp_db: Path) -> None:
        mem1 = SemanticMemory(db_path=tmp_db)
        mem1.add_fact("earth", "orbits", "sun")
        del mem1

        mem2 = SemanticMemory(db_path=tmp_db)
        facts = mem2.query_facts(subject="earth")
        assert len(facts) == 1
        assert facts[0].object == "sun"

    def test_upsert_replaces(self, memory: SemanticMemory) -> None:
        memory.add_fact("A", "is", "B", confidence=0.5)
        memory.add_fact("A", "is", "B", confidence=0.9)
        facts = memory.query_facts(subject="A", predicate="is", object="B")
        assert len(facts) == 1
        assert facts[0].confidence == pytest.approx(0.9)

    def test_infer_transitive(self, memory: SemanticMemory) -> None:
        memory.add_fact("A", "is_a", "B")
        memory.add_fact("B", "is_a", "C")
        inferred = memory.infer_transitive("A", "is_a")
        # Should find C through transitive inference
        assert any(f.object == "C" for f in inferred)


class TestFact:
    """Tests for the Fact dataclass."""

    def test_to_dict(self) -> None:
        fact = Fact(subject="X", predicate="is", object="Y", confidence=0.8, timestamp="2025-01-01", source="test")
        d = fact.to_dict()
        assert d["subject"] == "X"
        assert d["predicate"] == "is"
        assert d["object"] == "Y"
        assert d["confidence"] == 0.8

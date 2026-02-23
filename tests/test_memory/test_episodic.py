"""Tests for the episodic memory subsystem."""

from __future__ import annotations

from isaac.memory.episodic import Episode, EpisodicMemory


class TestEpisodicMemory:
    def test_record_and_recent(self) -> None:
        mem = EpisodicMemory(max_episodes=5)
        for i in range(3):
            mem.record(Episode(
                task=f"task-{i}",
                hypothesis=f"hyp-{i}",
                code=f"code-{i}",
                result_summary=f"result-{i}",
                success=i % 2 == 0,
            ))
        assert mem.size == 3
        recent = mem.recent(2)
        assert len(recent) == 2
        assert recent[-1].task == "task-2"

    def test_eviction(self) -> None:
        mem = EpisodicMemory(max_episodes=3)
        for i in range(5):
            mem.record(Episode(
                task=f"task-{i}", hypothesis="", code="", result_summary="", success=True
            ))
        assert mem.size == 3
        assert mem.recent(1)[0].task == "task-4"

    def test_search(self) -> None:
        mem = EpisodicMemory()
        mem.record(Episode(
            task="sort an array", hypothesis="use quicksort", code="",
            result_summary="", success=True
        ))
        mem.record(Episode(
            task="reverse a string", hypothesis="slice", code="", result_summary="", success=True
        ))
        results = mem.search("array")
        assert len(results) == 1
        assert results[0].task == "sort an array"

    def test_clear(self) -> None:
        mem = EpisodicMemory()
        mem.record(Episode(task="x", hypothesis="", code="", result_summary="", success=True))
        mem.clear()
        assert mem.size == 0

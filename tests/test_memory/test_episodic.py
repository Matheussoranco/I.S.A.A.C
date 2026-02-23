"""Tests for the episodic memory subsystem."""

from __future__ import annotations

from isaac.memory.episodic import (
    Episode,
    EpisodicMemory,
    get_episodic_memory,
    reset_episodic_memory,
)


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

    # --- New method tests ---

    def test_recent_failures(self) -> None:
        mem = EpisodicMemory()
        mem.record(Episode(task="ok", hypothesis="", code="", result_summary="", success=True))
        mem.record(Episode(task="fail-1", hypothesis="", code="", result_summary="", success=False))
        mem.record(Episode(task="ok2", hypothesis="", code="", result_summary="", success=True))
        mem.record(Episode(task="fail-2", hypothesis="", code="", result_summary="", success=False))
        failures = mem.recent_failures(5)
        assert len(failures) == 2
        assert failures[0].task == "fail-1"
        assert failures[1].task == "fail-2"

    def test_recent_successes(self) -> None:
        mem = EpisodicMemory()
        mem.record(Episode(task="ok-1", hypothesis="", code="", result_summary="", success=True))
        mem.record(Episode(task="fail", hypothesis="", code="", result_summary="", success=False))
        mem.record(Episode(task="ok-2", hypothesis="", code="", result_summary="", success=True))
        successes = mem.recent_successes(5)
        assert len(successes) == 2
        assert successes[0].task == "ok-1"

    def test_recent_failures_respects_limit(self) -> None:
        mem = EpisodicMemory()
        for i in range(10):
            mem.record(Episode(
                task=f"fail-{i}", hypothesis="", code="", result_summary="", success=False,
            ))
        assert len(mem.recent_failures(3)) == 3

    def test_summarise_recent_empty(self) -> None:
        mem = EpisodicMemory()
        assert mem.summarise_recent() == "No prior episodes."

    def test_summarise_recent_format(self) -> None:
        mem = EpisodicMemory()
        mem.record(Episode(
            task="sort numbers", hypothesis="quicksort", code="sorted()",
            result_summary="works", success=True,
        ))
        mem.record(Episode(
            task="reverse string", hypothesis="slice", code="s[::-1]",
            result_summary="crashed", success=False,
        ))
        text = mem.summarise_recent(5)
        assert "[SUCCESS]" in text
        assert "[FAILURE]" in text
        assert "sort numbers" in text
        assert "reverse string" in text

    def test_episode_node_and_iteration_fields(self) -> None:
        ep = Episode(
            task="t", hypothesis="h", code="c", result_summary="r",
            success=True, node="reflection", iteration=3,
        )
        assert ep.node == "reflection"
        assert ep.iteration == 3


class TestEpisodicMemorySingleton:
    def setup_method(self) -> None:
        reset_episodic_memory()

    def teardown_method(self) -> None:
        reset_episodic_memory()

    def test_get_returns_same_instance(self) -> None:
        m1 = get_episodic_memory()
        m2 = get_episodic_memory()
        assert m1 is m2

    def test_reset_creates_new_instance(self) -> None:
        m1 = get_episodic_memory()
        reset_episodic_memory()
        m2 = get_episodic_memory()
        assert m1 is not m2

    def test_reset_clears_data(self) -> None:
        mem = get_episodic_memory()
        mem.record(Episode(task="x", hypothesis="", code="", result_summary="", success=True))
        assert mem.size == 1
        reset_episodic_memory()
        assert get_episodic_memory().size == 0

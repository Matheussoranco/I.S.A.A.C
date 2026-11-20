from __future__ import annotations

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from isaac.agents.agent_loop import AgentLoop
from isaac.memory.episodic import Episode, EpisodicMemory
from isaac.memory.long_term import LongTermMemory
from isaac.specialists.base import SpecialistResult
from isaac.specialists.orchestrator import Orchestrator, SubTask


class Model:
    def invoke(self, messages):
        return AIMessage(content="done")

    def astream(self, messages):
        return Tokens()


class Tokens:
    sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.sent:
            raise StopAsyncIteration
        self.sent = True
        return AIMessage(content="done")


def test_fifty_completed_turns_consolidate_persistent_memory(tmp_path, monkeypatch):
    from isaac.memory import manager

    path = tmp_path / "memory.db"
    monkeypatch.setattr(
        manager,
        "get_settings",
        lambda: SimpleNamespace(
            isaac_home=tmp_path,
            memory_db_path=str(path),
            memory_consolidation_interval=50,
            improvement_enabled=False,
        ),
    )
    from isaac.memory import episodic

    episodes = EpisodicMemory(max_episodes=2)
    monkeypatch.setattr(episodic, "_episodic_memory", episodes)
    episodes.record(Episode("kept", "hypothesis", "", "result", True))
    with LongTermMemory(path) as memory:
        duplicate = memory.remember("duplicate memory", importance=0.3)
        keeper = memory.remember("duplicate memory", importance=0.9)
        weak = memory.remember("weak memory", importance=0.02)
    agent = AgentLoop([], llm=Model())
    for _ in range(49):
        assert agent.run("turn").completed
    with LongTermMemory(path) as memory:
        assert memory.size == 3
    assert not AgentLoop([], llm=Model(), should_stop=lambda: True).run("cancelled").completed
    with LongTermMemory(path) as memory:
        assert memory.size == 3

    assert asyncio.run(agent.arun("turn fifty")).completed
    with LongTermMemory(path) as memory:
        assert memory.get(duplicate) is None
        assert memory.get(weak) is None
        assert memory.get(keeper)["importance"] == 0.9
        assert [row["id"] for row in memory.recall("duplicate")] == [keeper]
        memory.remember("duplicate memory", importance=0.4)
    assert agent.run("turn fifty-one").completed
    with LongTermMemory(path) as memory:
        assert memory.size == 2
    assert episodes.size == 1
    assert episodes.recent(1)[0].task == "kept"


def test_orchestrator_counts_one_turn_not_each_specialist(tmp_path, monkeypatch):
    from isaac.memory import manager

    path = tmp_path / "memory.db"
    monkeypatch.setattr(
        manager,
        "get_settings",
        lambda: SimpleNamespace(
            isaac_home=tmp_path,
            memory_db_path=str(path),
            memory_consolidation_interval=5,
        ),
    )

    class Specialist:
        name = "generalist"

        def run(self, task, context=""):
            result = AgentLoop([], llm=Model()).run(task, context)
            return SpecialistResult(
                specialist=self.name, task=task, output=result.output, success=result.success
            )

    orchestrator = Orchestrator(
        planner=lambda *_: [SubTask(str(i), f"task {i}", "generalist") for i in range(2)],
        specialist_factory=lambda *_a, **_kw: Specialist(),
        manager_llm=Model(),
    )
    with LongTermMemory(path) as memory:
        for _ in range(2):
            memory.remember("duplicate", importance=0.9)
    for _ in range(4):
        assert orchestrator.run("goal").completed
    with LongTermMemory(path) as memory:
        assert memory.size == 2
    assert orchestrator.run("goal").completed
    with LongTermMemory(path) as memory:
        assert memory.size == 1


def test_turn_counter_serializes_connections(tmp_path):
    path = tmp_path / "memory.db"

    def record(_):
        with LongTermMemory(path) as memory:
            return memory.record_turn(5)

    with LongTermMemory(path) as memory:
        memory.remember("duplicate", importance=0.4)
        memory.remember("duplicate", importance=0.9)
    with ThreadPoolExecutor(max_workers=5) as pool:
        completed_passes = list(pool.map(record, range(11)))
    assert sum(completed_passes) == 2
    with LongTermMemory(path) as memory:
        assert memory.size == 1


def test_consolidation_decays_only_stale_unaccessed_weak_entries(tmp_path, monkeypatch):
    from isaac.memory import long_term

    class Clock(datetime):
        current = datetime(2025, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    monkeypatch.setattr(long_term, "datetime", Clock)
    with LongTermMemory(tmp_path / "memory.db") as memory:
        stale = memory.remember("stale", importance=0.4)
        used = memory.remember("used", importance=0.4)
        important = memory.remember("important", importance=0.8)
        memory.recall("used")
        Clock.current = datetime(2025, 2, 15, tzinfo=UTC)
        fresh = memory.remember("fresh", importance=0.4)
        memory.consolidate()
        assert memory.get(stale)["importance"] == pytest.approx(0.38)
        assert memory.get(used)["importance"] == 0.4
        assert memory.get(important)["importance"] == 0.8
        assert memory.get(fresh)["importance"] == 0.4


def test_memory_io_failure_does_not_replace_completed_result(tmp_path, monkeypatch, caplog):
    from isaac.memory import manager

    monkeypatch.setattr(
        manager,
        "get_settings",
        lambda: SimpleNamespace(
            isaac_home=tmp_path,
            memory_db_path=str(tmp_path),
            memory_consolidation_interval=50,
        ),
    )
    assert AgentLoop([], llm=Model()).run("turn").output == "done"
    assert "Could not record completed turn" in caplog.text


@pytest.mark.asyncio
async def test_stream_counts_only_after_exhaustion(tmp_path, monkeypatch):
    from isaac.memory import manager

    path = tmp_path / "memory.db"
    monkeypatch.setattr(
        manager,
        "get_settings",
        lambda: SimpleNamespace(
            isaac_home=tmp_path, memory_db_path=str(path), memory_consolidation_interval=5
        ),
    )
    with LongTermMemory(path) as memory:
        memory.remember("duplicate")
        memory.remember("duplicate")
        for _ in range(4):
            assert not memory.record_turn(5)
    agent = AgentLoop([], llm=Model())
    stream = agent.astream("partial")
    assert await anext(stream) == "done"
    await stream.aclose()
    with LongTermMemory(path) as memory:
        assert memory.size == 2
    assert [chunk async for chunk in agent.astream("complete")] == ["done"]
    with LongTermMemory(path) as memory:
        assert memory.size == 1


@pytest.fixture
def turn_at_threshold(tmp_path, monkeypatch):
    from isaac.memory import manager

    path = tmp_path / "memory.db"
    monkeypatch.setattr(
        manager,
        "get_settings",
        lambda: SimpleNamespace(
            isaac_home=tmp_path,
            memory_db_path=str(path),
            memory_consolidation_interval=5,
        ),
    )
    with LongTermMemory(path) as memory:
        memory.remember("duplicate")
        memory.remember("duplicate")
        for _ in range(4):
            assert not memory.record_turn(5)
    return path


def test_locked_memory_does_not_overrun_agent_wall_budget(turn_at_threshold):
    with closing(sqlite3.connect(turn_at_threshold)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        result = AgentLoop([], llm=Model(), max_wall_seconds=0.25).run("turn")
        assert time.monotonic() - start < 1
        assert result.stopped_reason == "budget_exhausted"
    with LongTermMemory(turn_at_threshold) as memory:
        assert memory.size == 2
        assert memory.record_turn(5)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancelled_turn_cannot_write_after_lock_release(
    turn_at_threshold, monkeypatch, streaming
):
    attempted = Event()
    finished = Event()
    original = LongTermMemory.record_turn

    def record(self, *args, **kwargs):
        attempted.set()
        try:
            return original(self, *args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(LongTermMemory, "record_turn", record)
    with closing(sqlite3.connect(turn_at_threshold)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        agent = AgentLoop([], llm=Model())

        async def run_turn():
            if streaming:
                return [chunk async for chunk in agent.astream("turn")]
            return await agent.arun("turn")

        task = asyncio.create_task(run_turn())
        try:
            assert await asyncio.to_thread(attempted.wait, 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert await asyncio.to_thread(finished.wait, 2)
    with LongTermMemory(turn_at_threshold) as memory:
        assert memory.size == 2
        assert memory.record_turn(5)


def test_transient_memory_lock_does_not_drop_completed_turn(turn_at_threshold, monkeypatch):
    attempted = Event()
    locked = Event()
    original = LongTermMemory.record_turn

    def record(self, *args, **kwargs):
        attempted.set()
        return original(self, *args, **kwargs)

    def hold_lock():
        with closing(sqlite3.connect(turn_at_threshold)) as lock:
            lock.execute("BEGIN IMMEDIATE")
            locked.set()
            if attempted.wait(2):
                time.sleep(0.12)

    monkeypatch.setattr(LongTermMemory, "record_turn", record)
    thread = Thread(target=hold_lock)
    thread.start()
    try:
        assert locked.wait(2)
        assert AgentLoop([], llm=Model(), max_wall_seconds=2).run("turn").completed
    finally:
        thread.join()
    with LongTermMemory(turn_at_threshold) as memory:
        assert memory.size == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_commit_started_before_cancellation_reports_completion(
    turn_at_threshold, monkeypatch, streaming
):
    in_commit = Event()
    release_commit = Event()
    connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)

        def trace(statement):
            if statement == "COMMIT":
                in_commit.set()
                release_commit.wait(3)

        connection.set_trace_callback(trace)
        return connection

    agent = AgentLoop([], llm=Model())

    async def run_turn():
        if streaming:
            return [chunk async for chunk in agent.astream("turn")] == ["done"]
        return (await agent.arun("turn")).completed

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", traced_connect)
        task = asyncio.create_task(run_turn())
        try:
            assert await asyncio.to_thread(in_commit.wait, 2)
            assert task.cancel()
            assert task.cancel()
            await asyncio.sleep(0)
        finally:
            release_commit.set()
        assert await task
        assert task.cancelling() == 0
    with LongTermMemory(turn_at_threshold) as memory:
        assert memory.size == 1

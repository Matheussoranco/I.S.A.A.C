"""Tests for the run-trace store."""

from __future__ import annotations

from isaac.agents.trace import TraceStore


def test_trace_store_records_full_run_lifecycle(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    rid = store.start_run("organise my downloads")
    store.record_event(rid, "iteration", {"n": 1})
    store.record_event(rid, "tool_call", {"name": "fs_list", "args": {"path": "~"}})
    store.record_event(rid, "final", {"text": "done"})
    store.finish_run(rid, stopped_reason="final", iterations=1, output="done")

    runs = store.recent_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == rid
    assert runs[0]["stopped_reason"] == "final"
    assert runs[0]["iterations"] == 1
    assert runs[0]["finished_at"] is not None

    events = store.run_events(rid)
    assert [e["kind"] for e in events] == ["iteration", "tool_call", "final"]
    assert "fs_list" in events[1]["data_json"]


def test_trace_store_handles_unserialisable_event_data(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    rid = store.start_run("t")
    store.record_event(rid, "weird", {"obj": object()})  # default=str fallback
    assert store.run_events(rid)[0]["kind"] == "weird"


def test_recent_runs_ordering_and_unknown_run(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    first = store.start_run("first")
    second = store.start_run("second")
    ids = [r["run_id"] for r in store.recent_runs()]
    assert set(ids) == {first, second}
    assert store.run_events("does-not-exist") == []

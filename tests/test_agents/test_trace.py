"""Tests for the run-trace store."""

from __future__ import annotations

import sqlite3

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


def test_trace_omits_task_arguments_and_output_by_default(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    secret = "private-account-number-123"
    rid = store.start_run(f"Look up {secret}")
    store.record_event(rid, "tool_call", {"name": "fs_list", "args": {"path": secret}})
    store.record_event(rid, "thought", {"text": secret})
    store.finish_run(rid, stopped_reason="final", iterations=1, output=secret)
    assert secret not in str(store.recent_runs())
    assert secret not in str(store.run_events(rid))
    assert "fs_list" in store.run_events(rid)[0]["data_json"]


def test_existing_trace_content_is_scrubbed_on_upgrade(tmp_path) -> None:
    db_path = tmp_path / "traces.db"
    secret = "private-account-number-123"
    legacy = TraceStore(db_path, include_content=True)
    rid = legacy.start_run(secret)
    legacy.record_event(rid, "tool_call", {"name": "fs_list", "args": {"path": secret}})
    legacy.finish_run(rid, stopped_reason="final", iterations=1, output=secret)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version=0")
    reopened = TraceStore(db_path)
    assert secret not in str(reopened.recent_runs())
    assert secret not in str(reopened.run_events(rid))


def test_trace_retention_removes_old_runs_and_events(tmp_path) -> None:
    db_path = tmp_path / "traces.db"
    store = TraceStore(db_path)
    old = store.start_run("old")
    store.record_event(old, "iteration", {"n": 1})
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE agent_runs SET started_at=0 WHERE run_id=?", (old,))
    store.start_run("new")
    assert all(run["run_id"] != old for run in store.recent_runs())
    assert store.run_events(old) == []


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

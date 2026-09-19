from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from isaac.core.state import SkillCandidate
from isaac.improvement.engine import ImprovementEngine
from isaac.memory import reminders
from isaac.memory.skill_library import SkillLibrary
from isaac.memory.skill_verification import Check, SkillVerifier, VerificationOutcome
from isaac.scheduler import heartbeat


def test_reminder_concurrent_writers_and_reopen(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        created = list(
            pool.map(lambda i: reminders.add_reminder(f"task {i}", isaac_home=tmp_path), range(64))
        )
    assert {r.id for r in reminders.list_reminders(isaac_home=tmp_path)} == {r.id for r in created}
    env = {**os.environ, "PYTHONPATH": str(Path(reminders.__file__).parents[2])}
    code = (
        "import sys; from pathlib import Path; from isaac.memory.reminders import add_reminder; "
        "[add_reminder(str(i), isaac_home=Path(sys.argv[1])) for i in range(12)]"
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        processes = list(
            pool.map(
                lambda _: subprocess.run(
                    [sys.executable, "-c", code, str(tmp_path)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ),
                range(3),
            )
        )
    assert all(p.returncode == 0 for p in processes)
    assert len(reminders.list_reminders(isaac_home=tmp_path)) == 100


def test_reminder_chronology_and_completion(tmp_path):
    early = reminders.add_reminder("early", "2026-09-20T10:00:00+05:00", isaac_home=tmp_path)
    later = reminders.add_reminder("later", "2026-09-20T04:00:00-03:00", isaac_home=tmp_path)
    naive = reminders.add_reminder("naive", "2026-09-20T06:00", isaac_home=tmp_path)
    reminders.add_reminder("undated", isaac_home=tmp_path)
    assert early.due_at == "2026-09-20T05:00:00+00:00"
    assert naive.due_at == "2026-09-20T06:00:00+00:00"
    assert reminders.due_reminders(isaac_home=tmp_path, now="2026-09-20T06:00Z") == [early, naive]
    assert reminders.complete_reminder(early.id[:12], isaac_home=tmp_path)
    assert reminders.complete_reminder(early.id, isaac_home=tmp_path)
    assert not reminders.complete_reminder("unknown", isaac_home=tmp_path)
    assert reminders.due_reminders(isaac_home=tmp_path, now=later.due_at) == [naive, later]
    assert len(reminders.list_reminders(include_done=True, isaac_home=tmp_path)) == 4


@pytest.mark.parametrize("value", ["", " ", None])
def test_empty_reminder_inputs_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        reminders.add_reminder(value, isaac_home=tmp_path)
    with pytest.raises(ValueError):
        reminders.complete_reminder(value, isaac_home=tmp_path)
    with pytest.raises(ValueError):
        reminders.parse_remind_args(value)


@pytest.mark.parametrize("when", ["bad", " ", "2026-02-30", None])
def test_invalid_reminder_times_rejected(tmp_path, when):
    with pytest.raises(ValueError):
        reminders.add_reminder("task", when, isaac_home=tmp_path)
    with pytest.raises(ValueError):
        reminders.due_reminders(isaac_home=tmp_path, now=when)


@pytest.mark.parametrize(
    "text",
    [
        "task @",
        "@ in 1h",
        "task @ in -1h",
        "task @ in 0m",
        "task @ in nanh",
        "task @ in 1w",
        "task @ tomorrow",
    ],
)
def test_invalid_reminder_syntax_rejected(text):
    with pytest.raises(ValueError):
        reminders.parse_remind_args(text)


def test_reminder_parser_normalizes_times():
    assert reminders.parse_remind_args("  task  ") == ("task", "")
    assert reminders.parse_remind_args("task @ 2026-09-20T10:00+03:00") == (
        "task",
        "2026-09-20T07:00:00+00:00",
    )
    before = datetime.now(timezone.utc).timestamp()
    body, due = reminders.parse_remind_args("task @ in 1.5h")
    assert body == "task"
    assert 5400 <= datetime.fromisoformat(due).timestamp() - before < 5402


def test_json_migration_is_atomic_and_prefixes_are_unambiguous(tmp_path):
    legacy = tmp_path / "reminders.json"
    first = dict(id="abc1", text="first", created_at="2026-01-01", due_at="2026-01-02")
    second = dict(first, id="abc2", text="second", done=True)
    legacy.write_text(json.dumps([first, {"id": "bad"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        reminders.list_reminders(isaac_home=tmp_path)
    legacy.write_text(json.dumps([first, second]), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=4) as pool:
        lists = list(
            pool.map(
                lambda _: reminders.list_reminders(include_done=True, isaac_home=tmp_path), range(4)
            )
        )
    assert all(len(items) == 2 for items in lists)
    with pytest.raises(ValueError, match="ambiguous"):
        reminders.complete_reminder("abc", isaac_home=tmp_path)
    assert len(reminders.list_reminders(isaac_home=tmp_path)) == 1
    legacy.write_text("invalid after migration", encoding="utf-8")
    assert reminders.complete_reminder("abc1", isaac_home=tmp_path)
    assert reminders.list_reminders(isaac_home=tmp_path) == []


def test_storage_errors_propagate(tmp_path):
    home = tmp_path / "file"
    home.write_text("blocked", encoding="utf-8")
    with pytest.raises(OSError):
        reminders.add_reminder("task", isaac_home=home)
    (tmp_path / "reminders.sqlite3").write_bytes(b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        reminders.list_reminders(isaac_home=tmp_path)


def test_notification_claims_retry_expiry_and_persistent_ack(tmp_path):
    reminder = reminders.add_reminder("due", "2026-01-01", isaac_home=tmp_path)
    options = dict(isaac_home=tmp_path, now="2026-01-02T00:00Z")
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: reminders.claim_due_reminders("log", **options), range(8)))
    assert sum(map(len, claims)) == 1
    first = next(batch[0] for batch in claims if batch)
    later = dict(options, now="2026-01-02T00:05Z")
    second = reminders.claim_due_reminders("log", **later)[0]
    with pytest.raises(ValueError, match="stale"):
        reminders.finish_reminder_notification(
            reminder.id,
            "log",
            first.token,
            delivered=True,
            **later,
        )
    reminders.finish_reminder_notification(
        reminder.id,
        "log",
        second.token,
        delivered=False,
        error="offline",
        **later,
    )
    third = reminders.claim_due_reminders("log", **later)[0]
    reminders.finish_reminder_notification(reminder.id, "log", third.token, delivered=True, **later)
    assert reminders.claim_due_reminders("log", **later) == []
    assert reminders.due_reminders(**later) == [reminder]
    assert len(reminders.claim_due_reminders("telegram:1", **later)) == 1
    reminders.complete_reminder(reminder.id, isaac_home=tmp_path)
    assert reminders.claim_due_reminders("telegram:2", **later) == []


def test_scheduler_retries_only_failed_recipients_and_drains_batches(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        isaac_home=tmp_path,
        telegram_bot_token="test",
        telegram_allowed_users=[1, 2],
    )
    monkeypatch.setattr(heartbeat, "_get_settings", lambda: settings)
    reminder = reminders.add_reminder("due", "2000-01-01", isaac_home=tmp_path)
    sent = []

    def send(text, recipient, token):
        sent.append(recipient)
        if recipient == "2" and sent.count("2") == 1:
            raise RuntimeError("offline")

    monkeypatch.setattr(heartbeat, "_send_reminder_notification", send)
    with pytest.raises(RuntimeError, match="offline"):
        heartbeat.reminders_job()
    heartbeat.reminders_job()
    heartbeat.reminders_job()
    assert sent == ["1", "2", "2"]
    assert reminders.due_reminders(isaac_home=tmp_path) == [reminder]
    for i in range(12):
        reminders.add_reminder(f"due {i}", "2000-01-01", isaac_home=tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: heartbeat.reminders_job(), range(4)))
    assert sent.count("1") == 13
    assert sent.count("2") == 14


@pytest.mark.parametrize(("status", "payload"), [(500, {}), (200, {"ok": False}), (200, [])])
def test_telegram_delivery_requires_positive_receipt(monkeypatch, status, payload):
    response = httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://example.test/secret-token"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="delivery failed") as error:
        heartbeat._send_reminder_notification("task", "1", "secret-token")
    assert "secret-token" not in str(error.value)


def test_scheduler_lifecycle_is_thread_safe(monkeypatch):
    from apscheduler.schedulers import background

    scheduler = Mock()
    factory = Mock(return_value=scheduler)
    monkeypatch.setattr(background, "BackgroundScheduler", factory)
    monkeypatch.setattr(heartbeat, "_scheduler", None)
    monkeypatch.setattr(heartbeat, "_get_settings", lambda: SimpleNamespace())
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: heartbeat.start_scheduler(), range(16)))
        assert all(
            pool.map(lambda i: heartbeat.register_callback(lambda: None, name=str(i)), range(8))
        )
        list(pool.map(lambda _: heartbeat.stop_scheduler(), range(16)))
    factory.assert_called_once()
    scheduler.start.assert_called_once()
    scheduler.shutdown.assert_called_once()


@pytest.fixture
def maintenance(monkeypatch):
    from isaac.improvement import performance, self_critique
    from isaac.memory import manager
    from isaac.security import audit

    tracker = Mock()
    tracker.prune.return_value = 3
    memory = Mock()
    monkeypatch.setattr(performance, "get_tracker", lambda: tracker)
    monkeypatch.setattr(manager, "get_memory_manager", lambda: memory)
    critique = Mock(
        return_value=SimpleNamespace(summary="observation", improvement_note="proposal")
    )
    monkeypatch.setattr(self_critique, "build_critique", critique)
    monkeypatch.setattr(audit, "audit", Mock())
    engine = ImprovementEngine()
    engine._curator = Mock()
    engine._curator.curate_all.return_value = []
    return engine, tracker, memory, critique


def test_recursive_cycle_reports_unverified_without_repeating(maintenance):
    engine, tracker, memory, critique = maintenance
    result = engine.run_cycle(recursive=True)
    assert result.refinement_status == "unsupported"
    assert not result.improvement_verified
    assert any("before/after" in error for error in result.errors)
    assert result.pruned_rows == 3
    assert result.finished_at >= result.started_at
    engine._curator.curate_all.assert_called_once()
    tracker.prune.assert_called_once()
    memory.consolidate.assert_called_once()
    critique.assert_called_once()


def test_cycle_reentrancy_is_process_wide_and_releases_on_failure(maintenance):
    engine, _, _, _ = maintenance
    entered = Event()
    release = Event()

    def curate():
        entered.set()
        assert release.wait(5)
        raise RuntimeError("curation failed")

    engine._curator.curate_all.side_effect = curate
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(engine.run_cycle)
        try:
            assert entered.wait(5)
            blocked = ImprovementEngine().run_cycle()
            assert any("re-entrancy" in error for error in blocked.errors)
        finally:
            release.set()
        result = future.result(timeout=5)
    assert any("curation failed" in error for error in result.errors)
    engine._curator.curate_all.side_effect = None
    assert engine.run_cycle().errors == []


@pytest.mark.parametrize(
    ("code", "schema", "verified"),
    [
        ("async def run():\n    raise RuntimeError('executed')", {"example": {}}, False),
        ("async def run():\n    return 42", {"example": {}}, True),
        ("def run(): return 1\nasync def _selftest():\n    assert run() == 2", {}, False),
        ("def run(): return 1\nasync def _selftest():\n    assert run() == 1", {}, True),
        ("def run():\n    raise RuntimeError('executed')", {"example": {}}, False),
        ("def run(): return 1", {"example": {}, "function": "missing"}, False),
        ("def run():\n    yield 1", {"example": {}}, False),
        ("def _selftest(): pass", {}, False),
    ],
)
def test_verifier_executes_async_and_empty_examples(code, schema, verified):
    candidate = SkillCandidate(name="candidate", code=code, input_schema=schema)
    result = SkillVerifier(require_sandbox=False).verify(candidate)
    assert result.verified is verified


@pytest.mark.parametrize(
    ("payload", "exit_code"),
    [
        ({"checks": []}, 0),
        ([], 0),
        (
            {
                "ok": True,
                "checks": [
                    {"name": name, "status": "passed"}
                    for name in ("import", "doctest", "selftest", "example")
                ],
                "callables": ["run"],
            },
            1,
        ),
    ],
)
def test_verifier_rejects_invalid_verdicts_and_failed_process(monkeypatch, payload, exit_code):
    from isaac.memory import skill_verification

    verifier = SkillVerifier(require_sandbox=True)
    monkeypatch.setattr(skill_verification, "_docker_available", lambda: True)
    monkeypatch.setattr(
        verifier,
        "_execute_in_docker",
        lambda *args: (
            "__ISAAC_SKILL_VERIFY__" + json.dumps(payload),
            "failed",
            exit_code,
        ),
    )
    assert not verifier.verify(SkillCandidate(name="run", code="def run(): return 1")).verified


def test_library_requires_behavior_and_marks_opt_out_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr(SkillLibrary, "_ensure_collection", lambda self: None)
    library = SkillLibrary(tmp_path)
    candidate = SkillCandidate(name="run", code="def run(): return 1")
    verifier = SkillVerifier(require_sandbox=False)
    assert not library.commit(candidate, verify=True, verifier=verifier)
    assert library.size == 0
    assert not (tmp_path / "run.py").exists()
    assert library.commit(candidate, verify=False)
    assert library.get_metadata("run")["verified"] is False
    import_only = Mock()
    import_only.verify.return_value = VerificationOutcome("run", True, evidence="import")
    assert not library.commit(candidate, verify=True, verifier=import_only)
    assert library.get_metadata("run")["verified"] is False


def test_library_concurrent_instances_and_reentrant_verifier(tmp_path, monkeypatch):
    monkeypatch.setattr(SkillLibrary, "_ensure_collection", lambda self: None)

    def commit(i):
        return SkillLibrary(tmp_path).commit(
            SkillCandidate(name=f"skill_{i}", code="def run(): return 1"),
            verify=False,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(commit, range(24)))
    library = SkillLibrary(tmp_path)
    assert library.size == 24
    candidate = SkillCandidate(name="outer", code="def run(): return 1")

    def verify(snapshot):
        nested = library.commit(candidate, verify=False)
        assert not nested and "re-entrant" in nested.reason
        snapshot.code = "corrupted"
        return VerificationOutcome(
            "outer",
            True,
            evidence="behaviour",
            checks=[Check("selftest", "passed")],
        )

    assert library.commit(candidate, verify=True, verifier=SimpleNamespace(verify=verify))
    assert library.get_code("outer") == candidate.code
    assert library.size == 25


def test_library_surfaces_persistence_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(SkillLibrary, "_ensure_collection", lambda self: None)
    library = SkillLibrary(tmp_path)
    monkeypatch.setattr(library, "_save_index", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        library.commit(SkillCandidate(name="run", code="def run(): pass"), verify=False)
    assert SkillLibrary(tmp_path).size == 0

"""Tests for the Cron Engine module."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import patch

import pytest


class TestCronTaskCRUD:
    """Tests for cron task CRUD operations."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        """Redirect cron data to a temp directory."""
        self._home = tmp_path / ".isaac"
        self._home.mkdir()
        self._patcher = patch(
            "isaac.background.cron_engine._isaac_home",
            return_value=self._home,
        )
        self._patcher.start()

    def teardown_method(self) -> None:
        self._patcher.stop()

    def test_add_task(self) -> None:
        from isaac.background.cron_engine import add_task, list_tasks

        task = add_task(name="Test Task", schedule="*/5 * * * *", command="echo hello")
        assert task.name == "Test Task"
        assert task.schedule == "*/5 * * * *"
        assert task.approved is False
        tasks = list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Test Task"

    def test_explicit_unattended_approval_is_persisted(self) -> None:
        from isaac.background.cron_engine import add_task, load_tasks

        add_task(
            name="Approved",
            schedule="*/5 * * * *",
            command="echo hello",
            approved=True,
        )
        assert load_tasks()[0].approved is True

    def test_remove_task(self) -> None:
        from isaac.background.cron_engine import add_task, list_tasks, remove_task

        task = add_task(name="To Remove", schedule="0 * * * *", command="echo bye")
        assert len(list_tasks()) == 1
        ok = remove_task(task.id)
        assert ok
        assert len(list_tasks()) == 0

    def test_remove_nonexistent(self) -> None:
        from isaac.background.cron_engine import remove_task

        assert not remove_task("nonexistent_id")

    def test_pause_and_resume(self) -> None:
        from isaac.background.cron_engine import add_task, load_tasks, pause_task, resume_task

        task = add_task(name="Pausable", schedule="0 * * * *", command="echo hi")
        assert pause_task(task.id)
        tasks = load_tasks()
        assert not tasks[0].enabled

        assert resume_task(task.id)
        tasks = load_tasks()
        assert tasks[0].enabled

    def test_list_tasks_empty(self) -> None:
        from isaac.background.cron_engine import list_tasks

        assert list_tasks() == []

    def test_multiple_tasks(self) -> None:
        from isaac.background.cron_engine import add_task, list_tasks

        add_task(name="A", schedule="0 * * * *", command="echo a")
        add_task(name="B", schedule="0 * * * *", command="echo b")
        add_task(name="C", schedule="0 * * * *", command="echo c")
        assert len(list_tasks()) == 3


class TestCronDaemon:
    """Tests for cron daemon start/stop."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._home = tmp_path / ".isaac"
        self._home.mkdir()
        self._patcher = patch(
            "isaac.background.cron_engine._isaac_home",
            return_value=self._home,
        )
        self._patcher.start()

    def teardown_method(self) -> None:
        from isaac.background.cron_engine import stop_cron_daemon

        stop_cron_daemon()
        self._patcher.stop()

    def test_start_and_stop(self) -> None:
        from isaac.background.cron_engine import (
            is_cron_running,
            start_cron_daemon,
            stop_cron_daemon,
        )

        start_cron_daemon(poll_seconds=1)
        assert is_cron_running()

        stop_cron_daemon()
        assert not is_cron_running()

    def test_pid_file_created(self) -> None:
        from isaac.background.cron_engine import start_cron_daemon, stop_cron_daemon

        start_cron_daemon(poll_seconds=1)
        assert (self._home / "cron.pid").exists()

        stop_cron_daemon()
        assert not (self._home / "cron.pid").exists()


class TestCronIsDue:
    """Tests for cron schedule evaluation."""

    def test_never_run_is_due(self) -> None:
        from isaac.background.cron_engine import CronTask, _is_due

        task = CronTask(schedule="* * * * *", last_run="")
        try:
            result = _is_due(task)
            assert result  # Never ran => due
        except ImportError:
            pytest.skip("croniter not installed")

    def test_recently_run_not_due(self) -> None:
        from datetime import datetime

        from isaac.background.cron_engine import CronTask, _is_due

        now = datetime.now(UTC).isoformat()
        task = CronTask(schedule="0 * * * *", last_run=now)
        try:
            result = _is_due(task)
            assert not result
        except ImportError:
            pytest.skip("croniter not installed")


class TestCronExecutionSafety:
    def test_unapproved_task_never_reaches_connector(self, tmp_path: Path) -> None:
        from isaac.background.cron_engine import CronTask, _execute_task

        task = CronTask(name="No grant", command="echo hello", approved=False)
        with (
            patch("isaac.background.cron_engine._isaac_home", return_value=tmp_path),
            patch("isaac.skills.connectors.registry.run_connector") as run,
        ):
            status = _execute_task(task)
        assert status == "approval_required"
        run.assert_not_called()

    def test_approved_task_still_cannot_override_critical_rule(self, tmp_path: Path) -> None:
        from isaac.background.cron_engine import CronTask, _execute_task

        task = CronTask(name="Unsafe", command="rm -rf /", approved=True)
        with (
            patch("isaac.background.cron_engine._isaac_home", return_value=tmp_path),
            patch("isaac.skills.connectors.registry.run_connector") as run,
        ):
            status = _execute_task(task)
        assert status == "blocked"
        run.assert_not_called()

    def test_approved_pre_run_script_is_reviewed_before_host_execution(
        self, tmp_path: Path
    ) -> None:
        from isaac.background.cron_engine import CronTask, _execute_task

        task = CronTask(
            name="Unsafe hook",
            command="summarise",
            pre_run_script="rm -rf /",
            no_agent=True,
            approved=True,
        )
        with (
            patch("isaac.background.cron_engine._isaac_home", return_value=tmp_path),
            patch("isaac.background.cron_engine._run_script") as script,
        ):
            status = _execute_task(task)
        assert status == "blocked"
        script.assert_not_called()


class TestCronManifestSafety:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self._home = tmp_path / ".isaac"
        self._home.mkdir()
        self._patcher = patch(
            "isaac.background.cron_engine._isaac_home",
            return_value=self._home,
        )
        self._patcher.start()

    def teardown_method(self) -> None:
        self._patcher.stop()

    def test_corrupt_manifest_is_not_overwritten(self) -> None:
        from isaac.background.cron_engine import CronManifestError, add_task

        manifest = self._home / "cron_tasks.json"
        manifest.write_text("{not json", encoding="utf-8")

        with pytest.raises(CronManifestError):
            add_task(name="New", schedule="0 * * * *", command="echo hi")

        assert manifest.read_text(encoding="utf-8") == "{not json"

    def test_malformed_schedule_does_not_raise(self) -> None:
        from datetime import datetime

        from isaac.background.cron_engine import CronTask, _is_due

        task = CronTask(schedule="not a schedule", last_run=datetime.now(UTC).isoformat())
        assert _is_due(task) is False

    def test_malformed_last_run_does_not_raise(self) -> None:
        from isaac.background.cron_engine import CronTask, _is_due

        assert _is_due(CronTask(schedule="* * * * *", last_run="yesterday")) is False


class TestConnectorParsing:
    def test_windows_path_is_not_a_connector_call(self, monkeypatch) -> None:
        from isaac.background.cron_engine import _parse_connector_call

        monkeypatch.setattr(
            "isaac.skills.connectors.registry.get_registry",
            lambda: {"shell": object()},
        )
        assert _parse_connector_call(r"C:\scripts\job.bat") is None

    def test_quoted_multi_word_shell_command_is_kept(self, monkeypatch) -> None:
        from isaac.background.cron_engine import _parse_connector_call

        monkeypatch.setattr(
            "isaac.skills.connectors.registry.get_registry",
            lambda: {"shell": object()},
        )
        parsed = _parse_connector_call('shell:command="echo hello world"')
        assert parsed == ("shell", {"command": "echo hello world"})

    def test_unbalanced_quote_is_rejected(self, monkeypatch) -> None:
        from isaac.background.cron_engine import _parse_connector_call

        monkeypatch.setattr(
            "isaac.skills.connectors.registry.get_registry",
            lambda: {"shell": object()},
        )
        assert _parse_connector_call('shell:command="echo hello') is None


class TestCronLocks:
    def test_stale_lock_is_reclaimed(self, tmp_path: Path) -> None:
        from isaac.background import cron_locks

        with patch("isaac.background.cron_locks._isaac_home", return_value=tmp_path):
            path = cron_locks._get_lock_path("abc")
            assert path is not None
            path.write_text("1 0", encoding="utf-8")
            import os

            os.utime(path, (0, 0))
            assert cron_locks._acquire_lock("abc") is True
            cron_locks._release_lock("abc")
            assert not path.exists()

    def test_invalid_task_id_never_builds_a_path(self) -> None:
        from isaac.background.cron_locks import _acquire_lock, _get_lock_path

        assert _get_lock_path("../../escape") is None
        assert _acquire_lock("../../escape") is False

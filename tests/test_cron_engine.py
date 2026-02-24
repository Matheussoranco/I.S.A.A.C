"""Tests for the Cron Engine module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
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
        tasks = list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Test Task"

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
        from isaac.background.cron_engine import is_cron_running, start_cron_daemon, stop_cron_daemon

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
        from datetime import datetime, timezone
        from isaac.background.cron_engine import CronTask, _is_due

        # Just ran, every hour schedule
        now = datetime.now(timezone.utc).isoformat()
        task = CronTask(schedule="0 * * * *", last_run=now)
        try:
            result = _is_due(task)
            assert not result
        except ImportError:
            pytest.skip("croniter not installed")

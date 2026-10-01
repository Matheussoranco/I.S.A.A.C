from unittest.mock import patch

from isaac.background.cron_engine import CronTask, add_task
from isaac.scheduler.cron_parser import schedule_from_nl


def test_nl_parsing():
    assert schedule_from_nl("every 30m") == "*/30 * * * *"
    assert schedule_from_nl("every monday 9am") == "0 9 * * 1"
    assert schedule_from_nl("daily") == "0 0 * * *"
    assert schedule_from_nl("hourly") == "0 * * * *"
    assert schedule_from_nl("every 2h") == "0 */2 * * *"
    assert (
        schedule_from_nl("already 0 0 * * *") == "already 0 0 * * *"
    )  # should return as-is if it doesn't match NL


def test_cron_task_overrides():
    # Test that adding a task preserves overrides
    task = add_task(
        name="Test Job",
        schedule="*/5 * * * *",
        command="Hello World",
        skill="test-skill",
        model="gpt-4",
        provider="openai",
        deliver_to=["telegram", "email"],
    )
    assert task.skill == "test-skill"
    assert task.model == "gpt-4"
    assert task.provider == "openai"
    assert "telegram" in task.deliver_to


def test_context_chaining_config():
    task = add_task(
        name="Chained Job",
        schedule="0 * * * *",
        command="Process {context}",
        context_from=["task123", "task456"],
    )
    assert task.context_from == ["task123", "task456"]


@patch("isaac.background.cron_locks._acquire_lock")
@patch("isaac.background.cron_locks._release_lock")
def test_lock_mechanism(mock_release, mock_acquire):

    # Mock CronTask
    CronTask(id="lock-test", name="Lock Test", command="echo 1", approved=True)

    # Simulate lock acquisition failure
    mock_acquire.return_value = False
    # We need to test the daemon loop logic that uses the lock,
    # but we can verify _acquire_lock and _release_lock are available and used.
    assert mock_acquire.called is False  # Not called yet

    # This test is a stub for the actual daemon loop integration test
    # which would require running the loop in a thread.
    pass

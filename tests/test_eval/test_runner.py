"""Tests for the eval runner + results store — fully offline (fake runner)."""

from __future__ import annotations

from isaac.eval.results import EvalStore
from isaac.eval.runner import TaskAnswer, run_suite
from isaac.eval.suite import EvalTask


def _tasks() -> list[EvalTask]:
    return [
        EvalTask(
            id="t-math",
            prompt="2+2?",
            category="reasoning",
            checks=[{"type": "numeric", "value": 4}],
        ),
        EvalTask(
            id="t-file",
            prompt="write hello.txt",
            category="file-org",
            checks=[{"type": "file_contains", "path": "hello.txt", "value": "hello"}],
        ),
        EvalTask(
            id="t-fail",
            prompt="impossible",
            category="reasoning",
            checks=[{"type": "contains", "value": "unicorn"}],
        ),
    ]


def _fake_runner(workspace):
    def run(task: EvalTask) -> TaskAnswer:
        if task.id == "t-math":
            return TaskAnswer(text="The answer is 4.")
        if task.id == "t-file":
            (workspace / "hello.txt").write_text("hello world", encoding="utf-8")
            return TaskAnswer(text="written")
        return TaskAnswer(text="no idea", stopped_reason="max_iterations")

    return run


def test_run_suite_scores_and_aggregates(tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    summary = run_suite(
        _tasks(),
        _fake_runner(ws),
        suite_name="unit",
        workspace=ws,
        model="fake-model",
        provider="fake",
    )
    assert summary.total == 3
    assert summary.passed == 2
    assert abs(summary.accuracy - 2 / 3) < 1e-9
    assert summary.by_category() == {"reasoning": (1, 2), "file-org": (1, 1)}
    failed = next(o for o in summary.outcomes if o.task_id == "t-fail")
    assert failed.stopped_reason == "max_iterations"


def test_run_suite_seeds_task_files(tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    task = EvalTask(
        id="t-seed",
        prompt="read it",
        files={"data/in.txt": "seeded-content", "../escape.txt": "nope"},
        checks=[{"type": "contains", "value": "seeded"}],
    )
    seen: dict[str, str] = {}

    def runner(t: EvalTask) -> TaskAnswer:
        seen["content"] = (ws / "data" / "in.txt").read_text(encoding="utf-8")
        return TaskAnswer(text=seen["content"])

    summary = run_suite([task], runner, workspace=ws, model="m", provider="p")
    assert summary.passed == 1
    assert seen["content"] == "seeded-content"
    assert not (tmp_path / "escape.txt").exists(), "seed files must not escape the workspace"


def test_runner_crash_is_contained(tmp_path) -> None:
    def runner(t: EvalTask) -> TaskAnswer:
        raise RuntimeError("model exploded")

    summary = run_suite([_tasks()[0]], runner, workspace=tmp_path, model="m", provider="p")
    assert summary.total == 1
    assert summary.passed == 0
    assert summary.outcomes[0].stopped_reason == "error"


def test_run_is_persisted_and_reportable(tmp_path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    store = EvalStore(tmp_path / "eval.db")
    summary = run_suite(
        _tasks(),
        _fake_runner(ws),
        suite_name="unit",
        workspace=ws,
        store=store,
        model="fake-model",
        provider="fake",
        runner_kind="agent",
    )

    runs = store.recent_runs()
    assert len(runs) == 1
    rec = runs[0]
    assert rec.run_id == summary.run_id
    assert rec.suite == "unit"
    assert rec.suite_hash == summary.suite_hash
    assert rec.model == "fake-model"
    assert (rec.passed, rec.total) == (2, 3)

    details = store.run_details(summary.run_id)
    assert {d["task_id"] for d in details} == {"t-math", "t-file", "t-fail"}
    failed_row = next(d for d in details if d["task_id"] == "t-fail")
    assert failed_row["passed"] == 0
    assert "unicorn" in failed_row["checks_json"]


def test_report_formatting(tmp_path) -> None:
    from isaac.eval.report import format_recent, format_summary

    ws = tmp_path / "ws"
    ws.mkdir()
    store = EvalStore(tmp_path / "eval.db")
    summary = run_suite(
        _tasks(),
        _fake_runner(ws),
        suite_name="unit",
        workspace=ws,
        store=store,
        model="fake-model",
        provider="fake",
    )
    text = format_summary(summary)
    assert "SCORE: 2/3" in text
    assert "[FAIL] t-fail" in text
    recent = format_recent(store)
    assert "fake-model" in recent
    assert "66.7%" in recent

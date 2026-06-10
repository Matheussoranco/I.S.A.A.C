"""Plain-text reporting for eval runs (used by ``isaac eval``)."""

from __future__ import annotations

from datetime import datetime

from isaac.eval.results import EvalStore
from isaac.eval.runner import EvalRunSummary


def format_summary(summary: EvalRunSummary) -> str:
    """Scoreboard for a just-finished run."""
    lines = [
        f"Run {summary.run_id} — suite '{summary.suite}' (hash {summary.suite_hash})",
        f"  model={summary.model} provider={summary.provider} "
        f"runner={summary.runner} git={summary.git_rev or 'n/a'}",
        "",
        f"  SCORE: {summary.passed}/{summary.total}  ({summary.accuracy:.1%})",
        "",
    ]
    for cat, (p, t) in sorted(summary.by_category().items()):
        lines.append(f"    {cat:<14s} {p}/{t}")
    lines.append("")
    for o in summary.outcomes:
        mark = "PASS" if o.passed else "FAIL"
        lines.append(f"  [{mark}] {o.task_id:<22s} {o.duration_ms:>8.0f}ms  {o.stopped_reason}")
        if not o.passed:
            for c in o.checks:
                if not c.passed:
                    lines.append(f"         x {c.type}: {c.detail}")
    return "\n".join(lines)


def format_recent(store: EvalStore, limit: int = 10) -> str:
    """Comparison table of the most recent persisted runs."""
    runs = store.recent_runs(limit)
    if not runs:
        return "No eval runs recorded yet. Run: isaac eval <suite.jsonl>"
    lines = [
        f"{'run_id':<14s} {'date':<17s} {'suite':<16s} {'model':<24s} "
        f"{'runner':<7s} {'score':<9s} acc"
    ]
    for r in runs:
        date = datetime.fromtimestamp(r.started_at).strftime("%Y-%m-%d %H:%M")
        lines.append(
            f"{r.run_id:<14s} {date:<17s} {r.suite:<16s} {r.model:<24s} "
            f"{r.runner:<7s} {r.passed}/{r.total:<7d} {r.accuracy:.1%}"
        )
    return "\n".join(lines)

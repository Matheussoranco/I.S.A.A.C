"""Evaluation harness — turn capability claims into measurements.

``isaac eval`` loads a task suite (JSONL), runs the agent or the specialist
team against every task, scores the answers with programmatic checkers, and
persists each run to a SQLite results DB so scores are reproducible and
comparable across models and versions.

This is Workstream 1 of ``docs/ROADMAP-1.0.md``: the word "SOTA" is only
allowed in public artifacts once a number produced by this harness backs it.
"""

from __future__ import annotations

from isaac.eval.checkers import CheckOutcome, run_check, score_answer
from isaac.eval.results import EvalStore
from isaac.eval.runner import EvalRunSummary, TaskAnswer, TaskOutcome, run_suite
from isaac.eval.suite import EvalTask, load_suite, suite_hash

__all__ = [
    "CheckOutcome",
    "EvalRunSummary",
    "EvalStore",
    "EvalTask",
    "TaskAnswer",
    "TaskOutcome",
    "load_suite",
    "run_check",
    "run_suite",
    "score_answer",
    "suite_hash",
]

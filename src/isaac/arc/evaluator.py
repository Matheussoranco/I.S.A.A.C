"""ARC-AGI evaluation harness.

Loads ARC tasks, runs the I.S.A.A.C. cognitive graph (or a standalone
solver) against them, and scores results.

Supports two solving modes:
1. **graph** — full cognitive loop (Perception → Plan → Synthesis → Execute → Reflect)
2. **dsl_search** — brute-force search over DSL primitive compositions
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from isaac.arc.dsl import PRIMITIVES, apply_program, compose
from isaac.arc.grid_ops import Grid, analyse_grid, format_grid_for_prompt, grid_diff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ArcPair:
    """A single input→output example."""

    input: Grid
    output: Grid


@dataclass
class ArcTask:
    """A full ARC task with training examples and test pairs."""

    id: str
    train: list[ArcPair]
    test: list[ArcPair]
    description: str = ""


@dataclass
class TaskResult:
    """Result of evaluating a single ARC task."""

    task_id: str
    correct: bool
    predicted: list[Grid | None]
    """One prediction per test pair (``None`` if no answer produced)."""
    program: list[dict[str, Any]] | str = ""
    """The program that produced the prediction (DSL steps or Python code)."""
    solve_time_ms: float = 0.0
    method: str = ""
    """Which solver produced this result."""


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    total_tasks: int = 0
    correct: int = 0
    accuracy: float = 0.0
    results: list[TaskResult] = field(default_factory=list)
    total_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(path: Path) -> list[ArcTask]:
    """Load ARC tasks from a JSON file.

    Expected format (same as ARC-AGI public dataset)::

        [
          {
            "id": "...",
            "train": [{"input": [[...]], "output": [[...]]}],
            "test":  [{"input": [[...]], "output": [[...]]}],
            "description": "..."   // optional
          }
        ]
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[ArcTask] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        train_pairs = [
            ArcPair(
                input=np.array(p["input"], dtype=int),
                output=np.array(p["output"], dtype=int),
            )
            for p in item.get("train", [])
        ]
        test_pairs = [
            ArcPair(
                input=np.array(p["input"], dtype=int),
                output=np.array(p["output"], dtype=int),
            )
            for p in item.get("test", [])
        ]
        tasks.append(ArcTask(
            id=item.get("id", f"task_{len(tasks)}"),
            train=train_pairs,
            test=test_pairs,
            description=item.get("description", ""),
        ))
    logger.info("Loaded %d ARC tasks from %s", len(tasks), path)
    return tasks


def load_tasks_from_dir(directory: Path) -> list[ArcTask]:
    """Load all ``.json`` task files from a directory."""
    tasks: list[ArcTask] = []
    for json_path in sorted(directory.glob("*.json")):
        tasks.extend(load_tasks(json_path))
    return tasks


# ---------------------------------------------------------------------------
# DSL brute-force solver
# ---------------------------------------------------------------------------


def _try_single_primitive(
    task: ArcTask,
) -> TaskResult | None:
    """Try each single DSL primitive and check if it solves all training pairs."""
    for name, fn in PRIMITIVES.items():
        try:
            if all(
                np.array_equal(fn(pair.input), pair.output)
                for pair in task.train
            ):
                # Verify on test pairs
                predictions = [fn(pair.input) for pair in task.test]
                correct = all(
                    np.array_equal(pred, pair.output)
                    for pred, pair in zip(predictions, task.test)
                )
                return TaskResult(
                    task_id=task.id,
                    correct=correct,
                    predicted=predictions,
                    program=[{"op": name}],
                    method="dsl_single",
                )
        except Exception:
            continue
    return None


def _try_two_primitive_composition(
    task: ArcTask,
    max_combinations: int = 2000,
) -> TaskResult | None:
    """Try all 2-primitive compositions."""
    names = list(PRIMITIVES.keys())
    tried = 0
    for name_a in names:
        for name_b in names:
            if tried >= max_combinations:
                return None
            tried += 1
            fn = compose(PRIMITIVES[name_a], PRIMITIVES[name_b])
            try:
                if all(
                    np.array_equal(fn(pair.input), pair.output)
                    for pair in task.train
                ):
                    predictions = [fn(pair.input) for pair in task.test]
                    correct = all(
                        np.array_equal(pred, pair.output)
                        for pred, pair in zip(predictions, task.test)
                    )
                    return TaskResult(
                        task_id=task.id,
                        correct=correct,
                        predicted=predictions,
                        program=[{"op": name_a}, {"op": name_b}],
                        method="dsl_compose_2",
                    )
            except Exception:
                continue
    return None


def solve_with_dsl(task: ArcTask) -> TaskResult:
    """Attempt to solve a task using DSL primitive search.

    Tries single primitives first, then 2-compositions.
    """
    t0 = time.perf_counter()

    # Level 1: single primitive
    result = _try_single_primitive(task)
    if result is not None:
        result.solve_time_ms = (time.perf_counter() - t0) * 1000
        return result

    # Level 2: two-primitive composition
    result = _try_two_primitive_composition(task)
    if result is not None:
        result.solve_time_ms = (time.perf_counter() - t0) * 1000
        return result

    elapsed = (time.perf_counter() - t0) * 1000
    return TaskResult(
        task_id=task.id,
        correct=False,
        predicted=[None] * len(task.test),
        program="unsolved",
        solve_time_ms=elapsed,
        method="dsl_search",
    )


# ---------------------------------------------------------------------------
# LLM-guided solver (uses the I.S.A.A.C. cognitive graph)
# ---------------------------------------------------------------------------


def build_arc_prompt(task: ArcTask) -> str:
    """Format an ARC task into a text prompt for the LLM-based solver."""
    lines: list[str] = []
    lines.append("## ARC Task")
    if task.description:
        lines.append(f"Description: {task.description}")
    lines.append("")

    for i, pair in enumerate(task.train):
        lines.append(f"### Training Example {i + 1}")
        lines.append("Input:")
        lines.append(format_grid_for_prompt(pair.input))
        lines.append("Output:")
        lines.append(format_grid_for_prompt(pair.output))

        # Add structural analysis
        diff = grid_diff(pair.input, pair.output)
        lines.append(f"Shape changed: {diff['shape_changed']}")
        lines.append(f"Cells changed: {diff['n_changed_cells']}")
        lines.append(f"Colour changes: added={diff['colour_changes']['added']}, "
                      f"removed={diff['colour_changes']['removed']}")
        lines.append("")

    for i, pair in enumerate(task.test):
        lines.append(f"### Test Input {i + 1}")
        lines.append(format_grid_for_prompt(pair.input))
        lines.append("")

    lines.append(
        "Write a Python function `solve(grid: np.ndarray) -> np.ndarray` that "
        "transforms the input grid to produce the output grid. "
        "The function should work for all training examples and the test input. "
        "Use only numpy. Respond with a fenced ```python``` code block."
    )
    return "\n".join(lines)


def solve_with_llm(
    task: ArcTask,
    llm: Any | None = None,
) -> TaskResult:
    """Use an LLM to generate a Python solve function for the task.

    Falls back to DSL search if the LLM is unavailable.
    """
    t0 = time.perf_counter()

    if llm is None:
        try:
            from isaac.llm.provider import get_llm
            llm = get_llm("strong")
        except Exception:
            logger.warning("LLM unavailable — falling back to DSL solver.")
            return solve_with_dsl(task)

    prompt_text = build_arc_prompt(task)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "You are an expert at solving ARC-AGI tasks. Analyse the input→output "
            "patterns in the training examples and write a Python function that "
            "implements the transformation. The function signature must be "
            "`solve(grid: np.ndarray) -> np.ndarray`. Use only numpy. "
            "Respond ONLY with a fenced Python code block."
        )),
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        # Extract code
        import re
        match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
        code = match.group(1).strip() if match else content.strip()

        # Execute the code to get the solve function
        namespace: dict[str, Any] = {"np": np, "numpy": np}
        exec(code, namespace)  # noqa: S102
        solve_fn = namespace.get("solve")

        if solve_fn is None:
            raise ValueError("No 'solve' function found in generated code.")

        # Validate on training data
        train_ok = all(
            np.array_equal(solve_fn(pair.input), pair.output)
            for pair in task.train
        )

        if not train_ok:
            logger.info("LLM solution failed training validation for %s.", task.id)
            # Fall back to DSL
            return solve_with_dsl(task)

        # Run on test data
        predictions = [solve_fn(pair.input) for pair in task.test]
        correct = all(
            np.array_equal(pred, pair.output)
            for pred, pair in zip(predictions, task.test)
        )

        elapsed = (time.perf_counter() - t0) * 1000
        return TaskResult(
            task_id=task.id,
            correct=correct,
            predicted=predictions,
            program=code,
            solve_time_ms=elapsed,
            method="llm",
        )

    except Exception as exc:
        logger.warning("LLM solver failed for %s: %s", task.id, exc)
        return solve_with_dsl(task)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


def evaluate(
    tasks: list[ArcTask],
    solver: str = "hybrid",
    llm: Any | None = None,
) -> EvalReport:
    """Run evaluation across all tasks.

    Parameters
    ----------
    tasks:
        List of ARC tasks to evaluate.
    solver:
        ``"dsl"`` — DSL search only.
        ``"llm"`` — LLM solver only.
        ``"hybrid"`` — try DSL first, then LLM for unsolved.
    llm:
        Optional LLM instance.  If ``None``, auto-loads from settings.

    Returns
    -------
    EvalReport
        Aggregate results with per-task details.
    """
    report = EvalReport(total_tasks=len(tasks))
    t_start = time.perf_counter()

    for task in tasks:
        logger.info("Evaluating task %s (%d train, %d test)...",
                     task.id, len(task.train), len(task.test))

        if solver == "dsl":
            result = solve_with_dsl(task)
        elif solver == "llm":
            result = solve_with_llm(task, llm)
        else:
            # Hybrid: DSL first, then LLM
            result = solve_with_dsl(task)
            if not result.correct:
                result = solve_with_llm(task, llm)

        report.results.append(result)
        if result.correct:
            report.correct += 1
            logger.info("  ✓ %s solved in %.1fms via %s",
                         task.id, result.solve_time_ms, result.method)
        else:
            logger.info("  ✗ %s unsolved (%.1fms)", task.id, result.solve_time_ms)

    report.total_time_ms = (time.perf_counter() - t_start) * 1000
    report.accuracy = report.correct / report.total_tasks if report.total_tasks else 0.0

    logger.info(
        "ARC Evaluation complete: %d/%d correct (%.1f%%) in %.1fs",
        report.correct,
        report.total_tasks,
        report.accuracy * 100,
        report.total_time_ms / 1000,
    )
    return report


def print_report(report: EvalReport) -> None:
    """Pretty-print an evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("ARC-AGI EVALUATION REPORT")
    print("=" * 60)
    print(f"  Tasks:    {report.total_tasks}")
    print(f"  Correct:  {report.correct}")
    print(f"  Accuracy: {report.accuracy:.1%}")
    print(f"  Time:     {report.total_time_ms / 1000:.2f}s")
    print("-" * 60)
    for r in report.results:
        status = "✓" if r.correct else "✗"
        print(f"  [{status}] {r.task_id:30s}  {r.solve_time_ms:8.1f}ms  {r.method}")
    print("=" * 60 + "\n")

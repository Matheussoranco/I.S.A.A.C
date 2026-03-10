"""LLM Self-Refinement Loop for ARC-AGI.

Implements test-time compute scaling via iterative hypothesis revision:

  generate → validate → diagnose failures → revise → repeat

This is the most important ingredient for SOTA ARC performance: rather than
making one LLM call and hoping it's correct, we use the training data as a
*verifiable oracle* and iterate until the program passes all training pairs
(or we exhaust the compute budget).

This is aligned with Chollet's philosophy of *effective compute at test time*:
the system should be able to think harder on harder problems, using more
iterations when the first attempt fails.

Neurosymbolic design:
  - The oracle (training pairs) is symbolic and exact — no ambiguity
  - Failures are diagnosed symbolically (cell-level diff, object-level change)
  - The LLM refines based on structured, verifiable feedback
  - Each iteration is a hypothesis-test-revise cycle (Popperian falsification)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac.arc.evaluator import ArcTask
from isaac.arc.grid_ops import Grid, format_grid_for_prompt

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


# ─────────────────────────────────────────────────────────────────────────────
# Failure diagnosis
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PairFailure:
    """Structured diagnosis of a single failing training pair."""

    pair_index: int
    input_grid: Grid
    expected_grid: Grid
    actual_grid: Grid | None
    error: str = ""
    n_wrong_cells: int = 0
    wrong_cells: list[dict[str, Any]] = field(default_factory=list)
    shape_mismatch: bool = False


def _diagnose_pair(
    pair_idx: int,
    input_grid: Grid,
    expected: Grid,
    solve_fn: Any,
) -> PairFailure | None:
    """Run solve_fn on input_grid and return a PairFailure if it doesn't match expected.

    Returns None if the pair passes (no failure).
    """
    try:
        actual = solve_fn(input_grid.copy())
    except Exception as exc:
        return PairFailure(
            pair_index=pair_idx,
            input_grid=input_grid,
            expected_grid=expected,
            actual_grid=None,
            error=str(exc)[:300],
        )

    if not isinstance(actual, np.ndarray):
        try:
            actual = np.array(actual, dtype=int)
        except Exception:
            return PairFailure(
                pair_index=pair_idx,
                input_grid=input_grid,
                expected_grid=expected,
                actual_grid=None,
                error="solve() did not return an ndarray",
            )

    if np.array_equal(actual, expected):
        return None  # Pass — no failure

    failure = PairFailure(
        pair_index=pair_idx,
        input_grid=input_grid,
        expected_grid=expected,
        actual_grid=actual,
    )

    if actual.shape != expected.shape:
        failure.shape_mismatch = True
        failure.error = (
            f"Shape mismatch: got {actual.shape}, expected {expected.shape}"
        )
        return failure

    diff_mask = actual != expected
    wrong_positions = list(zip(*np.where(diff_mask)))
    failure.n_wrong_cells = len(wrong_positions)
    failure.wrong_cells = [
        {
            "row": int(r),
            "col": int(c),
            "expected": int(expected[r, c]),
            "got": int(actual[r, c]),
        }
        for r, c in wrong_positions[:20]  # cap for prompt size
    ]
    return failure


def _format_failure(f: PairFailure) -> str:
    """Format a PairFailure into a human-readable block for the LLM prompt."""
    lines: list[str] = [f"### Pair {f.pair_index + 1}: FAILED"]

    lines.append("**Input:**")
    lines.append("```")
    lines.append(format_grid_for_prompt(f.input_grid))
    lines.append("```")

    lines.append("**Expected output:**")
    lines.append("```")
    lines.append(format_grid_for_prompt(f.expected_grid))
    lines.append("```")

    if f.actual_grid is not None:
        lines.append("**Your solve() produced:**")
        lines.append("```")
        lines.append(format_grid_for_prompt(f.actual_grid))
        lines.append("```")

        if f.shape_mismatch:
            lines.append(f"**Shape mismatch:** {f.error}")
        elif f.wrong_cells:
            lines.append(f"**{f.n_wrong_cells} cells differ:**")
            for cell in f.wrong_cells[:10]:
                lines.append(
                    f"  row={cell['row']}, col={cell['col']}: "
                    f"expected {cell['expected']}, got {cell['got']}"
                )
            if f.n_wrong_cells > 10:
                lines.append(f"  ... and {f.n_wrong_cells - 10} more")
    else:
        lines.append(f"**Runtime error:** {f.error}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Code execution helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_solve_fn(code: str) -> Any | None:
    """Compile and return the solve() function from a code string."""
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    try:
        exec(code, namespace)  # noqa: S102
        return namespace.get("solve")
    except Exception:
        return None


def _training_accuracy(code: str, task: ArcTask) -> tuple[float, list[PairFailure]]:
    """Return (accuracy_0_to_1, list_of_failures) for *code* on *task*."""
    solve_fn = _load_solve_fn(code)
    if solve_fn is None:
        return 0.0, []

    failures: list[PairFailure] = []
    for i, pair in enumerate(task.train):
        f = _diagnose_pair(i, pair.input, pair.output, solve_fn)
        if f is not None:
            failures.append(f)

    n_pass = len(task.train) - len(failures)
    acc = n_pass / len(task.train) if task.train else 0.0
    return acc, failures


# ─────────────────────────────────────────────────────────────────────────────
# Refinement prompt
# ─────────────────────────────────────────────────────────────────────────────


_REFINE_SYSTEM = (
    "You are an expert ARC-AGI solver performing iterative self-correction. "
    "You have been given:\n"
    "  1. Your current solve() function\n"
    "  2. Exactly which training pairs it FAILS on, with visual diffs\n\n"
    "Your job:\n"
    "  A. DIAGNOSE: Identify the root cause of each failure. Think step by step.\n"
    "     - Is the rule wrong?\n"
    "     - Does the rule handle all cases?\n"
    "     - Is there an off-by-one or colour confusion?\n"
    "  B. REVISE: Write a corrected solve() that passes ALL shown examples.\n\n"
    "RULES:\n"
    "  - Function signature: `def solve(grid: np.ndarray) -> np.ndarray:`\n"
    "  - Import only numpy (as np) and standard library modules\n"
    "  - Do NOT change the function name\n"
    "  - The function must be self-contained\n"
    "  - Fix the root cause, not just special-case the failing examples\n\n"
    "Respond with:\n"
    "  1. A brief diagnosis (2-5 sentences)\n"
    "  2. Your revised solve() in a fenced ```python``` block"
)


def _build_refinement_prompt(
    current_code: str,
    failures: list[PairFailure],
    iteration: int,
    n_train: int,
    analogy_hint: str = "",
) -> list[Any]:
    """Build the LLM refinement prompt showing failures with visual diffs."""
    from langchain_core.messages import HumanMessage, SystemMessage

    n_fail = len(failures)
    n_pass = n_train - n_fail

    failure_blocks = "\n\n".join(_format_failure(f) for f in failures)

    hint_section = ""
    if analogy_hint:
        hint_section = f"\n## Analogy hint (from symbolic analysis)\n{analogy_hint}\n"

    content = (
        f"## Refinement iteration {iteration}\n"
        f"**Training accuracy: {n_pass}/{n_train} pass, {n_fail}/{n_train} fail**\n\n"
        f"## Your current solve() function\n"
        f"```python\n{current_code}\n```\n\n"
        f"{hint_section}"
        f"## Failing training pairs\n\n"
        f"{failure_blocks}\n\n"
        "## Your task\n"
        "Diagnose the root cause of each failure, then write a corrected "
        "`solve()` function inside a ```python``` block."
    )

    return [SystemMessage(content=_REFINE_SYSTEM), HumanMessage(content=content)]


# ─────────────────────────────────────────────────────────────────────────────
# Main self-refinement loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RefinementResult:
    """Outcome of the self-refinement loop."""

    best_code: str
    best_accuracy: float
    iterations_run: int
    converged: bool
    """True if training accuracy reached 1.0."""
    history: list[tuple[int, float]] = field(default_factory=list)
    """(iteration, accuracy) for each round."""


def arc_self_refine(
    task: ArcTask,
    llm: Any,
    initial_code: str,
    max_iterations: int = 5,
    time_budget_s: float = 60.0,
    analogy_hint: str = "",
) -> RefinementResult:
    """Iteratively refine a solve() function until it passes all training pairs.

    Parameters
    ----------
    task:
        The ARC task with training pairs used as the verification oracle.
    llm:
        LangChain LLM instance.
    initial_code:
        Starting solve() function (e.g. from first LLM synthesis attempt).
    max_iterations:
        Maximum number of LLM refinement calls.
    time_budget_s:
        Wall-clock time limit.
    analogy_hint:
        Optional string from the analogy engine to include in prompts.

    Returns
    -------
    RefinementResult
        Best code found and convergence metadata.
    """
    t_start = time.perf_counter()

    # Evaluate initial code
    acc, failures = _training_accuracy(initial_code, task)
    best_code = initial_code
    best_acc = acc
    history = [(0, acc)]
    no_improvement_streak = 0

    logger.info(
        "ARC self-refine start: %.0f%% training accuracy, %d/%d fail",
        acc * 100, len(failures), len(task.train),
    )

    if acc == 1.0:
        return RefinementResult(
            best_code=best_code,
            best_accuracy=1.0,
            iterations_run=0,
            converged=True,
            history=history,
        )

    for iteration in range(1, max_iterations + 1):
        elapsed = time.perf_counter() - t_start
        if elapsed > time_budget_s:
            logger.info("ARC self-refine: time budget exhausted at iteration %d", iteration)
            break

        if not failures:
            break

        # Build refinement prompt and call LLM
        prompt = _build_refinement_prompt(
            current_code=best_code,
            failures=failures,
            iteration=iteration,
            n_train=len(task.train),
            analogy_hint=analogy_hint,
        )

        try:
            response = llm.invoke(prompt)
            content = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as exc:
            logger.warning("ARC self-refine: LLM call failed: %s", exc)
            break

        # Extract revised code
        match = _CODE_FENCE.search(content)
        if not match:
            logger.warning("ARC self-refine: no code block in LLM response (iter %d)", iteration)
            no_improvement_streak += 1
            if no_improvement_streak >= 2:
                break
            continue

        revised_code = match.group(1).strip()
        new_acc, new_failures = _training_accuracy(revised_code, task)
        history.append((iteration, new_acc))

        logger.info(
            "ARC self-refine iter %d: %.0f%% accuracy (%+.0f%%)",
            iteration, new_acc * 100, (new_acc - best_acc) * 100,
        )

        if new_acc > best_acc:
            best_code = revised_code
            best_acc = new_acc
            failures = new_failures
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            # Still update failures from best code for next iteration context
            # (don't revert to worse code)

        if best_acc == 1.0:
            logger.info("ARC self-refine: converged at iteration %d!", iteration)
            return RefinementResult(
                best_code=best_code,
                best_accuracy=1.0,
                iterations_run=iteration,
                converged=True,
                history=history,
            )

        if no_improvement_streak >= 2:
            logger.info("ARC self-refine: no improvement for 2 iterations — stopping")
            break

    return RefinementResult(
        best_code=best_code,
        best_accuracy=best_acc,
        iterations_run=len(history) - 1,
        converged=best_acc == 1.0,
        history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: refine + run on test pairs
# ─────────────────────────────────────────────────────────────────────────────


def refine_and_predict(
    task: ArcTask,
    llm: Any,
    initial_code: str,
    max_iterations: int = 5,
    time_budget_s: float = 60.0,
    analogy_hint: str = "",
) -> tuple[str, list[Grid | None], float]:
    """Run self-refinement and return (best_code, test_predictions, train_accuracy).

    This is the main entry point used by ``solver.py``.
    """
    result = arc_self_refine(
        task=task,
        llm=llm,
        initial_code=initial_code,
        max_iterations=max_iterations,
        time_budget_s=time_budget_s,
        analogy_hint=analogy_hint,
    )

    solve_fn = _load_solve_fn(result.best_code)
    predictions: list[Grid | None] = []
    if solve_fn is not None:
        for pair in task.test:
            try:
                predictions.append(solve_fn(pair.input.copy()))
            except Exception:
                predictions.append(None)
    else:
        predictions = [None] * len(task.test)

    return result.best_code, predictions, result.best_accuracy

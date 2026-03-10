"""ARC-AGI Program Synthesis Engine.

Implements a multi-strategy, test-time compute-scaling solver aligned with
François Chollet's philosophy:

1. **Analogy-guided hypothesis generation** — extract candidate rules from
   training pairs before searching (prunes the search space dramatically).
2. **Beam search over DSL compositions** — systematic enumeration with
   early pruning; depth-first when confidence is high.
3. **Parameterised primitive search** — test parametric variants (colour args,
   shift amounts, scale factors) guided by extracted deltas.
4. **LLM-backed custom code synthesis** — for tasks beyond the DSL,
   generate Python functions and validate against all training pairs.
5. **Ensemble verification** — multiple candidate programs voted/ranked by
   accuracy on training data before selecting the final answer.

The solver is designed to use *more* computation at test time when available
(like a human thinking harder on a difficult puzzle), consistent with Chollet's
emphasis on ``effective compute`` as the real measure of intelligence.
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac.arc.dsl import PRIMITIVES, apply_program, compose
from isaac.arc.evaluator import ArcTask, TaskResult
from isaac.arc.grid_ops import Grid

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate program dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CandidateProgram:
    """A candidate transformation program with its training accuracy."""

    ops: list[dict[str, Any]]
    """Serialised DSL operations."""
    train_accuracy: float = 0.0
    """Fraction of training pairs solved correctly (0.0–1.0)."""
    method: str = "dsl"
    score: float = 0.0
    """Composite score: accuracy + simplicity bonus."""


# ─────────────────────────────────────────────────────────────────────────────
# Parameterised search helpers
# ─────────────────────────────────────────────────────────────────────────────


def _colour_args_from_task(task: ArcTask) -> list[dict[str, int]]:
    """Extract all (from_colour, to_colour) pairs seen across training pairs."""
    colour_args: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in task.train:
        in_colours = set(np.unique(pair.input).tolist())
        out_colours = set(np.unique(pair.output).tolist())
        for fc in in_colours:
            for tc in out_colours:
                if fc != tc and (fc, tc) not in seen:
                    seen.add((fc, tc))
                    colour_args.append({"from_colour": fc, "to_colour": tc})
    return colour_args


def _shift_args_from_task(task: ArcTask) -> list[dict[str, int]]:
    """Generate plausible shift amounts based on grid dimensions."""
    max_dim = max(
        max(p.input.shape[0], p.input.shape[1]) for p in task.train
    )
    return [{"n": n} for n in range(1, min(max_dim, 8) + 1)]


def _scale_args_from_task(task: ArcTask) -> list[dict[str, int]]:
    """Generate scale factors consistent with output/input size ratios."""
    factors: list[dict[str, int]] = []
    seen: set[int] = set()
    for pair in task.train:
        for dim_in, dim_out in [
            (pair.input.shape[0], pair.output.shape[0]),
            (pair.input.shape[1], pair.output.shape[1]),
        ]:
            if dim_in > 0 and dim_out % dim_in == 0:
                f = dim_out // dim_in
                if f > 1 and f not in seen:
                    seen.add(f)
                    factors.append({"factor": f})
    return factors or [{"factor": 2}, {"factor": 3}]


def _tile_args_from_task(task: ArcTask) -> list[dict[str, int]]:
    """Generate tile (rows, cols) parameters from output/input ratios."""
    args: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in task.train:
        rr = pair.output.shape[0] / pair.input.shape[0] if pair.input.shape[0] > 0 else 0
        rc = pair.output.shape[1] / pair.input.shape[1] if pair.input.shape[1] > 0 else 0
        if rr == int(rr) and rc == int(rc) and int(rr) >= 1 and int(rc) >= 1:
            k = (int(rr), int(rc))
            if k not in seen:
                seen.add(k)
                args.append({"rows": k[0], "cols": k[1]})
    return args or [{"rows": 2, "cols": 2}, {"rows": 3, "cols": 3}]


def _fill_colour_arg_variants(task: ArcTask) -> list[dict[str, int]]:
    """Colour arg variants for fill_enclosed_regions."""
    colours: set[int] = set()
    for pair in task.train:
        colours.update(int(c) for c in np.unique(pair.output))
    bg = int(np.bincount(task.train[0].input.ravel()).argmax())
    colours.discard(bg)
    return [{"fill_col": c, "background": bg} for c in colours] or [{"fill_col": 1}]


# ─────────────────────────────────────────────────────────────────────────────
# Training accuracy evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_program(
    ops: list[dict[str, Any]],
    task: ArcTask,
) -> float:
    """Return the fraction of training pairs solved by *ops* (0.0–1.0)."""
    if not task.train:
        return 0.0
    correct = 0
    for pair in task.train:
        try:
            pred = apply_program(ops, pair.input)
            if np.array_equal(pred, pair.output):
                correct += 1
        except Exception:
            pass
    return correct / len(task.train)


def _score(candidate: CandidateProgram) -> float:
    """Composite score: accuracy with simplicity bonus for shorter programs."""
    simplicity_bonus = 0.01 * max(0, 5 - len(candidate.ops))
    return candidate.train_accuracy + simplicity_bonus


# ─────────────────────────────────────────────────────────────────────────────
# Beam search
# ─────────────────────────────────────────────────────────────────────────────


def _beam_search_dsl(
    task: ArcTask,
    max_depth: int = 3,
    beam_width: int = 20,
    time_budget_s: float = 10.0,
) -> list[CandidateProgram]:
    """Beam search over DSL primitive compositions up to *max_depth* steps.

    Uses analogy-guided priors to rank primitive candidates at each step.
    """
    from isaac.arc.analogy import run_analogy_engine

    # Build analogy to get priority hints
    train_dicts = [
        {"input": p.input.tolist(), "output": p.output.tolist()} for p in task.train
    ]
    analogy = run_analogy_engine(train_dicts)

    # Prioritised primitive list: DSL ops mentioned in top hypotheses first
    priority_ops: list[str] = []
    for hyp in analogy.hypotheses[:5]:
        for step in hyp.dsl_ops:
            if step.get("op") in PRIMITIVES and step["op"] not in priority_ops:
                priority_ops.append(step["op"])

    # Remaining ops in default order
    all_ops = priority_ops + [k for k in PRIMITIVES if k not in priority_ops]

    # Zero-arg ops (safe to call without extra args)
    zero_arg_ops = {
        "identity", "rotate_90", "rotate_180", "rotate_270",
        "flip_horizontal", "flip_vertical", "transpose", "diagonal_flip",
        "reflect_about_main_diagonal", "gravity_down", "gravity_up",
        "gravity_left", "gravity_right", "hollow_rectangle",
        "fill_enclosed_auto", "select_largest_object", "select_smallest_object",
        "recolour_by_size", "recolour_by_position", "outline_objects",
        "complete_symmetry_horizontal", "complete_symmetry_vertical",
        "mirror_objects_to_fill_symmetry", "connect_objects_horizontal",
        "connect_objects_vertical", "object_to_border", "center_object",
        "split_grid_horizontal", "split_grid_vertical", "count_to_cells",
        "crop_to_object", "normalise_to_square", "erode_objects", "expand_objects",
    }

    # Parametric ops with their arg variants
    parametric_ops: dict[str, list[dict[str, Any]]] = {
        "fill_colour": _colour_args_from_task(task),
        "keep_colour": [{"colour": c} for c in range(1, 10)],
        "remove_colour": [{"colour": c} for c in range(1, 10)],
        "scale_up": _scale_args_from_task(task),
        "tile_grid": _tile_args_from_task(task),
        "shift_right": _shift_args_from_task(task),
        "shift_left": _shift_args_from_task(task),
        "shift_down": _shift_args_from_task(task),
        "shift_up": _shift_args_from_task(task),
        "fill_enclosed_regions": _fill_colour_arg_variants(task),
        "pad_grid": [{"pad": n} for n in range(1, 4)],
        "add_border": [{"colour": c, "width": 1} for c in range(1, 10)],
        "remove_border": [{"n": n} for n in range(1, 4)],
        "crop_to_colour": [{"colour": c} for c in range(1, 10)],
    }

    t_start = time.perf_counter()
    solutions: list[CandidateProgram] = []

    # Build the candidate step list: (op_name, args_dict)
    def _step_candidates() -> list[tuple[str, dict[str, Any]]]:
        cands: list[tuple[str, dict[str, Any]]] = []
        for op in all_ops:
            if op in zero_arg_ops:
                cands.append((op, {}))
            elif op in parametric_ops:
                for args in parametric_ops[op]:
                    cands.append((op, args))
        return cands

    step_candidates = _step_candidates()

    # Beam: list of (ops_so_far, best_accuracy)
    beam: list[tuple[list[dict[str, Any]], float]] = [([], 0.0)]

    for depth in range(max_depth):
        if time.perf_counter() - t_start > time_budget_s:
            logger.debug("Beam search: time budget exhausted at depth %d", depth)
            break

        next_beam: list[tuple[list[dict[str, Any]], float]] = []

        for ops_prefix, _ in beam:
            for op_name, args in step_candidates:
                if time.perf_counter() - t_start > time_budget_s:
                    break
                new_ops = ops_prefix + [{"op": op_name, "args": args} if args else {"op": op_name}]
                acc = _evaluate_program(new_ops, task)
                if acc == 1.0:
                    cand = CandidateProgram(ops=new_ops, train_accuracy=1.0, method="dsl_beam")
                    cand.score = _score(cand)
                    solutions.append(cand)
                elif acc > 0.0:
                    next_beam.append((new_ops, acc))

        # Keep only the top beam_width candidates
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]

    # Also include partial solutions (top by accuracy)
    for ops, acc in beam[:5]:
        if acc > 0.0:
            cand = CandidateProgram(ops=ops, train_accuracy=acc, method="dsl_partial")
            cand.score = _score(cand)
            solutions.append(cand)

    solutions.sort(key=lambda c: c.score, reverse=True)
    return solutions


# ─────────────────────────────────────────────────────────────────────────────
# Analogy-direct solver
# ─────────────────────────────────────────────────────────────────────────────


def _analogy_direct_solve(task: ArcTask) -> list[CandidateProgram]:
    """Use analogy engine hypotheses directly as candidate programs."""
    from isaac.arc.analogy import run_analogy_engine

    train_dicts = [
        {"input": p.input.tolist(), "output": p.output.tolist()} for p in task.train
    ]
    analogy = run_analogy_engine(train_dicts)

    solutions: list[CandidateProgram] = []
    for hyp in analogy.hypotheses:
        if not hyp.dsl_ops or hyp.requires_custom_code:
            continue
        acc = _evaluate_program(hyp.dsl_ops, task)
        if acc > 0.0:
            cand = CandidateProgram(
                ops=hyp.dsl_ops,
                train_accuracy=acc,
                method=f"analogy:{hyp.name}",
            )
            cand.score = _score(cand)
            solutions.append(cand)

    solutions.sort(key=lambda c: c.score, reverse=True)
    return solutions


# ─────────────────────────────────────────────────────────────────────────────
# LLM synthesis
# ─────────────────────────────────────────────────────────────────────────────


def _llm_solve_enriched(
    task: ArcTask,
    llm: Any,
    analogy_ctx: str,
    object_ctx: str,
    prior_obs: list[str],
) -> tuple[str, float] | None:
    """Generate a Python solve() function via LLM with full symbolic context.

    Uses the ARC-specific chain-of-thought prompt enriched with:
    - Analogy engine findings
    - Object-level scene graph
    - Core knowledge prior observations

    Returns (code, train_accuracy) or None on failure.
    """
    try:
        from isaac.arc.grid_ops import format_grid_for_prompt
        from isaac.llm.prompts import arc_synthesis_prompt

        # Build training examples for prompt
        training_examples = [
            {
                "input_str": format_grid_for_prompt(p.input),
                "output_str": format_grid_for_prompt(p.output),
            }
            for p in task.train
        ]

        # Hypothesis from task description or analogy
        hypothesis = task.description or "Infer the transformation rule from examples."

        prompt = arc_synthesis_prompt(
            task_description=hypothesis,
            training_examples=training_examples,
            analogy_context=analogy_ctx + "\n\n" + object_ctx,
            prior_observations=prior_obs,
        )

        response = llm.invoke(prompt)
        content = (
            response.content if isinstance(response.content, str)
            else str(response.content)
        )

        import re
        match = re.search(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
        if not match:
            return None
        code = match.group(1).strip()

        if "def solve" not in code:
            return None

        # Validate on training data
        namespace: dict[str, Any] = {"np": np, "numpy": np}
        exec(code, namespace)  # noqa: S102
        solve_fn = namespace.get("solve")
        if solve_fn is None:
            return None

        correct = sum(
            1 for p in task.train
            if _safe_equal_fn(solve_fn, p.input, p.output)
        )
        acc = correct / len(task.train) if task.train else 0.0
        return code, acc

    except Exception as exc:
        logger.debug("Enriched LLM solve failed: %s", exc)
        return None


def _safe_equal_fn(fn: Any, inp: Grid, expected: Grid) -> bool:
    try:
        result = fn(inp.copy())
        return np.array_equal(np.array(result, dtype=int), expected)
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main synthesis engine — 5 strategies + self-refinement
# ─────────────────────────────────────────────────────────────────────────────


def synthesise(
    task: ArcTask,
    llm: Any | None = None,
    time_budget_s: float = 30.0,
    beam_width: int = 30,
    max_depth: int = 3,
    max_refine_iterations: int = 5,
) -> TaskResult:
    """Full multi-strategy program synthesis for a single ARC task.

    Strategy order (early exit on 100% training accuracy):

    1. **Analogy-direct** — test DSL hypotheses from the analogy engine (O(1))
    2. **Beam search** — systematic DSL composition search (guided by analogy)
    3. **Object-level synthesis** — symbolic per-object rule inference (neurosymbolic)
    4. **LLM synthesis** — chain-of-thought with prior + analogy + object context
    5. **LLM self-refinement** — iterative fix loop: failing pairs → diagnose → revise

    Each strategy uses more compute than the previous.  The system exits as
    soon as any strategy finds a program that passes all training pairs.
    This embodies Chollet's *test-time compute scaling* principle.

    Parameters
    ----------
    task:
        The ARC task with training and test pairs.
    llm:
        Optional LLM instance. Strategies 4+5 require it; 1-3 are LLM-free.
    time_budget_s:
        Total wall-clock seconds. Divided proportionally across strategies.
    beam_width:
        Beam width for DSL search.
    max_depth:
        Maximum DSL composition depth.
    max_refine_iterations:
        Maximum self-refinement iterations (strategy 5).
    """
    t_start = time.perf_counter()
    all_candidates: list[CandidateProgram] = []

    # Pre-compute symbolic context once (used by strategies 4+5)
    _analogy_ctx = ""
    _object_ctx = ""
    _prior_obs: list[str] = []
    _analogy_hint = ""

    def _build_symbolic_context() -> None:
        nonlocal _analogy_ctx, _object_ctx, _prior_obs, _analogy_hint
        if _analogy_ctx:
            return  # Already built
        try:
            from isaac.arc.analogy import run_analogy_engine, format_analogy_for_prompt
            from isaac.arc.priors import full_prior_analysis, describe_prior_analysis
            from isaac.arc.object_synthesis import build_object_context_for_llm

            train_dicts = [
                {"input": p.input.tolist(), "output": p.output.tolist()} for p in task.train
            ]
            analogy = run_analogy_engine(train_dicts)
            _analogy_ctx = format_analogy_for_prompt(analogy)
            _analogy_hint = (
                analogy.hypotheses[0].description if analogy.hypotheses else ""
            )

            train_pairs_np = [(p.input, p.output) for p in task.train]
            _object_ctx = build_object_context_for_llm(train_pairs_np)

            for pair in task.train[:2]:
                pa = full_prior_analysis(pair.input)
                _prior_obs.extend(describe_prior_analysis(pa)[:5])
        except Exception as exc:
            logger.debug("Symbolic context build failed: %s", exc)

    # ── Strategy 1: Analogy-direct ────────────────────────────────────────
    try:
        cands = _analogy_direct_solve(task)
        all_candidates.extend(cands)
        perfect = [c for c in cands if c.train_accuracy == 1.0]
        if perfect:
            best = perfect[0]
            logger.info("Solver[1/analogy-direct] solved %s", task.id)
            return _make_task_result(task, best, t_start)
    except Exception as exc:
        logger.debug("Strategy 1 failed: %s", exc)

    # ── Strategy 2: Beam search ───────────────────────────────────────────
    remaining = time_budget_s - (time.perf_counter() - t_start)
    if remaining > 1.0:
        try:
            cands = _beam_search_dsl(
                task,
                max_depth=max_depth,
                beam_width=beam_width,
                time_budget_s=remaining * 0.35,
            )
            all_candidates.extend(cands)
            perfect = [c for c in cands if c.train_accuracy == 1.0]
            if perfect:
                best = perfect[0]
                logger.info("Solver[2/beam-search] solved %s (depth=%d)",
                            task.id, len(best.ops))
                return _make_task_result(task, best, t_start)
        except Exception as exc:
            logger.debug("Strategy 2 failed: %s", exc)

    # ── Strategy 3: Object-level synthesis (neurosymbolic) ────────────────
    remaining = time_budget_s - (time.perf_counter() - t_start)
    if remaining > 1.0:
        try:
            from isaac.arc.object_synthesis import synthesise_from_object_rules
            result = synthesise_from_object_rules(task)
            if result is not None:
                code, acc = result
                cand = CandidateProgram(
                    ops=[{"op": "_custom_python", "code": code}],
                    train_accuracy=acc,
                    method="object_synthesis",
                )
                cand.score = _score(cand)
                all_candidates.append(cand)
                if acc == 1.0:
                    logger.info("Solver[3/object-synthesis] solved %s", task.id)
                    return _make_task_result(task, cand, t_start)
        except Exception as exc:
            logger.debug("Strategy 3 failed: %s", exc)

    # ── Strategy 4: LLM synthesis with enriched context ───────────────────
    initial_llm_code: str | None = None
    remaining = time_budget_s - (time.perf_counter() - t_start)
    if llm is not None and remaining > 3.0:
        _build_symbolic_context()
        try:
            result_llm = _llm_solve_enriched(
                task, llm, _analogy_ctx, _object_ctx, _prior_obs
            )
            if result_llm is not None:
                code, acc = result_llm
                initial_llm_code = code
                cand = CandidateProgram(
                    ops=[{"op": "_custom_python", "code": code}],
                    train_accuracy=acc,
                    method="llm_enriched",
                )
                cand.score = _score(cand)
                all_candidates.append(cand)
                if acc == 1.0:
                    logger.info("Solver[4/llm-enriched] solved %s", task.id)
                    return _make_task_result(task, cand, t_start)
                logger.info(
                    "Solver[4/llm-enriched] %.0f%% training — escalating to refinement",
                    acc * 100,
                )
        except Exception as exc:
            logger.debug("Strategy 4 failed: %s", exc)

    # ── Strategy 5: LLM self-refinement loop ──────────────────────────────
    remaining = time_budget_s - (time.perf_counter() - t_start)
    if llm is not None and remaining > 5.0:
        _build_symbolic_context()

        # Use LLM code from strategy 4 if available; else generate a quick initial
        starting_code = initial_llm_code
        if starting_code is None:
            try:
                from isaac.arc.evaluator import build_arc_prompt
                import re as _re
                prompt_text = build_arc_prompt(task)
                from langchain_core.messages import HumanMessage, SystemMessage
                quick_msgs = [
                    SystemMessage(content=(
                        "You are an ARC-AGI expert. Write a Python `solve(grid)` function. "
                        "Use only numpy. Respond ONLY with a fenced ```python``` block."
                    )),
                    HumanMessage(content=prompt_text),
                ]
                resp = llm.invoke(quick_msgs)
                cnt = resp.content if isinstance(resp.content, str) else str(resp.content)
                m = _re.search(r"```(?:python)?\s*\n(.*?)```", cnt, _re.DOTALL)
                if m:
                    starting_code = m.group(1).strip()
            except Exception as exc:
                logger.debug("Quick LLM code generation failed: %s", exc)

        if starting_code is not None:
            try:
                from isaac.arc.refinement import refine_and_predict

                refined_code, predictions, acc = refine_and_predict(
                    task=task,
                    llm=llm,
                    initial_code=starting_code,
                    max_iterations=max_refine_iterations,
                    time_budget_s=remaining * 0.9,
                    analogy_hint=_analogy_hint,
                )
                cand = CandidateProgram(
                    ops=[{"op": "_custom_python", "code": refined_code}],
                    train_accuracy=acc,
                    method="llm_self_refine",
                )
                cand.score = _score(cand)
                all_candidates.append(cand)

                if acc == 1.0:
                    logger.info("Solver[5/self-refine] solved %s", task.id)
                    # Build TaskResult from predictions (already computed in refinement)
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    correct = all(
                        pred is not None and np.array_equal(pred, pair.output)
                        for pred, pair in zip(predictions, task.test)
                    )
                    return TaskResult(
                        task_id=task.id,
                        correct=correct,
                        predicted=predictions,
                        program=refined_code,
                        solve_time_ms=elapsed_ms,
                        method="llm_self_refine",
                    )
                logger.info(
                    "Solver[5/self-refine] best: %.0f%% training for %s",
                    acc * 100, task.id,
                )
            except Exception as exc:
                logger.debug("Strategy 5 failed: %s", exc)

    # ── Return best partial solution ──────────────────────────────────────
    if all_candidates:
        best = max(all_candidates, key=lambda c: c.score)
        logger.info(
            "Solver: best partial for %s: %.0f%% training, method=%s",
            task.id, best.train_accuracy * 100, best.method,
        )
        return _make_task_result(task, best, t_start)

    elapsed = (time.perf_counter() - t_start) * 1000
    return TaskResult(
        task_id=task.id,
        correct=False,
        predicted=[None] * len(task.test),
        program="unsolved",
        solve_time_ms=elapsed,
        method="synthesis_engine",
    )


def _make_task_result(
    task: ArcTask,
    best: CandidateProgram,
    t_start: float,
) -> TaskResult:
    """Convert a CandidateProgram into a TaskResult by running on test pairs."""
    elapsed = (time.perf_counter() - t_start) * 1000

    # Custom Python code path
    if best.ops and best.ops[0].get("op") == "_custom_python":
        code = best.ops[0].get("code", "")
        try:
            namespace: dict[str, Any] = {"np": np, "numpy": np}
            exec(code, namespace)  # noqa: S102
            solve_fn = namespace.get("solve")
            if solve_fn is not None:
                predictions = [solve_fn(p.input) for p in task.test]
                correct = all(
                    np.array_equal(pred, p.output)
                    for pred, p in zip(predictions, task.test)
                )
                return TaskResult(
                    task_id=task.id,
                    correct=correct,
                    predicted=predictions,
                    program=code,
                    solve_time_ms=elapsed,
                    method=best.method,
                )
        except Exception:
            pass

    # DSL program path
    predictions = []
    for pair in task.test:
        try:
            pred = apply_program(best.ops, pair.input)
            predictions.append(pred)
        except Exception:
            predictions.append(None)

    correct = all(
        pred is not None and np.array_equal(pred, pair.output)
        for pred, pair in zip(predictions, task.test)
    )

    return TaskResult(
        task_id=task.id,
        correct=correct,
        predicted=predictions,
        program=best.ops,
        solve_time_ms=elapsed,
        method=best.method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch solve
# ─────────────────────────────────────────────────────────────────────────────


def solve_batch(
    tasks: list[ArcTask],
    llm: Any | None = None,
    time_budget_per_task_s: float = 30.0,
    beam_width: int = 30,
    max_depth: int = 3,
) -> list[TaskResult]:
    """Synthesise solutions for a list of ARC tasks.

    Parameters
    ----------
    tasks:
        List of ARC tasks.
    llm:
        Optional LLM for code generation fallback.
    time_budget_per_task_s:
        Per-task time budget in seconds.

    Returns
    -------
    list[TaskResult]
        One result per task.
    """
    results: list[TaskResult] = []
    for task in tasks:
        try:
            result = synthesise(
                task,
                llm=llm,
                time_budget_s=time_budget_per_task_s,
                beam_width=beam_width,
                max_depth=max_depth,
            )
        except Exception as exc:
            logger.error("Synthesis failed for task %s: %s", task.id, exc)
            result = TaskResult(
                task_id=task.id,
                correct=False,
                predicted=[None] * len(task.test),
                program=f"error: {exc}",
                method="synthesis_engine",
            )
        results.append(result)
    return results

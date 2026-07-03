"""ARC-AGI benchmark adapter — load ARC-AGI-1 tasks as :class:`EvalTask` suites.

ARC-AGI (Chollet, 2019 — https://github.com/fchollet/ARC-AGI) is a public,
ungated benchmark of abstraction-and-reasoning puzzles: infer a grid
transformation from a few training pairs, apply it to test inputs. It is one
of the public benchmarks named by the ROADMAP-1.0 §1 evidence gate, and the
cheapest-signal-first choice from WS1 (the solver already ships in
:mod:`isaac.arc`).

Usage::

    isaac eval <dir-with-task-jsons> --format arc
    isaac eval --format arc --download          # fetch the public dataset first

Scoring: a task passes only when **every** test output grid matches the
ground truth exactly (all-cells, all-pairs). This is *stricter* than the
official leaderboard protocol, which allows two attempts per test input
(pass@2); the solver produces a single attempt, so a score here is a
conservative lower bound on the official metric.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

from isaac.eval.suite import EvalTask

logger = logging.getLogger(__name__)

ARC_AGI_1_ZIP = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
ARC_SPLITS = ("training", "evaluation")


# ---------------------------------------------------------------------------
# Scoring (exact grid match, single attempt)
# ---------------------------------------------------------------------------


def score_arc_answer(answer_text: str, ground_truth_json: str) -> tuple[bool, str]:
    """Exact-match every predicted test grid against the ground truth.

    ``answer_text`` is the runner's JSON list of predicted grids (``null`` for
    unanswered pairs); ``ground_truth_json`` is the JSON list of expected
    grids. Returns ``(passed, detail)`` and never raises — a malformed answer
    scores as failed, matching the harness's one-bad-task-never-aborts rule.
    """
    expected = json.loads(ground_truth_json)
    try:
        predicted = json.loads(answer_text)
    except json.JSONDecodeError:
        return False, "answer is not valid JSON"
    if not isinstance(predicted, list) or len(predicted) != len(expected):
        return False, f"expected {len(expected)} prediction(s), got {predicted!r:.60}"
    exact = sum(1 for p, e in zip(predicted, expected, strict=True) if p == e)
    return exact == len(expected), f"{exact}/{len(expected)} test grids exact"


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_arc_tasks(
    split_dir: str | Path,
    *,
    time_budget_s: float = 30.0,
) -> list[EvalTask]:
    """Load every ``*.json`` ARC task in a split directory (official format:
    one file per task, ``{"train": [...], "test": [...]}``, id = file stem).

    ``time_budget_s`` becomes each task's ``timeout_seconds`` — the runner
    uses it as the solver's per-task test-time compute budget.
    """
    split = Path(split_dir)
    task_files = sorted(split.glob("*.json"))
    if not task_files:
        raise FileNotFoundError(
            f"No ARC task files (*.json) in {split}. Download the public dataset "
            "first (isaac eval --format arc --download)."
        )

    tasks: list[EvalTask] = []
    for path in task_files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON — {exc}") from exc
        test_pairs = obj.get("test") or []
        if not obj.get("train") or not test_pairs:
            logger.warning("ARC %s: missing train/test pairs; skipping", path.stem)
            continue
        expected = [p["output"] for p in test_pairs]
        tasks.append(
            EvalTask(
                id=path.stem,
                prompt=(
                    f"ARC-AGI task {path.stem}: infer the grid transformation from "
                    f"{len(obj['train'])} training pair(s) and apply it to "
                    f"{len(test_pairs)} test input(s)."
                ),
                checks=[{"type": "arc", "value": json.dumps(expected)}],
                category=f"arc-agi-1-{split.name}",
                timeout_seconds=time_budget_s,
            )
        )
    if not tasks:
        raise ValueError(f"No scoreable ARC tasks found in {split}.")
    return tasks


def arc_runner(split_dir: str | Path, llm: object | None = None):
    """A runner for ARC suites: solves each task with the bundled synthesis
    engine (:func:`isaac.arc.solver.synthesise`) instead of the AgentLoop,
    and answers with the JSON list of predicted test grids.

    ``llm=None`` keeps the run purely symbolic (strategies 1-3, LLM-free) —
    fully reproducible on any machine with no model configured.
    """
    import numpy as np

    from isaac.arc.evaluator import ArcPair, ArcTask
    from isaac.arc.solver import synthesise
    from isaac.eval.runner import TaskAnswer

    split = Path(split_dir)

    def run(task: EvalTask) -> TaskAnswer:
        obj = json.loads((split / f"{task.id}.json").read_text(encoding="utf-8"))
        arc_task = ArcTask(
            id=task.id,
            train=[
                ArcPair(
                    input=np.array(p["input"], dtype=int),
                    output=np.array(p["output"], dtype=int),
                )
                for p in obj["train"]
            ],
            test=[
                ArcPair(
                    input=np.array(p["input"], dtype=int),
                    output=np.array(p["output"], dtype=int),
                )
                for p in obj["test"]
            ],
        )
        result = synthesise(arc_task, llm=llm, time_budget_s=task.timeout_seconds)
        predicted = [p.tolist() if p is not None else None for p in result.predicted]
        return TaskAnswer(text=json.dumps(predicted), stopped_reason="final")

    return run


def download_arc(dest: str | Path | None = None) -> Path:
    """Download the public ARC-AGI-1 dataset (training + evaluation splits)
    from the official GitHub repository. No auth, no gating.

    Returns the ``evaluation`` split directory (400 tasks).
    """
    if dest is None:
        from isaac.config.settings import get_settings

        dest = get_settings().isaac_home / "datasets" / "arc-agi-1"
    dest = Path(dest)

    eval_dir = dest / "evaluation"
    if list(eval_dir.glob("*.json")):
        logger.info("ARC-AGI-1 already present at %s", dest)
        return eval_dir

    logger.info("Downloading ARC-AGI-1 from %s ...", ARC_AGI_1_ZIP)
    with urllib.request.urlopen(ARC_AGI_1_ZIP, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            # <repo>-master/data/<split>/<task>.json
            if (
                len(parts) == 4
                and parts[1] == "data"
                and parts[2] in ARC_SPLITS
                and parts[3].endswith(".json")
            ):
                target = dest / parts[2] / parts[3]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
    n = len(list(eval_dir.glob("*.json")))
    if not n:
        raise RuntimeError(f"ARC download produced no evaluation tasks under {dest}")
    logger.info("ARC-AGI-1 ready: %d evaluation tasks at %s", n, eval_dir)
    return eval_dir

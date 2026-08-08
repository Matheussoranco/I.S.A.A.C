"""Run the 1.5.0 self-improvement ablation on the golden suite.

Usage (from the repo root)::

    python evals/run_ablation.py --trials 3 --warmup 2 --out evals/results/ablation.json

Task selection is **pre-registered** and deterministic: the first
``--per-category`` tasks of each category in ``golden_v1.jsonl``, in file
order.  Stating the rule up front is the point — it removes the option of
quietly reselecting tasks once the numbers are in.

Every task is forced through the specialist ``team`` runner, because
MetaLearner-guided selection is what is being ablated and the single-agent
runner never consults it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


def pick_tasks(suite: Path, per_category: int) -> list:
    """First *per_category* tasks of each category, in file order."""
    from isaac.eval.suite import load_suite

    seen: dict[str, int] = {}
    picked = []
    for task in load_suite(suite):
        n = seen.get(task.category, 0)
        if n < per_category:
            seen[task.category] = n + 1
            task.runner = "team"  # the arm under test only exists on this path
            picked.append(task)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default=str(REPO / "evals" / "golden_v1.jsonl"))
    ap.add_argument("--trials", type=int, default=3, help="Paired trials per arm.")
    ap.add_argument("--warmup", type=int, default=2, help="History-building passes.")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--out", default=str(REPO / "evals" / "results" / "ablation_1.5.0.json"))
    ap.add_argument("--task-timeout", type=float, default=180.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    from isaac.config.settings import get_settings
    from isaac.eval.ablation import format_report, run_ablation
    from isaac.tools import register_all_tools

    register_all_tools()

    tasks = pick_tasks(Path(args.suite), args.per_category)
    for t in tasks:
        t.timeout_seconds = args.task_timeout

    settings = get_settings()
    total = (args.warmup + 2 * args.trials) * len(tasks)
    print(
        f"Ablation: {len(tasks)} tasks x ({args.warmup} warmup + "
        f"2 arms x {args.trials} trials) = {total} team runs",
        flush=True,
    )
    print(f"  model    : {settings.llm.model_name} ({settings.llm.llm_provider})", flush=True)
    print(f"  tasks    : {', '.join(t.id for t in tasks)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_event(kind: str, data: dict) -> None:
        if kind == "task_done":
            mark = "PASS" if data["passed"] else "FAIL"
            print(
                f"  [{data['arm']}/{data['trial']}] {data['task_id']:<18} {mark}",
                flush=True,
            )
        elif kind == "trial_done":
            print(
                f"== {data['arm']}/trial {data['trial']}: {data['accuracy']:.3f}",
                flush=True,
            )
        elif kind == "warmup_done":
            print(f"== warm-up history: {json.dumps(data['history'])}", flush=True)

    report = run_ablation(
        tasks,
        trials=args.trials,
        warmup_trials=args.warmup,
        suite_name=f"golden_v1[{args.per_category}-per-category]",
        on_event=on_event,
        checkpoint_path=out_path,
    )

    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    print("\n" + format_report(report), flush=True)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

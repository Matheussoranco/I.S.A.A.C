"""Turn an ablation JSON report into the Markdown tables used in the docs.

Usage::

    python evals/summarise_ablation.py evals/results/ablation_1.5.0.json

Works on a *partial* report too — ``run_ablation`` checkpoints after every
trial, and only complete ON/OFF pairs are counted, so an interrupted run still
yields an honest (if smaller) n rather than a lopsided comparison.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def paired_trials(report: dict) -> tuple[list[dict], list[dict]]:
    """Return (on, off) trials truncated to complete pairs only."""
    on = sorted((t for t in report["trials"] if t["arm"] == "on"), key=lambda t: t["trial"])
    off = sorted((t for t in report["trials"] if t["arm"] == "off"), key=lambda t: t["trial"])
    n = min(len(on), len(off))
    return on[:n], off[:n]


def specialist_usage(trials: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trials:
        for names in t.get("plans", {}).values():
            for name in names:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/results/ablation_1.5.0.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    on, off = paired_trials(report)
    n = len(on)

    if n == 0:
        print(f"No complete ON/OFF pair in {path} yet ({len(report['trials'])} trial(s) recorded).")
        return 1

    on_acc = [t["accuracy"] for t in on]
    off_acc = [t["accuracy"] for t in off]
    tasks = report["task_ids"]

    print(f"# Ablation summary — {report['suite']}")
    print()
    print(f"- model: `{report['model']}` ({report['provider']})")
    print(f"- git rev: `{report['git_rev']}`  suite hash: `{report['suite_hash']}`")
    print(f"- {len(tasks)} tasks, team runner, {report['warmup_trials']} warm-up pass(es)")
    print(f"- **n = {n} paired trials per arm**")
    print()
    print("| arm | mean accuracy | stdev | per-trial |")
    print("|---|---|---|---|")
    for label, acc in (("ON", on_acc), ("OFF", off_acc)):
        per = ", ".join(f"{a:.3f}" for a in acc)
        sd = statistics.stdev(acc) if len(acc) > 1 else 0.0
        print(f"| {label} | {statistics.fmean(acc):.3f} | {sd:.3f} | {per} |")
    print()

    delta = statistics.fmean(on_acc) - statistics.fmean(off_acc)
    print(f"- delta (ON - OFF): **{delta:+.3f}** accuracy points")
    print(f"- permutation p (paired by task): {report.get('permutation_p_by_task', 'n/a')}")
    print(f"- verdict: **{report.get('verdict', '?').upper()}**")
    print()

    on_counts = {t: 0 for t in tasks}
    off_counts = {t: 0 for t in tasks}
    for t in on:
        for tid, ok in t["passed"].items():
            on_counts[tid] += int(ok)
    for t in off:
        for tid, ok in t["passed"].items():
            off_counts[tid] += int(ok)

    print(f"| task | ON (/{n}) | OFF (/{n}) | delta |")
    print("|---|---|---|---|")
    for tid in tasks:
        d = on_counts[tid] - off_counts[tid]
        print(f"| `{tid}` | {on_counts[tid]} | {off_counts[tid]} | {d:+d} |")
    print()

    print("Specialist dispatch counts (how the arms actually differed):")
    print()
    print("| specialist | ON | OFF |")
    print("|---|---|---|")
    u_on, u_off = specialist_usage(on), specialist_usage(off)
    for name in sorted(set(u_on) | set(u_off), key=lambda k: -(u_on.get(k, 0) + u_off.get(k, 0))):
        print(f"| {name} | {u_on.get(name, 0)} | {u_off.get(name, 0)} |")
    print()

    if report.get("warmup_history"):
        print("Warm-up history the ON arm consumed:")
        print()
        print("| specialist | wins | runs | raw win-rate | smoothed score |")
        print("|---|---|---|---|---|")
        for row in report["warmup_history"]:
            print(
                f"| {row['name']} | {row['wins']} | {row['runs']} | "
                f"{row['raw_win_rate']:.2f} | {row['score']:.3f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

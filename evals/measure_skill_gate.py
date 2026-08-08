"""Measure the skill promotion gate on real LLM-generalised skills.

The unit tests prove the gate rejects hand-built broken candidates.  That is
not the same as knowing how often it fires on *actual* model output, which is
the number worth reporting.  This script reproduces exactly what
``skill_abstraction_node`` does — take concrete task code, ask the model to
generalise it into a reusable function — and pushes each result through
:class:`~isaac.memory.skill_library.SkillLibrary`'s gate, then prints the
promote/reject counts and the reasons.

Usage::

    python evals/measure_skill_gate.py --out evals/results/skill_gate.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: Concrete, task-shaped snippets of the kind Reflection hands to the
#: abstraction node.  Deliberately varied: some are trivially generalisable,
#: some carry task-local state that a careless generalisation will strand.
SNIPPETS: list[tuple[str, str]] = [
    (
        "count_words",
        "text = open('report.txt').read()\nprint(len(text.split()))",
    ),
    (
        "sum_csv_column",
        "import csv\nrows = list(csv.DictReader(open('sales.csv')))\n"
        "print(sum(float(r['amount']) for r in rows))",
    ),
    (
        "rotate_grid",
        "grid = [[1,2],[3,4]]\nprint([list(r) for r in zip(*grid[::-1])])",
    ),
    (
        "dedupe_lines",
        "lines = open('log.txt').read().splitlines()\n"
        "seen = set()\nout = [x for x in lines if not (x in seen or seen.add(x))]\n"
        "print('\\n'.join(out))",
    ),
    (
        "fizzbuzz",
        "for i in range(1, 16):\n"
        "    print('FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else "
        "'Buzz' if i%5==0 else i)",
    ),
    (
        "parse_iso_dates",
        "from datetime import datetime\n"
        "d = datetime.fromisoformat('2026-01-02')\nprint((d.weekday()))",
    ),
    (
        "flatten_json",
        "import json\nobj = json.load(open('data.json'))\n"
        "flat = {}\n"
        "def walk(o, p=''):\n"
        "    for k, v in o.items():\n"
        "        if isinstance(v, dict): walk(v, p+k+'.')\n"
        "        else: flat[p+k] = v\n"
        "walk(obj)\nprint(flat)",
    ),
    (
        "top_n_frequent",
        "from collections import Counter\n"
        "words = open('book.txt').read().lower().split()\n"
        "print(Counter(words).most_common(5))",
    ),
    (
        "rename_by_extension",
        "import os\nfor f in os.listdir('.'):\n"
        "    if f.endswith('.txt'): os.rename(f, f[:-4] + '.md')",
    ),
    (
        "http_status_check",
        "import urllib.request\n"
        "print(urllib.request.urlopen('http://example.com').status)",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "evals" / "results" / "skill_gate.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

    from isaac.core.state import SkillCandidate
    from isaac.llm.prompts import skill_abstraction_prompt
    from isaac.llm.provider import get_llm
    from isaac.memory.skill_library import SkillLibrary
    from isaac.nodes.skill_abstraction import _extract_code

    llm = get_llm("strong")
    records: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="isaac-gate-") as tmp:
        lib = SkillLibrary(Path(tmp))
        for name, concrete in SNIPPETS:
            try:
                response = llm.invoke(
                    skill_abstraction_prompt(concrete_code=concrete, task_context=name)
                )
                content = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                generalised = _extract_code(content)
            except Exception as exc:
                print(f"  {name:22s} LLM FAILED: {exc}", flush=True)
                continue

            outcome = lib.commit(
                SkillCandidate(name=name, code=generalised, task_context=name, success_count=1)
            )
            mark = "PROMOTE" if outcome.promoted else "REJECT "
            print(f"  {name:22s} {mark} {outcome.evidence:10s} {outcome.reason[:70]}", flush=True)
            records.append(
                {
                    "name": name,
                    "promoted": outcome.promoted,
                    "evidence": outcome.evidence,
                    "reason": outcome.reason,
                    "code": generalised,
                }
            )

        stats = lib.promotion_stats()

    payload = {"stats": stats, "candidates": records}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

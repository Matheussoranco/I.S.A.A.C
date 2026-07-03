"""Programmatic answer checkers.

Every checker is a small pure function over ``(answer_text, workspace_dir)``;
a task passes only when **all** of its checks pass.  Checkers are deliberately
dumb and deterministic — no LLM judging — so a score means the same thing on
every machine.

Checker specs (the ``checks`` entries in a suite)::

    {"type": "contains",      "value": "...", "case_sensitive": false}
    {"type": "not_contains",  "value": "..."}
    {"type": "any_of",        "values": ["a", "b"]}        # at least one substring
    {"type": "all_of",        "values": ["a", "b"]}        # every substring
    {"type": "regex",         "pattern": "..."}
    {"type": "numeric",       "value": 408, "tolerance": 0}  # any number in answer
    {"type": "min_length",    "value": 100}
    {"type": "file_exists",   "path": "out/report.md"}        # workspace-relative
    {"type": "file_contains", "path": "out/report.md", "value": "..."}
    {"type": "file_regex",    "path": "out/report.md", "pattern": "..."}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class CheckOutcome:
    """Result of one checker against one answer."""

    type: str
    passed: bool
    detail: str = ""


def _read_workspace_file(workspace: Path, rel: str) -> str | None:
    target = (workspace / rel).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError:
        return None  # path escapes the workspace — treat as missing
    if not target.is_file():
        return None
    return target.read_text(encoding="utf-8", errors="replace")


def run_check(spec: dict, answer: str, workspace: Path) -> CheckOutcome:
    """Evaluate one checker spec. Unknown/malformed specs fail loudly in the
    outcome (never raise) so one bad task cannot abort a whole run."""
    kind = str(spec.get("type", ""))
    try:
        if kind in ("contains", "not_contains"):
            needle = str(spec["value"])
            hay = answer if spec.get("case_sensitive", False) else answer.lower()
            needle_cmp = needle if spec.get("case_sensitive", False) else needle.lower()
            found = needle_cmp in hay
            passed = found if kind == "contains" else not found
            return CheckOutcome(kind, passed, f"'{needle}' {'found' if found else 'not found'}")

        if kind in ("any_of", "all_of"):
            values = [str(v) for v in spec["values"]]
            hay = answer.lower()
            hits = [v for v in values if v.lower() in hay]
            passed = bool(hits) if kind == "any_of" else len(hits) == len(values)
            return CheckOutcome(kind, passed, f"matched {len(hits)}/{len(values)}")

        if kind == "regex":
            pattern = str(spec["pattern"])
            m = re.search(pattern, answer, re.IGNORECASE | re.MULTILINE)
            return CheckOutcome(kind, m is not None, f"pattern {pattern!r}")

        if kind == "numeric":
            expected = float(spec["value"])
            tolerance = float(spec.get("tolerance", 0.0))
            for raw in _NUMBER_RE.findall(answer):
                if abs(float(raw.replace(",", ".")) - expected) <= tolerance:
                    return CheckOutcome(kind, True, f"found {raw}")
            return CheckOutcome(kind, False, f"no number within {tolerance} of {expected}")

        if kind == "gaia":
            # Official GAIA quasi-exact match against the 'FINAL ANSWER:' line.
            from isaac.eval.gaia import extract_final_answer, question_scorer

            ground_truth = str(spec["value"])
            extracted = extract_final_answer(answer)
            passed = question_scorer(extracted, ground_truth)
            return CheckOutcome(kind, passed, f"answered {extracted!r}, expected {ground_truth!r}")

        if kind == "arc":
            # Exact grid match on every ARC test pair (single attempt).
            from isaac.eval.arc import score_arc_answer

            passed, detail = score_arc_answer(answer, str(spec["value"]))
            return CheckOutcome(kind, passed, detail)

        if kind == "min_length":
            n = int(spec["value"])
            return CheckOutcome(kind, len(answer.strip()) >= n, f"{len(answer.strip())} >= {n}")

        if kind == "file_exists":
            rel = str(spec["path"])
            content = _read_workspace_file(workspace, rel)
            return CheckOutcome(kind, content is not None, rel)

        if kind in ("file_contains", "file_regex"):
            rel = str(spec["path"])
            content = _read_workspace_file(workspace, rel)
            if content is None:
                return CheckOutcome(kind, False, f"{rel} missing")
            if kind == "file_contains":
                needle = str(spec["value"])
                return CheckOutcome(kind, needle.lower() in content.lower(), rel)
            m = re.search(str(spec["pattern"]), content, re.IGNORECASE | re.MULTILINE)
            return CheckOutcome(kind, m is not None, rel)

        return CheckOutcome(kind or "unknown", False, f"unknown checker type {kind!r}")
    except (KeyError, ValueError, TypeError, re.error) as exc:
        return CheckOutcome(kind or "unknown", False, f"malformed spec: {exc}")


def score_answer(checks: list[dict], answer: str, workspace: Path) -> list[CheckOutcome]:
    """Run every check; the task passes when all outcomes pass."""
    return [run_check(spec, answer, workspace) for spec in checks]

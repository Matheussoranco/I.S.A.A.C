"""Task-suite loading for the evaluation harness.

A suite is a JSONL file — one task per line::

    {"id": "math-001", "category": "reasoning",
     "prompt": "What is 17 * 24? Reply with just the number.",
     "checks": [{"type": "numeric", "value": 408}]}

Fields
------
id            unique task identifier (required)
prompt        the task given to the agent (required)
checks        list of checker specs (required, see :mod:`isaac.eval.checkers`)
category      free-form grouping label for per-category scores
runner        "agent" (default) or "team" (specialist orchestrator)
tools         restrict the agent to these tool names (agent runner only)
files         {relative_path: content} seeded into the workspace before the run
max_iterations / timeout_seconds   per-task loop caps
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalTask:
    """One scoreable task in a suite."""

    id: str
    prompt: str
    checks: list[dict] = field(default_factory=list)
    category: str = "general"
    runner: str = "agent"  # "agent" | "team"
    tools: list[str] | None = None
    files: dict[str, str] = field(default_factory=dict)
    file_paths: dict[str, str] = field(default_factory=dict)
    """Binary attachments: {workspace-relative dest: absolute source path}.
    Copied (not inlined) into the workspace before the run — used by dataset
    adapters (e.g. GAIA) whose tasks ship xlsx/pdf/png/mp3 files."""
    max_iterations: int = 12
    timeout_seconds: float = 300.0


def load_suite(path: str | Path) -> list[EvalTask]:
    """Load a JSONL task suite, validating ids are present and unique."""
    p = Path(path)
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name}:{lineno}: invalid JSON — {exc}") from exc
        if not obj.get("id") or not obj.get("prompt") or not obj.get("checks"):
            raise ValueError(f"{p.name}:{lineno}: 'id', 'prompt', and 'checks' are required")
        if obj["id"] in seen:
            raise ValueError(f"{p.name}:{lineno}: duplicate task id '{obj['id']}'")
        seen.add(obj["id"])
        tasks.append(
            EvalTask(
                id=str(obj["id"]),
                prompt=str(obj["prompt"]),
                checks=list(obj["checks"]),
                category=str(obj.get("category", "general")),
                runner=str(obj.get("runner", "agent")),
                tools=list(obj["tools"]) if obj.get("tools") else None,
                files={str(k): str(v) for k, v in (obj.get("files") or {}).items()},
                max_iterations=int(obj.get("max_iterations", 12)),
                timeout_seconds=float(obj.get("timeout_seconds", 300.0)),
            )
        )
    if not tasks:
        raise ValueError(f"{p}: suite contains no tasks")
    return tasks


def suite_hash(tasks: list[EvalTask]) -> str:
    """Stable content hash of a suite — recorded with every run so a score is
    only comparable to runs of the *identical* task set."""
    entries = []
    for t in sorted(tasks, key=lambda t: t.id):
        entry: dict = {
            "id": t.id,
            "prompt": t.prompt,
            "checks": t.checks,
            "files": t.files,
        }
        # Key added only when present so suites without binary attachments
        # (e.g. golden_v1, whose hash is already published) keep their hash.
        if t.file_paths:
            entry["file_paths"] = sorted(t.file_paths)
        entries.append(entry)
    canonical = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

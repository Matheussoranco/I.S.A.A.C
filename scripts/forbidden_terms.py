#!/usr/bin/env python3
"""Forbidden Terms Gate — CI check for prohibited marketing claims.

ADR-001: No "SOTA" claims — README/docs/CLI describe "competitive local-first
autonomous agent framework" only. CI enforces grep check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    (r"\bSOTA\b", "SOTA"),
    (r"state.of.the.art", "state-of-the-art"),
    (r"best.in.class", "best-in-class"),
]

# Files/patterns to exclude from checking
EXCLUDE_PATTERNS = [
    r"\.git/",
    r"__pycache__/",
    r"\.pyc$",
    r"\.pytest_cache/",
    r"\.venv/",
    r"node_modules/",
    r"dist/",
    r"build/",
    r"\.egg-info/",
]

# Allow-list for specific occurrences that are explicitly permitted
# Format: (file_pattern, line_regex, description)
ALLOWLIST = [
    # ROADMAP-1.0.md discusses the SOTA gate itself — meta discussion is allowed
    (r"docs/ROADMAP-1\.0\.md", r".*", "Roadmap document discusses SOTA gate policy"),
    # eval/__init__.py documents the gate implementation
    (r"src/isaac/eval/__init__.py", r".*", "Eval module documents SOTA gate"),
    # eval/gaia.py references the gate
    (r"src/isaac/eval/gaia\.py", r".*", "GAIA eval references SOTA gate"),
    # README.md references the gate policy
    (r"README\.md", r".*", "README documents SOTA gate policy"),
    # CHANGELOG.md references the gate
    (r"CHANGELOG\.md", r".*", "Changelog documents SOTA gate"),
    # LIMITATIONS.md references the gate
    (r"LIMITATIONS\.md", r".*", "Limitations document discusses SOTA gate"),
    # PRODUCT_ROADMAP.md references the gate
    (r"PRODUCT_ROADMAP\.md", r".*", "Product roadmap documents SOTA gate"),
]


def is_excluded(path: Path) -> bool:
    path_str = str(path)
    return any(re.search(pat, path_str) for pat in EXCLUDE_PATTERNS)


def is_allowed(path: Path, line: str) -> bool:
    # Normalize to forward slashes for pattern matching
    path_str = str(path).replace("\\", "/")
    for file_pat, line_pat, _desc in ALLOWLIST:
        if re.search(file_pat, path_str) and re.search(line_pat, line):
            return True
    return False


def check_file(path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_number, matched_term, line) for violations."""
    violations: list[tuple[int, str, str]] = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return violations

    for i, line in enumerate(content.splitlines(), 1):
        if is_allowed(path, line):
            continue
        for pattern, term in FORBIDDEN_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                violations.append((i, term, line.strip()))
    return violations


def main() -> int:
    import subprocess

    root = Path.cwd()
    all_violations = []

    # Use git ls-files to get only tracked files (faster, excludes .gitignore)
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True, timeout=10
        )
        tracked_files = result.stdout.strip().splitlines()
    except Exception:
        tracked_files = []

    for file_str in tracked_files:
        path = root / file_str
        if not path.is_file() or is_excluded(path):
            continue
        # Only check text files
        text_suffixes = {
            ".py",
            ".md",
            ".txt",
            ".rst",
            ".yml",
            ".yaml",
            ".toml",
            ".cfg",
            ".ini",
            ".json",
        }
        if path.suffix in text_suffixes or path.name in {"Dockerfile", "Makefile"}:
            violations = check_file(path)
            if violations:
                for line_num, term, line in violations:
                    all_violations.append((file_str, line_num, term, line))

    if all_violations:
        print("FORBIDDEN TERMS DETECTED:")
        print("=" * 60)
        for rel_path, line_num, term, line in all_violations:
            print(f"{rel_path}:{line_num}: [{term}] {line}")
        print("=" * 60)
        print(f"Total violations: {len(all_violations)}")
        msg = (
            "\nADR-001: No 'SOTA' claims — use 'competitive local-first autonomous agent framework'"
        )
        print(msg)
        return 1

    print("No forbidden terms found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

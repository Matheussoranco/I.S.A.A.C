"""Bundle the ARC DSL modules for execution inside the code sandbox."""

from __future__ import annotations

from pathlib import Path

_ARC_MODULES = ("grid_ops", "priors", "dsl")


def arc_sandbox_files() -> dict[str, str]:
    """Return sandbox-relative source files for the ``isaac_arc`` package."""
    source_dir = Path(__file__).resolve().parent
    files = {"isaac_arc/__init__.py": ""}
    for module in _ARC_MODULES:
        source = (source_dir / f"{module}.py").read_text(encoding="utf-8")
        files[f"isaac_arc/{module}.py"] = source.replace("from isaac.arc.", "from isaac_arc.")
    return files

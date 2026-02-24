"""File Tool — Scoped file operations within ~/.isaac/workspace/.

Every path is resolved and validated to stay inside the workspace
boundary.  Delete operations carry risk_level 5 and always require
human approval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from isaac.config.settings import get_settings
from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    """Return the workspace root, creating it if necessary."""
    root = get_settings().isaac_home / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_resolve(relative: str) -> Path | None:
    """Resolve *relative* under workspace root.

    Returns ``None`` if the resolved path escapes the boundary.
    """
    root = _workspace_root()
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


class FileReadTool(IsaacTool):
    """Read a file from the Isaac workspace."""

    name = "file_read"
    description = "Read the contents of a file within the Isaac workspace."
    risk_level = 1
    requires_approval = False
    sandbox_required = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing 'path' parameter.")

        target = _safe_resolve(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        if not target.is_file():
            return ToolResult(success=False, error=f"File not found: {path_str}")

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            max_chars = int(kwargs.get("max_chars", 50_000))
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... truncated at {max_chars} chars ..."
            return ToolResult(success=True, output=content)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FileWriteTool(IsaacTool):
    """Write a file to the Isaac workspace."""

    name = "file_write"
    description = "Write content to a file within the Isaac workspace."
    risk_level = 2
    requires_approval = False
    sandbox_required = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        content: str = kwargs.get("content", "")
        if not path_str:
            return ToolResult(success=False, error="Missing 'path' parameter.")

        target = _safe_resolve(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Wrote {len(content)} chars to {path_str}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FileListTool(IsaacTool):
    """List files and directories in the Isaac workspace."""

    name = "file_list"
    description = "List contents of a directory within the Isaac workspace."
    risk_level = 1
    requires_approval = False
    sandbox_required = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", ".")

        target = _safe_resolve(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path_str}")

        try:
            entries: list[str] = []
            for child in sorted(target.iterdir()):
                kind = "dir" if child.is_dir() else "file"
                entries.append(f"[{kind}] {child.name}")
            return ToolResult(success=True, output="\n".join(entries) or "(empty)")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FileDeleteTool(IsaacTool):
    """Delete a file from the Isaac workspace.  Risk level 5 — always needs approval."""

    name = "file_delete"
    description = "Delete a file within the Isaac workspace. High risk — requires human approval."
    risk_level = 5
    requires_approval = True
    sandbox_required = False

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing 'path' parameter.")

        target = _safe_resolve(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        if not target.exists():
            return ToolResult(success=False, error=f"Does not exist: {path_str}")

        # Refuse directory delete — only files
        if target.is_dir():
            return ToolResult(
                success=False,
                error="Directory deletion is not supported. Delete files individually.",
            )

        try:
            target.unlink()
            return ToolResult(success=True, output=f"Deleted {path_str}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

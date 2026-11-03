"""File Tool — Scoped file operations within ~/.isaac/workspace/.

Every path is resolved and validated to stay inside the workspace
boundary.  Delete operations carry risk_level 5 and always require
human approval.
"""

from __future__ import annotations

import logging
import os
from itertools import islice
from pathlib import Path
from typing import Any

from isaac.config.settings import get_settings
from isaac.security.file_mutation import write_text
from isaac.security.path_policy import is_sensitive_path
from isaac.security.workspace import current_workspace
from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_LIST_ENTRIES = 2_000


def _workspace_root() -> Path:
    """Return the workspace root, creating it if necessary."""
    root = current_workspace() or get_settings().isaac_home / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_resolve(relative: str) -> Path | None:
    """Resolve *relative* under workspace root.

    Returns ``None`` if the resolved path escapes the boundary.
    """
    root = _workspace_root()
    target = (root / relative).resolve()
    if is_sensitive_path(root / relative) or is_sensitive_path(target):
        return None
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _safe_link_path(relative: str) -> Path | None:
    """Validate a path without following its final symlink component.

    Deleting or renaming a link must act on the link itself, never on the
    file it points at.
    """
    root = _workspace_root().resolve()
    raw = Path(relative)
    if raw.name in ("", ".", ".."):
        return None
    try:
        candidate = (root / raw.parent).resolve() / raw.name
    except (OSError, RuntimeError):
        return None
    if is_sensitive_path(raw) or is_sensitive_path(candidate):
        return None
    if not candidate.parent.is_relative_to(root):
        return None
    return candidate


class FileReadTool(IsaacTool):
    """Read a file from the Isaac workspace."""

    name = "file_read"
    description = "Read the contents of a file within the Isaac workspace."
    risk_level = 1
    requires_approval = False
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the file to read.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Truncate output to this many characters (default 50000).",
            },
        },
        "required": ["path"],
    }

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
            max_chars = max(0, int(kwargs.get("max_chars", 50_000)))
            with target.open("r", encoding="utf-8", errors="replace") as handle:
                content = handle.read(max_chars + 1)
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... truncated at {max_chars} chars ..."
            return ToolResult(success=True, output=content)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FileWriteTool(IsaacTool):
    """Write a file to the Isaac workspace."""

    name = "file_write"
    description = "Write content to a file within the Isaac workspace (creates parent dirs)."
    risk_level = 2
    requires_approval = False
    sandbox_required = False
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the file to write.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Replace existing content with approval.",
            },
            "content": {"type": "string", "description": "Full text content to write."},
        },
        "required": ["path", "content"],
    }

    def approval_required(self, **kwargs: Any) -> bool:
        return bool(kwargs.get("overwrite"))

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
            write_text(target, content, overwrite=bool(kwargs.get("overwrite")))
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
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative directory to list (default '.').",
            },
        },
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", ".")

        target = _safe_resolve(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path_str}")

        try:
            children = sorted(islice(target.iterdir(), _MAX_LIST_ENTRIES + 1))
            entries: list[str] = []
            for child in children[:_MAX_LIST_ENTRIES]:
                kind = "dir" if child.is_dir() else "file"
                entries.append(f"[{kind}] {child.name}")
            if len(children) > _MAX_LIST_ENTRIES:
                entries.append(f"... truncated at {_MAX_LIST_ENTRIES} entries ...")
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
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path of the file to delete.",
            },
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str: str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="Missing 'path' parameter.")

        target = _safe_link_path(path_str)
        if target is None:
            return ToolResult(success=False, error="Path escapes workspace boundary.")

        if not os.path.lexists(target):
            return ToolResult(success=False, error=f"Does not exist: {path_str}")

        if not target.is_symlink() and target.is_dir():
            return ToolResult(
                success=False,
                error="Directory deletion is not supported. Delete files individually.",
            )

        try:
            target.unlink()
            return ToolResult(success=True, output=f"Deleted {path_str}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

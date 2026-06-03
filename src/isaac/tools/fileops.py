"""Host File-Operations Tools — work on the user's *real* files, safely.

The workspace :mod:`isaac.tools.file` tools are sandboxed to
``~/.isaac/workspace`` — great for scratch output, useless for "organise my
Downloads folder".  These tools operate on the host filesystem but confine
every path to :attr:`Settings.allowed_paths` (default: the user's home
directory), so the file-organizer and designer specialists can read, sort,
move, and lay out files where the user actually keeps them — without ever
touching system paths like ``C:\\Windows`` or ``/etc``.

Risk grading:
    fs_list / fs_info / fs_read   risk 1  (read-only)
    fs_mkdir                      risk 2
    fs_write / fs_move / fs_copy  risk 3
Destructive deletion is intentionally **not** offered — organisers should move
unwanted files to an archive folder, which is reversible.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from isaac.tools.base import IsaacTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 50_000
_MAX_LIST_ENTRIES = 2_000


def _allowed_roots() -> list[Path]:
    """Return the configured allow-listed root directories (resolved)."""
    try:
        from isaac.config.settings import get_settings

        raw = get_settings().allowed_paths or [str(Path.home())]
    except Exception:
        raw = [str(Path.home())]
    roots: list[Path] = []
    for r in raw:
        try:
            roots.append(Path(r).expanduser().resolve())
        except Exception:
            continue
    return roots or [Path.home().resolve()]


def _resolve(path_str: str) -> Path | None:
    """Resolve *path_str* and confirm it lies within an allow-listed root.

    Returns ``None`` if the path escapes every allowed root.
    """
    if not path_str:
        return None
    try:
        target = Path(path_str).expanduser().resolve()
    except Exception:
        return None
    for root in _allowed_roots():
        try:
            target.relative_to(root)
            return target
        except ValueError:
            continue
    return None


def _denied(path_str: str) -> ToolResult:
    roots = ", ".join(str(r) for r in _allowed_roots())
    return ToolResult(
        success=False,
        error=(
            f"Path '{path_str}' is outside the allowed roots ({roots}). "
            "Set ISAAC_ALLOWED_PATHS to widen access."
        ),
    )


class FsListTool(IsaacTool):
    """List a directory on the host (within allowed paths)."""

    name = "fs_list"
    description = (
        "List files and folders at an absolute host path (within allowed roots). "
        "Set 'recursive' to walk subdirectories and 'pattern' (glob, e.g. '*.pdf') "
        "to filter. Returns name, type, size, and modified time."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute directory path to list."},
            "recursive": {"type": "boolean", "description": "Walk subdirectories (default false)."},
            "pattern": {"type": "string", "description": "Glob filter, e.g. '*.jpg' (optional)."},
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str = str(kwargs.get("path", ""))
        target = _resolve(path_str)
        if target is None:
            return _denied(path_str)
        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path_str}")

        recursive = bool(kwargs.get("recursive", False))
        pattern = str(kwargs.get("pattern", "") or "")
        try:
            it = target.rglob(pattern or "*") if recursive else target.glob(pattern or "*")
            lines: list[str] = []
            for child in sorted(it):
                if len(lines) >= _MAX_LIST_ENTRIES:
                    lines.append(f"... truncated at {_MAX_LIST_ENTRIES} entries ...")
                    break
                try:
                    st = child.stat()
                    kind = "dir " if child.is_dir() else "file"
                    size = "" if child.is_dir() else f"{st.st_size:>10}B"
                    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    rel = child.relative_to(target)
                    lines.append(f"[{kind}] {size:>12}  {mtime}  {rel}")
                except OSError:
                    continue
            return ToolResult(success=True, output="\n".join(lines) or "(empty)")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsInfoTool(IsaacTool):
    """Return metadata about a host path (within allowed paths)."""

    name = "fs_info"
    description = (
        "Get metadata for an absolute host path: type, size, created/modified time, "
        "and (for directories) immediate child count. Use to inspect files before acting."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute path to inspect."}},
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str = str(kwargs.get("path", ""))
        target = _resolve(path_str)
        if target is None:
            return _denied(path_str)
        if not target.exists():
            return ToolResult(success=False, error=f"Does not exist: {path_str}")
        try:
            st = target.stat()
            info = {
                "path": str(target),
                "type": "dir" if target.is_dir() else "file",
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "created": datetime.fromtimestamp(st.st_ctime).isoformat(timespec="seconds"),
            }
            if target.is_dir():
                try:
                    info["children"] = sum(1 for _ in target.iterdir())
                except OSError:
                    info["children"] = -1
            else:
                info["suffix"] = target.suffix
            body = "\n".join(f"{k}: {v}" for k, v in info.items())
            return ToolResult(success=True, output=body, metadata=info)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsReadTool(IsaacTool):
    """Read a text file from the host (within allowed paths)."""

    name = "fs_read"
    description = (
        "Read the text contents of an absolute host file path (within allowed roots). "
        "Truncates to 50k characters. Use for source files, configs, notes, and data."
    )
    risk_level = 1
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute file path to read."},
            "max_chars": {"type": "integer", "description": "Truncate to N chars (default 50000)."},
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str = str(kwargs.get("path", ""))
        target = _resolve(path_str)
        if target is None:
            return _denied(path_str)
        if not target.is_file():
            return ToolResult(success=False, error=f"File not found: {path_str}")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            limit = int(kwargs.get("max_chars", _MAX_READ_CHARS))
            if len(content) > limit:
                content = content[:limit] + f"\n\n... truncated at {limit} chars ..."
            return ToolResult(success=True, output=content)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsWriteTool(IsaacTool):
    """Write a text file to the host (within allowed paths)."""

    name = "fs_write"
    description = (
        "Write text content to an absolute host file path (within allowed roots), "
        "creating parent directories. Use to save generated documents, designs, code, "
        "or reports where the user keeps their files."
    )
    risk_level = 3
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute file path to write."},
            "content": {"type": "string", "description": "Full text content."},
            "append": {
                "type": "boolean",
                "description": "Append instead of overwrite (default false).",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str = str(kwargs.get("path", ""))
        content = str(kwargs.get("content", ""))
        target = _resolve(path_str)
        if target is None:
            return _denied(path_str)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if bool(kwargs.get("append", False)) else "w"
            with target.open(mode, encoding="utf-8") as fh:
                fh.write(content)
            return ToolResult(success=True, output=f"Wrote {len(content)} chars to {target}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsMkdirTool(IsaacTool):
    """Create a directory on the host (within allowed paths)."""

    name = "fs_mkdir"
    description = (
        "Create a directory (and parents) at an absolute host path (within allowed roots). "
        "Use to build a folder structure when organising files."
    )
    risk_level = 2
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute directory path to create."}
        },
        "required": ["path"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        path_str = str(kwargs.get("path", ""))
        target = _resolve(path_str)
        if target is None:
            return _denied(path_str)
        try:
            target.mkdir(parents=True, exist_ok=True)
            return ToolResult(success=True, output=f"Created directory {target}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsMoveTool(IsaacTool):
    """Move or rename a file/folder on the host (within allowed paths)."""

    name = "fs_move"
    description = (
        "Move or rename a file or folder. Both 'src' and 'dest' must be absolute "
        "paths within allowed roots. Refuses to overwrite an existing destination "
        "unless 'overwrite' is true. This is the primary tool for organising files."
    )
    risk_level = 3
    parameters = {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Absolute source path."},
            "dest": {"type": "string", "description": "Absolute destination path."},
            "overwrite": {
                "type": "boolean",
                "description": "Replace existing dest (default false).",
            },
        },
        "required": ["src", "dest"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        src_str, dest_str = str(kwargs.get("src", "")), str(kwargs.get("dest", ""))
        src, dest = _resolve(src_str), _resolve(dest_str)
        if src is None:
            return _denied(src_str)
        if dest is None:
            return _denied(dest_str)
        if not src.exists():
            return ToolResult(success=False, error=f"Source does not exist: {src_str}")
        if dest.exists() and not bool(kwargs.get("overwrite", False)):
            return ToolResult(
                success=False,
                error=f"Destination already exists: {dest_str} (set overwrite=true to replace).",
            )
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.is_file():
                dest.unlink()
            shutil.move(str(src), str(dest))
            return ToolResult(success=True, output=f"Moved {src} -> {dest}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FsCopyTool(IsaacTool):
    """Copy a file/folder on the host (within allowed paths)."""

    name = "fs_copy"
    description = (
        "Copy a file or folder. Both 'src' and 'dest' must be absolute paths within "
        "allowed roots. Folders are copied recursively."
    )
    risk_level = 3
    parameters = {
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Absolute source path."},
            "dest": {"type": "string", "description": "Absolute destination path."},
        },
        "required": ["src", "dest"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        src_str, dest_str = str(kwargs.get("src", "")), str(kwargs.get("dest", ""))
        src, dest = _resolve(src_str), _resolve(dest_str)
        if src is None:
            return _denied(src_str)
        if dest is None:
            return _denied(dest_str)
        if not src.exists():
            return ToolResult(success=False, error=f"Source does not exist: {src_str}")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(str(src), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(src), str(dest))
            return ToolResult(success=True, output=f"Copied {src} -> {dest}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

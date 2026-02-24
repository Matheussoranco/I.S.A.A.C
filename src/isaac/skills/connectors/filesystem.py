"""FileSystemConnector — controlled local file operations.

Operates ONLY within user-configured ALLOWED_PATHS.  Every path is
validated before execution.  Write operations create a ``.isaac_backup``
copy first.  File deletion is never permitted.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class FileSystemConnector(BaseConnector):
    """Read, write, list, and search files within allowed directories."""

    name = "filesystem"
    description = (
        "Read, write, list, and search files on the local filesystem. "
        "Operations are restricted to user-configured allowed paths."
    )
    requires_env: list[str] = []

    def __init__(self) -> None:
        try:
            from isaac.config.settings import settings
            self._allowed: list[Path] = [
                Path(p).expanduser().resolve() for p in settings.allowed_paths
            ]
        except Exception:
            self._allowed = [
                Path("~/Documents").expanduser().resolve(),
                Path("~/Downloads").expanduser().resolve(),
                Path("~/Desktop").expanduser().resolve(),
            ]

    def _validate_path(self, path: str) -> Path:
        """Resolve and validate that *path* is within ALLOWED_PATHS.

        Raises
        ------
        PermissionError
            If the path is outside all allowed directories.
        """
        resolved = Path(path).expanduser().resolve()
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        raise PermissionError(
            f"Path {resolved} is outside allowed directories: {self._allowed}"
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a filesystem operation.

        Parameters
        ----------
        action : str
            One of ``read_file``, ``write_file``, ``list_directory``, ``search_files``.
        path : str
            Target file or directory path.
        content : str
            Content to write (for ``write_file``).
        query : str
            Search string (for ``search_files``).
        directory : str
            Directory to search in (for ``search_files``).
        """
        action: str = kwargs.get("action", "")
        path: str = kwargs.get("path", "")

        if action == "read_file":
            return self._read_file(path)
        elif action == "write_file":
            content: str = kwargs.get("content", "")
            return self._write_file(path, content)
        elif action == "list_directory":
            return self._list_directory(path)
        elif action == "search_files":
            query: str = kwargs.get("query", "")
            directory: str = kwargs.get("directory", path)
            return self._search_files(query, directory)
        else:
            return {"error": f"Unknown action: {action}"}

    def _read_file(self, path: str) -> dict[str, Any]:
        try:
            validated = self._validate_path(path)
            if not validated.is_file():
                return {"error": f"Not a file: {path}"}
            content = validated.read_text(encoding="utf-8", errors="replace")
            return {"path": str(validated), "content": content, "size": len(content)}
        except PermissionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Read failed: {exc}"}

    def _write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            validated = self._validate_path(path)
            # Create backup if file exists
            if validated.is_file():
                backup = validated.with_suffix(validated.suffix + ".isaac_backup")
                shutil.copy2(str(validated), str(backup))
                logger.info("FileSystem: backup created at %s", backup)
            validated.parent.mkdir(parents=True, exist_ok=True)
            validated.write_text(content, encoding="utf-8")
            return {"path": str(validated), "written": len(content), "success": True}
        except PermissionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Write failed: {exc}"}

    def _list_directory(self, path: str) -> dict[str, Any]:
        try:
            validated = self._validate_path(path)
            if not validated.is_dir():
                return {"error": f"Not a directory: {path}"}
            entries: list[dict[str, Any]] = []
            for entry in sorted(validated.iterdir()):
                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                })
            return {"path": str(validated), "entries": entries}
        except PermissionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"List failed: {exc}"}

    def _search_files(self, query: str, directory: str) -> dict[str, Any]:
        if not query:
            return {"error": "No search query provided."}
        try:
            validated = self._validate_path(directory)
            if not validated.is_dir():
                return {"error": f"Not a directory: {directory}"}

            matches: list[str] = []
            for f in validated.rglob("*"):
                if f.is_file() and query.lower() in f.name.lower():
                    matches.append(str(f))
                if len(matches) >= 50:
                    break
            return {"directory": str(validated), "query": query, "matches": matches}
        except PermissionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Search failed: {exc}"}

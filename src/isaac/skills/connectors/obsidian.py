"""ObsidianConnector — Local Obsidian vault file access.

Requires ``OBSIDIAN_VAULT_PATH`` environment variable pointing to the
root of an Obsidian vault on the host filesystem.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class ObsidianConnector(BaseConnector):
    """Read, write, search, and list notes in a local Obsidian vault."""

    name = "obsidian"
    description = (
        "Access a local Obsidian vault: read, write, search, and list markdown notes. "
        "Requires OBSIDIAN_VAULT_PATH."
    )
    requires_env: list[str] = ["OBSIDIAN_VAULT_PATH"]

    def _vault_root(self) -> Path:
        return Path(os.environ["OBSIDIAN_VAULT_PATH"]).resolve()

    def _validate_path(self, target: Path) -> Path:
        """Ensure *target* is within the vault root."""
        resolved = target.resolve()
        vault = self._vault_root()
        if not str(resolved).startswith(str(vault)):
            raise PermissionError(f"Path escapes the vault: {resolved}")
        return resolved

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run an Obsidian vault operation.

        Parameters
        ----------
        action : str
            ``"list"`` — list notes, ``"read"`` — read a note,
            ``"write"`` — create/update a note, ``"search"`` — full-text
            search across notes.
        path : str
            Relative path within the vault (for ``read`` / ``write``).
        content : str
            Markdown content (for ``write``).
        query : str
            Search term (for ``search``).
        folder : str
            Subfolder to restrict listing / search (default ``""`` = root).
        """
        action: str = kwargs.get("action", "list")

        try:
            handlers = {
                "list": self._list_notes,
                "read": self._read_note,
                "write": self._write_note,
                "search": self._search_notes,
            }
            handler = handlers.get(action)
            if handler is None:
                return {"error": f"Unknown action: {action}"}
            return handler(**kwargs)
        except PermissionError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            logger.error("Obsidian %s failed: %s", action, exc)
            return {"error": str(exc)}

    def _list_notes(self, **kwargs: Any) -> dict[str, Any]:
        vault = self._vault_root()
        folder = kwargs.get("folder", "")
        base = self._validate_path(vault / folder) if folder else vault

        notes: list[str] = []
        for p in sorted(base.rglob("*.md")):
            # Skip hidden directories (.obsidian, .trash)
            parts = p.relative_to(vault).parts
            if any(part.startswith(".") for part in parts):
                continue
            notes.append(str(p.relative_to(vault)))
        return {"vault": str(vault), "notes": notes[:200]}

    def _read_note(self, **kwargs: Any) -> dict[str, Any]:
        vault = self._vault_root()
        rel_path = kwargs.get("path", "")
        if not rel_path:
            return {"error": "Missing 'path'"}

        target = self._validate_path(vault / rel_path)
        if not target.exists():
            return {"error": f"Note not found: {rel_path}"}

        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "path": rel_path,
            "content": content[:20_000],
            "size_bytes": target.stat().st_size,
        }

    def _write_note(self, **kwargs: Any) -> dict[str, Any]:
        vault = self._vault_root()
        rel_path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not rel_path:
            return {"error": "Missing 'path'"}

        target = self._validate_path(vault / rel_path)
        existed = target.exists()

        # Backup existing notes
        if existed:
            backup = target.with_suffix(".md.isaac_backup")
            backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        return {
            "status": "updated" if existed else "created",
            "path": rel_path,
            "size_bytes": len(content.encode("utf-8")),
        }

    def _search_notes(self, **kwargs: Any) -> dict[str, Any]:
        vault = self._vault_root()
        query = kwargs.get("query", "").lower()
        folder = kwargs.get("folder", "")
        if not query:
            return {"error": "Missing 'query'"}

        base = self._validate_path(vault / folder) if folder else vault

        matches: list[dict[str, Any]] = []
        for p in base.rglob("*.md"):
            parts = p.relative_to(vault).parts
            if any(part.startswith(".") for part in parts):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if query in text.lower():
                # Extract a snippet around the first match
                idx = text.lower().index(query)
                start = max(0, idx - 80)
                end = min(len(text), idx + len(query) + 80)
                snippet = text[start:end].replace("\n", " ")
                matches.append(
                    {
                        "path": str(p.relative_to(vault)),
                        "snippet": snippet,
                    }
                )
            if len(matches) >= 20:
                break

        return {"query": query, "matches": matches}

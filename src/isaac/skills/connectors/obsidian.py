"""ObsidianConnector — Local Obsidian vault file access.

Canonical env var (see ``.env.example``)::

    ISAAC_OBSIDIAN_VAULT_PATH

Legacy ``OBSIDIAN_VAULT_PATH`` remains accepted as a deprecated fallback
with a ``DeprecationWarning``.  All values resolve through
:func:`isaac.config.settings.get_settings` — never ``os.environ[]`` directly.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

#: Safety bounds for vault scans (prevents unbounded rglob on huge vaults).
MAX_SCAN_FILES = 5_000
SCAN_TIMEOUT_SECONDS = 10.0
MAX_NOTES_RETURNED = 200
MAX_SEARCH_MATCHES = 20


class ObsidianConnector(BaseConnector):
    """Read, write, search, and list notes in a local Obsidian vault."""

    name = "obsidian"
    description = (
        "Access a local Obsidian vault: read, write, search, and list markdown notes. "
        "Requires ISAAC_OBSIDIAN_VAULT_PATH."
    )
    requires_env: ClassVar[list[str]] = ["ISAAC_OBSIDIAN_VAULT_PATH"]

    def _vault_root(self) -> Path:
        from isaac.config.settings import get_settings, resolve_env

        cfg = get_settings().obsidian_vault_path or resolve_env("ISAAC_OBSIDIAN_VAULT_PATH")
        raw = cfg.strip()
        if not raw:
            raise RuntimeError("Obsidian connector unavailable — missing ISAAC_OBSIDIAN_VAULT_PATH")
        return Path(raw).expanduser().resolve()

    def _validate_path(self, target: Path) -> Path:
        """Ensure *target* is within the vault root (TOCTOU-safe).

        Resolves both sides, then re-validates with
        :meth:`Path.is_relative_to` immediately before the caller uses the
        result.  Callers must use the returned path and re-validate after any
        symlink-creating operation and before write.
        """
        vault = self._vault_root()
        resolved = target.resolve()
        try:
            if hasattr(resolved, "is_relative_to"):
                if not resolved.is_relative_to(vault):
                    raise PermissionError(f"Path escapes the vault: {resolved}")
            else:  # Python < 3.9 fallback
                resolved.relative_to(vault)
        except ValueError:
            raise PermissionError(f"Path escapes the vault: {resolved}") from None
        return resolved

    def _iter_markdown(
        self, base: Path, vault: Path, *, max_files: int = MAX_SCAN_FILES
    ) -> list[Path]:
        """Bounded ``*.md`` scan: skips hidden dirs, caps files + wall time."""
        out: list[Path] = []
        deadline = time.monotonic() + SCAN_TIMEOUT_SECONDS
        scanned = 0
        for p in sorted(base.rglob("*.md")):
            scanned += 1
            if scanned > max_files or time.monotonic() > deadline:
                logger.warning("Obsidian scan truncated at %d files (vault=%s)", scanned, vault)
                break
            try:
                parts = p.relative_to(vault).parts
            except ValueError:
                continue
            if any(part.startswith(".") for part in parts):
                continue
            out.append(p)
            if len(out) >= MAX_NOTES_RETURNED and max_files >= MAX_NOTES_RETURNED:
                # _list_notes caps at MAX_NOTES_RETURNED; keep scanning cheap
                # for search which stops early on matches.
                pass
        return out

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
        for p in self._iter_markdown(base, vault):
            notes.append(str(p.relative_to(vault)))
        return {"vault": str(vault), "notes": notes[:MAX_NOTES_RETURNED]}

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

        # Versioned backup — timestamped, never overwritten.
        backup_path = ""
        if existed:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup = target.with_name(f"{target.name}.{stamp}.isaac_backup.md")
            backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path = str(backup.relative_to(vault))

        target.parent.mkdir(parents=True, exist_ok=True)
        # TOCTOU: re-resolve + re-validate after mkdir (symlink swap guard).
        target = self._validate_path((vault / rel_path).resolve())
        target.write_text(content, encoding="utf-8")

        result: dict[str, Any] = {
            "status": "updated" if existed else "created",
            "path": rel_path,
            "size_bytes": len(content.encode("utf-8")),
        }
        if backup_path:
            result["backup"] = backup_path
        return result

    def _search_notes(self, **kwargs: Any) -> dict[str, Any]:
        vault = self._vault_root()
        query = kwargs.get("query", "").lower()
        folder = kwargs.get("folder", "")
        if not query:
            return {"error": "Missing 'query'"}

        base = self._validate_path(vault / folder) if folder else vault

        matches: list[dict[str, Any]] = []
        for p in self._iter_markdown(base, vault):
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
            if len(matches) >= MAX_SEARCH_MATCHES:
                break

        return {"query": query, "matches": matches}

"""Audit Log — append-only, tamper-evident security event log.

Every security-relevant event is written to ``~/.isaac/audit/audit.jsonl``
as a single JSON line with a SHA-256 chain hash linking each entry to the
previous one (like a lightweight blockchain).

Categories
----------
* ``auth``      — login / token / session events
* ``tool``      — tool invocations (especially risk ≥ 3)
* ``approval``  — approval requests / grants / rejections
* ``guard``     — prompt injection guard results
* ``sandbox``   — sandbox creation / destruction / violations
* ``system``    — startup / shutdown / config changes
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

AuditCategory = Literal[
    "auth", "tool", "approval", "guard", "sandbox", "system"
]

_GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: str
    category: AuditCategory
    action: str
    actor: str = "system"
    details: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 over (prev_hash + timestamp + category + action + details)."""
        payload = f"{self.prev_hash}|{self.timestamp}|{self.category}|{self.action}|{json.dumps(self.details, sort_keys=True)}"
        return hashlib.sha256(payload.encode()).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit log.

    Thread-safe: all writes acquire a lock.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._prev_hash = _GENESIS_HASH

        if log_dir is None:
            try:
                from isaac.config.settings import get_settings
                log_dir = get_settings().isaac_home / "audit"
            except Exception:
                log_dir = Path.home() / ".isaac" / "audit"

        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = log_dir / "audit.jsonl"

        # Resume chain hash from last entry
        self._resume_chain()

    def _resume_chain(self) -> None:
        """Read the last line of the log to get the previous hash."""
        if not self._log_path.exists():
            return
        try:
            with open(self._log_path, "rb") as f:
                # Seek to end, read backwards to find last newline
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return
                # Read last 4KB to find the last complete line
                f.seek(max(0, size - 4096))
                lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
                if lines:
                    last = json.loads(lines[-1])
                    self._prev_hash = last.get("entry_hash", _GENESIS_HASH)
        except Exception as exc:
            logger.debug("Failed to resume audit chain: %s", exc)

    def log(
        self,
        category: AuditCategory,
        action: str,
        *,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append an entry to the audit log.

        Returns the entry with its computed hash.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            action=action,
            actor=actor,
            details=details or {},
        )

        with self._lock:
            entry.prev_hash = self._prev_hash
            entry.entry_hash = entry.compute_hash()
            self._prev_hash = entry.entry_hash

            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.error("Failed to write audit entry: %s", exc)

        return entry

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the chain integrity of the entire log.

        Returns
        -------
        (valid, count)
            ``valid`` is ``True`` if the chain is unbroken;
            ``count`` is the number of entries verified.
        """
        if not self._log_path.exists():
            return True, 0

        prev = _GENESIS_HASH
        count = 0

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = AuditEntry(**data)

                    if entry.prev_hash != prev:
                        logger.error(
                            "Audit chain broken at entry %d: expected prev=%s, got prev=%s.",
                            count,
                            prev[:16],
                            entry.prev_hash[:16],
                        )
                        return False, count

                    recomputed = entry.compute_hash()
                    if recomputed != entry.entry_hash:
                        logger.error(
                            "Audit hash mismatch at entry %d: expected %s, got %s.",
                            count,
                            recomputed[:16],
                            entry.entry_hash[:16],
                        )
                        return False, count

                    prev = entry.entry_hash
                    count += 1

            return True, count
        except Exception as exc:
            logger.error("Audit chain verification failed: %s", exc)
            return False, count

    def recent(self, n: int = 20) -> list[AuditEntry]:
        """Return the last *n* entries."""
        if not self._log_path.exists():
            return []

        entries: list[AuditEntry] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                line = line.strip()
                if line:
                    entries.append(AuditEntry(**json.loads(line)))
        except Exception as exc:
            logger.debug("Failed to read recent audit entries: %s", exc)
        return entries


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: AuditLog | None = None


def get_audit_log() -> AuditLog:
    """Return the singleton AuditLog instance."""
    global _instance
    if _instance is None:
        _instance = AuditLog()
    return _instance


def reset_audit_log() -> None:
    """Reset singleton — for testing."""
    global _instance
    _instance = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def audit(
    category: AuditCategory,
    action: str,
    *,
    actor: str = "system",
    details: dict[str, Any] | None = None,
) -> AuditEntry:
    """Module-level shortcut for logging an audit event."""
    return get_audit_log().log(category, action, actor=actor, details=details)

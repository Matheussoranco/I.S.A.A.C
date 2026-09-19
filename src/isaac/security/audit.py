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
* ``constitution`` — constitutional action reviews
* ``security``  — other security policy events
* ``system``    — startup / shutdown / config changes
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

AuditCategory = Literal[
    "auth",
    "tool",
    "approval",
    "guard",
    "sandbox",
    "constitution",
    "security",
    "system",
]

_GENESIS_HASH = "0" * 64


@contextmanager
def _process_lock(path: Path):
    """Serialize the tail-read and append across independent processes."""
    with path.open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl  # type: ignore[import-not-found]

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def _decode_entry(data: dict[str, Any]) -> AuditEntry:
    # Records written before hash versioning use the original actor-less hash.
    return AuditEntry(**{**data, "hash_version": data.get("hash_version", 1)})


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
    # Version 1 entries predate actor binding (kept verifiable on read).
    # All newly-created entries default to version 2, which binds actor
    # into the hash chain.
    hash_version: int = 2

    def compute_hash(self) -> str:
        """Compute the versioned SHA-256 integrity hash for this entry."""
        details_json = json.dumps(self.details, sort_keys=True)
        if self.hash_version >= 2:
            payload = (
                f"{self.prev_hash}|{self.timestamp}|{self.category}|{self.action}|"
                f"{self.actor}|{details_json}"
            )
        else:
            payload = (
                f"{self.prev_hash}|{self.timestamp}|{self.category}|{self.action}|{details_json}"
            )
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
        self._lock_path = log_dir / "audit.lock"

        # Resume chain hash from last entry
        with _process_lock(self._lock_path):
            self._resume_chain()

    def _resume_chain(self) -> None:
        """Read the last line of the log to get the previous hash.

        Reverse block scan until a complete line is found (handles records
        larger than 4KB); never resets to genesis on corruption — fail-closed.
        Corruption raises and logs an alert; the last good hash is mirrored
        to a ``prev_hash`` sidecar for forensics.
        """
        if not self._log_path.exists():
            self._prev_hash = _GENESIS_HASH
            return
        try:
            with open(self._log_path, "rb") as f:
                # Seek to end, read backwards to find last newline
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    self._prev_hash = _GENESIS_HASH
                    return
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    raise ValueError("incomplete final audit record")
                position, tail = size - 1, b""
                while position > 0:
                    block = min(4096, position)
                    position -= block
                    f.seek(position)
                    tail = f.read(block) + tail
                    if b"\n" in tail:
                        tail = tail.rsplit(b"\n", 1)[1]
                        break
                last = _decode_entry(json.loads(tail.decode("utf-8")))
                if last.compute_hash() != last.entry_hash:
                    raise ValueError("final audit record hash mismatch")
                self._prev_hash = last.entry_hash
            self._write_sidecar()
        except Exception as exc:
            logger.error("Audit log corrupt, refusing to reset to genesis: %s", exc)
            raise ValueError(f"Cannot append to corrupt audit log: {exc}") from exc

    def _write_sidecar(self) -> None:
        """Mirror the last good hash to a ``prev_hash`` sidecar (best-effort)."""
        try:
            (self._log_path.parent / "prev_hash").write_text(self._prev_hash, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write audit prev_hash sidecar: %s", exc)

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
            hash_version=2,
        )

        with self._lock, _process_lock(self._lock_path):
            # Another instance/process may have appended since our last write.
            self._resume_chain()
            entry.prev_hash = self._prev_hash
            entry.entry_hash = entry.compute_hash()

            try:
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self._prev_hash = entry.entry_hash
                self._write_sidecar()
            except Exception as exc:
                logger.error("Failed to write audit entry: %s", exc)
                raise

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
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = _decode_entry(data)

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
        if n <= 0 or not self._log_path.exists():
            return []

        entries: list[AuditEntry] = []
        try:
            with open(self._log_path, encoding="utf-8") as f:
                lines = deque(f, maxlen=n)
            for line in lines:
                line = line.strip()
                if line:
                    entries.append(_decode_entry(json.loads(line)))
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

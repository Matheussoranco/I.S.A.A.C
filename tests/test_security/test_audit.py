"""Tests for the Audit Log — hash-chained, append-only log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaac.security.audit import AuditEntry, AuditLog, _GENESIS_HASH


@pytest.fixture()
def audit_dir(tmp_path: Path) -> Path:
    return tmp_path / "audit"


@pytest.fixture()
def audit_log(audit_dir: Path) -> AuditLog:
    return AuditLog(log_dir=audit_dir)


class TestAuditEntry:
    def test_compute_hash_deterministic(self) -> None:
        e1 = AuditEntry(
            timestamp="2025-01-01T00:00:00Z",
            category="system",
            action="startup",
            prev_hash=_GENESIS_HASH,
        )
        h1 = e1.compute_hash()
        h2 = e1.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_data(self) -> None:
        e1 = AuditEntry(
            timestamp="2025-01-01T00:00:00Z",
            category="system",
            action="startup",
            prev_hash=_GENESIS_HASH,
        )
        e2 = AuditEntry(
            timestamp="2025-01-01T00:00:00Z",
            category="system",
            action="shutdown",
            prev_hash=_GENESIS_HASH,
        )
        assert e1.compute_hash() != e2.compute_hash()


class TestAuditLog:
    def test_log_creates_file(self, audit_log: AuditLog, audit_dir: Path) -> None:
        audit_log.log("system", "startup")
        assert (audit_dir / "audit.jsonl").exists()

    def test_log_returns_entry_with_hash(self, audit_log: AuditLog) -> None:
        entry = audit_log.log("tool", "execute", actor="test", details={"tool": "search"})
        assert entry.entry_hash != ""
        assert entry.prev_hash == _GENESIS_HASH

    def test_chain_links_entries(self, audit_log: AuditLog) -> None:
        e1 = audit_log.log("system", "startup")
        e2 = audit_log.log("system", "shutdown")
        assert e2.prev_hash == e1.entry_hash

    def test_verify_chain_valid(self, audit_log: AuditLog) -> None:
        for i in range(5):
            audit_log.log("system", f"event_{i}")
        valid, count = audit_log.verify_chain()
        assert valid is True
        assert count == 5

    def test_verify_chain_detects_tampering(self, audit_log: AuditLog, audit_dir: Path) -> None:
        for i in range(3):
            audit_log.log("system", f"event_{i}")

        # Tamper with the log
        log_path = audit_dir / "audit.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        entry = json.loads(lines[1])
        entry["action"] = "TAMPERED"
        lines[1] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        valid, count = audit_log.verify_chain()
        assert valid is False

    def test_recent(self, audit_log: AuditLog) -> None:
        for i in range(10):
            audit_log.log("system", f"event_{i}")
        recent = audit_log.recent(n=3)
        assert len(recent) == 3
        assert recent[-1].action == "event_9"

    def test_empty_log_recent(self, audit_log: AuditLog) -> None:
        assert audit_log.recent() == []

    def test_empty_log_verify(self, audit_log: AuditLog) -> None:
        valid, count = audit_log.verify_chain()
        assert valid is True
        assert count == 0

    def test_resume_chain_on_reopen(self, audit_dir: Path) -> None:
        log1 = AuditLog(log_dir=audit_dir)
        e1 = log1.log("system", "startup")
        del log1

        log2 = AuditLog(log_dir=audit_dir)
        e2 = log2.log("system", "resumed")
        assert e2.prev_hash == e1.entry_hash

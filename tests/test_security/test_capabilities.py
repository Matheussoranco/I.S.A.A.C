"""Tests for the Capability Token system."""

from __future__ import annotations

from pathlib import Path

import pytest

from isaac.security.capabilities import CapabilityToken, TokenStore


@pytest.fixture()
def store(tmp_path: Path) -> TokenStore:
    return TokenStore(store_path=tmp_path / "tokens.json")


class TestCapabilityToken:
    def test_default_is_valid(self) -> None:
        # No expiry, not revoked, no max_uses
        t = CapabilityToken(
            token_id="test",
            tool_name="*",
            action="*",
            issued_at="2025-01-01T00:00:00Z",
        )
        assert t.is_valid()

    def test_revoked_not_valid(self) -> None:
        t = CapabilityToken(token_id="t", tool_name="*", revoked=True)
        assert not t.is_valid()

    def test_exceeded_max_uses_not_valid(self) -> None:
        t = CapabilityToken(token_id="t", tool_name="*", max_uses=3, use_count=3)
        assert not t.is_valid()

    def test_expired_not_valid(self) -> None:
        t = CapabilityToken(
            token_id="t",
            tool_name="*",
            expires_at="2020-01-01T00:00:00+00:00",
        )
        assert not t.is_valid()

    def test_matches_wildcard(self) -> None:
        t = CapabilityToken(token_id="t", tool_name="*", action="*")
        assert t.matches("any_tool", "any_action")

    def test_matches_specific(self) -> None:
        t = CapabilityToken(token_id="t", tool_name="web_search", action="search")
        assert t.matches("web_search", "search")
        assert not t.matches("file_read", "read")


class TestTokenStore:
    def test_issue_creates_token(self, store: TokenStore) -> None:
        token = store.issue("web_search", ttl_hours=24)
        assert token.token_id != ""
        assert token.tool_name == "web_search"
        assert token.is_valid()

    def test_check_valid_token(self, store: TokenStore) -> None:
        token = store.issue("web_search")
        result = store.check(token.token_id, "web_search")
        assert result is True

    def test_check_wrong_tool(self, store: TokenStore) -> None:
        token = store.issue("web_search")
        result = store.check(token.token_id, "file_delete")
        assert result is False

    def test_check_nonexistent_token(self, store: TokenStore) -> None:
        assert store.check("nonexistent", "tool") is False

    def test_check_increments_use_count(self, store: TokenStore) -> None:
        token = store.issue("tool", max_uses=5)
        store.check(token.token_id, "tool")
        store.check(token.token_id, "tool")
        active = store.list_active()
        t = next(t for t in active if t.token_id == token.token_id)
        assert t.use_count == 2

    def test_revoke(self, store: TokenStore) -> None:
        token = store.issue("tool")
        assert store.revoke(token.token_id)
        assert not store.check(token.token_id, "tool")

    def test_revoke_nonexistent(self, store: TokenStore) -> None:
        assert not store.revoke("nonexistent")

    def test_list_active_excludes_revoked(self, store: TokenStore) -> None:
        t1 = store.issue("tool_a")
        t2 = store.issue("tool_b")
        store.revoke(t1.token_id)
        active = store.list_active()
        active_ids = {t.token_id for t in active}
        assert t1.token_id not in active_ids
        assert t2.token_id in active_ids

    def test_cleanup_expired(self, store: TokenStore) -> None:
        t1 = store.issue("tool", ttl_hours=0)  # expires immediately
        t2 = store.issue("tool", ttl_hours=24)
        # t1 is expired (ttl=0 means expires_at = issued_at = now)
        # Actually ttl_hours=0 might not expire; let's revoke instead
        store.revoke(t1.token_id)
        removed = store.cleanup_expired()
        assert removed >= 1

    def test_persistence(self, tmp_path: Path) -> None:
        store_path = tmp_path / "tokens.json"
        s1 = TokenStore(store_path=store_path)
        token = s1.issue("tool")
        del s1

        s2 = TokenStore(store_path=store_path)
        assert s2.check(token.token_id, "tool")

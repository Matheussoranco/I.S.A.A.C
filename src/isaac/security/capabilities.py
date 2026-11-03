"""Capability Tokens — fine-grained permission system for tool access.

Each tool invocation must present a valid capability token that grants
the specific action.  Tokens are:
- Scoped to a tool name and optional action.
- Time-limited (expiry).
- Revocable by the operator.
- Logged in the audit trail.

The token store persists to ``~/.isaac/security/tokens.json``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CapabilityToken:
    """A capability token granting access to a specific tool/action."""

    token_id: str = ""
    """Unique (opaque) identifier."""
    tool_name: str = ""
    """Tool this token grants access to. ``"*"`` means all tools."""
    action: str = "*"
    """Specific action within the tool. ``"*"`` means all actions."""
    issued_at: str = ""
    expires_at: str = ""
    issued_by: str = "system"
    revoked: bool = False
    max_uses: int = 0
    """0 = unlimited."""
    use_count: int = 0

    def is_valid(self) -> bool:
        """Check if this token is currently valid (not expired, not revoked, under usage limit)."""
        if self.revoked:
            return False
        if self.max_uses > 0 and self.use_count >= self.max_uses:
            return False
        if self.expires_at:
            try:
                exp = datetime.fromisoformat(self.expires_at)
                if datetime.now(UTC) > exp:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def matches(self, tool_name: str, action: str = "*") -> bool:
        """Check if this token matches the requested tool/action."""
        if not self.is_valid():
            return False
        tool_ok = self.tool_name == "*" or self.tool_name == tool_name
        action_ok = self.action == "*" or self.action == action
        return tool_ok and action_ok


class TokenStore:
    """Persistent store for capability tokens."""

    def __init__(self, store_path: Path | None = None) -> None:
        if store_path is None:
            try:
                from isaac.config.settings import get_settings

                store_path = get_settings().isaac_home / "security" / "tokens.json"
            except Exception:
                store_path = Path.home() / ".isaac" / "security" / "tokens.json"

        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens: dict[str, CapabilityToken] = {}
        self._lock = threading.RLock()
        self._mtime_ns: int | None = None
        self._load()

    def _load(self) -> None:
        try:
            self._mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            self._mtime_ns = None
            self._tokens.clear()
            return
        except OSError:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._tokens = {tid: CapabilityToken(**tdata) for tid, tdata in data.items()}
        except Exception as exc:
            logger.error("Failed to load token store: %s", exc)
            self._tokens.clear()

    def _refresh(self) -> None:
        """Pick up grants and revocations written by another process."""
        try:
            mtime_ns = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime_ns = None
        except OSError:
            return
        if mtime_ns != self._mtime_ns:
            self._load()

    def _save(self) -> None:
        """Atomically persist grants before reporting an authorization change."""
        temporary = self._path.with_name(f".{self._path.name}.{secrets.token_hex(8)}.tmp")
        try:
            data = {tid: asdict(t) for tid, t in self._tokens.items()}
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            try:
                self._mtime_ns = self._path.stat().st_mtime_ns
            except OSError:
                self._mtime_ns = None
        except Exception as exc:
            logger.error("Failed to save token store: %s", exc)
            temporary.unlink(missing_ok=True)
            raise

    def issue(
        self,
        tool_name: str,
        *,
        action: str = "*",
        ttl_hours: float = 24,
        issued_by: str = "system",
        max_uses: int = 0,
    ) -> CapabilityToken:
        """Issue a new capability token."""
        with self._lock:
            self._refresh()
            now = datetime.now(UTC)
            token = CapabilityToken(
                token_id=secrets.token_hex(16),
                tool_name=tool_name,
                action=action,
                issued_at=now.isoformat(),
                expires_at=(now + timedelta(hours=ttl_hours)).isoformat(),
                issued_by=issued_by,
                max_uses=max_uses,
            )
            self._tokens[token.token_id] = token
            try:
                self._save()
            except Exception:
                del self._tokens[token.token_id]
                raise

        # Audit
        try:
            from isaac.security.audit import audit

            audit(
                "auth",
                "token_issued",
                actor=issued_by,
                details={"token_id": token.token_id, "tool": tool_name, "ttl_hours": ttl_hours},
            )
        except Exception:
            pass

        logger.info(
            "Issued token %s for tool '%s' (ttl=%.2fh).",
            token.token_id[:8],
            tool_name,
            ttl_hours,
        )
        return token

    def check(self, token_id: str, tool_name: str, action: str = "*") -> bool:
        """Validate a token for a tool/action.  Increments use_count if valid."""
        with self._lock:
            self._refresh()
            token = self._tokens.get(token_id)
            if token is None:
                return False

            if not token.matches(tool_name, action):
                return False

            token.use_count += 1
            exhausted = token.max_uses > 0 and token.use_count >= token.max_uses
            try:
                if exhausted:
                    del self._tokens[token.token_id]
                self._save()
            except Exception:
                token.use_count -= 1
                self._tokens.setdefault(token.token_id, token)
                return False

        # Audit
        try:
            from isaac.security.audit import audit

            audit(
                "auth",
                "token_used",
                details={
                    "token_id": token_id,
                    "tool": tool_name,
                    "action": action,
                    "uses": token.use_count,
                },
            )
        except Exception:
            pass

        return True

    def consume_matching(self, tool_name: str, action: str = "*") -> CapabilityToken | None:
        """Atomically consume and return one matching active token.

        This is the execution-facing API.  Callers do not need to expose opaque
        token IDs to an LLM; an operator-issued grant can be consumed internally
        immediately before the action it authorizes.
        """
        consumed: CapabilityToken | None = None
        with self._lock:
            self._refresh()
            for tid, token in list(self._tokens.items()):
                if token.matches(tool_name, action):
                    token.use_count += 1
                    exhausted = token.max_uses > 0 and token.use_count >= token.max_uses
                    try:
                        if exhausted:
                            del self._tokens[tid]
                        self._save()
                    except Exception:
                        token.use_count -= 1
                        self._tokens.setdefault(tid, token)
                        return None
                    consumed = token
                    break
        if consumed is not None:
            try:
                from isaac.security.audit import audit

                audit(
                    "auth",
                    "token_used",
                    details={
                        "token_id": consumed.token_id,
                        "tool": tool_name,
                        "action": action,
                        "uses": consumed.use_count,
                    },
                )
            except Exception:
                pass
        return consumed

    def revoke(self, token_id: str, *, revoked_by: str = "system") -> bool:
        """Revoke a token. Returns True if found and revoked."""
        with self._lock:
            self._refresh()
            token = self._tokens.get(token_id)
            if token is None:
                return False

            token.revoked = True
            try:
                self._save()
            except Exception:
                token.revoked = False
                return False

        try:
            from isaac.security.audit import audit

            audit("auth", "token_revoked", actor=revoked_by, details={"token_id": token_id})
        except Exception:
            pass

        return True

    def list_active(self) -> list[CapabilityToken]:
        """Return all active (non-revoked, non-expired) tokens."""
        with self._lock:
            self._refresh()
            return [t for t in self._tokens.values() if t.is_valid()]

    def cleanup_expired(self) -> int:
        """Remove expired or revoked tokens. Returns count removed."""
        with self._lock:
            self._refresh()
            to_remove = [tid for tid, t in self._tokens.items() if not t.is_valid()]
            for tid in to_remove:
                del self._tokens[tid]
            if to_remove:
                try:
                    self._save()
                except Exception:
                    self._load()
                    raise
        return len(to_remove)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: TokenStore | None = None


def get_token_store() -> TokenStore:
    """Return the singleton TokenStore."""
    global _instance
    if _instance is None:
        _instance = TokenStore()
    return _instance


def reset_token_store() -> None:
    """Reset singleton — for testing."""
    global _instance
    _instance = None

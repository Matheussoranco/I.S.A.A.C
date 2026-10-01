"""Management UI backend for provider API keys.
Integrates with the OS keyring via isaac.security.credentials.
"""

from __future__ import annotations

from fastapi import HTTPException

from isaac.security import credentials


def list_keys() -> dict[str, bool]:
    """Check which providers have keys available."""
    providers = ["openai", "anthropic"]
    return {p: credentials.credential_available(p) for p in providers}


def set_key(provider: str, value: str) -> dict:
    """Set or remove an API key in the keyring."""
    try:
        credentials.set_credential(provider, value)
        return {"ok": True, "provider": provider}
    except Exception as e:
        raise HTTPException(500, str(e)) from e

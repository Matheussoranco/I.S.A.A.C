"""Shared host-path security policy.

Every component that can reach the user's real filesystem must use this
module.  Keeping the credential deny-list in one place prevents a safer tool
surface from being bypassed through an older connector implementation.
"""

from __future__ import annotations

from pathlib import Path

DENIED_DIR_NAMES = frozenset(
    {
        ".ssh",
        ".aws",
        ".azure",
        ".gcloud",
        ".gnupg",
        ".kube",
        ".docker",
        ".password-store",
        ".mozilla",
        ".thunderbird",
    }
)

DENIED_FILE_NAMES = frozenset(
    {
        ".netrc",
        "_netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "credentials.json",
        "id_rsa",
        "id_ecdsa",
        "id_ed25519",
        "id_dsa",
    }
)

DENIED_SUFFIXES = frozenset({".pem", ".key", ".pfx", ".p12", ".kdbx"})


def is_sensitive_path(path: Path) -> bool:
    """Return whether *path* is, or is inside, a credential location."""
    resolved_parts = tuple(part.lower() for part in path.parts)
    if any(part in DENIED_DIR_NAMES for part in resolved_parts[:-1]):
        return True
    name = path.name.lower()
    return (
        name in DENIED_DIR_NAMES
        or name in DENIED_FILE_NAMES
        or name.startswith(".env")
        or path.suffix.lower() in DENIED_SUFFIXES
    )

"""Profile-aware writable configuration (Hermes-mirror, Phase 1.2).

Each profile owns a directory under ``~/.isaac/profiles/<name>/`` containing
a ``config.yaml`` whose keys mirror the Settings schema (nested, e.g.
``llm: {model_name: ...}``).  Profile overrides are layered on top of
environment variables and defaults (precedence: profile > env > default).

The active profile is chosen via the ``ISAAC_PROFILE`` env var
(default ``"default"``).  ``ISAAC_HOME``, when set, relocates the
``~/.isaac`` root — mostly useful for hermetic tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PROFILE = "default"


def _isaac_root() -> Path:
    """Root ISAAC data dir; honours ISAAC_HOME for tests/sandboxes."""
    override = os.environ.get("ISAAC_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".isaac"


def get_active_profile() -> str:
    """Return the active profile name (``ISAAC_PROFILE`` or ``"default"``)."""
    return (os.environ.get("ISAAC_PROFILE") or "").strip() or DEFAULT_PROFILE


def profile_dir(name: str) -> Path:
    """Return the directory for profile *name*, creating it on demand."""
    name = (name or "").strip() or DEFAULT_PROFILE
    d = _isaac_root() / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def profile_config_path(name: str) -> Path:
    """Return the ``config.yaml`` path inside profile *name*'s directory."""
    return profile_dir(name) / "config.yaml"


def list_profiles() -> list[str]:
    """List profile names (subdirs of ``~/.isaac/profiles/``)."""
    root = _isaac_root() / "profiles"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_profile_overrides(name: str) -> dict[str, Any]:
    """Parse the profile's config.yaml into a nested dict.

    Returns an empty dict when the file is missing, empty, invalid, or
    pyyaml is not installed — this function never raises.
    """
    try:
        path = profile_config_path(name)
    except Exception:
        return {}
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_profile_overrides(name: str, data: dict[str, Any]) -> Path:
    """Write *data* as YAML to the profile's config.yaml; returns the path."""
    import yaml  # type: ignore[import-untyped]

    path = profile_config_path(name)
    path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def sniff_scalar(raw: str) -> Any:
    """Best-effort scalar type-sniff for CLI input: int/float/bool else str."""
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw

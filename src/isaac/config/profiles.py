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

import yaml  # type: ignore[import-untyped]

from isaac.config.settings import ProfileConfigError, validate_profile_overrides
from isaac.security.file_mutation import write_text

DEFAULT_PROFILE = "default"

_MAX_PROFILE_NAME_LENGTH = 64
_INVALID_PROFILE_CHARS = frozenset('/\\:*?"<>|')


def validate_profile_name(name: str) -> str:
    """Return a filesystem-safe profile name or raise ``ValueError``."""
    name = (name or "").strip() or DEFAULT_PROFILE
    if (
        name in {".", ".."}
        or len(name) > _MAX_PROFILE_NAME_LENGTH
        or name.startswith(".")
        or name.endswith((".", " "))
        or any(char in _INVALID_PROFILE_CHARS or ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Invalid profile name: {name!r}")
    return name


def _isaac_root() -> Path:
    """Root ISAAC data dir; honours ISAAC_HOME for tests/sandboxes."""
    override = (os.environ.get("ISAAC_HOME") or os.environ.get("ISAAC_ISAAC_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".isaac"


def get_active_profile() -> str:
    """Return the active profile name (``ISAAC_PROFILE`` or ``"default"``)."""
    return (os.environ.get("ISAAC_PROFILE") or "").strip() or DEFAULT_PROFILE


def profile_dir(name: str) -> Path:
    """Return the directory for profile *name*, creating it on demand."""
    name = validate_profile_name(name)
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


def load_profile_overrides(name: str, *, strict: bool = False) -> dict[str, Any]:
    """Read a profile's YAML, optionally reporting invalid configuration."""
    try:
        try:
            path = profile_config_path(name)
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (ValueError, OSError, yaml.YAMLError, RecursionError):
            raise ProfileConfigError("Could not read config.yaml as YAML.") from None
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ProfileConfigError("config.yaml must contain a mapping.")
        validate_profile_overrides(data)
        return data
    except ProfileConfigError:
        if strict:
            raise
        return {}


def save_profile_overrides(name: str, data: dict[str, Any]) -> Path:
    """Validate and atomically replace the profile's YAML."""
    validate_profile_overrides(data)
    path = profile_config_path(name)
    content = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    write_text(path, content, overwrite=True)
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

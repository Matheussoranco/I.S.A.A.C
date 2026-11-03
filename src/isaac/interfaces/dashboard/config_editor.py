"""Visual configuration editor backend for Isaac profiles.

Handles YAML read/write and validation against the Settings schema.
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]
from fastapi import HTTPException

from isaac.config.profiles import (
    get_active_profile,
    profile_config_path,
    save_profile_overrides,
)


def _config_path(profile_name: str | None) -> Path:
    try:
        return profile_config_path(profile_name or get_active_profile())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def get_config_content(profile_name: str | None = None) -> dict:
    """Read the config.yaml for a given profile."""
    path = _config_path(profile_name)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(500, f"Error reading config: {e}") from e
    return data if isinstance(data, dict) else {}


def update_config(payload: dict, profile_name: str | None = None) -> dict:
    """Update the config.yaml for a given profile."""
    path = _config_path(profile_name)
    try:
        save_profile_overrides(path.parent.name, payload)
    except Exception as e:
        raise HTTPException(500, f"Error writing config: {e}") from e
    return {"ok": True, "profile": path.parent.name}

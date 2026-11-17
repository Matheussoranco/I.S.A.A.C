"""Visual configuration editor backend for Isaac profiles.

Handles YAML read/write and validation against the Settings schema.
"""

from __future__ import annotations

from fastapi import HTTPException

from isaac.config.profiles import (
    get_active_profile,
    load_profile_overrides,
    save_profile_overrides,
    validate_profile_name,
)
from isaac.config.settings import ProfileConfigError, clear_settings_cache


def _profile_name(profile_name: str | None) -> str:
    try:
        return validate_profile_name(profile_name or get_active_profile())
    except ValueError as exc:
        raise HTTPException(400, "Invalid profile name.") from exc


def get_config_content(profile_name: str | None = None) -> dict:
    """Return the validated non-secret profile settings."""
    name = _profile_name(profile_name)
    try:
        return load_profile_overrides(name, strict=True)
    except ProfileConfigError as exc:
        raise HTTPException(400, str(exc)) from exc


def update_config(payload: dict, profile_name: str | None = None) -> dict:
    """Save a validated profile configuration."""
    name = _profile_name(profile_name)
    try:
        save_profile_overrides(name, payload)
    except ProfileConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not write config.yaml.") from exc
    clear_settings_cache()
    return {"ok": True, "profile": name}

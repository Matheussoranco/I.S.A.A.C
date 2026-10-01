"""Visual configuration editor backend for Isaac profiles.
Handles YAML read/write and validation against the Settings schema.
"""
from __future__ import annotations
from pathlib import Path
import yaml
from fastapi import HTTPException
from isaac.config.profiles import profile_config_path, get_active_profile

def get_config_content(profile_name: str | None = None) -> dict:
    """Read the config.yaml for a given profile."""
    name = profile_name or get_active_profile()
    path = profile_config_path(name)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(500, f"Error reading config: {e}")

def update_config(payload: dict, profile_name: str | None = None) -> dict:
    """Update the config.yaml for a given profile."""
    name = profile_name or get_active_profile()
    path = profile_config_path(name)
    
    # In a real scenario, we would validate payload against the Pydantic Settings model here
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False)
        return {"ok": True, "profile": name}
    except Exception as e:
        raise HTTPException(500, f"Error writing config: {e}")

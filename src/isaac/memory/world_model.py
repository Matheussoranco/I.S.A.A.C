"""WorldModel builder and sync utilities.

Provides helpers to snapshot the environment (via sandbox introspection)
and apply incremental updates from Perception observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

from isaac.core.state import WorldModel


def empty_world() -> WorldModel:
    """Return a blank world model."""
    return WorldModel()


def merge_observations(
    model: WorldModel,
    new_observations: list[str],
) -> WorldModel:
    """Return a *new* WorldModel with ``new_observations`` appended.

    Older observations beyond a rolling window of 50 are discarded.
    """
    combined = model.observations + new_observations
    return replace(
        model,
        files=dict(model.files),
        resources=dict(model.resources),
        constraints=list(model.constraints),
        observations=combined[-50:],
    )


def update_files(model: WorldModel, files: dict[str, str]) -> WorldModel:
    """Return a new WorldModel with the ``files`` dict replaced/updated."""
    return replace(model, files={**model.files, **files})


def update_resources(model: WorldModel, resources: dict[str, Any]) -> WorldModel:
    """Return a new WorldModel with updated resource snapshots."""
    return replace(model, resources={**model.resources, **resources})


def content_hash(data: str) -> str:
    """SHA-256 hex-digest of *data* (used for file deduplication)."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]

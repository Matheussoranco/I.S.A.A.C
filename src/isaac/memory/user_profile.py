"""User Profile — persistent JSON-based user preferences and metadata.

Tracks the user's name, preferences, interaction count, and inferred
interests.  Updated automatically after each session by the Reflection node.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE: dict[str, Any] = {
    "name": "",
    "preferences": {},
    "interaction_count": 0,
    "first_seen": "",
    "last_seen": "",
    "tags": [],
}


class UserProfile:
    """JSON-file-backed user profile.

    Parameters
    ----------
    profile_path:
        Path to the JSON file.  Created on first access.
    """

    def __init__(self, profile_path: str | Path | None = None) -> None:
        if profile_path is None:
            try:
                from isaac.config.settings import settings
                profile_path = Path(settings.user_profile_path).expanduser()
            except Exception:
                profile_path = Path.home() / ".isaac" / "user_profile.json"

        self._path = Path(profile_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Load profile from disk, or create a default."""
        if self._path.is_file():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                merged = dict(_DEFAULT_PROFILE)
                merged.update(raw)
                return merged
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load user profile: %s", exc)
        return dict(_DEFAULT_PROFILE)

    def save(self) -> None:
        """Persist current profile to disk."""
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @name.setter
    def name(self, value: str) -> None:
        self._data["name"] = value

    @property
    def preferences(self) -> dict[str, Any]:
        return self._data.get("preferences", {})

    @property
    def interaction_count(self) -> int:
        return self._data.get("interaction_count", 0)

    @property
    def first_seen(self) -> str:
        return self._data.get("first_seen", "")

    @property
    def last_seen(self) -> str:
        return self._data.get("last_seen", "")

    @property
    def tags(self) -> list[str]:
        return self._data.get("tags", [])

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        if "preferences" not in self._data:
            self._data["preferences"] = {}
        self._data["preferences"][key] = value

    def add_tag(self, tag: str) -> None:
        """Add an inferred interest tag (deduplicated)."""
        tags = self._data.setdefault("tags", [])
        if tag not in tags:
            tags.append(tag)

    def record_interaction(self) -> None:
        """Increment interaction count and update timestamps."""
        now = datetime.now(tz=timezone.utc).isoformat()
        self._data["interaction_count"] = self._data.get("interaction_count", 0) + 1
        if not self._data.get("first_seen"):
            self._data["first_seen"] = now
        self._data["last_seen"] = now

    def update_after_session(
        self,
        inferred_tags: list[str] | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> None:
        """Batch update at session end — called by the Reflection node."""
        self.record_interaction()
        if inferred_tags:
            for tag in inferred_tags:
                self.add_tag(tag)
        if preferences:
            for k, v in preferences.items():
                self.set_preference(k, v)
        self.save()

    def to_context_string(self) -> str:
        """Format for injection into LLM prompts."""
        parts: list[str] = []
        if self.name:
            parts.append(f"User name: {self.name}")
        if self.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in self.preferences.items())
            parts.append(f"Preferences: {prefs}")
        if self.tags:
            parts.append(f"Interests: {', '.join(self.tags)}")
        parts.append(f"Interactions: {self.interaction_count}")
        return "\n".join(parts) if parts else ""

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the raw profile data."""
        return dict(self._data)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: UserProfile | None = None


def get_user_profile() -> UserProfile:
    """Return the singleton UserProfile instance."""
    global _instance  # noqa: PLW0603
    if _instance is None:
        _instance = UserProfile()
    return _instance


def reset_user_profile() -> None:
    """Reset the singleton (used in tests)."""
    global _instance  # noqa: PLW0603
    _instance = None

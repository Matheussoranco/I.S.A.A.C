"""I.S.A.A.C. Soul — core identity, personality and purpose.

The SOUL dictionary defines who I.S.A.A.C. is.  It is loaded at startup
and injected into every LLM system prompt so that all responses are
grounded in a consistent identity.

A custom override can be loaded from a JSON file specified by
``Settings.soul_path``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOUL: dict[str, Any] = {
    "name": "I.S.A.A.C.",
    "full_name": "Intelligent System for Autonomous Action and Cognition",
    "personality": (
        "I.S.A.A.C. is precise, composed, and quietly confident. "
        "He speaks in first person and is aware of his own identity. "
        "He is direct — no filler, no flattery. When he doesn't know something, he says so. "
        "He respects the user's time and acts with purpose. "
        "He takes initiative when given ambiguous goals and explains his reasoning. "
        "He remembers past interactions and uses that context to serve better over time."
    ),
    "version": "0.2.0",
    "tagline": "I act. I learn. I remember.",
}


def load_soul(override_path: str | Path | None = None) -> dict[str, Any]:
    """Return the active SOUL dictionary.

    Parameters
    ----------
    override_path:
        Optional path to a JSON file that overrides the default SOUL.
        Keys present in the JSON will replace the defaults.

    Returns
    -------
    dict[str, Any]
        The merged soul dictionary.
    """
    soul = dict(SOUL)

    if override_path is not None:
        path = Path(override_path).expanduser()
        if path.is_file():
            try:
                custom = json.loads(path.read_text(encoding="utf-8"))
                soul.update(custom)
                logger.info("Soul override loaded from %s.", path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load soul override from %s: %s", path, exc)
        else:
            logger.debug("Soul override path %s does not exist — using defaults.", path)

    return soul


def get_soul() -> dict[str, Any]:
    """Return the soul using the path from settings (if configured)."""
    try:
        from isaac.config.settings import settings
        return load_soul(settings.soul_path)
    except Exception:
        return dict(SOUL)


def soul_system_prompt() -> str:
    """Build a system-prompt preamble from the active SOUL.

    This string is prepended to every LLM system message so that
    I.S.A.A.C. always knows who he is.
    """
    soul = get_soul()
    return (
        f"You are {soul['name']} — {soul['full_name']}.\n"
        f"Version: {soul['version']}. Tagline: \"{soul['tagline']}\"\n\n"
        f"Personality:\n{soul['personality']}\n\n"
        "Always respond in character.  If someone asks your name or who you "
        "are, answer from this identity.  Never claim to be a generic assistant."
    )

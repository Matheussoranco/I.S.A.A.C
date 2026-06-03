"""Persona Builder — let the user define, store, and activate custom personas.

The built-in identity lives in :mod:`isaac.identity.soul` as the ``SOUL`` dict.
This module lets a user craft their *own* persona — a name, voice, values, and
expertise — persist it, and make it the active identity that every node and
specialist speaks through.

Storage layout (under ``<isaac_home>/personas/``)::

    personas/
      atlas.json        ← a saved persona (a "soul" dict)
      sage.json
      active.txt        ← slug of the currently-active persona

Activating a persona writes its dict to the *active soul file* (``soul_path``
from settings, or ``<isaac_home>/soul.json`` by default) and invalidates the
cached soul prompt, so the change takes effect immediately in-process.  Set
``ISAAC_SOUL_PATH`` to that file in your ``.env`` to make it persist across
restarts.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REQUIRED_PERSONALITY = "personality"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return Path(get_settings().isaac_home)
    except Exception:  # pragma: no cover - settings should always load
        return Path.home() / ".isaac"


def personas_dir() -> Path:
    """Return ``<isaac_home>/personas``, creating it if necessary."""
    d = _isaac_home() / "personas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def active_soul_path() -> Path:
    """Return the active soul JSON path (``settings.soul_path`` or a default)."""
    try:
        from isaac.config.settings import get_settings

        configured = (get_settings().soul_path or "").strip()
    except Exception:  # pragma: no cover
        configured = ""
    if configured:
        return Path(configured).expanduser()
    return _isaac_home() / "soul.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Lowercase, kebab-case a name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "persona"


def _as_list(value: Any) -> list[str]:
    """Normalise a list-or-comma-string into a clean ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        items = [str(value)]
    return [s.strip() for s in items if str(s).strip()]


def _join_human(items: list[str]) -> str:
    """Join a list into a readable 'a, b and c' phrase."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


# ---------------------------------------------------------------------------
# Persona construction
# ---------------------------------------------------------------------------


def build_persona_from_answers(answers: dict[str, Any]) -> dict[str, Any]:
    """Turn a Q&A dict into a persona ("soul") dict.

    Recognised keys (all optional except ``name``): ``name``, ``full_name``,
    ``role``, ``tone``, ``values`` (list or comma-string), ``expertise`` (list
    or comma-string), ``quirks`` (free text), ``tagline``.

    Args:
        answers: The user's answers.

    Returns:
        A persona dict with keys ``name``, ``full_name``, ``personality``,
        ``version``, ``tagline``, ``traits``, ``values``.

    Raises:
        ValueError: If no ``name`` is supplied.
    """
    name = str(answers.get("name", "")).strip()
    if not name:
        raise ValueError("A persona needs a 'name'.")

    role = str(answers.get("role", "")).strip()
    tone = str(answers.get("tone", "")).strip()
    quirks = str(answers.get("quirks", "")).strip()
    values = _as_list(answers.get("values"))
    expertise = _as_list(answers.get("expertise"))
    full_name = str(answers.get("full_name", "")).strip() or name
    tagline = str(answers.get("tagline", "")).strip()

    # Compose a coherent first-person personality paragraph.
    sentences: list[str] = []
    sentences.append(f"I am {name}" + (f", a {role}." if role else "."))
    if tone:
        sentences.append(f"I communicate in a {tone} manner.")
    if values:
        sentences.append(f"I value {_join_human(values)}.")
    if expertise:
        sentences.append(f"I specialise in {_join_human(expertise)}.")
    if quirks:
        sentences.append(quirks if quirks.endswith(".") else quirks + ".")
    sentences.append("I remember past interactions and use that context to serve better over time.")
    personality = " ".join(sentences)

    # Derive a few short traits from the tone and quirks.
    traits = _as_list(tone.replace(" and ", ",").replace("/", ","))[:3]
    if not traits:
        traits = ["adaptive"]

    return {
        "name": name,
        "full_name": full_name,
        "personality": personality,
        "version": "1.0.0",
        "tagline": tagline,
        "traits": traits,
        "values": values,
    }


def interactive_questions() -> list[dict[str, Any]]:
    """Return the ordered question specs a CLI can drive to build a persona."""
    return [
        {"key": "name", "prompt": "Persona name", "required": True},
        {
            "key": "role",
            "prompt": "Role / what they are (e.g. research assistant)",
            "required": False,
        },
        {"key": "tone", "prompt": "Tone of voice (e.g. warm and concise)", "required": False},
        {"key": "values", "prompt": "Core values (comma-separated)", "required": False},
        {"key": "expertise", "prompt": "Areas of expertise (comma-separated)", "required": False},
        {"key": "quirks", "prompt": "Any quirks or signature habits", "required": False},
        {"key": "tagline", "prompt": "A short tagline", "required": False},
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def list_personas() -> list[str]:
    """Return the slugs of all saved personas, sorted."""
    return sorted(p.stem for p in personas_dir().glob("*.json"))


def save_persona(data: dict[str, Any], slug: str | None = None) -> Path:
    """Validate, normalise, and persist a persona; return its file path.

    Args:
        data: The persona dict (must include a non-empty ``name`` and
            ``personality``).
        slug: Override the auto-derived slug.

    Returns:
        The path the persona was written to.

    Raises:
        ValueError: If ``name`` or ``personality`` is missing.
    """
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("A persona needs a 'name'.")
    if not str(data.get(_REQUIRED_PERSONALITY, "")).strip():
        raise ValueError("A persona needs a 'personality'.")

    persona = dict(data)
    persona.setdefault("full_name", name)
    persona.setdefault("version", "1.0.0")
    persona.setdefault("tagline", "")

    slug = slug or slugify(name)
    path = personas_dir() / f"{slug}.json"
    path.write_text(json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved persona %r to %s", slug, path)
    return path


def load_persona(slug: str) -> dict[str, Any]:
    """Load and return a saved persona dict.

    Raises:
        FileNotFoundError: If no persona is saved under *slug*.
    """
    path = personas_dir() / f"{slug}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No persona saved under slug {slug!r}.")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_persona(slug: str) -> bool:
    """Delete a saved persona; return whether it existed."""
    path = personas_dir() / f"{slug}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def active_persona() -> str | None:
    """Return the slug of the active persona, or ``None``."""
    pointer = personas_dir() / "active.txt"
    if not pointer.is_file():
        return None
    slug = pointer.read_text(encoding="utf-8").strip()
    return slug or None


def activate_persona(slug: str) -> dict[str, Any]:
    """Make the persona under *slug* the active identity.

    Writes the persona to the active soul file, records the slug, points the
    in-process settings at it, and invalidates the cached soul prompt so the
    change is effective immediately.

    Returns:
        The activated persona dict.

    Raises:
        FileNotFoundError: If no persona is saved under *slug*.
    """
    persona = load_persona(slug)

    soul_path = active_soul_path()
    soul_path.parent.mkdir(parents=True, exist_ok=True)
    soul_path.write_text(json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8")

    (personas_dir() / "active.txt").write_text(slug, encoding="utf-8")

    # Point the running process at the new soul and clear the cached prompt.
    try:
        from isaac.config.settings import get_settings

        get_settings().soul_path = str(soul_path)
    except Exception:  # pragma: no cover - best-effort
        logger.debug("Could not update settings.soul_path", exc_info=True)
    try:
        from isaac.identity.soul import invalidate_soul_cache

        invalidate_soul_cache()
    except Exception:  # pragma: no cover - best-effort
        logger.debug("Could not invalidate soul cache", exc_info=True)

    logger.info("Activated persona %r", slug)
    return persona


def create_and_activate(
    answers: dict[str, Any], *, activate: bool = True
) -> tuple[str, dict[str, Any]]:
    """Build a persona from answers, save it, and optionally activate it.

    Returns:
        A ``(slug, persona)`` tuple.
    """
    persona = build_persona_from_answers(answers)
    slug = slugify(persona["name"])
    save_persona(persona, slug=slug)
    if activate:
        activate_persona(slug)
    return slug, persona


# ---------------------------------------------------------------------------
# Example personas
# ---------------------------------------------------------------------------


EXAMPLE_PERSONAS: dict[str, dict[str, Any]] = {
    "atlas": {
        "name": "Atlas",
        "full_name": "Atlas — Principal Engineer",
        "personality": (
            "I am Atlas, a principal software engineer. I communicate tersely and "
            "precisely, leading with the answer and the trade-offs. I value "
            "correctness, simplicity and reproducibility. I specialise in systems "
            "design, performance and debugging. I never pad my replies; when I am "
            "unsure I say so and propose how to find out."
        ),
        "version": "1.0.0",
        "tagline": "Right answer, least words.",
        "traits": ["terse", "precise", "rigorous"],
        "values": ["correctness", "simplicity", "reproducibility"],
    },
    "sage": {
        "name": "Sage",
        "full_name": "Sage — Patient Mentor",
        "personality": (
            "I am Sage, a patient teacher. I communicate warmly and clearly, "
            "checking understanding as I go. I value curiosity, empathy and "
            "growth. I specialise in explaining hard ideas with analogies and "
            "small steps. I encourage questions and never make anyone feel slow."
        ),
        "version": "1.0.0",
        "tagline": "Understand it, don't just memorise it.",
        "traits": ["warm", "patient", "clear"],
        "values": ["curiosity", "empathy", "growth"],
    },
}


def install_examples() -> list[str]:
    """Save any bundled example persona not already present; return saved slugs."""
    existing = set(list_personas())
    saved: list[str] = []
    for slug, data in EXAMPLE_PERSONAS.items():
        if slug not in existing:
            save_persona(data, slug=slug)
            saved.append(slug)
    return saved

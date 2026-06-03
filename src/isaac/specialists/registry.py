"""Name-keyed registry of :class:`~isaac.specialists.base.Specialist` classes.

The orchestrator looks specialists up by their short ``name`` (e.g. ``"coder"``)
rather than importing concrete classes, so the roster can grow without changing
call sites.  Concrete specialists live in :mod:`isaac.specialists.roster` and
register themselves via the :func:`register_specialist` decorator.

To avoid a circular import (``roster`` imports :func:`register_specialist` from
this module), the roster is **not** imported at module top.  Instead every
public lookup function first calls :func:`_ensure_builtins_loaded`, which imports
the roster exactly once; importing the roster runs the decorators and populates
:data:`SPECIALIST_CLASSES`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from isaac.specialists.base import Specialist

logger = logging.getLogger(__name__)

#: Registry mapping ``Specialist.name`` -> concrete ``Specialist`` subclass.
SPECIALIST_CLASSES: dict[str, type[Specialist]] = {}

#: Guard so the built-in roster is imported at most once.
_builtins_loaded = False


def register_specialist(cls: type[Specialist]) -> type[Specialist]:
    """Register *cls* under its ``name`` attribute and return it unchanged.

    Intended as a class decorator on concrete :class:`Specialist` subclasses.

    Args:
        cls: The specialist subclass to register.

    Returns:
        The same class, so the decorator is transparent.
    """
    name = getattr(cls, "name", "") or ""
    key = name.strip().lower()
    if not key:
        raise ValueError(f"Specialist {cls!r} must define a non-empty 'name'")
    if key in SPECIALIST_CLASSES and SPECIALIST_CLASSES[key] is not cls:
        logger.warning("Overriding already-registered specialist %r", key)
    SPECIALIST_CLASSES[key] = cls
    return cls


def _ensure_builtins_loaded() -> None:
    """Import the built-in roster once so the registry is populated.

    Importing :mod:`isaac.specialists.roster` triggers the
    :func:`register_specialist` decorators on the nine built-in specialists.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    # Set the flag *before* importing so a re-entrant import (roster importing
    # this module) does not recurse back into a second load.
    _builtins_loaded = True
    try:
        import isaac.specialists.roster  # noqa: F401  (side-effect import)
    except Exception:  # pragma: no cover - defensive; roster import shouldn't fail
        _builtins_loaded = False
        logger.exception("Failed to import the built-in specialist roster")
        raise


def get_specialist_class(name: str) -> type[Specialist] | None:
    """Return the specialist class registered under *name*, or ``None``.

    The lookup is case-insensitive.
    """
    _ensure_builtins_loaded()
    return SPECIALIST_CLASSES.get(name.strip().lower())


def get_specialist(name: str, **kwargs) -> Specialist:
    """Instantiate the specialist registered under *name*.

    Args:
        name: The specialist's short name (case-insensitive).
        **kwargs: Forwarded to the specialist's constructor.

    Returns:
        A new specialist instance.

    Raises:
        KeyError: If no specialist is registered under *name*.
    """
    _ensure_builtins_loaded()
    cls = SPECIALIST_CLASSES.get(name.strip().lower())
    if cls is None:
        raise KeyError(f"No specialist registered under name {name!r}")
    return cls(**kwargs)


def specialist_names() -> list[str]:
    """Return the names of all registered specialists."""
    _ensure_builtins_loaded()
    return list(SPECIALIST_CLASSES.keys())


def list_specialists() -> list[dict]:
    """Return a routing card (:meth:`Specialist.card`) for every specialist."""
    _ensure_builtins_loaded()
    return [cls().card() for cls in SPECIALIST_CLASSES.values()]


def all_specialists(**kwargs) -> dict[str, Specialist]:
    """Instantiate every registered specialist, keyed by name.

    Args:
        **kwargs: Forwarded to every specialist's constructor.

    Returns:
        A mapping of ``name`` -> freshly constructed specialist instance.
    """
    _ensure_builtins_loaded()
    return {name: cls(**kwargs) for name, cls in SPECIALIST_CLASSES.items()}

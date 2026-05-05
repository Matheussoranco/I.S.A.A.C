"""Expert registry — singleton holding all available experts.

Experts self-register at import-time via :func:`register_default_experts`,
which is called lazily the first time :func:`get_registry` is invoked.
External code can register custom experts via ``get_registry().register(...)``.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

from isaac.experts.base import Expert

logger = logging.getLogger(__name__)


class ExpertRegistry:
    """Thread-safe registry of :class:`Expert` instances."""

    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}
        self._lock = threading.RLock()

    def register(self, expert: Expert) -> None:
        with self._lock:
            if expert.name in self._experts:
                logger.debug("Replacing expert %r in registry.", expert.name)
            self._experts[expert.name] = expert

    def unregister(self, name: str) -> bool:
        with self._lock:
            return self._experts.pop(name, None) is not None

    def get(self, name: str) -> Expert | None:
        return self._experts.get(name)

    def all(self) -> list[Expert]:
        with self._lock:
            return list(self._experts.values())

    def by_domain(self, domain: str) -> list[Expert]:
        return [e for e in self.all() if domain in e.domains]

    def names(self) -> list[str]:
        return [e.name for e in self.all()]


# ---------------------------------------------------------------------------
# Default experts bootstrap
# ---------------------------------------------------------------------------

_instance: ExpertRegistry | None = None
_init_lock = threading.Lock()


def register_default_experts(registry: ExpertRegistry) -> None:
    """Register the bundled experts. Each import is wrapped in try/except so
    that an optional dependency (sympy, networkx, z3) doesn't kill the whole
    registry."""
    # Language expert is mandatory — wraps the local LLM.
    try:
        from isaac.experts.language import LanguageExpert
        registry.register(LanguageExpert())
    except Exception as exc:
        logger.error("Failed to register LanguageExpert: %s", exc)

    # Optional experts — log & continue on import failure.
    optional = [
        ("isaac.experts.math", "MathExpert"),
        ("isaac.experts.code", "CodeExpert"),
        ("isaac.experts.kg", "KGExpert"),
        ("isaac.experts.arc", "ArcExpert"),
        ("isaac.experts.logic", "LogicExpert"),
        ("isaac.experts.vision", "VisionExpert"),
    ]
    for module, cls_name in optional:
        try:
            mod = __import__(module, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            registry.register(cls())
        except Exception as exc:
            logger.debug("Optional expert %s.%s not available: %s", module, cls_name, exc)


def get_registry() -> ExpertRegistry:
    """Return the global registry, initialising it on first call."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = ExpertRegistry()
                register_default_experts(_instance)
    return _instance


def reset_registry() -> None:
    """Reset the singleton — for testing only."""
    global _instance
    _instance = None


def iter_experts() -> Iterable[Expert]:
    yield from get_registry().all()

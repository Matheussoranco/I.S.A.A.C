"""Identity — I.S.A.A.C.'s sense of self, personality, and purpose."""

from __future__ import annotations

from isaac.identity.persona_builder import (
    activate_persona,
    active_persona,
    build_persona_from_answers,
    create_and_activate,
    install_examples,
    list_personas,
    load_persona,
    save_persona,
)
from isaac.identity.soul import get_soul, invalidate_soul_cache, soul_system_prompt

__all__ = [
    "activate_persona",
    "active_persona",
    "build_persona_from_answers",
    "create_and_activate",
    "get_soul",
    "install_examples",
    "invalidate_soul_cache",
    "list_personas",
    "load_persona",
    "save_persona",
    "soul_system_prompt",
]

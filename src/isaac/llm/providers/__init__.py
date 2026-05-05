"""Pluggable LLM provider backends.

Each provider lives in its own submodule and exports a ``build()`` function
that returns a LangChain ``BaseChatModel``.  Providers register themselves
into the :data:`PROVIDERS` registry so :func:`get_provider` can resolve
them by name.

The default routing prefers local backends in this order:

    1. ``ollama``           — local first-class
    2. ``llamacpp``         — local llama.cpp HTTP server
    3. ``openai_compat``    — generic OpenAI-compatible (LM Studio, vLLM, ...)
    4. ``openai``           — cloud
    5. ``anthropic``        — cloud
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from . import anthropic as _anthropic
from . import llamacpp as _llamacpp
from . import ollama as _ollama
from . import openai as _openai
from . import openai_compat as _openai_compat

ProviderBuilder = Callable[..., Any]

PROVIDERS: Dict[str, ProviderBuilder] = {
    "ollama": _ollama.build,
    "llamacpp": _llamacpp.build,
    "openai_compat": _openai_compat.build,
    "openai": _openai.build,
    "anthropic": _anthropic.build,
}

LOCAL_PROVIDERS: tuple[str, ...] = ("ollama", "llamacpp", "openai_compat")
"""Providers considered fully local (no outbound traffic to vendor APIs)."""


def get_provider(name: str) -> ProviderBuilder:
    """Return the builder for the named provider, or raise KeyError."""
    key = name.lower().strip()
    if key not in PROVIDERS:
        raise KeyError(
            f"Unknown LLM provider: {name!r}. "
            f"Known: {', '.join(sorted(PROVIDERS))}."
        )
    return PROVIDERS[key]


def is_local(name: str) -> bool:
    """Return True if the named provider runs locally."""
    return name.lower().strip() in LOCAL_PROVIDERS

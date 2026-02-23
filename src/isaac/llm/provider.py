"""LLM provider factory.

Returns a ``BaseChatModel`` configured from :pymod:`isaac.config.settings`.
Supports OpenAI and Anthropic backends — switchable via the
``ISAAC_LLM_PROVIDER`` environment variable.

Tiered models
-------------
* ``get_llm()``           — default (uses ``model_name``).
* ``get_llm("fast")``     — cheap/fast model for Perception & Planner.
* ``get_llm("strong")``   — powerful model for Synthesis, Reflection, Skill Abstraction.

When a tier override is not configured, it falls back to the default model.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

ModelTier = Literal["default", "fast", "strong"]


@lru_cache(maxsize=4)
def get_llm(tier: ModelTier = "default") -> BaseChatModel:
    """Build and cache a chat model for the given *tier*.

    Parameters
    ----------
    tier:
        ``"default"``  — uses ``ISAAC_MODEL_NAME``.
        ``"fast"``     — uses ``ISAAC_FAST_MODEL`` (falls back to default).
        ``"strong"``   — uses ``ISAAC_STRONG_MODEL`` (falls back to default).

    Raises
    ------
    ValueError
        If the configured provider is not ``"openai"`` or ``"anthropic"``.
    """
    from isaac.config.settings import settings

    cfg = settings.llm
    provider = cfg.llm_provider.lower()

    # Resolve model name and temperature for the requested tier
    if tier == "fast" and cfg.fast_model:
        model_name = cfg.fast_model
        temperature = cfg.fast_temperature if cfg.fast_temperature >= 0 else cfg.temperature
    elif tier == "strong" and cfg.strong_model:
        model_name = cfg.strong_model
        temperature = cfg.strong_temperature if cfg.strong_temperature >= 0 else cfg.temperature
    else:
        model_name = cfg.model_name
        temperature = cfg.temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": model_name,
            "temperature": temperature,
            "api_key": settings.openai_api_key or "ollama",  # type: ignore[arg-type]
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,  # type: ignore[arg-type]
            temperature=temperature,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    msg = f"Unsupported LLM provider: {provider!r}. Use 'openai' or 'anthropic'."
    raise ValueError(msg)

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


@lru_cache(maxsize=1)
def get_perception_llm() -> BaseChatModel:
    """Return a token-capped LLM specifically for the Perception node.

    Capped at 200 tokens because we only need a short JSON response:
    ``{"observations": [...], "hypothesis": "...", "task_mode": "..."}``.  
    Limiting output tokens is the single fastest win for Ollama.
    """
    from isaac.config.settings import settings

    cfg = settings.llm
    provider = cfg.llm_provider.lower()
    model_name = cfg.fast_model or cfg.model_name
    temperature = cfg.fast_temperature if cfg.fast_temperature >= 0 else cfg.temperature

    if provider in ("openai", "ollama"):
        kwargs: dict = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": 200,
            "api_key": settings.openai_api_key or "ollama",  # type: ignore[arg-type]
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        try:
            # Try native Ollama client first (faster, no HTTP overhead)
            from langchain_ollama import ChatOllama
            from isaac.config.settings import settings as s
            return ChatOllama(
                model=model_name,
                base_url=s.ollama_base_url or "http://localhost:11434",
                temperature=temperature,
                num_predict=200,  # Ollama-specific token cap
            )
        except ImportError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=200,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    # Fallback — same as fast but with token cap via bind
    base = get_llm("fast")
    return base.bind(max_tokens=200)  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_direct_response_llm() -> BaseChatModel:
    """Return a token-capped LLM for the DirectResponse fast-path node.

    400 tokens is enough for conversational replies, greetings, and short
    answers.  Keeping it capped prevents runaway generation on a local model.
    """
    from isaac.config.settings import settings

    cfg = settings.llm
    provider = cfg.llm_provider.lower()
    model_name = cfg.fast_model or cfg.model_name
    temperature = cfg.fast_temperature if cfg.fast_temperature >= 0 else cfg.temperature

    if provider in ("openai", "ollama"):
        try:
            from langchain_ollama import ChatOllama
            from isaac.config.settings import settings as s
            return ChatOllama(
                model=model_name,
                base_url=s.ollama_base_url or "http://localhost:11434",
                temperature=temperature,
                num_predict=400,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=400,
                api_key=settings.openai_api_key or "ollama",  # type: ignore[arg-type]
                **(dict(base_url=cfg.base_url) if cfg.base_url else {}),
            )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=400,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    base = get_llm("fast")
    return base.bind(max_tokens=400)  # type: ignore[return-value]


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

"""LLM provider factory.

Returns a ``BaseChatModel`` configured from :pymod:`isaac.config.settings`.
The default backend is **local**: Ollama running ``qwen3.6``.  Cloud
backends (OpenAI, Anthropic) and the other local ones (llama.cpp, any
OpenAI-compatible server) are selected via the ``ISAAC_LLM_PROVIDER``
environment variable.

Tiered models
-------------
* ``get_llm()``           — default (uses ``model_name``).
* ``get_llm("fast")``     — cheap/fast model for Perception & Planner.
* ``get_llm("strong")``   — powerful model for Synthesis, Reflection, Skill Abstraction.

When a tier override is not configured, it falls back to the default model.

Local-first failure mode
------------------------
Before the first Ollama-backed model is built, the factory preflights the
daemon and the model tag (``ISAAC_OLLAMA_PREFLIGHT``, on by default).  A
missing daemon or un-pulled model raises
:class:`~isaac.llm.providers.ollama.OllamaUnavailableError` naming the exact
``ollama serve`` / ``ollama pull <model>`` command — it never silently
redirects the request to a billable cloud API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

ModelTier = Literal["default", "fast", "strong"]


def _ollama_preflight(model: str) -> None:
    """Preflight the Ollama daemon + *model*, unless disabled in settings.

    Cached per (model, base_url) so the probe costs at most one HTTP call per
    distinct model per process.
    """
    from isaac.config.settings import settings

    if not getattr(settings, "ollama_preflight", True):
        return
    _preflight_once(model, settings.ollama_base_url or "")


@lru_cache(maxsize=8)
def _preflight_once(model: str, base_url: str) -> None:
    from isaac.llm.providers.ollama import DEFAULT_BASE_URL, preflight

    preflight(model, base_url or DEFAULT_BASE_URL)


def _build_ollama(model: str, temperature: float, max_tokens: int | None = None) -> BaseChatModel:
    """Build an Ollama chat model after preflighting daemon + model tag."""
    from isaac.config.settings import settings
    from isaac.llm.providers.ollama import DEFAULT_BASE_URL
    from isaac.llm.providers.ollama import build as build_ollama

    _ollama_preflight(model)
    return build_ollama(
        model=model,
        base_url=settings.ollama_base_url or DEFAULT_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _capped_llm(max_tokens: int) -> BaseChatModel:
    """Return a token-capped model on the *fast* tier for the given budget.

    Shared by :func:`get_perception_llm` and :func:`get_direct_response_llm`,
    each of which caches its own instance.
    """
    from isaac.config.settings import settings

    cfg = settings.llm
    provider = cfg.llm_provider.lower()
    model_name = cfg.fast_model or cfg.model_name
    temperature = cfg.fast_temperature if cfg.fast_temperature >= 0 else cfg.temperature

    if provider == "ollama":
        return _build_ollama(model_name, temperature, max_tokens=max_tokens)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": settings.openai_api_key or None,
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=settings.anthropic_api_key or None,  # type: ignore[arg-type]
        )

    # llamacpp / openai_compat and anything else: reuse the tier builder and
    # bind the cap at call time.
    base = get_llm("fast")
    return base.bind(max_tokens=max_tokens)  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_perception_llm() -> BaseChatModel:
    """Return a token-capped LLM specifically for the Perception node.

    Capped at 200 tokens because we only need a short JSON response:
    ``{"observations": [...], "hypothesis": "...", "task_mode": "..."}``.
    Limiting output tokens is the single fastest win for Ollama.
    """
    return _capped_llm(200)


@lru_cache(maxsize=1)
def get_direct_response_llm() -> BaseChatModel:
    """Return a token-capped LLM for the DirectResponse fast-path node.

    400 tokens is enough for conversational replies, greetings, and short
    answers.  Keeping it capped prevents runaway generation on a local model.
    """
    return _capped_llm(400)


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
        If the configured provider is not one of ``ollama``, ``llamacpp``,
        ``openai_compat``, ``openai`` or ``anthropic``.
    isaac.llm.providers.ollama.OllamaUnavailableError
        If the (default) Ollama backend is selected but the daemon is down or
        the configured model has not been pulled.
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

    # ── Local-first providers ───────────────────────────────────────────
    # Delegate to the dedicated builders in isaac.llm.providers so the default
    # configuration (``ISAAC_LLM_PROVIDER=ollama``) actually works for the
    # AgentLoop, the specialist agents, and every tier-resolving caller.
    if provider == "ollama":
        return _build_ollama(model_name, temperature)

    if provider == "llamacpp":
        from isaac.llm.providers.llamacpp import build as build_llamacpp

        return build_llamacpp(
            model=settings.llamacpp_model or model_name,
            base_url=settings.llamacpp_base_url or "http://localhost:8080",
            temperature=temperature,
        )

    if provider == "openai_compat":
        from isaac.llm.providers.openai_compat import build as build_compat

        return build_compat(
            model=settings.openai_compat_model or model_name,
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            temperature=temperature,
        )

    # ── Cloud providers (opt-in) ────────────────────────────────────────
    if provider == "openai":
        _require_key(settings.openai_api_key, "openai", "OPENAI_API_KEY")
        from langchain_openai import ChatOpenAI

        kwargs: dict = {
            "model": model_name,
            "temperature": temperature,
            "api_key": settings.openai_api_key or None,  # type: ignore[arg-type]
        }
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        _require_key(settings.anthropic_api_key, "anthropic", "ANTHROPIC_API_KEY")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,  # type: ignore[arg-type]
            temperature=temperature,
            api_key=settings.anthropic_api_key or None,  # type: ignore[arg-type]
        )

    msg = (
        f"Unsupported LLM provider: {provider!r}. Use one of: "
        "ollama, llamacpp, openai_compat, openai, anthropic."
    )
    raise ValueError(msg)


def _require_key(value: str, provider: str, env_var: str) -> None:
    """Validate that an explicitly selected cloud provider has its API key.

    Key validation is *conditional*: a default (Ollama) install needs no keys
    at all, so nothing is checked until a cloud provider is actually chosen.
    A blank ``base_url`` matters here — an OpenAI-compatible local server
    reached through ``ISAAC_LLM_PROVIDER=openai`` is accepted keyless.
    """
    from isaac.config.settings import settings

    if value:
        return
    if provider == "openai" and settings.llm.base_url:
        return  # local OpenAI-compatible endpoint — no key needed
    raise ValueError(
        f"ISAAC_LLM_PROVIDER={provider!r} was selected but {env_var} is not set.\n"
        f"Either export {env_var}=... or switch back to the local default:\n"
        "    ISAAC_LLM_PROVIDER=ollama\n"
        "    ollama pull qwen3.6\n"
    )

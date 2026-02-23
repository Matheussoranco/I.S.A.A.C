"""LLM provider factory.

Returns a ``BaseChatModel`` configured from :pymod:`isaac.config.settings`.
Supports OpenAI and Anthropic backends — switchable via the
``ISAAC_LLM_PROVIDER`` environment variable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Build and cache the chat model singleton.

    Raises
    ------
    ValueError
        If the configured provider is not ``"openai"`` or ``"anthropic"``.
    """
    from isaac.config.settings import settings  # noqa: PLC0415

    cfg = settings.llm
    provider = cfg.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

        return ChatAnthropic(
            model=cfg.model_name,  # type: ignore[arg-type]
            temperature=cfg.temperature,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    msg = f"Unsupported LLM provider: {provider!r}. Use 'openai' or 'anthropic'."
    raise ValueError(msg)

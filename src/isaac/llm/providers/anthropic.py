"""Anthropic Claude cloud provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def build(
    model: str = "claude-haiku-4-5-20251001",
    api_key: str = "",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,  # noqa: ARG001
    **extra: Any,
) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key or None,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(extra)
    return ChatAnthropic(**kwargs)

"""OpenAI cloud provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def build(
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,  # noqa: ARG001
    **extra: Any,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key or None,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(extra)
    return ChatOpenAI(**kwargs)

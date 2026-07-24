"""OpenAI cloud provider — opt-in.

``model`` is intentionally required: I.S.A.A.C. defaults to a local backend
(see :pymod:`isaac.llm.providers.ollama`), so nothing here should ever be
able to pick a billable model on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def build(
    model: str,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,
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

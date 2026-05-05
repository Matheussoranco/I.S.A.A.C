"""Generic OpenAI-compatible HTTP provider.

Targets any server that speaks the OpenAI Chat Completions API:

* LM Studio
* vLLM
* TGI (with the ``--openai`` adapter)
* LiteLLM proxy
* Together.ai / DeepInfra / Groq / Mistral (cloud, OpenAI-compatible)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def build(
    model: str,
    base_url: str,
    api_key: str = "not-needed",
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,  # noqa: ARG001
    **extra: Any,
) -> BaseChatModel:
    """Build a ChatOpenAI client against any OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base,
        "api_key": api_key or "not-needed",
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(extra)
    return ChatOpenAI(**kwargs)

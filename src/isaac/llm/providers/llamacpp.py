"""llama.cpp HTTP server provider.

Targets the OpenAI-compatible endpoint exposed by ``llama.cpp``'s
``server`` binary (``./server -m model.gguf --port 8080``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_BASE_URL = "http://localhost:8080"


def build(
    model: str = "local-model",
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,  # noqa: ARG001
    **extra: Any,
) -> BaseChatModel:
    """Build an OpenAI-compatible client pointed at a llama.cpp server."""
    from langchain_openai import ChatOpenAI

    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    kwargs: dict[str, Any] = {
        "model": model,
        "base_url": base,
        "api_key": "llamacpp",
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(extra)
    return ChatOpenAI(**kwargs)


def health_check(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> bool:
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url.rstrip('/')}/health")
            return r.status_code == 200
    except Exception:
        return False

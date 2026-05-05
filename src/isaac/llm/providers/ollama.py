"""Ollama provider — first-class local LLM backend.

Uses the native ``langchain_ollama.ChatOllama`` client when available
and falls back to an OpenAI-compatible HTTP shim against Ollama's
``/v1`` endpoint otherwise.

Ollama exposes both text-only and vision-language models (e.g.
``llava``, ``qwen2.5-vl``, ``llama3.2-vision``).  The ``vision=True``
flag is informational — model selection is the user's responsibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"


def build(
    model: str = "qwen2.5-coder:7b",
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,  # noqa: ARG001 — informational only
    **extra: Any,
) -> BaseChatModel:
    """Construct a ChatOllama (or OpenAI-shim) client.

    Parameters
    ----------
    model:
        Ollama model tag (``mistral``, ``llava:13b``, ...).
    base_url:
        Ollama server URL.
    temperature:
        Sampling temperature.
    max_tokens:
        Hard cap on generated tokens (``num_predict`` in Ollama).
    vision:
        Hint that the caller expects multimodal input.  No effect on
        construction — but downstream code can inspect this.
    extra:
        Arbitrary kwargs forwarded to the underlying client.

    Returns
    -------
    BaseChatModel
    """
    base_url = base_url.rstrip("/")
    try:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        kwargs.update(extra)
        return ChatOllama(**kwargs)
    except ImportError:
        logger.debug("langchain_ollama not installed — falling back to OpenAI shim.")
        from langchain_openai import ChatOpenAI

        kwargs2: dict[str, Any] = {
            "model": model,
            "base_url": f"{base_url}/v1",
            "api_key": "ollama",  # Ollama ignores it but the client requires one
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs2["max_tokens"] = max_tokens
        kwargs2.update(extra)
        return ChatOpenAI(**kwargs2)


def health_check(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> bool:
    """Return True iff the Ollama server is reachable."""
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:  # pragma: no cover — network paths
        return False


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 5.0) -> list[str]:
    """Return the list of installed Ollama model tags."""
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
            data = r.json()
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:  # pragma: no cover
        return []

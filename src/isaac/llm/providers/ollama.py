"""Ollama provider — the default, first-class local LLM backend.

I.S.A.A.C. is local-first: with no API keys configured at all, every LLM
call is resolved here against a local Ollama daemon running
:data:`DEFAULT_MODEL`.

Uses the native ``langchain_ollama.ChatOllama`` client when available
and falls back to an OpenAI-compatible HTTP shim against Ollama's
``/v1`` endpoint otherwise.

Ollama exposes both text-only and vision-language models (e.g.
``llava``, ``qwen2.5-vl``, ``llama3.2-vision``).  The ``vision=True``
flag is informational — model selection is the user's responsibility.

Failure mode
------------
When the daemon is down or the configured model was never pulled, calling
:func:`preflight` raises :class:`OllamaUnavailableError` with the exact
shell command to run.  I.S.A.A.C. deliberately does *not* fall back to a
billable cloud API on its own — switching to a cloud provider is always an
explicit, user-made configuration choice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"

DEFAULT_MODEL = "qwen3.6"
"""Default local model tag.  Install it with ``ollama pull qwen3.6``.

Single source of truth for the project-wide default model:
:pymod:`isaac.config.settings` imports it rather than repeating the literal.
"""


class OllamaUnavailableError(RuntimeError):
    """The Ollama daemon is unreachable, or the requested model is not installed.

    The message is written to be pasted straight into a terminal — it names
    the exact ``ollama serve`` / ``ollama pull <model>`` command needed.
    """


def _normalise_tag(tag: str) -> str:
    """Return *tag* with an explicit version, matching Ollama's own display.

    ``ollama pull qwen3.6`` installs a model that ``/api/tags`` reports as
    ``qwen3.6:latest``.  Comparing raw strings would report a freshly pulled
    model as missing, so both sides are normalised before comparison.
    """
    tag = tag.strip()
    return tag if ":" in tag else f"{tag}:latest"


def is_model_installed(model: str, installed: Iterable[str]) -> bool:
    """Return True when *model* matches one of the *installed* Ollama tags."""
    want = _normalise_tag(model)
    return any(_normalise_tag(tag) == want for tag in installed)


def daemon_unreachable_message(base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL) -> str:
    """Return the actionable error text for an unreachable Ollama daemon."""
    return (
        f"Ollama is not reachable at {base_url}.\n"
        "\n"
        "I.S.A.A.C. runs local-first and will not silently fall back to a paid "
        "cloud API. Fix one of the following:\n"
        "\n"
        "  1. Start the daemon:   ollama serve\n"
        "  2. Install Ollama:     https://ollama.com/download\n"
        f"  3. Pull the model:     ollama pull {model}\n"
        "  4. Or point I.S.A.A.C. somewhere else, e.g.\n"
        "         ISAAC_OLLAMA_BASE_URL=http://<host>:11434\n"
        "         ISAAC_LLM_PROVIDER=anthropic   (with ANTHROPIC_API_KEY)\n"
        "         ISAAC_LLM_PROVIDER=openai      (with OPENAI_API_KEY)\n"
    )


def model_missing_message(
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    installed: Iterable[str] = (),
) -> str:
    """Return the actionable error text for a model that was never pulled."""
    have = ", ".join(sorted(installed)) or "(none)"
    return (
        f"Ollama is running at {base_url}, but the model {model!r} is not installed.\n"
        "\n"
        "Install it with:\n"
        "\n"
        f"    ollama pull {model}\n"
        "\n"
        f"Models currently installed: {have}\n"
        "\n"
        "Or point I.S.A.A.C. at a model you already have:\n"
        "    ISAAC_MODEL_NAME=<tag>\n"
        "    ISAAC_OLLAMA_LIGHT_MODEL=<tag>\n"
        "    ISAAC_OLLAMA_HEAVY_MODEL=<tag>\n"
    )


def preflight(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 5.0,
) -> None:
    """Verify the daemon is up and *model* is installed, or raise.

    Raises
    ------
    OllamaUnavailableError
        With a message naming the exact command to run.
    """
    base_url = base_url or DEFAULT_BASE_URL
    if not health_check(base_url, timeout=timeout):
        raise OllamaUnavailableError(daemon_unreachable_message(base_url, model))

    installed = list_models(base_url, timeout=timeout)
    # An empty list means the tags endpoint answered but told us nothing
    # useful; do not block on a check we cannot actually perform.
    if installed and not is_model_installed(model, installed):
        raise OllamaUnavailableError(model_missing_message(model, base_url, installed))


def build(
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    vision: bool = False,
    check: bool = False,
    **extra: Any,
) -> BaseChatModel:
    """Construct a ChatOllama (or OpenAI-shim) client.

    Parameters
    ----------
    model:
        Ollama model tag (``qwen3.6``, ``llava:13b``, ...).
    base_url:
        Ollama server URL.
    temperature:
        Sampling temperature.
    max_tokens:
        Hard cap on generated tokens (``num_predict`` in Ollama).
    vision:
        Hint that the caller expects multimodal input.  No effect on
        construction — but downstream code can inspect this.
    check:
        Run :func:`preflight` before constructing the client.  Off by default
        so construction stays offline-pure (and unit-testable); callers that
        are about to issue a real request opt in.
    extra:
        Arbitrary kwargs forwarded to the underlying client.

    Returns
    -------
    BaseChatModel
    """
    base_url = base_url.rstrip("/")
    if check:
        preflight(model, base_url)
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

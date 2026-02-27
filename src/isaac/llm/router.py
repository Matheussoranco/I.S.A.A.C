"""Ollama-first LLM Router — intelligent model selection by task complexity.

Routes tasks to the appropriate LLM based on complexity level:

* ``simple`` / ``moderate`` → local Ollama lightweight model (fast, private)
* ``complex`` / ``reasoning`` → local Ollama heavy model OR API fallback

All routing decisions are logged.  The router health-checks Ollama on first
use and falls back gracefully if unavailable.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels for LLM routing."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    REASONING = "reasoning"


class LLMRouter:
    """Route tasks to the appropriate LLM based on complexity.

    Ollama-first: always prefers local models for privacy.  Falls back
    to the configured API provider only when Ollama is unavailable or
    the task requires a model tier beyond local capacity.

    Parameters
    ----------
    ollama_base_url:
        Base URL for the Ollama API (default: ``http://localhost:11434``).
    light_model:
        Ollama model identifier for simple/moderate tasks.
    heavy_model:
        Ollama model identifier for complex/reasoning tasks.
    fallback_provider:
        Provider name (``"openai"`` or ``"anthropic"``) for API fallback.
    """

    _HEALTH_TTL_SECONDS: float = 60.0
    """Re-check Ollama availability at most once every 60 s."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        light_model: str = "qwen2.5-coder:7b",
        heavy_model: str = "qwen2.5-coder:7b",
        fallback_provider: str = "",
    ) -> None:
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._light_model = light_model
        self._heavy_model = heavy_model
        self._fallback_provider = fallback_provider
        self._ollama_available: bool | None = None  # lazy check
        self._last_health_check: float = 0.0  # epoch seconds

    # -- Health check -------------------------------------------------------

    def _check_ollama_health(self) -> bool:
        """Ping Ollama to verify it is running and responsive.

        Result is cached for ``_HEALTH_TTL_SECONDS`` seconds so that a
        transient restart is detected within one TTL cycle rather than
        never (the old permanent-cache behaviour).
        """
        now = time.monotonic()
        if (
            self._ollama_available is not None
            and (now - self._last_health_check) < self._HEALTH_TTL_SECONDS
        ):
            return self._ollama_available

        import urllib.request
        import urllib.error

        try:
            url = f"{self._ollama_base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._ollama_available = resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            logger.warning(
                "Ollama is not available at %s — will use fallback provider.",
                self._ollama_base_url,
            )
            self._ollama_available = False

        self._last_health_check = time.monotonic()
        if self._ollama_available:
            logger.info("Ollama is available at %s.", self._ollama_base_url)
        return bool(self._ollama_available)

    @property
    def ollama_available(self) -> bool:
        """Whether Ollama was reachable on the last health check."""
        return self._check_ollama_health()

    # -- Model construction -------------------------------------------------

    def _build_ollama_model(self, model_name: str) -> BaseChatModel:
        """Build a LangChain ChatOllama instance."""
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model_name,
                base_url=self._ollama_base_url,
                temperature=0.2,
            )
        except ImportError:
            # Fallback to OpenAI-compatible endpoint
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                base_url=f"{self._ollama_base_url}/v1",
                api_key="ollama",  # type: ignore[arg-type]
                temperature=0.2,
            )

    def _build_fallback_model(self) -> BaseChatModel:
        """Build a model using the configured API fallback provider."""
        from isaac.llm.provider import get_llm

        return get_llm("strong")

    # -- Public API ---------------------------------------------------------

    def route(self, task_complexity: str | TaskComplexity) -> BaseChatModel:
        """Return the appropriate LLM for the given task complexity.

        Parameters
        ----------
        task_complexity:
            One of ``"simple"``, ``"moderate"``, ``"complex"``, ``"reasoning"``.

        Returns
        -------
        BaseChatModel
            A LangChain chat model ready for ``.invoke()``.
        """
        complexity = TaskComplexity(task_complexity)

        if self._check_ollama_health():
            if complexity in (TaskComplexity.SIMPLE, TaskComplexity.MODERATE):
                logger.info("Router: %s → Ollama light (%s)", complexity.value, self._light_model)
                return self._build_ollama_model(self._light_model)
            else:
                logger.info("Router: %s → Ollama heavy (%s)", complexity.value, self._heavy_model)
                return self._build_ollama_model(self._heavy_model)

        # Ollama unavailable — fallback
        if self._fallback_provider:
            logger.info(
                "Router: Ollama unavailable, falling back to %s for %s task.",
                self._fallback_provider,
                complexity.value,
            )
            return self._build_fallback_model()

        # Last resort: try Ollama anyway (may raise)
        logger.warning("Router: no fallback configured — attempting Ollama regardless.")
        model = self._light_model if complexity in (
            TaskComplexity.SIMPLE, TaskComplexity.MODERATE,
        ) else self._heavy_model
        return self._build_ollama_model(model)

    def route_for_guard(self) -> BaseChatModel:
        """Return the model for the Prompt Injection Guard.

        Always uses the local Ollama light model for privacy — never
        sends guard prompts to external APIs.
        """
        if self._check_ollama_health():
            return self._build_ollama_model(self._light_model)

        # If Ollama is down, still try the local model
        logger.warning("Router: Ollama unavailable for guard — attempting anyway.")
        return self._build_ollama_model(self._light_model)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Return the module-level LLM router singleton."""
    global _router  # noqa: PLW0603
    if _router is None:
        from isaac.config.settings import settings

        _router = LLMRouter(
            ollama_base_url=settings.ollama_base_url,
            light_model=settings.ollama_light_model,
            heavy_model=settings.ollama_heavy_model,
            fallback_provider=settings.llm_fallback_provider,
        )
    return _router


def reset_router() -> None:
    """Reset the singleton (used in tests)."""
    global _router  # noqa: PLW0603
    _router = None

"""Modality-aware LLM router.

Extends the complexity-tier routing of :pymod:`isaac.llm.router` with a
*modality* dimension.  Each (modality, complexity) pair maps to a
specific model + provider, with graceful fallback chains.

Modalities
----------
* ``text``    — standard chat
* ``vision``  — text + image input (VLM)
* ``audio``   — text + audio input (rare; usually STT happens first)

Complexity tiers
----------------
* ``fast``     — cheap, low-latency model (perception, classification)
* ``default``  — main reasoning model (synthesis, planning)
* ``strong``   — large model for hard tasks (reflection, debugging)

Health checks for local providers are cached (``_HEALTH_TTL_SECONDS``).
A ``RouteError`` is raised when no backend is reachable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from isaac.llm.providers import (
    LOCAL_PROVIDERS,
    PROVIDERS,
    get_provider,
    is_local,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


class Complexity(str, Enum):
    FAST = "fast"
    DEFAULT = "default"
    STRONG = "strong"


class RouteError(RuntimeError):
    """Raised when no provider can satisfy a routing request."""


@dataclass(frozen=True)
class Route:
    """Concrete (provider, model) pair for a single routing decision."""

    provider: str
    model: str
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int | None = None
    api_key: str = ""

    def build(self, **overrides: Any) -> BaseChatModel:
        builder = get_provider(self.provider)
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.base_url:
            params["base_url"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        params.update(overrides)
        return builder(**params)


class MultimodalRouter:
    """Route requests across (modality × complexity) and local/cloud providers.

    The router consults a list of preferred backends (``preferred_chain``)
    and picks the first one that is healthy.  Each backend can specify a
    distinct model per (modality, complexity).
    """

    _HEALTH_TTL_SECONDS = 60.0

    def __init__(
        self,
        text_routes: dict[Complexity, Route],
        vision_routes: dict[Complexity, Route] | None = None,
        audio_routes: dict[Complexity, Route] | None = None,
        local_first: bool = True,
        fallback_chain: list[str] | None = None,
    ) -> None:
        self._text = text_routes
        self._vision = vision_routes or {}
        self._audio = audio_routes or {}
        self._local_first = local_first
        self._fallback_chain = fallback_chain or []
        self._health: dict[str, tuple[bool, float]] = {}

    # -- Health check -------------------------------------------------------

    def _is_healthy(self, route: Route) -> bool:
        """Cheap (cached) health check for a route's provider."""
        if not is_local(route.provider):
            return True  # cloud — assumed reachable, errors surface on call

        key = f"{route.provider}|{route.base_url}"
        cached = self._health.get(key)
        now = time.monotonic()
        if cached is not None and (now - cached[1]) < self._HEALTH_TTL_SECONDS:
            return cached[0]

        ok = self._probe(route)
        self._health[key] = (ok, now)
        if ok:
            logger.info("Router: %s healthy at %s", route.provider, route.base_url or "default")
        else:
            logger.warning(
                "Router: %s NOT healthy at %s — will skip.",
                route.provider, route.base_url or "default",
            )
        return ok

    @staticmethod
    def _probe(route: Route) -> bool:
        if route.provider == "ollama":
            from isaac.llm.providers.ollama import health_check
            return health_check(route.base_url or "http://localhost:11434")
        if route.provider == "llamacpp":
            from isaac.llm.providers.llamacpp import health_check
            return health_check(route.base_url or "http://localhost:8080")
        if route.provider == "openai_compat":
            # No standard health endpoint — try a HEAD on the base URL
            import httpx
            try:
                with httpx.Client(timeout=3.0) as client:
                    client.get(route.base_url.rstrip("/v").rstrip("/") + "/v1/models")
                return True
            except Exception:
                return False
        return True

    # -- Public API ---------------------------------------------------------

    def route(
        self,
        modality: Modality | str = Modality.TEXT,
        complexity: Complexity | str = Complexity.DEFAULT,
    ) -> BaseChatModel:
        """Return a built model for the requested (modality, complexity)."""
        modality = Modality(modality)
        complexity = Complexity(complexity)

        table = {
            Modality.TEXT: self._text,
            Modality.VISION: self._vision,
            Modality.AUDIO: self._audio,
        }[modality]

        primary = table.get(complexity) or table.get(Complexity.DEFAULT)
        if primary is None:
            raise RouteError(f"No route configured for modality={modality.value}.")

        if self._is_healthy(primary):
            return primary.build()

        # Try fallback chain (each entry is a provider name)
        for provider_name in self._fallback_chain:
            for c in (complexity, Complexity.DEFAULT, Complexity.FAST):
                candidate = table.get(c)
                if candidate is None or candidate.provider != provider_name:
                    continue
                if self._is_healthy(candidate):
                    logger.info(
                        "Router: primary %s unavailable, falling back to %s.",
                        primary.provider, candidate.provider,
                    )
                    return candidate.build()

        raise RouteError(
            f"No healthy provider for modality={modality.value} complexity={complexity.value}. "
            f"Tried primary={primary.provider} and fallback_chain={self._fallback_chain}."
        )

    def known_providers(self) -> list[str]:
        """Return all registered provider names."""
        return sorted(PROVIDERS)

    def local_providers(self) -> list[str]:
        return list(LOCAL_PROVIDERS)


# ---------------------------------------------------------------------------
# Default router from settings
# ---------------------------------------------------------------------------

_router: MultimodalRouter | None = None


def get_multimodal_router() -> MultimodalRouter:
    """Return a process-wide router built from :data:`isaac.config.settings`."""
    global _router  # noqa: PLW0603
    if _router is not None:
        return _router

    from isaac.config.settings import settings

    cfg = settings.llm

    # ── Text routes (one per complexity tier) ──────────────────────────────
    fast_model = cfg.fast_model or settings.ollama_light_model
    default_model = cfg.model_name or settings.ollama_light_model
    strong_model = cfg.strong_model or settings.ollama_heavy_model

    text_routes: dict[Complexity, Route] = {}
    if settings.local_first:
        text_routes[Complexity.FAST] = Route(
            provider="ollama",
            model=fast_model,
            base_url=settings.ollama_base_url,
            temperature=cfg.fast_temperature if cfg.fast_temperature >= 0 else cfg.temperature,
            max_tokens=200,
        )
        text_routes[Complexity.DEFAULT] = Route(
            provider="ollama",
            model=default_model,
            base_url=settings.ollama_base_url,
            temperature=cfg.temperature,
        )
        text_routes[Complexity.STRONG] = Route(
            provider="ollama",
            model=strong_model,
            base_url=settings.ollama_base_url,
            temperature=cfg.strong_temperature if cfg.strong_temperature >= 0 else cfg.temperature,
        )

    # ── Vision routes ──────────────────────────────────────────────────────
    vision_routes: dict[Complexity, Route] = {}
    if settings.vision_enabled:
        vision_routes[Complexity.DEFAULT] = Route(
            provider="ollama",
            model=settings.vision_model,
            base_url=settings.ollama_base_url,
            temperature=cfg.temperature,
        )
        vision_routes[Complexity.STRONG] = Route(
            provider="ollama",
            model=settings.vision_strong_model or settings.vision_model,
            base_url=settings.ollama_base_url,
            temperature=cfg.temperature,
        )

    # ── Fallback chain ─────────────────────────────────────────────────────
    fallback: list[str] = []
    if settings.llm_fallback_provider:
        fallback.append(settings.llm_fallback_provider.lower())
    # Also append the legacy primary provider so cloud users keep working
    if cfg.llm_provider not in fallback and cfg.llm_provider in PROVIDERS:
        fallback.append(cfg.llm_provider)

    _router = MultimodalRouter(
        text_routes=text_routes,
        vision_routes=vision_routes,
        local_first=settings.local_first,
        fallback_chain=fallback,
    )
    return _router


def reset_multimodal_router() -> None:
    """Reset the singleton (used in tests)."""
    global _router  # noqa: PLW0603
    _router = None

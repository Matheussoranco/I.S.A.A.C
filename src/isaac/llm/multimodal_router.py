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
from enum import StrEnum
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


class Modality(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


class Complexity(StrEnum):
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


RouteSpec = Route | list[Route]


class MultimodalRouter:
    """Route requests across (modality × complexity) and local/cloud providers.

    Each (modality, complexity) slot holds an ordered candidate list; the
    first healthy backend is built.  Local providers are probed, cloud
    providers are assumed reachable and surface errors on call.
    """

    _HEALTH_TTL_SECONDS = 60.0

    def __init__(
        self,
        text_routes: dict[Complexity, RouteSpec],
        vision_routes: dict[Complexity, RouteSpec] | None = None,
        audio_routes: dict[Complexity, RouteSpec] | None = None,
    ) -> None:
        self._text = self._normalise(text_routes)
        self._vision = self._normalise(vision_routes or {})
        self._audio = self._normalise(audio_routes or {})
        self._health: dict[str, tuple[bool, float]] = {}

    @staticmethod
    def _normalise(routes: dict[Complexity, RouteSpec]) -> dict[Complexity, list[Route]]:
        return {
            complexity: [route] if isinstance(route, Route) else list(route)
            for complexity, route in routes.items()
        }

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
                route.provider,
                route.base_url or "default",
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

        candidates = table.get(complexity) or table.get(Complexity.DEFAULT) or []
        if not candidates:
            raise RouteError(f"No route configured for modality={modality.value}.")

        primary = candidates[0]
        for candidate in candidates:
            if self._is_healthy(candidate):
                if candidate is not primary:
                    logger.info(
                        "Router: primary %s unavailable, falling back to %s.",
                        primary.provider,
                        candidate.provider,
                    )
                return candidate.build()

        detail = (
            f"No healthy provider for modality={modality.value} complexity={complexity.value}. "
            f"Tried providers={[route.provider for route in candidates]}."
        )
        if primary.provider == "ollama":
            # The default backend: say exactly which command fixes it rather
            # than leaving the user to guess (and never reroute to a paid API).
            from isaac.llm.providers.ollama import (
                DEFAULT_BASE_URL,
                daemon_unreachable_message,
            )

            detail = (
                f"{detail}\n\n"
                f"{daemon_unreachable_message(primary.base_url or DEFAULT_BASE_URL, primary.model)}"
            )
        raise RouteError(detail)

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
    global _router
    if _router is not None:
        return _router

    from isaac.config.settings import settings

    cfg = settings.llm
    primary = cfg.llm_provider.lower().strip()

    providers = [primary]
    if settings.local_first:
        providers.extend(p for p in LOCAL_PROVIDERS if p not in providers)
    fallback = settings.llm_fallback_provider.lower().strip()
    if fallback and fallback not in providers:
        providers.append(fallback)

    text_routes: dict[Complexity, RouteSpec] = {}
    for tier in Complexity:
        routes = [
            route
            for provider in providers
            if (route := _build_route(settings, provider, tier)) is not None
        ]
        if routes:
            text_routes[tier] = routes

    vision_routes: dict[Complexity, RouteSpec] = {}
    if settings.vision_enabled:
        cloud = _build_cloud_vision_route(settings)
        for tier in (Complexity.DEFAULT, Complexity.STRONG):
            local = _build_route(settings, "ollama", tier, vision=True)
            ordered = [local, cloud] if settings.local_first else [cloud, local]
            routes = [route for route in ordered if route is not None]
            if routes:
                vision_routes[tier] = routes

    _router = MultimodalRouter(text_routes=text_routes, vision_routes=vision_routes)
    return _router


def _tier_model(settings: Any, provider: str, tier: Complexity) -> str:
    cfg = settings.llm
    if tier == Complexity.FAST:
        return cfg.fast_model or (
            settings.ollama_light_model if provider == "ollama" else cfg.model_name
        )
    if tier == Complexity.STRONG:
        return cfg.strong_model or (
            settings.ollama_heavy_model if provider == "ollama" else cfg.model_name
        )
    return cfg.model_name


def _tier_temperature(cfg: Any, tier: Complexity) -> float:
    if tier == Complexity.FAST and cfg.fast_temperature >= 0:
        return float(cfg.fast_temperature)
    if tier == Complexity.STRONG and cfg.strong_temperature >= 0:
        return float(cfg.strong_temperature)
    return float(cfg.temperature)


def _build_route(
    settings: Any,
    provider: str,
    tier: Complexity,
    *,
    vision: bool = False,
) -> Route | None:
    from isaac.llm.provider import provider_api_key

    cfg = settings.llm
    provider = provider.lower().strip()
    if provider == "ollama":
        if vision:
            model = (
                settings.vision_strong_model
                if tier == Complexity.STRONG and settings.vision_strong_model
                else settings.vision_model
            )
        else:
            model = _tier_model(settings, provider, tier)
        return Route(
            provider=provider,
            model=model,
            base_url=settings.ollama_base_url,
            temperature=_tier_temperature(cfg, tier),
        )
    model = _tier_model(settings, provider, tier)
    if provider == "llamacpp":
        return Route(
            provider=provider,
            model=settings.llamacpp_model or model,
            base_url=settings.llamacpp_base_url,
            temperature=_tier_temperature(cfg, tier),
        )
    if provider == "openai_compat":
        if not settings.openai_compat_base_url.strip():
            return None
        return Route(
            provider=provider,
            model=settings.openai_compat_model or model,
            base_url=settings.openai_compat_base_url,
            temperature=_tier_temperature(cfg, tier),
            api_key=settings.openai_compat_api_key,
        )
    if provider in {"openai", "anthropic"}:
        try:
            api_key = provider_api_key(provider, settings)
        except ValueError:
            return None
        return Route(
            provider=provider,
            model=model,
            base_url=cfg.base_url if provider == "openai" else "",
            temperature=_tier_temperature(cfg, tier),
            api_key=api_key,
        )
    return None


def _build_cloud_vision_route(settings: Any) -> Route | None:
    from isaac.llm.provider import provider_api_key

    configured = (settings.vision_cloud_provider or "").strip()
    if not configured:
        return None
    lowered = configured.lower()
    if "gemini" in lowered:
        return None
    provider = "anthropic" if ("claude" in lowered or lowered == "anthropic") else "openai"
    model = configured
    if lowered in {"anthropic", "claude"}:
        model = "claude-3-5-sonnet-20240620"
    elif lowered == "openai":
        model = "gpt-4o"
    try:
        api_key = provider_api_key(provider, settings)
    except ValueError:
        return None
    return Route(
        provider=provider,
        model=model,
        base_url=settings.llm.base_url if provider == "openai" else "",
        temperature=settings.llm.temperature,
        api_key=api_key,
    )


def reset_multimodal_router() -> None:
    """Reset the singleton (used in tests)."""
    global _router
    _router = None

"""Tests for the LLM Router — model selection logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from isaac.llm.router import LLMRouter, TaskComplexity, get_router


class TestLLMRouter:
    """Unit tests for LLMRouter."""

    def test_init_defaults(self) -> None:
        router = LLMRouter()
        assert router._ollama_base_url == "http://localhost:11434"
        assert router._light_model == "qwen2.5-coder:7b"
        assert router._heavy_model == "qwen2.5-coder:7b"
        assert router._fallback_provider == ""
        assert router._ollama_available is None  # lazy

    def test_route_simple_uses_light_model(self) -> None:
        router = LLMRouter()
        router._ollama_available = True  # skip health check
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            model = router.route(TaskComplexity.SIMPLE)
            mock_build.assert_called_once_with("qwen2.5-coder:7b")

    def test_route_moderate_uses_light_model(self) -> None:
        router = LLMRouter()
        router._ollama_available = True
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            router.route(TaskComplexity.MODERATE)
            mock_build.assert_called_once_with("qwen2.5-coder:7b")

    def test_route_complex_uses_heavy_model(self) -> None:
        router = LLMRouter(heavy_model="llama3:70b")
        router._ollama_available = True
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            router.route(TaskComplexity.COMPLEX)
            mock_build.assert_called_once_with("llama3:70b")

    def test_route_reasoning_uses_heavy_model(self) -> None:
        router = LLMRouter(heavy_model="llama3:70b")
        router._ollama_available = True
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            router.route(TaskComplexity.REASONING)
            mock_build.assert_called_once_with("llama3:70b")

    def test_route_fallback_when_ollama_unavailable(self) -> None:
        router = LLMRouter(fallback_provider="openai")
        router._ollama_available = False
        with patch.object(router, "_build_fallback_model") as mock_fb:
            mock_fb.return_value = MagicMock()
            router.route(TaskComplexity.SIMPLE)
            mock_fb.assert_called_once()

    def test_route_without_fallback_attempts_ollama(self) -> None:
        router = LLMRouter(fallback_provider="")
        router._ollama_available = False
        # Without a fallback, it still attempts Ollama (may raise from build)
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            router.route(TaskComplexity.SIMPLE)
            mock_build.assert_called_once()

    def test_health_check_caches(self) -> None:
        router = LLMRouter()
        router._ollama_available = True
        # Second call should use cached value without HTTP
        assert router.ollama_available is True

    def test_route_for_guard_always_local(self) -> None:
        router = LLMRouter()
        router._ollama_available = True
        with patch.object(router, "_build_ollama_model") as mock_build:
            mock_build.return_value = MagicMock()
            router.route_for_guard()
            mock_build.assert_called_once()

    def test_task_complexity_enum_values(self) -> None:
        assert TaskComplexity.SIMPLE.value == "simple"
        assert TaskComplexity.MODERATE.value == "moderate"
        assert TaskComplexity.COMPLEX.value == "complex"
        assert TaskComplexity.REASONING.value == "reasoning"

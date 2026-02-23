"""Tests for the tiered LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGetLLMTiers:
    """Verify that get_llm respects tier-specific model overrides."""

    def setup_method(self) -> None:
        """Clear the lru_cache between tests."""
        from isaac.llm.provider import get_llm

        get_llm.cache_clear()

    def _make_mock_settings(self, **overrides):
        """Build a mock settings object with LLM config."""
        mock_llm_cfg = MagicMock()
        mock_llm_cfg.llm_provider = "openai"
        mock_llm_cfg.model_name = "gpt-4o"
        mock_llm_cfg.temperature = 0.2
        mock_llm_cfg.base_url = ""
        mock_llm_cfg.fast_model = ""
        mock_llm_cfg.fast_temperature = -1.0
        mock_llm_cfg.strong_model = ""
        mock_llm_cfg.strong_temperature = -1.0

        for k, v in overrides.items():
            setattr(mock_llm_cfg, k, v)

        mock_settings = MagicMock()
        mock_settings.llm = mock_llm_cfg
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_api_key = "test-key"
        return mock_settings

    def test_default_tier(self) -> None:
        from isaac.llm.provider import get_llm

        mock_chat_cls = MagicMock(return_value="mock_llm")
        mock_settings = self._make_mock_settings()

        with patch("isaac.config.settings.settings", mock_settings), \
             patch("langchain_openai.ChatOpenAI", mock_chat_cls):
            get_llm.cache_clear()
            result = get_llm("default")

        mock_chat_cls.assert_called_once()
        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.2

    def test_fast_tier_uses_fast_model(self) -> None:
        from isaac.llm.provider import get_llm

        mock_chat_cls = MagicMock(return_value="mock_llm")
        mock_settings = self._make_mock_settings(
            fast_model="gpt-4o-mini", fast_temperature=0.1,
        )

        with patch("isaac.config.settings.settings", mock_settings), \
             patch("langchain_openai.ChatOpenAI", mock_chat_cls):
            get_llm.cache_clear()
            result = get_llm("fast")

        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.1

    def test_strong_tier_uses_strong_model(self) -> None:
        from isaac.llm.provider import get_llm

        mock_chat_cls = MagicMock(return_value="mock_llm")
        mock_settings = self._make_mock_settings(
            strong_model="o3", strong_temperature=0.7,
        )

        with patch("isaac.config.settings.settings", mock_settings), \
             patch("langchain_openai.ChatOpenAI", mock_chat_cls):
            get_llm.cache_clear()
            result = get_llm("strong")

        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["model"] == "o3"
        assert call_kwargs["temperature"] == 0.7

    def test_fast_falls_back_when_empty(self) -> None:
        from isaac.llm.provider import get_llm

        mock_chat_cls = MagicMock(return_value="mock_llm")
        mock_settings = self._make_mock_settings(fast_model="")

        with patch("isaac.config.settings.settings", mock_settings), \
             patch("langchain_openai.ChatOpenAI", mock_chat_cls):
            get_llm.cache_clear()
            result = get_llm("fast")

        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"  # fallback to default

    def test_temperature_inherits_when_negative(self) -> None:
        from isaac.llm.provider import get_llm

        mock_chat_cls = MagicMock(return_value="mock_llm")
        mock_settings = self._make_mock_settings(
            fast_model="gpt-4o-mini", fast_temperature=-1.0, temperature=0.5,
        )

        with patch("isaac.config.settings.settings", mock_settings), \
             patch("langchain_openai.ChatOpenAI", mock_chat_cls):
            get_llm.cache_clear()
            result = get_llm("fast")

        call_kwargs = mock_chat_cls.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_unsupported_provider_raises(self) -> None:
        from isaac.llm.provider import get_llm

        mock_settings = self._make_mock_settings(llm_provider="unsupported")

        with patch("isaac.config.settings.settings", mock_settings):
            get_llm.cache_clear()
            with pytest.raises(ValueError, match="Unsupported"):
                get_llm("default")

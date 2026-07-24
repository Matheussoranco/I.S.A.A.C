"""The local-first default: Ollama + ``qwen3.6``, no API keys required.

Covers three guarantees:

1. A fresh install with **no** environment at all resolves to Ollama running
   the default local model.
2. A missing daemon or an un-pulled model produces an actionable error naming
   the exact ``ollama serve`` / ``ollama pull`` command — never a cryptic
   client error and never a silent hop to a billable cloud API.
3. Cloud providers stay fully selectable, and their API keys are validated
   *only* when one of them is actually chosen.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from isaac.config.settings import DEFAULT_LOCAL_MODEL, LLMSettings, Settings
from isaac.llm.providers.ollama import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaUnavailableError,
    is_model_installed,
    preflight,
)

_CLEARED_ENV = (
    "ISAAC_LLM_PROVIDER",
    "ISAAC_MODEL_NAME",
    "ISAAC_OLLAMA_BASE_URL",
    "ISAAC_OLLAMA_LIGHT_MODEL",
    "ISAAC_OLLAMA_HEAVY_MODEL",
    "ISAAC_SUBAGENT_MODEL",
    "OLLAMA_LIGHT_MODEL",
    "OLLAMA_HEAVY_MODEL",
    "OLLAMA_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture()
def pristine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a fresh machine: no ISAAC/provider env vars, no ``.env``."""
    for var in _CLEARED_ENV:
        monkeypatch.delenv(var, raising=False)


class TestLocalFirstDefaults:
    def test_default_model_tag_is_qwen36(self) -> None:
        assert DEFAULT_MODEL == "qwen3.6"
        assert DEFAULT_LOCAL_MODEL is DEFAULT_MODEL

    def test_llm_settings_default_to_ollama(self, pristine_env: None) -> None:
        cfg = LLMSettings(_env_file=None)
        assert cfg.llm_provider == "ollama"
        assert cfg.model_name == "qwen3.6"

    def test_settings_load_with_no_api_keys(self, pristine_env: None) -> None:
        s = Settings(_env_file=None, llm=LLMSettings(_env_file=None))
        # No credentials anywhere — and nothing raises.
        assert s.openai_api_key == ""
        assert s.anthropic_api_key == ""
        # ...yet the agent is fully configured, locally.
        assert s.llm.llm_provider == "ollama"
        assert s.ollama_base_url == DEFAULT_BASE_URL
        assert s.ollama_light_model == "qwen3.6"
        assert s.ollama_heavy_model == "qwen3.6"
        # And it will not quietly reach for a paid API.
        assert s.llm_fallback_provider == ""
        assert s.local_first is True

    def test_no_retired_or_stale_model_defaults(self, pristine_env: None) -> None:
        s = Settings(_env_file=None, llm=LLMSettings(_env_file=None))
        # claude-3.5-sonnet was retired in Oct 2025; claude-sonnet-4-6 is a
        # generation behind. Neither may reappear as a default.
        assert s.subagent_model == "claude-opus-4-8"
        blob = " ".join([s.subagent_model, s.llm.model_name, s.llm.fast_model, s.llm.strong_model])
        assert "claude-3.5-sonnet" not in blob
        assert "claude-sonnet-4-6" not in blob


class TestModelTagMatching:
    @pytest.mark.parametrize(
        ("model", "installed", "expected"),
        [
            # `ollama pull qwen3.6` is reported by /api/tags as qwen3.6:latest
            ("qwen3.6", ["qwen3.6:latest"], True),
            ("qwen3.6:latest", ["qwen3.6"], True),
            ("qwen3.6", ["qwen3.6"], True),
            ("qwen3.6", ["qwen3.5:cloud", "llava:7b"], False),
            ("qwen3.6", [], False),
            ("qwen3.6:35b", ["qwen3.6:latest"], False),
        ],
    )
    def test_is_model_installed(self, model: str, installed: list[str], expected: bool) -> None:
        assert is_model_installed(model, installed) is expected


class TestPreflightFailureMode:
    def test_daemon_down_names_serve_and_pull(self) -> None:
        with (
            patch("isaac.llm.providers.ollama.health_check", return_value=False),
            pytest.raises(OllamaUnavailableError) as exc,
        ):
            preflight("qwen3.6", DEFAULT_BASE_URL)
        msg = str(exc.value)
        assert "ollama serve" in msg
        assert "ollama pull qwen3.6" in msg
        assert DEFAULT_BASE_URL in msg
        # The cloud escape hatch is offered, not taken.
        assert "ISAAC_LLM_PROVIDER=anthropic" in msg

    def test_model_not_pulled_names_exact_command(self) -> None:
        with (
            patch("isaac.llm.providers.ollama.health_check", return_value=True),
            patch(
                "isaac.llm.providers.ollama.list_models",
                return_value=["qwen3.5:cloud", "llava:7b"],
            ),
            pytest.raises(OllamaUnavailableError) as exc,
        ):
            preflight("qwen3.6", DEFAULT_BASE_URL)
        msg = str(exc.value)
        assert "ollama pull qwen3.6" in msg
        # Tells the user what they *do* have, so the fix is obvious.
        assert "qwen3.5:cloud" in msg

    def test_installed_model_passes(self) -> None:
        with (
            patch("isaac.llm.providers.ollama.health_check", return_value=True),
            patch("isaac.llm.providers.ollama.list_models", return_value=["qwen3.6:latest"]),
        ):
            preflight("qwen3.6", DEFAULT_BASE_URL)  # does not raise

    def test_unknown_tag_list_does_not_block(self) -> None:
        """An empty tag list means 'cannot verify', not 'not installed'."""
        with (
            patch("isaac.llm.providers.ollama.health_check", return_value=True),
            patch("isaac.llm.providers.ollama.list_models", return_value=[]),
        ):
            preflight("qwen3.6", DEFAULT_BASE_URL)  # does not raise

    def test_get_llm_surfaces_the_actionable_error(self) -> None:
        """The failure reaches callers of get_llm rather than being swallowed."""
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="ollama", model_name="qwen3.6")
        boom = OllamaUnavailableError(
            "Ollama is not reachable at http://localhost:11434.\n"
            "  1. Start the daemon:   ollama serve\n"
            "  3. Pull the model:     ollama pull qwen3.6\n"
        )
        provider_mod._preflight_once.cache_clear()
        with (
            patch("isaac.config.settings.settings", settings),
            patch("isaac.llm.providers.ollama.preflight", side_effect=boom),
        ):
            provider_mod.get_llm.cache_clear()
            with pytest.raises(OllamaUnavailableError, match=r"ollama pull qwen3\.6"):
                provider_mod.get_llm("default")
        provider_mod._preflight_once.cache_clear()
        provider_mod.get_llm.cache_clear()

    def test_preflight_can_be_disabled(self) -> None:
        """``ISAAC_OLLAMA_PREFLIGHT=false`` skips the probe entirely."""
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="ollama", model_name="qwen3.6")
        settings.ollama_preflight = False
        probe = MagicMock(side_effect=AssertionError("preflight should not run"))
        provider_mod._preflight_once.cache_clear()
        with (
            patch("isaac.config.settings.settings", settings),
            patch("isaac.llm.providers.ollama.preflight", probe),
            patch("isaac.llm.providers.ollama.build", return_value="stub"),
        ):
            provider_mod.get_llm.cache_clear()
            assert provider_mod.get_llm("default") == "stub"
        probe.assert_not_called()
        provider_mod.get_llm.cache_clear()


class TestCloudProvidersStaySelectable:
    def test_anthropic_requires_its_key(self) -> None:
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="anthropic", model_name="claude-opus-4-8")
        settings.anthropic_api_key = ""
        with patch("isaac.config.settings.settings", settings):
            provider_mod.get_llm.cache_clear()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                provider_mod.get_llm("default")
        provider_mod.get_llm.cache_clear()

    def test_anthropic_builds_when_key_present(self) -> None:
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="anthropic", model_name="claude-opus-4-8")
        settings.anthropic_api_key = "sk-ant-xxx"
        chat_cls = MagicMock(return_value="anthropic-model")
        with (
            patch("isaac.config.settings.settings", settings),
            patch("langchain_anthropic.ChatAnthropic", chat_cls),
        ):
            provider_mod.get_llm.cache_clear()
            assert provider_mod.get_llm("default") == "anthropic-model"
        assert chat_cls.call_args[1]["model"] == "claude-opus-4-8"
        provider_mod.get_llm.cache_clear()

    def test_openai_compatible_local_server_needs_no_key(self) -> None:
        """``ISAAC_LLM_PROVIDER=openai`` + a base URL is a local server, not a bill."""
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="openai", model_name="qwen3.6")
        settings.openai_api_key = ""
        settings.llm.base_url = "http://localhost:11434/v1"
        chat_cls = MagicMock(return_value="compat-model")
        with (
            patch("isaac.config.settings.settings", settings),
            patch("langchain_openai.ChatOpenAI", chat_cls),
        ):
            provider_mod.get_llm.cache_clear()
            assert provider_mod.get_llm("default") == "compat-model"
        provider_mod.get_llm.cache_clear()

    def test_perception_llm_honours_a_cloud_provider(self) -> None:
        """The token-capped nodes must not force a local client on cloud users."""
        from isaac.llm import provider as provider_mod

        settings = _mock_settings(llm_provider="anthropic", model_name="claude-opus-4-8")
        settings.anthropic_api_key = "sk-ant-xxx"
        chat_cls = MagicMock(return_value="anthropic-capped")
        with (
            patch("isaac.config.settings.settings", settings),
            patch("langchain_anthropic.ChatAnthropic", chat_cls),
        ):
            provider_mod.get_perception_llm.cache_clear()
            assert provider_mod.get_perception_llm() == "anthropic-capped"
        assert chat_cls.call_args[1]["max_tokens"] == 200
        provider_mod.get_perception_llm.cache_clear()


def _mock_settings(**llm_overrides: object) -> MagicMock:
    """A settings double with a local-first shape, tweakable per test."""
    llm_cfg = MagicMock()
    llm_cfg.llm_provider = "ollama"
    llm_cfg.model_name = DEFAULT_MODEL
    llm_cfg.temperature = 0.2
    llm_cfg.base_url = ""
    llm_cfg.fast_model = ""
    llm_cfg.fast_temperature = -1.0
    llm_cfg.strong_model = ""
    llm_cfg.strong_temperature = -1.0
    for key, value in llm_overrides.items():
        setattr(llm_cfg, key, value)

    settings = MagicMock()
    settings.llm = llm_cfg
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.ollama_base_url = DEFAULT_BASE_URL
    settings.ollama_preflight = True
    return settings

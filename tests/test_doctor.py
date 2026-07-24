"""Tests for the ``isaac doctor`` preflight checks.

Ollama reachability is stubbed by the autouse ``stub_ollama`` fixture, so
these run offline even though Ollama is the default provider.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaac.doctor import CheckResult, _check_ollama, has_failures, run_checks


def test_run_checks_covers_core_and_never_raises() -> None:
    results = run_checks()
    names = {r.name for r in results}
    assert {"python", "settings", "ollama", "docker", "cloud-fallback"} <= names
    assert any(n.startswith("extra:") for n in names)
    assert all(r.status in {"ok", "warn", "fail"} for r in results)


def test_python_check_passes_on_supported_interpreter() -> None:
    py = next(r for r in run_checks() if r.name == "python")
    assert py.status == "ok"


def test_settings_check_reads_nested_llm_config() -> None:
    s = next(r for r in run_checks() if r.name == "settings")
    assert s.status == "ok", s.detail
    assert "provider=" in s.detail


def test_has_failures_only_counts_fail() -> None:
    assert has_failures([CheckResult("x", "fail", "")]) is True
    assert has_failures([CheckResult("x", "warn", ""), CheckResult("y", "ok", "")]) is False


def _local_settings() -> MagicMock:
    """Settings double for a stock local-first install."""
    s = MagicMock()
    s.llm.llm_provider = "ollama"
    s.llm.model_name = "qwen3.6"
    s.ollama_base_url = "http://localhost:11434"
    s.ollama_light_model = "qwen3.6"
    s.ollama_heavy_model = "qwen3.6"
    return s


def test_ollama_check_fails_loudly_when_daemon_is_down() -> None:
    """Ollama is the default provider, so an unreachable daemon is fatal."""
    with (
        patch("isaac.config.settings.get_settings", return_value=_local_settings()),
        patch("isaac.llm.providers.ollama.health_check", return_value=False),
    ):
        r = _check_ollama()
    assert r.status == "fail"
    assert "ollama serve" in r.detail


def test_ollama_check_names_the_pull_command_for_a_missing_model() -> None:
    with (
        patch("isaac.config.settings.get_settings", return_value=_local_settings()),
        patch("isaac.llm.providers.ollama.health_check", return_value=True),
        patch("isaac.llm.providers.ollama.list_models", return_value=["qwen3.5:cloud"]),
    ):
        r = _check_ollama()
    assert r.status == "fail"
    assert "ollama pull qwen3.6" in r.detail


def test_ollama_check_accepts_latest_suffixed_tags() -> None:
    """`ollama pull qwen3.6` shows up as `qwen3.6:latest` — not a missing model."""
    with (
        patch("isaac.config.settings.get_settings", return_value=_local_settings()),
        patch("isaac.llm.providers.ollama.health_check", return_value=True),
        patch("isaac.llm.providers.ollama.list_models", return_value=["qwen3.6:latest"]),
    ):
        r = _check_ollama()
    assert r.status == "ok", r.detail


def test_ollama_check_only_warns_when_a_cloud_provider_is_selected() -> None:
    s = _local_settings()
    s.llm.llm_provider = "anthropic"
    with (
        patch("isaac.config.settings.get_settings", return_value=s),
        patch("isaac.llm.providers.ollama.health_check", return_value=False),
    ):
        r = _check_ollama()
    assert r.status == "warn"

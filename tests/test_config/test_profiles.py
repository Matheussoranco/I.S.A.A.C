"""Tests for profile-aware configuration (Phase 1.2, Hermes-mirror).

Hermetic: every test points ``ISAAC_HOME`` at a tmp dir so the real
``~/.isaac`` is never touched.
"""

from __future__ import annotations

import pytest

from isaac.config import profiles
from isaac.config.settings import clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ISAAC_HOME", str(tmp_path / ".isaac"))
    monkeypatch.delenv("ISAAC_PROFILE", raising=False)
    # Neutralise the repo's real .env (baked into model_config at import):
    # point every settings class at a nonexistent env file.
    from isaac.config import settings as st

    dead = str(tmp_path / "no-such.env")
    for cls in (st.Settings, st.LLMSettings, st.SandboxSettings,
                st.UISandboxSettings, st.GraphSettings):
        monkeypatch.setitem(cls.model_config, "env_file", dead)
    clear_settings_cache()
    yield
    clear_settings_cache()
    monkeypatch.delenv("ISAAC_PROFILE", raising=False)


# ── profiles.py ───────────────────────────────────────────────────────


def test_default_profile_is_default():
    assert profiles.get_active_profile() == "default"


def test_active_profile_from_env(monkeypatch):
    monkeypatch.setenv("ISAAC_PROFILE", "work")
    assert profiles.get_active_profile() == "work"


def test_profile_dir_created_on_demand(tmp_path):
    d = profiles.profile_dir("alpha")
    assert d == tmp_path / ".isaac" / "profiles" / "alpha"
    assert d.is_dir()


def test_profile_config_path(tmp_path):
    p = profiles.profile_config_path("alpha")
    assert p == tmp_path / ".isaac" / "profiles" / "alpha" / "config.yaml"


def test_list_profiles(tmp_path):
    assert profiles.list_profiles() == []
    profiles.profile_dir("b")
    profiles.profile_dir("a")
    (tmp_path / ".isaac" / "profiles" / "notadir.txt").write_text("x")
    assert profiles.list_profiles() == ["a", "b"]


def test_load_overrides_missing_file_returns_empty():
    assert profiles.load_profile_overrides("ghost") == {}


def test_load_overrides_invalid_yaml_returns_empty():
    path = profiles.profile_config_path("bad")
    path.write_text("[unclosed bracket", encoding="utf-8")
    assert profiles.load_profile_overrides("bad") == {}


def test_load_overrides_roundtrip():
    profiles.save_profile_overrides("x", {"llm": {"model_name": "m"}, "temperature": 1})
    assert profiles.load_profile_overrides("x") == {
        "llm": {"model_name": "m"},
        "temperature": 1,
    }


# ── merge precedence: profile > env > default ─────────────────────────


def test_default_value_when_nothing_set(monkeypatch):
    monkeypatch.delenv("ISAAC_LLM_PROVIDER", raising=False)
    s = get_settings()
    assert s.llm.llm_provider == "ollama"


def test_env_beats_default(monkeypatch):
    monkeypatch.setenv("ISAAC_LLM_PROVIDER", "openai")
    s = get_settings()
    assert s.llm.llm_provider == "openai"


def test_profile_beats_env_and_default(monkeypatch):
    monkeypatch.setenv("ISAAC_LLM_PROVIDER", "openai")
    profiles.save_profile_overrides("default", {"llm": {"llm_provider": "anthropic"}})
    s = get_settings()
    assert s.llm.llm_provider == "anthropic"


def test_profile_deep_merge_keeps_sibling_fields():
    profiles.save_profile_overrides("default", {"llm": {"temperature": 0.9}})
    s = get_settings()
    assert s.llm.temperature == 0.9
    assert s.llm.llm_provider == "ollama"  # untouched by the override


def test_inactive_profile_overrides_ignored():
    profiles.save_profile_overrides("other", {"agent_name": "HAL"})
    s = get_settings()
    assert s.agent_name == "I.S.A.A.C."
    clear_settings_cache()
    import os

    os.environ["ISAAC_PROFILE"] = "other"
    s2 = get_settings()
    assert s2.agent_name == "HAL"


def test_invalid_override_warns_not_raises():
    profiles.save_profile_overrides(
        "default", {"llm": {"temperature": 99.0}}  # exceeds le=2.0 constraint
    )
    clear_settings_cache()
    with pytest.warns(UserWarning, match="profile config.yaml"):
        s = get_settings()
    assert s.llm.temperature == 0.2  # fell back to env/default


def test_clear_settings_cache():
    s1 = get_settings()
    assert get_settings() is s1
    clear_settings_cache()
    assert get_settings() is not s1


def test_sniff_scalar():
    assert profiles.sniff_scalar("42") == 42
    assert profiles.sniff_scalar("0.5") == 0.5
    assert profiles.sniff_scalar("true") is True
    assert profiles.sniff_scalar("FALSE") is False
    assert profiles.sniff_scalar("qwen3.6") == "qwen3.6"


# ── CLI: set / unset / profiles / profile-create ──────────────────────


@pytest.fixture
def runner():
    from typer.testing import CliRunner

    from isaac.cli import app

    return CliRunner(), app


def test_cli_set_and_get(runner):
    cli, app = runner
    r = cli.invoke(app, ["config", "set", "llm.model_name", "qwen3.6"])
    assert r.exit_code == 0, r.output
    assert "Set default.llm.model_name = 'qwen3.6'" in r.output

    clear_settings_cache()
    r = cli.invoke(app, ["config", "get", "llm.model_name"])
    assert r.exit_code == 0, r.output
    assert "qwen3.6" in r.output


def test_cli_set_type_sniffing(runner):
    cli, app = runner
    cli.invoke(app, ["config", "set", "llm.temperature", "0.7"])
    data = profiles.load_profile_overrides("default")
    assert data["llm"]["temperature"] == 0.7
    cli.invoke(app, ["config", "set", "heartbeat_interval_minutes", "5"])
    data = profiles.load_profile_overrides("default")
    assert data["heartbeat_interval_minutes"] == 5
    assert isinstance(data["heartbeat_interval_minutes"], int)


def test_cli_unset(runner):
    cli, app = runner
    cli.invoke(app, ["config", "set", "agent_name", "HAL"])
    assert profiles.load_profile_overrides("default").get("agent_name") == "HAL"
    r = cli.invoke(app, ["config", "unset", "agent_name"])
    assert r.exit_code == 0, r.output
    assert "agent_name" not in profiles.load_profile_overrides("default")


def test_cli_profiles_marks_active(runner):
    cli, app = runner
    cli.invoke(app, ["config", "profile-create", "work"])
    r = cli.invoke(app, ["config", "profiles"])
    assert r.exit_code == 0, r.output
    assert "* default" in r.output
    assert "  work" in r.output
    assert "ISAAC_PROFILE" in r.output  # sticky-hint


def test_cli_profile_flag(runner, monkeypatch):
    cli, app = runner
    cli.invoke(app, ["config", "--profile", "work", "set", "agent_name", "JARVIS"])
    assert profiles.load_profile_overrides("work").get("agent_name") == "JARVIS"
    assert profiles.load_profile_overrides("default") == {}
    monkeypatch.delenv("ISAAC_PROFILE", raising=False)


def test_cli_get_redacts_secrets(runner, monkeypatch):
    cli, app = runner
    monkeypatch.setenv("ISAAC_OPENAI_API_KEY", "sk-supersecret")
    clear_settings_cache()
    r = cli.invoke(app, ["config", "get", "openai_api_key"])
    assert r.exit_code == 0
    assert "sk-supersecret" not in r.output
    assert "<set:" in r.output

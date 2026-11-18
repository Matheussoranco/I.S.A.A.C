"""Tests for profile-aware configuration (Phase 1.2, Hermes-mirror).

Hermetic: every test points ``ISAAC_HOME`` at a tmp dir so the real
``~/.isaac`` is never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    for cls in (
        st.Settings,
        st.LLMSettings,
        st.SandboxSettings,
        st.UISandboxSettings,
        st.GraphSettings,
    ):
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


@pytest.mark.parametrize(
    "name",
    ["../evil", "..", ".", "/etc/passwd", "a/b", "a\\b", ".hidden", "x" * 65, "a:b"],
)
def test_profile_names_cannot_escape_the_profiles_directory(name):
    with pytest.raises(ValueError):
        profiles.profile_dir(name)


def test_load_overrides_invalid_yaml_returns_empty():
    path = profiles.profile_config_path("bad")
    path.write_text("[unclosed bracket", encoding="utf-8")
    assert profiles.load_profile_overrides("bad") == {}


def test_load_overrides_roundtrip():
    profiles.save_profile_overrides(
        "x", {"llm": {"model_name": "m"}, "heartbeat_interval_minutes": 1}
    )
    assert profiles.load_profile_overrides("x") == {
        "llm": {"model_name": "m"},
        "heartbeat_interval_minutes": 1,
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
    profiles.profile_config_path("default").write_text(
        "llm:\n  temperature: 99.0\n", encoding="utf-8"
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


@pytest.mark.parametrize(
    ("key", "env_name"),
    [
        ("openai_api_key", "ISAAC_OPENAI_API_KEY"),
        ("email_smtp_password", "ISAAC_EMAIL_SMTP_PASSWORD"),
    ],
)
def test_cli_get_redacts_secrets(runner, monkeypatch, key, env_name):
    cli, app = runner
    credential = "test-credential-from-env"
    monkeypatch.setenv(env_name, credential)
    clear_settings_cache()
    r = cli.invoke(app, ["config", "get", key])
    assert r.exit_code == 0
    assert credential not in r.output
    assert "<set:" in r.output


@pytest.mark.parametrize("key", ["openai_api_key", "email_smtp_password", "telegram_bot_token"])
def test_config_set_rejects_credential_fields_without_echoing_values(runner, key):
    cli, app = runner
    secret = "test-credential-not-for-yaml"
    result = cli.invoke(app, ["config", "set", key, secret])

    assert result.exit_code != 0
    assert ".env" in result.output
    assert secret not in result.output
    assert not profiles.profile_config_path("default").exists()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [("llm.temperature", "99", "llm.temperature"), ("llm.not_a_setting", "1", "Unknown")],
)
def test_config_set_preserves_file_on_validation_failure(runner, key, value, message):
    cli, app = runner
    assert cli.invoke(app, ["config", "set", "agent_name", "HAL"]).exit_code == 0
    path = profiles.profile_config_path("default")
    original = path.read_bytes()

    result = cli.invoke(app, ["config", "set", key, value])

    assert result.exit_code != 0
    assert message in result.output
    assert path.read_bytes() == original


def test_config_set_does_not_overwrite_malformed_yaml(runner):
    cli, app = runner
    path = profiles.profile_config_path("default")
    original = b"agent_name: [broken\n"
    path.write_bytes(original)

    result = cli.invoke(app, ["config", "set", "agent_name", "HAL"])

    assert result.exit_code != 0
    assert "config.yaml" in result.output
    assert path.read_bytes() == original


def test_profile_yaml_cannot_override_environment_credentials(monkeypatch):
    monkeypatch.setenv("ISAAC_EMAIL_SMTP_PASSWORD", "test-env-credential")
    profiles.profile_config_path("default").write_text(
        "email_smtp_password: test-yaml-credential\n", encoding="utf-8"
    )

    with pytest.warns(UserWarning, match=r"email_smtp_password.*\.env") as caught:
        settings = get_settings()

    assert settings.email_smtp_password == "test-env-credential"
    assert "test-yaml-credential" not in str(caught[0].message)
    assert "test-env-credential" not in str(caught[0].message)


def test_dotenv_supplies_credentials_without_changing_profile_yaml(tmp_path, monkeypatch):
    from isaac.config.settings import Settings

    dotenv = tmp_path / ".env"
    dotenv.write_text("ISAAC_EMAIL_SMTP_PASSWORD=test-dotenv-credential\n", encoding="utf-8")
    monkeypatch.delenv("ISAAC_EMAIL_SMTP_PASSWORD", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", str(dotenv))
    path = profiles.save_profile_overrides("default", {"agent_name": "HAL"})

    settings = get_settings()

    assert settings.agent_name == "HAL"
    assert settings.email_smtp_password == "test-dotenv-credential"
    assert "test-dotenv-credential" not in path.read_text(encoding="utf-8")


def test_profile_cli_creates_lists_and_switches(runner, tmp_path):
    cli, app = runner

    for name in ("research", "coding"):
        result = cli.invoke(app, ["profile", "create", name])
        assert result.exit_code == 0, result.output
        assert (tmp_path / ".isaac" / "profiles" / name).is_dir()
    result = cli.invoke(app, ["profile", "switch", "coding"])
    assert result.exit_code == 0, result.output
    current = cli.invoke(app, ["profile", "current"])
    assert current.exit_code == 0, current.output
    assert current.output.strip() == "coding"
    listing = cli.invoke(app, ["profile", "list"])
    assert listing.exit_code == 0, listing.output
    assert listing.output.splitlines() == ["* coding", "  default", "  research"]


def test_named_profiles_do_not_share_keyring_credentials(monkeypatch):
    from types import SimpleNamespace

    from isaac.security.credentials import get_credential, set_credential

    passwords = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(
            get_password=lambda service, provider: passwords.get((service, provider)),
            set_password=lambda service, provider, value: passwords.update(
                {(service, provider): value}
            ),
        ),
    )
    set_credential("openai", "default-test-key")
    profiles.profile_dir("research")
    profiles.switch_profile("research")
    assert get_credential("openai") == ""
    set_credential("openai", "research-test-key")
    assert get_credential("openai") == "research-test-key"
    profiles.switch_profile("default")
    assert get_credential("openai") == "default-test-key"


@pytest.mark.parametrize(
    ("dotenv_key", "exported_key", "keyring_key", "expected"),
    [
        ("dotenv-test-key", "", "keyring-test-key", "dotenv-test-key"),
        ("dotenv-test-key", "exported-test-key", "keyring-test-key", "exported-test-key"),
        ("changeme", "", "keyring-test-key", "keyring-test-key"),
        ("changeme", "", "", ""),
    ],
)
def test_cloud_stt_and_availability_use_the_same_profile_credential(
    monkeypatch, tmp_path, dotenv_key, exported_key, keyring_key, expected
):
    from types import SimpleNamespace

    from isaac.multimodal.voice.stt import STTManager
    from isaac.security.credentials import credential_available, get_credential, set_credential

    monkeypatch.delenv("ISAAC_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    passwords = {}
    monkeypatch.setitem(
        sys.modules,
        "keyring",
        SimpleNamespace(
            get_password=lambda service, provider: passwords.get((service, provider)),
            set_password=lambda service, provider, value: passwords.update(
                {(service, provider): value}
            ),
        ),
    )
    set_credential("openai", "default-test-key")
    profile = profiles.profile_dir("research")
    (profile / ".env").write_text(f"ISAAC_OPENAI_API_KEY={dotenv_key}\n", encoding="utf-8")
    profiles.switch_profile("research")
    if keyring_key:
        set_credential("openai", keyring_key)
    if exported_key:
        monkeypatch.setenv("OPENAI_API_KEY", exported_key)

    # Disable only the local backend and replace the remote SDK at its boundary.
    monkeypatch.setattr("isaac.multimodal.voice.stt.WhisperModel", None)

    class FakeOpenAI:
        def __init__(self, *, api_key):
            assert api_key == expected
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(create=lambda **_: SimpleNamespace(text="heard"))
            )

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    audio = tmp_path / "prompt.wav"
    audio.write_bytes(b"test audio")
    assert get_credential("openai") == expected
    assert credential_available("openai") is bool(expected)
    stt = STTManager()
    if expected:
        assert stt.transcribe(audio) == "heard"
    else:
        with pytest.raises(RuntimeError, match="All STT providers failed"):
            stt.transcribe(audio)


def test_profile_switch_preserves_selection_on_error_and_respects_env(runner, monkeypatch):
    cli, app = runner
    assert cli.invoke(app, ["profile", "create", "research"]).exit_code == 0
    assert cli.invoke(app, ["profile", "switch", "research"]).exit_code == 0
    for name in ("unknown", "AUX", "../outside"):
        assert cli.invoke(app, ["profile", "switch", name]).exit_code != 0
    assert cli.invoke(app, ["profile", "current"]).output.strip() == "research"

    with monkeypatch.context() as env:
        env.setenv("ISAAC_PROFILE", "default")
        assert cli.invoke(app, ["profile", "current"]).output.strip() == "default"
    assert cli.invoke(app, ["profile", "current"]).output.strip() == "research"


def test_profile_switch_isolates_config_secrets_skills_memory_and_kanban(runner, tmp_path):
    import os
    import subprocess

    cli, app = runner
    roots = {}
    for name in ("research", "coding"):
        result = cli.invoke(app, ["profile", "create", name])
        assert result.exit_code == 0, result.output
        roots[name] = tmp_path / ".isaac" / "profiles" / name
        (roots[name] / ".env").write_text(
            f"ISAAC_EMAIL_SMTP_PASSWORD={name}-test-secret\nISAAC_MODEL_NAME={name}-model\n"
            f"OPENAI_API_KEY={name}-test-key\n",
            encoding="utf-8",
        )
        profiles.save_profile_overrides(name, {"agent_name": name})

    probe = """
import sys
from pathlib import Path
from isaac.background.kanban import get_board
from isaac.config.settings import get_settings
from isaac.memory.long_term import LongTermMemory
from isaac.memory.skill_library import SkillLibrary

name, home = sys.argv[1], Path(sys.argv[2])
settings = get_settings()
assert settings.agent_name == name
assert settings.email_smtp_password == name + '-test-secret'
assert settings.openai_api_key == name + '-test-key'
assert settings.llm.model_name == name + '-model'
assert settings.isaac_home == home
assert settings.skills_dir == home / 'skills'
assert SkillLibrary(settings.skills_dir).list_names() == []
with LongTermMemory() as memory:
    assert (home / 'long_term_memory.db').is_file()
    assert memory.size == 0
    memory.remember(name)
board = get_board()
assert board.db_path == home / 'kanban.db'
assert board.get_board_state()['total'] == 0
board.create_task(name)
board.close()
"""
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "EMAIL_SMTP_PASSWORD",
            "ISAAC_EMAIL_SMTP_PASSWORD",
            "ISAAC_MODEL_NAME",
            "ISAAC_KANBAN_DB",
            "OPENAI_API_KEY",
            "ISAAC_OPENAI_API_KEY",
        }
    }
    for name, root in roots.items():
        result = cli.invoke(app, ["profile", "switch", name])
        assert result.exit_code == 0, result.output
        child = subprocess.run(
            [sys.executable, "-c", probe, name, str(root)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("store", ["kanban", "long_term_memory", "user_profile"])
def test_profile_storage_fails_closed_on_invalid_dotenv(store, tmp_path, monkeypatch):
    from isaac.background.kanban import KanbanBoard
    from isaac.config.settings import ProfileConfigError
    from isaac.memory.long_term import LongTermMemory
    from isaac.memory.user_profile import UserProfile

    default_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: default_home)
    monkeypatch.delenv("ISAAC_KANBAN_DB", raising=False)
    monkeypatch.delenv("ISAAC_HEARTBEAT_INTERVAL_MINUTES", raising=False)
    profile = profiles.profile_dir("research")
    (profile / ".env").write_text("ISAAC_HEARTBEAT_INTERVAL_MINUTES=invalid\n", encoding="utf-8")
    profiles.switch_profile("research")

    open_store = {
        "kanban": KanbanBoard,
        "long_term_memory": LongTermMemory,
        "user_profile": UserProfile,
    }[store]
    with pytest.raises(ProfileConfigError, match="Invalid environment settings"):
        # Close any unexpectedly opened SQLite store, including when testing the old behavior.
        opened = open_store()
        if isinstance(opened, (KanbanBoard, LongTermMemory)):
            opened.close()
    assert not default_home.exists()


def test_config_check_accepts_valid_selected_profile(runner):
    cli, app = runner
    profiles.save_profile_overrides("work", {"llm": {"temperature": 0.5}})

    result = cli.invoke(app, ["config", "check", "--profile", "work"])

    assert result.exit_code == 0, result.output
    assert "work" in result.output
    assert "valid" in result.output
    assert profiles.get_active_profile() == "default"


def test_config_check_accepts_missing_overrides(runner, tmp_path):
    cli, app = runner

    result = cli.invoke(app, ["config", "check"])

    assert result.exit_code == 0, result.output
    assert "no overrides" in result.output.lower()
    assert not (tmp_path / ".isaac").exists()


def test_config_check_accepts_explicit_yaml_merge_override(runner):
    cli, app = runner
    profiles.profile_config_path("default").write_bytes(
        b"llm:\n  <<: {model_name: local, temperature: 0.5}\n  temperature: 0.4\n"
    )

    result = cli.invoke(app, ["config", "check"])

    assert result.exit_code == 0, result.output
    assert get_settings().llm.temperature == 0.4


@pytest.mark.parametrize(
    ("original", "message"),
    [
        (b"openai_api_key: test-config-secret\n", ".env"),
        (b"<<: {llm: {openai_api_key: test-config-secret}}\nllm: {}\n", ".env"),
        (b"agent_name: !!timestamp test-config-secret\n", "as YAML"),
        (b"agent_name: !!bool test-config-secret\n", "as YAML"),
    ],
)
def test_config_check_reports_invalid_settings_without_values(runner, original, message):
    cli, app = runner
    path = profiles.profile_config_path("default")
    path.write_bytes(original)

    result = cli.invoke(app, ["config", "check"])

    assert result.exit_code == 1
    assert message in result.output
    assert "test-config-secret" not in result.output
    assert path.read_bytes() == original


@pytest.fixture
def config_editor(tmp_path, monkeypatch):
    script = tmp_path / "config editor.py"
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", f'"{sys.executable}" "{script}"')
    return script


def test_config_edit_can_repair_yaml_and_preserves_comments(runner, config_editor):
    cli, app = runner
    path = profiles.profile_config_path("default")
    path.write_bytes(b"agent_name: [broken\n")
    replacement = b"# Keep this note\nagent_name: HAL\n"
    config_editor.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"assert Path(sys.argv[1]).read_bytes() == {path.read_bytes()!r}\n"
        f"Path(sys.argv[1]).write_bytes({replacement!r})\n",
        encoding="utf-8",
    )

    result = cli.invoke(app, ["config", "edit"])

    assert result.exit_code == 0, result.output
    assert path.read_bytes() == replacement
    assert get_settings().agent_name == "HAL"


def test_config_edit_creates_the_selected_profile_using_visual(runner, config_editor, monkeypatch):
    cli, app = runner
    monkeypatch.setenv("VISUAL", f'"{sys.executable}" "{config_editor}"')
    monkeypatch.setenv("EDITOR", "not-the-configured-editor")
    config_editor.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_bytes(b'agent_name: HAL\\n')\n",
        encoding="utf-8",
    )

    result = cli.invoke(app, ["config", "edit", "--profile", "work"])

    assert result.exit_code == 0, result.output
    assert profiles.load_profile_overrides("work") == {"agent_name": "HAL"}
    assert profiles.get_active_profile() == "default"
    assert profiles.list_profiles() == ["work"]


def test_config_edit_leaves_missing_file_missing_when_unchanged(runner, config_editor, tmp_path):
    cli, app = runner
    config_editor.write_text("pass\n", encoding="utf-8")

    result = cli.invoke(app, ["config", "edit"])

    assert result.exit_code == 0, result.output
    assert "No changes" in result.output
    assert not (tmp_path / ".isaac").exists()


def test_config_edit_preserves_a_save_made_while_editor_is_open(runner, config_editor):
    cli, app = runner
    path = profiles.save_profile_overrides("default", {"agent_name": "Before"})
    config_editor.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from isaac.config.profiles import save_profile_overrides\n"
        "save_profile_overrides('default', {'agent_name': 'Concurrent'})\n"
        "Path(sys.argv[1]).write_bytes(b'agent_name: Draft\\n')\n",
        encoding="utf-8",
    )

    result = cli.invoke(app, ["config", "edit"])

    assert result.exit_code == 1
    assert "changed while editing" in result.output
    assert path.read_bytes() == b"agent_name: Concurrent\n"


def test_config_set_does_not_overwrite_a_newer_save(runner, monkeypatch):
    cli, app = runner
    profiles.save_profile_overrides("default", {"agent_name": "Before"})

    def parse_after_concurrent_save(value):
        profiles.save_profile_overrides("default", {"agent_name": "Concurrent"})
        return float(value)

    monkeypatch.setattr(profiles, "sniff_scalar", parse_after_concurrent_save)
    result = cli.invoke(app, ["config", "set", "llm.temperature", "0.4"])

    assert result.exit_code != 0
    assert "changed while editing" in result.output
    assert profiles.load_profile_overrides("default") == {"agent_name": "Concurrent"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows system editor")
def test_config_edit_launches_default_windows_editor_by_absolute_path(runner, monkeypatch):
    from subprocess import CompletedProcess

    cli, app = runner
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    launches = []

    def fake_run(args, *, check):
        launches.append(Path(args[0]))
        return CompletedProcess(args, 0)

    monkeypatch.setattr("isaac.cli.config_editor.subprocess.run", fake_run)
    result = cli.invoke(app, ["config", "edit"])

    assert result.exit_code == 0, result.output
    assert len(launches) == 1
    assert launches[0].is_absolute()
    assert launches[0].name == "notepad.exe"


@pytest.mark.parametrize(
    ("replacement", "exit_code", "message"),
    [
        (b"llm: {temperature: 99}\n", 0, "llm.temperature"),
        (b"openai_api_key: test-config-secret\n", 0, ".env"),
        (b"agent_name: HAL\nagent_name: Other\n", 0, "duplicate"),
        (b"agent_name: replacement\n", 3, "editor exited"),
    ],
)
def test_config_edit_rejects_unsuccessful_edits(
    runner, config_editor, replacement, exit_code, message
):
    cli, app = runner
    path = profiles.save_profile_overrides("default", {"agent_name": "HAL"})
    original = path.read_bytes()
    config_editor.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"Path(sys.argv[1]).write_bytes({replacement!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )

    result = cli.invoke(app, ["config", "edit"])

    assert result.exit_code != 0
    assert message in result.output.lower()
    assert "test-config-secret" not in result.output
    assert path.read_bytes() == original

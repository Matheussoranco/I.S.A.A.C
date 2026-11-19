"""Environment-driven application settings.

All values are loaded from environment variables (prefix ``ISAAC_``) or a
``.env`` file at the project root.  See ``.env.example`` for the full list.

Defaults are **local-first**: a fresh install with no API keys whatsoever
talks to a local Ollama daemon running :data:`DEFAULT_LOCAL_MODEL`.  The
cloud providers (``openai``, ``anthropic``) remain fully supported — they
are opt-in via ``ISAAC_LLM_PROVIDER`` plus the matching API key, never a
silent fallback.
"""

from __future__ import annotations

import os
import warnings
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict

# Single source of truth for the default local model tag (``ollama pull qwen3.6``).
from isaac.llm.providers.ollama import DEFAULT_BASE_URL as DEFAULT_OLLAMA_BASE_URL
from isaac.llm.providers.ollama import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL

_SECRET_SETTINGS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "elevenlabs_api_key",
        "openai_compat_api_key",
        "telegram_bot_token",
        "github_token",
        "email_password",
        "email_smtp_password",
        "caldav_password",
    }
)


class ProfileConfigError(ValueError):
    """A profile configuration error that is safe to display."""


def is_secret_setting(name: str) -> bool:
    """Return whether a setting belongs outside profile YAML."""
    return name in _SECRET_SETTINGS


# Resolve .env relative to this file so it is found regardless of CWD.
# Layout: src/isaac/config/settings.py → src/isaac/config → src/isaac → src → project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
_loading_profile: ContextVar[str | None] = ContextVar("loading_profile", default=None)


def default_isaac_home() -> Path:
    """Root data directory, honouring ``ISAAC_HOME`` (and its legacy alias)."""
    from isaac.config.profiles import isaac_root

    return isaac_root()


def _load_settings_model[Model: BaseSettings](model: type[Model]) -> Model:
    from isaac.config.profiles import DEFAULT_PROFILE, profile_dir

    profile = _loading_profile.get()
    if profile is None:
        return model()
    home = profile_dir(profile, create=False)
    dotenv = home / ".env"
    if dotenv.resolve() != dotenv:
        raise ProfileConfigError("Profile .env must not be a filesystem link.")
    if profile != DEFAULT_PROFILE:
        return model(_env_file=dotenv)
    legacy = model.model_config.get("env_file")
    if not dotenv.is_file():
        return model()
    legacy_files = (legacy,) if isinstance(legacy, (str, Path)) else tuple(legacy or ())
    return model(_env_file=(*legacy_files, dotenv))


def parse_id_list(value: object) -> list[str]:
    """Normalise a comma-separated string or iterable of IDs into strings."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


class LLMSettings(BaseSettings):
    """LLM provider configuration.

    Defaults to the local Ollama daemon running :data:`DEFAULT_LOCAL_MODEL`,
    so I.S.A.A.C. is fully functional with no API keys at all.

    Supports tiered models: a *fast* model for lightweight tasks (perception,
    planning) and a *strong* model for heavy lifting (synthesis, reflection).
    When tier-specific fields are blank, they fall back to the default model.
    """

    model_config = SettingsConfigDict(
        env_prefix="ISAAC_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    llm_provider: Literal["ollama", "llamacpp", "openai_compat", "openai", "anthropic"] = "ollama"
    """Primary backend.  ``ollama`` (local) by default; the cloud providers are
    opt-in and require the matching ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``."""
    model_name: str = DEFAULT_LOCAL_MODEL
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    base_url: str = ""  # Custom API base URL (e.g. http://localhost:11434/v1 for Ollama)

    # Tier overrides — leave blank to inherit from the defaults above.
    fast_model: str = ""
    """Lightweight model for Perception & Planner (e.g. qwen3.6, gpt-5-mini)."""
    fast_temperature: float = Field(default=-1.0, ge=-1.0, le=2.0)
    """Temperature for the fast model (-1 means inherit from ``temperature``)."""
    strong_model: str = ""
    """Powerful model for Synthesis & Reflection (e.g. qwen3.6, claude-opus-4-8)."""
    strong_temperature: float = Field(default=-1.0, ge=-1.0, le=2.0)
    """Temperature for the strong model (-1 means inherit from ``temperature``)."""


class SandboxSettings(BaseSettings):
    """Docker sandbox constraints for code-execution containers."""

    model_config = SettingsConfigDict(
        env_prefix="ISAAC_SANDBOX_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    image: str = "isaac-sandbox:latest"
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit: str = "256m"
    cpu_limit: float = Field(default=1.0, ge=0.1, le=8.0)
    pids_limit: int = Field(default=64, ge=8, le=512)
    network: str = "none"
    tmpfs_size: str = "64m"


class UISandboxSettings(BaseSettings):
    """Docker sandbox constraints for virtual-desktop Computer-Use containers."""

    model_config = SettingsConfigDict(
        env_prefix="ISAAC_UI_SANDBOX_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    image: str = "isaac-ui-sandbox:latest"
    timeout_seconds: int = Field(default=120, ge=10, le=600)
    memory_limit: str = "1g"
    cpu_limit: float = Field(default=1.5, ge=0.1, le=8.0)
    pids_limit: int = Field(default=256, ge=8, le=1024)
    #: 'none' blocks all network; 'bridge' allows outbound (needed for browser tasks)
    network: str = "none"
    allow_browser_network: bool = False
    vnc_enabled: bool = False
    vnc_port: int = Field(default=5900, ge=1024, le=65535)
    screen_width: int = 1280
    screen_height: int = 720
    screen_depth: int = 24
    max_ui_cycles: int = Field(default=20, ge=1, le=100)
    """Maximum screenshot→action iterations per active PlanStep."""


class GraphSettings(BaseSettings):
    """Cognitive-loop tuning knobs."""

    model_config = SettingsConfigDict(
        env_prefix="ISAAC_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_retries: int = Field(default=3, ge=1, le=20)
    max_iterations: int = Field(default=10, ge=1, le=100)
    max_ui_cycles: int = Field(default=20, ge=1, le=100)
    """Upper bound on ComputerUse screenshot→action loop per step."""


class HeartbeatSettings(BaseModel):
    interval_min: int | None = Field(default=None, ge=1, le=1440)
    prompt: str = ""


class Settings(BaseSettings):
    """Top-level settings aggregator.

    Canonical env names use the ``ISAAC_`` prefix (e.g.
    ``ISAAC_EMAIL_IMAP_HOST``, ``ISAAC_OBSIDIAN_VAULT_PATH``,
    ``ISAAC_GITHUB_TOKEN``, ``ISAAC_SHELL_UNRESTRICTED``,
    ``ISAAC_SANDBOX_TIMEOUT_SECONDS``).

    Legacy non-prefixed names (``EMAIL_IMAP_HOST``, ``OBSIDIAN_VAULT_PATH``,
    ``GITHUB_TOKEN``, …) remain accepted as a deprecated fallback via a
    secondary env source — a ``DeprecationWarning`` is emitted when one is
    used.  See :data:`LEGACY_ENV_FALLBACK` and :func:`warn_on_legacy_env`.
    """

    model_config = SettingsConfigDict(
        env_prefix="ISAAC_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Add a legacy non-prefixed env source as lowest-priority fallback.

        Canonical process environment and dotenv values win over legacy
        names, with process environment first within each pair.
        """
        from pydantic_settings.sources import EnvSettingsSource

        legacy_env = EnvSettingsSource(settings_cls, env_prefix="")
        legacy_dotenv = DotEnvSettingsSource(
            settings_cls, env_file=dotenv_settings.env_file, env_prefix=""
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            legacy_env,
            legacy_dotenv,
            file_secret_settings,
        )

    llm: LLMSettings = Field(default_factory=lambda: _load_settings_model(LLMSettings))
    sandbox: SandboxSettings = Field(default_factory=lambda: _load_settings_model(SandboxSettings))
    ui_sandbox: UISandboxSettings = Field(
        default_factory=lambda: _load_settings_model(UISandboxSettings)
    )
    graph: GraphSettings = Field(default_factory=lambda: _load_settings_model(GraphSettings))
    heartbeat: HeartbeatSettings = Field(default_factory=HeartbeatSettings)
    skills_dir: Path = Field(default_factory=lambda: default_isaac_home() / "skills")
    """Persistent skill library directory (absolute, anchored to isaac_home)."""

    # API keys (read from env without prefix).  Optional: only consulted when
    # a cloud provider is explicitly selected or configured as a fallback.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""
    api_key: str = ""
    """Bearer token required by the local OpenAI-compatible API proxy."""

    # Ollama-first LLM routing (the default backend)
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_light_model: str = DEFAULT_LOCAL_MODEL
    ollama_heavy_model: str = DEFAULT_LOCAL_MODEL
    ollama_preflight: bool = True
    """Verify the daemon is up and the model is pulled before the first call.

    On failure I.S.A.A.C. raises ``OllamaUnavailableError`` naming the exact
    ``ollama pull <model>`` command, instead of surfacing a cryptic client
    error.  Set to ``false`` to skip the check (e.g. offline test runs)."""
    llm_fallback_provider: str = ""
    """Fallback provider name when the primary one is unhealthy.

    Empty by default: I.S.A.A.C. never falls back to a billable cloud API
    unless you name one here."""

    # llama.cpp HTTP server
    llamacpp_base_url: str = "http://localhost:8080"
    llamacpp_model: str = "local-model"

    # Generic OpenAI-compatible endpoint (LM Studio, vLLM, LiteLLM, ...)
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    # Multimodal routing toggle
    local_first: bool = True
    """When True, the multimodal router prefers local backends over cloud."""

    # ── Vision (multimodal) ────────────────────────────────────────────
    vision_enabled: bool = True
    """If False, vision routes are not registered and image input is text-only."""
    vision_model: str = "llava:7b"
    """Default vision-language model tag (Ollama by default)."""
    vision_strong_model: str = ""
    """Optional larger VLM for hard visual reasoning."""
    vision_cloud_provider: str = "gpt-4o"
    """Cloud fallback VLM when the local one fails: ``gpt-4o``, ``claude``, ``gemini``."""

    # ── Voice (STT / TTS) ──────────────────────────────────────────────
    voice_enabled: bool = True
    """Master switch for the voice subsystem."""
    voice_device: Literal["auto", "cpu", "cuda"] = "auto"
    voice_stt_model: str = "base"
    """faster-whisper model size (tiny / base / small / medium / large-v3)."""
    voice_stt_language: str = ""  # auto-detect
    voice_stt_compute_type: str = "int8"
    voice_tts_voice: str = "en_US-lessac-medium"
    """Piper voice file name (looked up under PIPER_VOICE_DIR or ~/.isaac/voices)."""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    """ElevenLabs voice ID used by the ``elevenlabs`` TTS provider."""
    voice_tts_rate: int = 175
    """Base speech rate in words/min for the pyttsx3 engine."""

    # ── Self-improvement engine ────────────────────────────────────────
    improvement_enabled: bool = True
    """When True, the scheduler runs a periodic improvement cycle."""
    improvement_interval_minutes: int = Field(default=240, ge=10, le=10080)
    improvement_promote_runs: int = 10
    improvement_promote_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    improvement_deprecate_runs: int = 8
    improvement_deprecate_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # Telegram gateway
    telegram_bot_token: str = ""
    telegram_allowed_users: str = ""
    """Comma-separated list of allowed Telegram user IDs."""

    # Heartbeat scheduler
    heartbeat_interval_minutes: int = Field(default=15, ge=1, le=1440)

    # Security
    guard_suspicion_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    """PromptInjectionGuard threshold (0.0–1.0). Above this → sanitize/reject."""

    # Isaac workspace
    isaac_home: Path = Field(default_factory=default_isaac_home)
    """Root directory for Isaac persistent data (memory, audit, workspace)."""

    # ── Identity & Soul ─────────────────────────────────────────────────
    agent_name: str = "I.S.A.A.C."
    """Display name of the agent."""
    soul_path: str = ""
    """Path to a custom soul JSON file (overrides built-in SOUL)."""

    # ── Long-term Memory ────────────────────────────────────────────────
    memory_db_path: str = ""
    """SQLite DB path for long-term memory (default: ~/.isaac/long_term_memory.db)."""
    user_profile_path: str = ""
    """JSON file path for user profile (default: ~/.isaac/user_profile.json)."""
    memory_consolidation_interval: int = Field(default=50, ge=5, le=1000)
    """Number of interactions between automatic memory consolidation runs."""

    # ── Connectors ──────────────────────────────────────────────────────
    allowed_paths: list[str] = Field(
        default_factory=lambda: [str(default_isaac_home() / "workspace")]
    )
    """Directories accessible by the FileSystemConnector.

    Defaults to ``~/.isaac/workspace`` (created on first :func:`get_settings`
    call) instead of the whole home directory — least privilege by default.
    Override via ``ISAAC_ALLOWED_PATHS`` (JSON list) or constructor arg.
    """
    shell_allowed_commands: list[str] = Field(default_factory=list)
    """Commands the ShellConnector/ShellTool may execute (empty = use default set)."""
    shell_unrestricted: bool = False
    """Deprecated compatibility flag. True blocks host commands.

    Arbitrary host execution cannot guarantee filesystem confinement. Use the
    Docker code tool instead. There is no unsafe override.
    """
    shell_tool_timeout: int = Field(default=30, ge=1, le=600)
    """Default timeout (seconds) for the host ShellTool.

    Canonical env: ``ISAAC_SHELL_TOOL_TIMEOUT`` (not ``TIMEOUT``)."""
    shell_pass_secrets: bool = False
    """Deprecated compatibility flag; no host child processes are launched."""
    connector_audit_log: str = ""
    """Path for connector audit log (default: ~/.isaac/connector_audit.log)."""
    searx_url: str = ""
    """Self-hosted Searx base URL used as the web-search fallback."""

    # Connector env-vars (optional — loaded from environment)
    github_token: str = ""
    email_imap_host: str = ""
    email_user: str = ""
    email_password: str = ""
    email_imap_port: int = 993
    obsidian_vault_path: str = ""

    # Email — SMTP (outbound)
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""

    # CalDAV
    caldav_url: str = ""
    caldav_username: str = ""
    caldav_password: str = ""

    # ── Background / Cron ───────────────────────────────────────────────
    cron_poll_seconds: int = Field(default=30, ge=5, le=600)
    """Seconds between cron daemon poll cycles."""
    cron_enabled: bool = False
    """Whether to auto-start the cron daemon on boot."""

    # ── Multimodal ──────────────────────────────────────────────────────
    tts_engine: str = "auto"
    """Default TTS engine for the ``isaac speak`` command: pyttsx3/kokoro/openai/auto."""
    tts_voice: str = "default"
    """Default voice ID for ``isaac speak`` (engine-specific)."""

    # ── MCP Server ──────────────────────────────────────────────────────
    mcp_enabled: bool = True
    """Expose I.S.A.A.C. as an MCP tool provider (used by claude mcp-serve)."""

    # ── Self-Improvement ────────────────────────────────────────────────
    meta_learner_db_path: str = ""
    """SQLite path for MetaLearner outcomes (default: ~/.isaac/meta_learner.db)."""
    meta_specialist_selection: bool = False
    """Bias Orchestrator specialist selection with MetaLearner win-rates.

    When True the roster handed to the planner is ordered by each specialist's
    Bayesian-smoothed win-rate and annotated with its track record, and an
    unknown specialist name resolves to the best-scoring member instead of the
    generalist.  **Off by default**: the 1.5.0 ablation
    (``docs/ROADMAP-1.0.md`` §7) measured no end-to-end gain, and this project
    does not ship unmeasured behaviour as a default.  See
    :mod:`isaac.meta.specialist_selector`."""
    skill_verification_enabled: bool = True
    """Require a skill to pass a verification run before it enters the library.

    When True, :meth:`isaac.memory.skill_library.SkillLibrary.commit` re-parses,
    imports, and executes a generated skill (self-test / doctests when present)
    and refuses to promote it on failure, recording the rejection instead."""
    skill_verification_timeout: int = Field(default=20, ge=1, le=300)
    """Wall-clock budget (seconds) for one skill verification run."""
    skill_verification_require_sandbox: bool = True
    """Refuse to promote any skill when the Docker sandbox is unavailable.

    On by default: generated code is never imported by a host Python process.
    Set to false only for trusted development fixtures that deliberately test
    the fallback verifier."""
    parallel_synthesis_enabled: bool = False
    """Enable parallel Claude sub-agent synthesis for independent plan steps."""
    parallel_synthesis_min_steps: int = Field(default=2, ge=2, le=10)
    """Minimum independent steps required to trigger parallel synthesis."""

    # ── Sub-agents ──────────────────────────────────────────────────────
    subagent_model: str = "claude-opus-4-8"
    """Cloud model used by ClaudeSubAgent when it is explicitly pointed at one.

    Sub-agents resolve through :func:`isaac.llm.provider.get_llm` and therefore
    run on the local default (Ollama + ``qwen3.6``) unless the Anthropic
    provider is selected."""
    subagent_max_workers: int = Field(default=4, ge=1, le=16)
    """Max concurrent sub-agents in ParallelSubAgentPool."""


# Lazy module-level singleton — use get_settings(); ``settings`` stays as a
# backwards-compatible alias resolved via PEP 562 __getattr__ so importing
# this module never constructs Settings (nor emits warnings) as a side effect.
_settings: tuple[str, Settings] | None = None


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge *override* over *base* (dicts recurse, else override wins)."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, val in override.items():
            merged[key] = _deep_merge(merged[key], val) if key in merged else val
        return merged
    return override


def validate_profile_overrides(overrides: dict[str, Any]) -> None:
    """Validate YAML settings without loading environment values or credentials."""
    _validate_fields(Settings, overrides)


def _validate_fields(model: type[BaseModel], data: dict[str, Any], prefix: str = "") -> None:
    for key, value in data.items():
        if isinstance(key, str) and is_secret_setting(key):
            raise ProfileConfigError(f"{key} belongs in .env or the environment, not config.yaml.")
        if key not in model.model_fields:
            section = f" under {prefix.rstrip('.')}" if prefix else ""
            raise ProfileConfigError(f"Unknown setting{section} in config.yaml.")
        field = model.model_fields[key]
        path = f"{prefix}{key}"
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if not isinstance(value, dict):
                raise ProfileConfigError(f"{path} must be a mapping in config.yaml.")
            _validate_fields(annotation, value, f"{path}.")
        else:
            adapter: TypeAdapter[Any] = TypeAdapter(field.rebuild_annotation())
            try:
                adapter.validate_python(value)
            except ValidationError:
                raise ProfileConfigError(f"Invalid value for {path} in config.yaml.") from None


def _apply_profile_overrides(settings: Settings, profile: str) -> Settings:
    from isaac.config.profiles import load_profile_overrides

    try:
        overrides = load_profile_overrides(profile, strict=True)
        if overrides:
            return Settings(_env_file=None, **_deep_merge(settings.model_dump(), overrides))
    except ProfileConfigError as exc:
        warnings.warn(f"Ignoring profile config.yaml: {exc}", UserWarning, stacklevel=2)
    except ValidationError:
        warnings.warn(
            "Ignoring incompatible profile config.yaml overrides.", UserWarning, stacklevel=2
        )
    return settings


def load_profile_settings(profile: str) -> Settings:
    """Load one profile without changing the active process or its environment."""
    from isaac.config.profiles import DEFAULT_PROFILE, profile_dir, validate_profile_name

    profile = validate_profile_name(profile)
    token = _loading_profile.set(profile)
    try:
        settings = _apply_profile_overrides(_load_settings_model(Settings), profile)
    except ValidationError:
        raise ProfileConfigError("Invalid environment settings for this profile.") from None
    finally:
        _loading_profile.reset(token)
    if profile == DEFAULT_PROFILE:
        return settings

    home = profile_dir(profile, create=False)
    workspace = str(default_isaac_home() / "workspace")
    return settings.model_copy(
        update={
            "isaac_home": home,
            "skills_dir": home / "skills",
            "memory_db_path": "",
            "user_profile_path": "",
            "meta_learner_db_path": "",
            "allowed_paths": [
                str(home / "workspace") if p == workspace else p for p in settings.allowed_paths
            ],
        }
    )


def clear_settings_cache() -> None:
    """Drop the cached Settings singleton (tests / profile switches)."""
    global _settings
    _settings = None


def get_settings() -> Settings:
    """Return the cached settings for the active profile.

    The workspace directory (``allowed_paths[0]`` when it is the default
    ``~/.isaac/workspace``) is created here so import has no filesystem
    side effects.
    """
    from isaac.config.profiles import get_active_profile

    global _settings
    profile = get_active_profile()
    cached = _settings
    if cached is None or cached[0] != profile:
        active = load_profile_settings(profile)
        try:
            for p in active.allowed_paths:
                pp = Path(p)
                try:
                    if pp.resolve().is_relative_to(active.isaac_home):
                        pp.mkdir(parents=True, exist_ok=True)
                except Exception:
                    if str(pp).startswith(str(active.isaac_home)):
                        pp.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        cached = _settings = (profile, active)
    warn_on_legacy_env()
    return cached[1]


def __getattr__(name: str):
    # Compat shim: ``from isaac.config.settings import settings`` keeps working
    # but resolves lazily through get_settings() instead of at import time.
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Canonical / legacy env-name handling ─────────────────────────────────
#
# Canonical names carry the ``ISAAC_`` prefix and are what ``.env.example``
# documents.  Legacy non-prefixed names are accepted only as a deprecated
# fallback (see ``Settings.settings_customise_sources``).

#: canonical ``ISAAC_`` name -> deprecated legacy name(s)
LEGACY_ENV_FALLBACK: dict[str, list[str]] = {
    "ISAAC_EMAIL_IMAP_HOST": ["EMAIL_IMAP_HOST"],
    "ISAAC_EMAIL_USER": ["EMAIL_USER"],
    "ISAAC_EMAIL_PASSWORD": ["EMAIL_PASSWORD"],
    "ISAAC_EMAIL_IMAP_PORT": ["EMAIL_IMAP_PORT"],
    "ISAAC_EMAIL_SMTP_HOST": ["EMAIL_SMTP_HOST"],
    "ISAAC_EMAIL_SMTP_PORT": ["EMAIL_SMTP_PORT"],
    "ISAAC_EMAIL_SMTP_USER": ["EMAIL_SMTP_USER"],
    "ISAAC_EMAIL_SMTP_PASSWORD": ["EMAIL_SMTP_PASSWORD"],
    "ISAAC_OBSIDIAN_VAULT_PATH": ["OBSIDIAN_VAULT_PATH"],
    "ISAAC_GITHUB_TOKEN": ["GITHUB_TOKEN"],
    "ISAAC_CALDAV_URL": ["CALDAV_URL"],
    "ISAAC_CALDAV_USERNAME": ["CALDAV_USERNAME"],
    "ISAAC_CALDAV_PASSWORD": ["CALDAV_PASSWORD"],
    "ISAAC_TELEGRAM_BOT_TOKEN": ["TELEGRAM_BOT_TOKEN"],
    "ISAAC_TELEGRAM_ALLOWED_USERS": ["TELEGRAM_ALLOWED_USERS"],
    "ISAAC_SHELL_TOOL_TIMEOUT": ["SHELL_TOOL_TIMEOUT"],
    "ISAAC_SHELL_PASS_SECRETS": ["SHELL_PASS_SECRETS"],
    "ISAAC_OPENAI_API_KEY": ["OPENAI_API_KEY"],
    "ISAAC_ANTHROPIC_API_KEY": ["ANTHROPIC_API_KEY"],
    "ISAAC_GEMINI_API_KEY": ["GEMINI_API_KEY"],
    "ISAAC_ELEVENLABS_API_KEY": ["ELEVENLABS_API_KEY"],
}

_warned_legacy: set[str] = set()


def warn_on_legacy_env() -> None:
    """Emit a one-time ``DeprecationWarning`` per legacy env var in use."""
    import logging as _logging

    _log = _logging.getLogger(__name__)
    for canonical, legacies in LEGACY_ENV_FALLBACK.items():
        if os.environ.get(canonical):
            continue
        for legacy in legacies:
            if os.environ.get(legacy) and legacy not in _warned_legacy:
                _warned_legacy.add(legacy)
                msg = (
                    f"Env var {legacy} is deprecated; "
                    f"use {canonical} instead. "
                    f"Legacy fallback will be removed in a future release."
                )
                warnings.warn(msg, DeprecationWarning, stacklevel=3)
                _log.warning(msg)


def resolve_env(canonical: str, default: str = "") -> str:
    """Return ``canonical`` env value, falling back to legacy name(s).

    Emits a ``DeprecationWarning`` when the legacy name is used.
    Never logs or prints the value itself (secrets-safe).
    """
    val = os.environ.get(canonical, "")
    if val:
        return val
    for legacy in LEGACY_ENV_FALLBACK.get(canonical, []):
        val = os.environ.get(legacy, "")
        if val:
            warn_on_legacy_env()
            return val
    return default


def env_is_set(canonical: str) -> bool:
    """Check whether a canonical env var (or its legacy fallback) is set."""
    if os.environ.get(canonical):
        return True
    return any(os.environ.get(leg) for leg in LEGACY_ENV_FALLBACK.get(canonical, []))


def known_env_keys() -> set[str]:
    """Return every recognised env key (canonical + legacy + nested)."""
    keys = {"ISAAC_HOME", "ISAAC_PROFILE", "ISAAC_KANBAN_DB"}
    # Top-level Settings: canonical ISAAC_<FIELD> + legacy bare UPPER.
    for name in Settings.model_fields:
        if name in ("llm", "sandbox", "ui_sandbox", "graph"):
            continue
        keys.add(f"ISAAC_{name.upper()}")
        keys.add(name.upper())
    # Nested models with their own prefixes.
    for cls, prefix in (
        (LLMSettings, "ISAAC_"),
        (SandboxSettings, "ISAAC_SANDBOX_"),
        (UISandboxSettings, "ISAAC_UI_SANDBOX_"),
        (GraphSettings, "ISAAC_"),
    ):
        for name in cls.model_fields:
            keys.add(f"{prefix}{name.upper()}")
    # Non-prefixed secrets documented as canonical in .env.example.
    keys.update({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})
    # Legacy connector names (deprecated but accepted).
    for legacies in LEGACY_ENV_FALLBACK.values():
        keys.update(legacies)
        keys.update(LEGACY_ENV_FALLBACK.keys())
    return keys


def find_unknown_env_vars(env: dict[str, str] | None = None) -> list[str]:
    """Return sorted ``ISAAC_*`` keys in *env* not recognised by Settings.

    Only ``ISAAC_*`` (plus the documented ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY``) are checked — unrelated host variables are ignored.
    Used by ``isaac doctor --strict`` and the ``.env.example`` CI test.
    """
    import re as _re

    source = dict(os.environ) if env is None else dict(env)
    known = known_env_keys()
    # Correct nested keys use double prefixes occasionally (e.g.
    # ISAAC_SANDBOX_TIMEOUT_SECONDS); those are already in known_env_keys.
    # Anything starting with ISAAC_ that is not known is a probable typo —
    # e.g. ISAAC_TIMEOUT or ISAAC_SANDBOX_TIMEOUT instead of
    # ISAAC_SANDBOX_TIMEOUT_SECONDS.
    unknown = sorted(
        k
        for k in source
        if k not in known
        and (k.startswith("ISAAC_") or _re.fullmatch(r"(OPENAI|ANTHROPIC)_API_KEY", k))
    )
    return unknown


def parse_dotenv_example(path: Path | str | None = None) -> dict[str, str]:
    """Parse ``.env.example`` into ``{KEY: value}`` without importing secrets."""
    p = Path(path) if path else _PROJECT_ROOT / ".env.example"
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def validate_dotenv_example(path: Path | str | None = None) -> list[str]:
    """Return unknown keys found in ``.env.example`` (empty = valid)."""
    return [k for k in parse_dotenv_example(path) if k not in known_env_keys()]

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

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the default local model tag (``ollama pull qwen3.6``).
from isaac.llm.providers.ollama import DEFAULT_BASE_URL as DEFAULT_OLLAMA_BASE_URL
from isaac.llm.providers.ollama import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL

# Resolve .env relative to this file so it is found regardless of CWD.
# Layout: src/isaac/config/settings.py → src/isaac/config → src/isaac → src → project_root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


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


class Settings(BaseSettings):
    """Top-level settings aggregator."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    ui_sandbox: UISandboxSettings = Field(default_factory=UISandboxSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    skills_dir: Path = Field(default_factory=lambda: Path.home() / ".isaac" / "skills")
    """Persistent skill library directory (absolute, anchored to isaac_home)."""

    # API keys (read from env without prefix).  Optional: only consulted when
    # a cloud provider is explicitly selected or configured as a fallback.
    openai_api_key: str = ""
    anthropic_api_key: str = ""

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
    voice_tts_rate: int = 175
    voice_tts_sample_rate: int = 22050

    # ── Self-improvement engine ────────────────────────────────────────
    improvement_enabled: bool = False
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
    isaac_home: Path = Path.home() / ".isaac"
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
    allowed_paths: list[str] = Field(default_factory=lambda: [str(Path.home())])
    """Directories accessible by the FileSystemConnector."""
    shell_allowed_commands: list[str] = Field(default_factory=list)
    """Commands the ShellConnector/ShellTool may execute (empty = use default set)."""
    shell_unrestricted: bool = False
    """When True, the host ShellTool runs commands through the platform shell
    (enabling pipes, redirects, and aliases) instead of the strict allow-list +
    metacharacter block.  Constitutional review still hard-denies critical
    patterns (rm -rf /, fork bombs, disk writes, …) in either mode.  Off by
    default — opt in only on a trusted machine."""
    shell_tool_timeout: int = Field(default=30, ge=1, le=600)
    """Default timeout (seconds) for the host ShellTool."""
    connector_audit_log: str = ""
    """Path for connector audit log (default: ~/.isaac/connector_audit.log)."""

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
    whisper_model: str = "base"
    """faster-whisper model size: tiny/base/small/medium/large-v3."""
    tts_engine: str = "auto"
    """TTS engine: pyttsx3 / kokoro / openai / auto."""
    tts_voice: str = "default"
    """Voice ID for TTS (engine-specific)."""

    # ── MCP Server ──────────────────────────────────────────────────────
    mcp_enabled: bool = True
    """Expose I.S.A.A.C. as an MCP tool provider (used by claude mcp-serve)."""

    # ── Self-Improvement ────────────────────────────────────────────────
    meta_learner_db_path: str = ""
    """SQLite path for MetaLearner outcomes (default: ~/.isaac/meta_learner.db)."""
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


# Module-level singleton — import and use directly.
settings = Settings()


def get_settings() -> Settings:
    """Return the module-level Settings singleton."""
    return settings

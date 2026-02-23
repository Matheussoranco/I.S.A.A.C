"""Environment-driven application settings.

All values are loaded from environment variables (prefix ``ISAAC_``) or a
``.env`` file at the project root.  See ``.env.example`` for the full list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="ISAAC_")

    llm_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str = "gpt-4o"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    base_url: str = ""  # Custom API base URL (e.g. http://localhost:11434/v1 for Ollama)


class SandboxSettings(BaseSettings):
    """Docker sandbox constraints for code-execution containers."""

    model_config = SettingsConfigDict(env_prefix="ISAAC_SANDBOX_")

    image: str = "isaac-sandbox:latest"
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit: str = "256m"
    cpu_limit: float = Field(default=1.0, ge=0.1, le=8.0)
    pids_limit: int = Field(default=64, ge=8, le=512)
    network: str = "none"
    tmpfs_size: str = "64m"


class UISandboxSettings(BaseSettings):
    """Docker sandbox constraints for virtual-desktop Computer-Use containers."""

    model_config = SettingsConfigDict(env_prefix="ISAAC_UI_SANDBOX_")

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

    model_config = SettingsConfigDict(env_prefix="ISAAC_")

    max_retries: int = Field(default=3, ge=1, le=20)
    max_iterations: int = Field(default=10, ge=1, le=100)
    max_ui_cycles: int = Field(default=20, ge=1, le=100)
    """Upper bound on ComputerUse screenshot→action loop per step."""


class Settings(BaseSettings):
    """Top-level settings aggregator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    ui_sandbox: UISandboxSettings = Field(default_factory=UISandboxSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    skills_dir: Path = Path("skills")

    # API keys (read from env without prefix)
    openai_api_key: str = ""
    anthropic_api_key: str = ""


# Module-level singleton — import and use directly.
settings = Settings()

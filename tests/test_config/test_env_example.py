"""CI guard: ``.env.example`` keys must match ``Settings`` fields.

Fails on unknown keys (probable typos such as ``ISAAC_TIMEOUT`` instead of
``ISAAC_SANDBOX_TIMEOUT_SECONDS``).  ``extra="ignore"`` stays the runtime
default for backwards compatibility — this test plus
``isaac doctor --strict`` provide the strict validation path.
"""

from __future__ import annotations


def test_env_example_has_no_unknown_keys() -> None:
    from isaac.config.settings import known_env_keys, parse_dotenv_example

    example = parse_dotenv_example()
    assert example, ".env.example parsed empty — check path"
    known = known_env_keys()
    unknown = sorted(k for k in example if k not in known)
    assert not unknown, (
        f"Unknown key(s) in .env.example: {unknown}. "
        f"Use canonical names (e.g. ISAAC_SANDBOX_TIMEOUT_SECONDS, "
        f"ISAAC_SHELL_TOOL_TIMEOUT, ISAAC_OBSIDIAN_VAULT_PATH)."
    )


def test_canonical_connector_keys_are_known() -> None:
    from isaac.config.settings import known_env_keys

    known = known_env_keys()
    for key in (
        "ISAAC_EMAIL_IMAP_HOST",
        "ISAAC_EMAIL_USER",
        "ISAAC_EMAIL_PASSWORD",
        "ISAAC_EMAIL_IMAP_PORT",
        "ISAAC_OBSIDIAN_VAULT_PATH",
        "ISAAC_GITHUB_TOKEN",
        "ISAAC_SHELL_UNRESTRICTED",
        "ISAAC_SHELL_TOOL_TIMEOUT",
        "ISAAC_SANDBOX_TIMEOUT_SECONDS",
    ):
        assert key in known, f"Canonical key missing from Settings: {key}"


def test_strict_detector_flags_typos() -> None:
    from isaac.config.settings import find_unknown_env_vars

    unknown = find_unknown_env_vars({"ISAAC_TIMEOUT": "30", "ISAAC_SANDBOX_TIMEOUT_SECONDS": "30"})
    assert "ISAAC_TIMEOUT" in unknown
    assert "ISAAC_SANDBOX_TIMEOUT_SECONDS" not in unknown

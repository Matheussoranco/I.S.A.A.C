"""Tests for secrets redaction."""

from __future__ import annotations

from isaac.security.redact import contains_secret, redact_secrets


def test_provider_tokens_are_redacted() -> None:
    text = (
        "key=sk-abcdefghijklmnopqrstuvwx123456 and "
        "aws AKIAIOSFODNN7EXAMPLE plus ghp_ABCDEFGHIJKLMNOPQRSTUVWX"
    )
    out = redact_secrets(text)
    assert "sk-abcdefghijklmnopqrstuvwx123456" not in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "ghp_ABCDEFGHIJKLMNOPQRSTUVWX" not in out
    assert "[REDACTED:openai-key]" in out
    assert "[REDACTED:aws-access-key]" in out
    assert "[REDACTED:github-token]" in out


def test_private_key_block_is_redacted() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    out = redact_secrets(f"found this:\n{pem}\nend")
    assert "MIIEowIBAAKCAQEA" not in out
    assert "[REDACTED:private-key]" in out


def test_credential_assignments_are_redacted_but_keys_kept() -> None:
    out = redact_secrets("config: password=hunter2secret api_key: abc123def456")
    assert "hunter2secret" not in out
    assert "abc123def456" not in out
    assert "password=" in out  # the variable name survives, the value does not


def test_normal_text_untouched() -> None:
    text = "The password policy requires 12 characters. See docs/security.md."
    assert redact_secrets(text) == text
    assert contains_secret(text) is False


def test_contains_secret_detects() -> None:
    assert contains_secret("token: xoxb-123456789012-abcdefghij") is True

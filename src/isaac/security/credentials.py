"""Cloud credentials stored in the operating system's secure keyring."""

from __future__ import annotations

import os

_SERVICE = "I.S.A.A.C. Agent"
_ENV_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_credential(provider: str) -> str:
    """Return an environment credential first, then the OS keyring value."""
    provider = _validated_provider(provider)
    from_env = os.environ.get(_ENV_NAMES[provider], "").strip()
    if from_env:
        return from_env
    try:
        import keyring

        return (keyring.get_password(_SERVICE, provider) or "").strip()
    except Exception:
        return ""


def set_credential(provider: str, value: str) -> None:
    """Persist or remove a cloud credential in the OS keyring."""
    provider = _validated_provider(provider)
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("Instale a dependência desktop para salvar credenciais.") from exc

    value = value.strip()
    try:
        if value:
            keyring.set_password(_SERVICE, provider, value)
            return
        try:
            keyring.delete_password(_SERVICE, provider)
        except keyring.errors.PasswordDeleteError:
            return
    except Exception as exc:
        raise RuntimeError(
            "O Gerenciador de Credenciais do Windows não pôde salvar a chave."
        ) from exc


def credential_available(provider: str) -> bool:
    return bool(get_credential(provider))


def _validated_provider(provider: str) -> str:
    normalized = provider.lower().strip()
    if normalized not in _ENV_NAMES:
        raise ValueError("Only OpenAI and Anthropic credentials are supported.")
    return normalized


__all__ = ["credential_available", "get_credential", "set_credential"]

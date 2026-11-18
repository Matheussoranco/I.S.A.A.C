"""Cloud credentials stored in the operating system's secure keyring."""

from __future__ import annotations

import os

_SERVICE = "I.S.A.A.C. Agent"
_ENV_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def get_credential(provider: str) -> str:
    """Return an exported credential first, then the active profile's keyring value."""
    provider = _validated_provider(provider)
    from_env = os.environ.get(_ENV_NAMES[provider], "").strip()
    if from_env:
        return from_env
    service = _profile_service()
    try:
        import keyring

        return (keyring.get_password(service, provider) or "").strip()
    except Exception:
        return ""


def set_credential(provider: str, value: str) -> None:
    """Persist or remove a cloud credential in the OS keyring."""
    provider = _validated_provider(provider)
    service = _profile_service()
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("Instale a dependência desktop para salvar credenciais.") from exc

    value = value.strip()
    try:
        if value:
            keyring.set_password(service, provider, value)
            return
        try:
            keyring.delete_password(service, provider)
        except keyring.errors.PasswordDeleteError:
            return
    except Exception as exc:
        raise RuntimeError(
            "O Gerenciador de Credenciais do Windows não pôde salvar a chave."
        ) from exc


def credential_available(provider: str) -> bool:
    """Return whether the provider has a credential in the active profile."""
    from isaac.config.settings import get_settings

    provider = _validated_provider(provider)
    return bool(getattr(get_settings(), f"{provider}_api_key").strip() or get_credential(provider))


def _profile_service() -> str:
    from isaac.config.profiles import DEFAULT_PROFILE, get_active_profile, profile_dir

    profile = get_active_profile()
    if profile == DEFAULT_PROFILE:
        return _SERVICE
    path = profile_dir(profile, create=False)
    return f"{_SERVICE}:{os.path.normcase(path)}"


def _validated_provider(provider: str) -> str:
    normalized = provider.lower().strip()
    if normalized not in _ENV_NAMES:
        raise ValueError("Only OpenAI and Anthropic credentials are supported.")
    return normalized


__all__ = ["credential_available", "get_credential", "set_credential"]

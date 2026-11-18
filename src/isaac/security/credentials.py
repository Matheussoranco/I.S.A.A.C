"""Cloud credentials stored in the operating system's secure keyring."""

from __future__ import annotations

import os

_SERVICE = "I.S.A.A.C. Agent"
_ENV_NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def is_usable_key(value: str) -> bool:
    """Reject blank credentials and known placeholders."""
    value = value.strip().lower()
    return bool(value) and value not in {
        "...",
        "changeme",
        "change-me",
        "your-api-key",
        "your_api_key",
        "sk-...",
        "sk-your-key",
        "sk-your-key-here",
        "your-key-here",
        "not-needed",
        "none",
        "null",
    }


def get_credential(provider: str) -> str:
    """Resolve a provider credential from the environment, profile, or keyring."""
    from isaac.config.settings import get_settings

    provider = _validated_provider(provider)
    for name in (f"ISAAC_{_ENV_NAMES[provider]}", _ENV_NAMES[provider]):
        exported = os.environ.get(name, "").strip()
        if is_usable_key(exported):
            return exported
    configured = getattr(get_settings(), f"{provider}_api_key").strip()
    if is_usable_key(configured):
        return configured
    service = _profile_service()
    try:
        import keyring

        stored = (keyring.get_password(service, provider) or "").strip()
    except Exception:
        return ""
    return stored if is_usable_key(stored) else ""


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
    """Return whether a usable credential exists for the provider in the active profile."""
    return bool(get_credential(provider))


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

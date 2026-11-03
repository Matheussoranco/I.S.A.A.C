"""Tests for the API-key dependency."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from isaac.security.api_auth import validate_api_key


@pytest.mark.asyncio
async def test_non_ascii_header_is_rejected_not_raised(monkeypatch) -> None:
    from isaac.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "correct-key", raising=False)
    with pytest.raises(HTTPException) as exc:
        await validate_api_key("pässword")
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_bearer_token_is_accepted(monkeypatch) -> None:
    from isaac.config.settings import get_settings

    monkeypatch.setattr(get_settings(), "api_key", "correct-key", raising=False)
    assert await validate_api_key("Bearer correct-key") == "correct-key"

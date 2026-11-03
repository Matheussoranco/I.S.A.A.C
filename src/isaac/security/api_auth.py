from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def validate_api_key(api_key: str = Security(api_key_header)):
    """Validate the ``Authorization`` header against ``ISAAC_API_KEY``."""
    from isaac.config.settings import get_settings

    expected_key = get_settings().api_key.strip()
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API proxy is disabled. Set ISAAC_API_KEY to enable it.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key missing. Please provide 'Authorization' header.",
        )

    actual_key = api_key[7:] if api_key.startswith("Bearer ") else api_key

    if not secrets.compare_digest(actual_key.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")

    return actual_key


from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from isaac.config.settings import settings

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)):
    \"\"\"
    Simple API key validation middleware.
    Validates the 'Authorization' header against ISAAC_API_KEY in settings.
    \"\"\"
    expected_key = getattr(settings, "ISAAC_API_KEY", None)
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key missing. Please provide 'Authorization' header."
        )
    
    # Standardize key by stripping 'Bearer ' prefix if present
    actual_key = api_key[7:] if api_key.startswith("Bearer ") else api_key
    
    if expected_key and actual_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )
            
    return actual_key


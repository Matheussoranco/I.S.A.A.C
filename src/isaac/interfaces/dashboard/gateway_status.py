"""Live log viewer and status for connected gateways."""
from __future__ import annotations
from fastapi import HTTPException
from isaac.interfaces.gateway_dispatcher import load_enabled_gateways
from isaac.config.settings import settings

def get_gateway_status() -> dict:
    """List enabled gateways and their basic status."""
    try:
        gateways = load_enabled_gateways(settings)
        return {
            "enabled": [g.name for g in gateways],
            "count": len(gateways)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch gateway status: {e}")

"""Status for configured gateways."""

from __future__ import annotations

import os
from typing import Any

from isaac.interfaces import gateway_dispatcher
from isaac.interfaces.gateway_base import Gateway


def get_gateway_status() -> dict[str, Any]:
    """List configured gateways without failing on unconfigured ones."""
    raw = os.environ.get("ISAAC_GATEWAYS", gateway_dispatcher.DEFAULT_GATEWAYS)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    entries: list[dict[str, Any]] = []
    for name in names:
        cls = Gateway.GATEWAY_REGISTRY.get(name)
        if cls is None:
            entries.append({"name": name, "configured": False, "error": "unknown gateway"})
            continue
        try:
            gateway = cls()
        except Exception as exc:
            entries.append({"name": name, "configured": False, "error": str(exc)})
        else:
            entries.append({"name": name, "configured": True, "running": gateway.running})
    enabled = [entry["name"] for entry in entries if entry["configured"]]
    return {"enabled": enabled, "count": len(enabled), "gateways": entries}

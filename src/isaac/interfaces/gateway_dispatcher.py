"""Gateway dispatcher — load and run enabled gateways from configuration.

Selection
---------
The ``ISAAC_GATEWAYS`` environment variable holds a comma-separated list of
gateway names (default: ``telegram``).  Each name must be present in
``Gateway.GATEWAY_REGISTRY``, populated by importing the gateway modules
(which self-register via ``@Gateway.register``).
"""

from __future__ import annotations

import asyncio
import logging
import os

from isaac.interfaces import discord_gateway as _discord_gateway  # noqa: F401
from isaac.interfaces import telegram_gateway as _telegram_gateway  # noqa: F401
from isaac.interfaces.gateway_base import AgentRunner, Gateway

logger = logging.getLogger(__name__)

DEFAULT_GATEWAYS = "telegram"


def load_enabled_gateways(
    settings: object | None = None,
    *,
    agent_runner: AgentRunner | None = None,
) -> list[Gateway]:
    """Instantiate every gateway named in ``ISAAC_GATEWAYS``.

    ``settings`` is accepted for future per-gateway configuration and may be
    ``None``.  Unknown names raise ValueError listing the known gateways.
    """
    raw = os.environ.get("ISAAC_GATEWAYS", DEFAULT_GATEWAYS)
    names = [n.strip() for n in raw.split(",") if n.strip()]

    gateways: list[Gateway] = []
    for name in names:
        cls = Gateway.GATEWAY_REGISTRY.get(name)
        if cls is None:
            known = ", ".join(sorted(Gateway.GATEWAY_REGISTRY)) or "<none>"
            raise ValueError(f"Unknown gateway {name!r} in ISAAC_GATEWAYS. Known gateways: {known}")
        gateways.append(cls(agent_runner=agent_runner))
    return gateways


async def run_all(gateways: list[Gateway]) -> None:
    """Run every gateway concurrently until cancelled, then stop them all."""
    if not gateways:
        logger.warning("No gateways enabled — nothing to run.")
        return

    async def _run(gateway: Gateway) -> None:
        try:
            await gateway.start()
        except Exception:
            logger.exception("Gateway %s crashed", gateway.name)
            raise

    tasks = [asyncio.create_task(_run(g), name=f"gateway:{g.name}") for g in gateways]
    try:
        await asyncio.gather(*tasks)
    finally:
        for g in gateways:
            try:
                await g.stop()
            except Exception:
                logger.debug("Error stopping gateway %s", g.name, exc_info=True)

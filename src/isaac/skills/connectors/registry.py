"""Connector Registry — auto-discover, manage, and audit connectors.

Provides :func:`get_registry` which lazily imports every
:class:`~isaac.skills.connectors.base.BaseConnector` subclass from the
``isaac.skills.connectors`` package, checks availability, and exposes
them as a dict keyed by connector name.

Audit logging goes to ``~/.isaac/connector_audit.log``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaac.skills.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_AUDIT_LOG: Path | None = None
_registry: dict[str, BaseConnector] | None = None


def _audit_path() -> Path:
    try:
        from isaac.config.settings import get_settings

        home = get_settings().isaac_home
    except Exception:
        home = Path.home() / ".isaac"
    path = home / "connector_audit.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def audit_connector(connector_name: str, action: str, detail: str = "") -> None:
    """Append a timestamped entry to the connector audit log."""
    try:
        path = _audit_path()
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts}  connector={connector_name}  action={action}"
        if detail:
            line += f"  detail={detail}"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.debug("Audit write failed: %s", exc)


def _discover_connectors() -> dict[str, BaseConnector]:
    """Import all modules in ``isaac.skills.connectors`` and collect subclasses."""
    import isaac.skills.connectors as pkg

    registry: dict[str, BaseConnector] = {}

    for _finder, mod_name, _is_pkg in pkgutil.iter_modules(pkg.__path__):
        if mod_name in ("base", "registry", "__init__"):
            continue
        try:
            module = importlib.import_module(f"isaac.skills.connectors.{mod_name}")
        except Exception as exc:
            logger.warning("Could not import connector module %s: %s", mod_name, exc)
            continue

        for _attr_name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseConnector) and obj is not BaseConnector:
                instance = obj()
                registry[instance.name] = instance

    return registry


def get_registry() -> dict[str, BaseConnector]:
    """Return the connector registry (lazily discovered)."""
    global _registry
    if _registry is None:
        _registry = _discover_connectors()
    return _registry


def reset_registry() -> None:
    """Force re-discovery on next :func:`get_registry` call."""
    global _registry
    _registry = None


def get_available_connectors() -> dict[str, BaseConnector]:
    """Return only connectors whose env requirements are satisfied."""
    return {k: v for k, v in get_registry().items() if v.is_available()}


def list_connector_schemas() -> list[dict[str, Any]]:
    """Return JSON-serialisable schemas for all registered connectors."""
    return [c.to_schema() for c in get_registry().values()]


def run_connector(name: str, **kwargs: Any) -> dict[str, Any]:
    """Run a connector by name with audit logging.

    Returns the connector result dict, or an error dict if the connector
    is not found / not available.
    """
    reg = get_registry()
    connector = reg.get(name)
    if connector is None:
        return {"error": f"Unknown connector: {name}. Available: {sorted(reg)}"}
    if not connector.is_available():
        missing = [e for e in connector.requires_env if not __import__("os").environ.get(e)]
        return {"error": f"Connector '{name}' unavailable — missing env: {missing}"}

    audit_connector(name, "invoke", str(kwargs)[:200])
    try:
        result = connector.run(**kwargs)
        audit_connector(name, "success", str(result)[:200])
        return result
    except Exception as exc:
        audit_connector(name, "error", str(exc)[:200])
        return {"error": str(exc)}

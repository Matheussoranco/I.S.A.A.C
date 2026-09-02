"""ConnectorExecution Node — runs host-side connectors between Planner and Synthesis.

This node inspects the current plan to determine if any external-world
connectors (web search, GitHub, filesystem, etc.) should be invoked
*before* Synthesis generates code.  Results are placed in
``state["connector_results"]`` for Synthesis to reference.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.core.state import IsaacState, PlanStep

logger = logging.getLogger(__name__)

# These connector actions are read-only in this node's fixed argument mapper.
# Shell is intentionally absent: it needs an operator-issued token.
_AUTO_AUTHORIZED_CONNECTORS = frozenset(
    {"web_search", "web_fetch", "github", "filesystem", "obsidian", "calendar", "email"}
)

# Keywords that hint the plan step needs a connector
_CONNECTOR_HINTS: dict[str, str] = {
    "search": "web_search",
    "look up": "web_search",
    "find online": "web_search",
    "google": "web_search",
    "fetch": "web_fetch",
    "download": "web_fetch",
    "scrape": "web_fetch",
    "github": "github",
    "repository": "github",
    "issue": "github",
    "read file": "filesystem",
    "write file": "filesystem",
    "list files": "filesystem",
    "obsidian": "obsidian",
    "vault": "obsidian",
    "note": "obsidian",
    "calendar": "calendar",
    "event": "calendar",
    "schedule": "calendar",
    "email": "email",
    "inbox": "email",
    "shell": "shell",
    "run command": "shell",
    "terminal": "shell",
}


def _detect_connectors(description: str) -> list[str]:
    """Return connector names that match keywords in the step description."""
    lower = description.lower()
    found: list[str] = []
    for keyword, connector in _CONNECTOR_HINTS.items():
        if keyword in lower and connector not in found:
            found.append(connector)
    return found


def _extract_kwargs_from_description(connector_name: str, description: str) -> dict[str, Any]:
    """Best-effort extraction of kwargs from the plan step description."""
    kwargs: dict[str, Any] = {}

    if connector_name == "web_search":
        # Use the full description as the query
        kwargs["query"] = description[:200]
        kwargs["max_results"] = 5

    elif connector_name == "web_fetch":
        # Look for URLs in the description
        import re

        urls = re.findall(r"https?://[^\s\"'>]+", description)
        if urls:
            kwargs["url"] = urls[0]

    elif connector_name == "github":
        kwargs["action"] = "list_repos"

    elif connector_name == "filesystem":
        kwargs["action"] = "list_directory"
        try:
            from isaac.config.settings import get_settings

            kwargs["path"] = str(get_settings().allowed_paths[0])
        except Exception:
            kwargs["path"] = "."

    elif connector_name == "obsidian":
        kwargs["action"] = "list"

    elif connector_name == "calendar":
        kwargs["action"] = "read"

    elif connector_name == "email":
        kwargs["action"] = "list"
        kwargs["limit"] = 5

    elif connector_name == "shell":
        kwargs["command"] = "echo 'connector probe'"

    return kwargs


def connector_execution_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: ConnectorExecution.

    Scans the active PlanStep's description for connector hints.  For each
    matched connector that is available, runs it and collects results.
    Results are appended to ``connector_results`` for downstream nodes.
    """
    from isaac.skills.connectors.registry import get_available_connectors, run_connector

    plan: list[PlanStep] = state.get("plan", [])
    active = next((s for s in plan if s.status == "active"), None)
    if active is None:
        logger.debug("ConnectorExecution: no active step — skipping.")
        return {}

    available = get_available_connectors()
    if not available:
        logger.debug("ConnectorExecution: no connectors available — skipping.")
        return {}

    detected = _detect_connectors(active.description)
    if not detected:
        logger.debug("ConnectorExecution: no connector hints in step '%s'.", active.description)
        return {}

    results: list[dict[str, Any]] = []
    for connector_name in detected:
        if connector_name not in available:
            logger.debug("ConnectorExecution: '%s' detected but not available.", connector_name)
            continue

        capability_token = ""
        try:
            from isaac.security.capabilities import get_token_store

            store = get_token_store()
            matching = next(
                (t for t in store.list_active() if t.matches(connector_name, "execute")),
                None,
            )
            if matching is not None:
                capability_token = matching.token_id
            elif connector_name in _AUTO_AUTHORIZED_CONNECTORS:
                token = store.issue(
                    connector_name,
                    action="execute",
                    ttl_hours=1 / 60,
                    issued_by="connector_execution:read_only",
                    max_uses=1,
                )
                capability_token = token.token_id
            else:
                results.append(
                    {
                        "connector": connector_name,
                        "step_id": active.id,
                        "kwargs": {},
                        "result": {
                            "error": (
                                f"Connector '{connector_name}' requires an operator-issued "
                                "capability token."
                            )
                        },
                    }
                )
                continue
        except Exception as cap_exc:
            logger.error("ConnectorExecution: capability authorization failed: %s", cap_exc)
            results.append(
                {
                    "connector": connector_name,
                    "step_id": active.id,
                    "kwargs": {},
                    "result": {"error": "Connector authorization failed."},
                }
            )
            continue

        kwargs = _extract_kwargs_from_description(connector_name, active.description)
        logger.info("ConnectorExecution: invoking '%s' with %s", connector_name, kwargs)
        result = run_connector(connector_name, capability_token=capability_token, **kwargs)
        results.append(
            {
                "connector": connector_name,
                "step_id": active.id,
                "kwargs": kwargs,
                "result": result,
            }
        )

    if results:
        logger.info("ConnectorExecution: %d connector(s) returned results.", len(results))

        # Put compact, provenance-labelled connector observations into the
        # WorldModel consumed by Synthesis.  Previously connector_results were
        # populated but the sequential synthesis prompt never saw them.
        world_model = state.get("world_model")
        if world_model is not None:
            for item in results[-5:]:
                world_model.observations.append(
                    f"[UNTRUSTED CONNECTOR {item['connector']}] {str(item['result'])[:1000]}"
                )
            return {"connector_results": results, "world_model": world_model}

    return {"connector_results": results}

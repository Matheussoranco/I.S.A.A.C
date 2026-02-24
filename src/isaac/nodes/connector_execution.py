"""ConnectorExecution Node — runs host-side connectors between Planner and Synthesis.

This node inspects the current plan to determine if any external-world
connectors (web search, GitHub, filesystem, etc.) should be invoked
*before* Synthesis generates code.  Results are placed in
``state["connector_results"]`` for Synthesis to reference.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from isaac.core.state import IsaacState, PlanStep

logger = logging.getLogger(__name__)

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
        kwargs["action"] = "list"
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

        kwargs = _extract_kwargs_from_description(connector_name, active.description)
        logger.info("ConnectorExecution: invoking '%s' with %s", connector_name, kwargs)
        result = run_connector(connector_name, **kwargs)
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

    return {"connector_results": results}

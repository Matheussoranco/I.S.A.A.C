"""Connector graph-node authorization and provenance regressions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from isaac.core.state import PlanStep, make_initial_state
from isaac.nodes.connector_execution import connector_execution_node


def test_read_only_connector_result_reaches_world_model() -> None:
    state = make_initial_state()
    state["plan"] = [PlanStep(id="s1", description="search for release notes", status="active")]
    connector = MagicMock()

    with (
        patch(
            "isaac.skills.connectors.registry.get_available_connectors",
            return_value={"web_search": connector},
        ),
        patch(
            "isaac.skills.connectors.registry.run_connector",
            return_value={"results": [{"title": "release"}]},
        ) as run,
    ):
        result = connector_execution_node(state)

    assert run.call_args.kwargs["capability_token"]
    observations = result["world_model"].observations
    assert observations[-1].startswith("[UNTRUSTED CONNECTOR web_search]")


def test_shell_hint_requires_operator_token() -> None:
    state = make_initial_state()
    state["plan"] = [PlanStep(id="s1", description="run command in shell", status="active")]
    connector = MagicMock()

    with (
        patch(
            "isaac.skills.connectors.registry.get_available_connectors",
            return_value={"shell": connector},
        ),
        patch("isaac.skills.connectors.registry.run_connector") as run,
    ):
        result = connector_execution_node(state)

    run.assert_not_called()
    error = result["connector_results"][0]["result"]["error"]
    assert "operator-issued" in error

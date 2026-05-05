"""Sandbox Node — executes code in Docker and populates execution_logs.

This node contains **zero** LLM calls.  Its only job is to delegate the
``code_buffer`` to the :class:`~isaac.sandbox.executor.CodeExecutor` and
record the raw output.
"""

from __future__ import annotations

import logging
from typing import Any

from isaac.core.state import IsaacState
from isaac.sandbox.executor import CodeExecutor

logger = logging.getLogger(__name__)


def sandbox_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: Sandbox.

    Takes ``code_buffer`` from the state, executes it inside an ephemeral
    Docker container, and appends the ``ExecutionResult`` to
    ``execution_logs``.
    """
    code = state.get("code_buffer", "")
    if not code.strip():
        logger.warning("Sandbox: empty code_buffer — skipping execution.")
        from isaac.core.state import ExecutionResult

        return {
            "execution_logs": [
                ExecutionResult(
                    stdout="",
                    stderr="Empty code buffer — nothing executed.",
                    exit_code=1,
                    duration_ms=0.0,
                )
            ],
            "current_phase": "sandbox",
        }

    # Constitutional safety review — block hard violations before docker exec.
    try:
        from isaac.security.constitution import review

        decision = review(
            action_kind="sandbox_code",
            action=code,
            context={"phase": "sandbox", "code_len": len(code)},
            use_llm=False,
        )
        if not decision.allow:
            from isaac.core.state import ExecutionResult

            logger.warning(
                "Sandbox: constitution blocked execution — %s",
                decision.reason or "policy violation",
            )
            return {
                "execution_logs": [
                    ExecutionResult(
                        stdout="",
                        stderr=(
                            "[constitution] Action blocked: "
                            + (decision.reason or "policy violation")
                            + " | violations: "
                            + ", ".join(v.rule for v in decision.violations)
                        ),
                        exit_code=126,
                        duration_ms=0.0,
                    )
                ],
                "current_phase": "sandbox",
                "constitution_decision": decision.to_dict(),
            }
    except Exception as exc:
        logger.debug("Constitution review skipped (%s)", exc)

    executor = CodeExecutor()
    try:
        result = executor.execute(code)
    finally:
        executor.close()

    logger.info(
        "Sandbox: exit_code=%d  stdout=%d chars  stderr=%d chars",
        result.exit_code,
        len(result.stdout),
        len(result.stderr),
    )

    return {
        "execution_logs": [result],
        "current_phase": "sandbox",
    }

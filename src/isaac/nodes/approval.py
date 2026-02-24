"""AwaitApproval node — pauses the graph until an operator approves or rejects.

When a tool with ``requires_approval=True`` is invoked the Synthesis node
appends a :class:`PendingApproval` to the state.  The AwaitApproval node
polls for resolution (via Telegram or CLI depending on interface) and
either proceeds with tool execution or records a rejection error.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from isaac.core.state import IsaacState, PendingApproval, ErrorEntry

logger = logging.getLogger(__name__)

# Maximum seconds to wait for an approval before auto-rejecting.
_APPROVAL_TIMEOUT = 300  # 5 minutes


async def _poll_approval(approval: PendingApproval, timeout: float = _APPROVAL_TIMEOUT) -> bool:
    """Wait until the approval is resolved or times out.

    In a real deployment the Telegram gateway or CLI interface mutates the
    ``PendingApproval.approved`` field on the shared object.  This loop
    checks every second.

    Returns ``True`` if approved, ``False`` otherwise.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if approval.approved is not None:
            return approval.approved
        await asyncio.sleep(1.0)

    # Timed out — auto-reject
    approval.approved = False
    approval.resolved_by = "timeout"
    return False


def await_approval_node(state: IsaacState) -> dict[str, Any]:
    """LangGraph node: pause graph for pending approvals.

    For each unresolved :class:`PendingApproval`:
    1. Notify the operator (via Telegram if available, else console).
    2. Wait for ``approved`` to flip to ``True`` or ``False``.
    3. If approved, execute the tool and return the result.
    4. If rejected, record an error.

    Returns a partial state update.
    """
    pending = [a for a in state.get("pending_approvals", []) if a.approved is None]

    if not pending:
        return {"current_phase": "await_approval"}

    errors: list[ErrorEntry] = []
    execution_outputs: list[str] = []

    for approval in pending:
        # Notify operator
        _notify_operator(approval)

        # Block until resolved (sync wrapper around async poll)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're likely inside an async runner — use the event loop directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    approved = pool.submit(_sync_poll, approval).result()
            else:
                approved = loop.run_until_complete(_poll_approval(approval))
        except RuntimeError:
            # No event loop — create one
            approved = asyncio.run(_poll_approval(approval))

        if approved:
            logger.info("Approval granted for tool '%s'.", approval.tool_name)
            result = _execute_approved_tool(approval)
            execution_outputs.append(result)
        else:
            reason = f"Tool '{approval.tool_name}' rejected by {approval.resolved_by or 'timeout'}."
            logger.warning(reason)
            errors.append(
                ErrorEntry(
                    node="await_approval",
                    message=reason,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

    update: dict[str, Any] = {"current_phase": "await_approval"}
    if errors:
        update["errors"] = errors
    return update


def _sync_poll(approval: PendingApproval) -> bool:
    """Synchronous wrapper for the async poll — runs a new event loop."""
    return asyncio.run(_poll_approval(approval))


def _notify_operator(approval: PendingApproval) -> None:
    """Send an approval request to the operator.

    Tries Telegram first, falls back to console print.
    """
    message = (
        f"🔐 Approval Required\n"
        f"Tool: {approval.tool_name}\n"
        f"Risk: {approval.risk_level}/5\n"
        f"Reason: {approval.reason}\n"
        f"Args: {approval.tool_args}\n\n"
        f"Reply /approve or /reject"
    )

    try:
        from isaac.interfaces.telegram_gateway import send_notification
        send_notification(message)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Telegram notification failed: %s", exc)

    # Always print to console as fallback
    print(f"\n{'='*60}")
    print(message)
    print(f"{'='*60}\n")


def _execute_approved_tool(approval: PendingApproval) -> str:
    """Execute the approved tool and return a summary string."""
    try:
        from isaac.tools.base import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get(approval.tool_name)

        if tool is None:
            return f"Tool '{approval.tool_name}' not found in registry."

        result = asyncio.run(tool.execute(**approval.tool_args))
        if result.success:
            return f"Tool '{approval.tool_name}' executed: {result.output}"
        else:
            return f"Tool '{approval.tool_name}' failed: {result.error}"
    except Exception as exc:
        logger.error("Failed to execute approved tool '%s': %s", approval.tool_name, exc)
        return f"Execution error: {exc}"

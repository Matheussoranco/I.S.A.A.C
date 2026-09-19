"""Telemetry decorator — wraps any cognitive node so per-run metrics are
captured and forwarded to the performance tracker.

Usage::

    @track_node("perception")
    def perception_node(state): ...

The decorator captures:
* duration in milliseconds
* observed outcome (True / False / unknown), separate from task correctness
* iteration & session id (read from state)
* error message (truncated to 500 chars)

It is fail-safe — telemetry errors are swallowed so they never crash the graph.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def observed_success(value: Any) -> bool | None:
    """Extract an explicit execution outcome; normal return alone is unknown."""
    if isinstance(value, Mapping):
        if value.get("errors") or value.get("error"):
            return False
        if "execution_logs" in value:
            logs = value["execution_logs"]
            outcomes = [observed_success(log) for log in logs]
            if False in outcomes:
                return False
            return True if outcomes and all(v is True for v in outcomes) else None
        if "exit_code" in value:
            return value["exit_code"] == 0
        if isinstance(value.get("success"), bool):
            return value["success"]
    if hasattr(value, "exit_code"):
        return bool(value.exit_code == 0)
    if hasattr(value, "success") and isinstance(value.success, bool):
        return value.success
    return None


def track_node(node_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a node function with timing + success tracking."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.monotonic()
            err = ""
            success: bool | None = None
            state = args[0] if args and isinstance(args[0], Mapping) else {}
            try:
                result = func(*args, **kwargs)
                success = observed_success(result)
                return result
            except Exception as exc:
                success = False
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    from isaac.improvement.performance import get_tracker

                    duration_ms = (time.monotonic() - start) * 1000.0
                    iteration = int(state.get("iteration", 0))
                    session_id = str(state.get("session_id", ""))
                    get_tracker().record_node(
                        node=node_name,
                        duration_ms=duration_ms,
                        success=success,
                        iteration=iteration,
                        session_id=session_id,
                        error=err,
                    )
                except Exception:  # never let telemetry break the graph
                    pass

        return wrapper

    return decorator


def track_skill(skill_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a skill callable with success / duration tracking."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            err = ""
            success: bool | None = None
            try:
                result = func(*args, **kwargs)
                success = observed_success(result)
                return result
            except Exception as exc:
                success = False
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    from isaac.improvement.performance import get_tracker

                    duration_ms = (time.monotonic() - start) * 1000.0
                    get_tracker().record_skill(
                        skill_name=skill_name,
                        duration_ms=duration_ms,
                        success=success,
                        error=err,
                    )
                except Exception:
                    pass

        return wrapper

    return decorator

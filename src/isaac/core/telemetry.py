"""Telemetry decorator — wraps any cognitive node so per-run metrics are
captured and forwarded to the performance tracker.

Usage::

    @track_node("perception")
    def perception_node(state): ...

The decorator captures:
* duration in milliseconds
* success (no exception raised)
* iteration & session id (read from state)
* error message (truncated to 500 chars)

It is fail-safe — telemetry errors are swallowed so they never crash the graph.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def track_node(node_name: str) -> Callable[[F], F]:
    """Wrap a node function with timing + success tracking."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(state: dict[str, Any]) -> Any:
            start = time.monotonic()
            err = ""
            success = True
            try:
                return func(state)
            except Exception as exc:
                success = False
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                try:
                    from isaac.improvement.performance import get_tracker
                    duration_ms = (time.monotonic() - start) * 1000.0
                    iteration = int(state.get("iteration", 0)) if isinstance(state, dict) else 0
                    session_id = (
                        str(state.get("session_id", "")) if isinstance(state, dict) else ""
                    )
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

        return wrapper  # type: ignore[return-value]

    return decorator


def track_skill(skill_name: str) -> Callable[[F], F]:
    """Wrap a skill callable with success / duration tracking."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            err = ""
            success = True
            try:
                return func(*args, **kwargs)
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

        return wrapper  # type: ignore[return-value]

    return decorator

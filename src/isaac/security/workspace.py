"""Per-run filesystem scope, propagated explicitly to worker threads."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_workspace: ContextVar[Path | None] = ContextVar("isaac_workspace", default=None)


def current_workspace() -> Path | None:
    return _workspace.get()


@contextmanager
def workspace_scope(root: Path) -> Iterator[Path]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = _workspace.set(root)
    try:
        yield root
    finally:
        _workspace.reset(token)


def allowed_roots() -> list[Path]:
    scoped = current_workspace()
    if scoped is not None:
        return [scoped]
    from isaac.config.settings import get_settings

    # Empty configuration must never widen access to the user's home.
    return [Path(p).expanduser().resolve() for p in get_settings().allowed_paths]


def resolve_allowed(path: str | Path) -> Path | None:
    from isaac.security.path_policy import is_sensitive_path

    try:
        raw = Path(path).expanduser()
        if not str(path) or not raw.is_absolute():
            return None
        target = raw.resolve()
        if is_sensitive_path(raw) or is_sensitive_path(target):
            return None
        if any(target.is_relative_to(root) for root in allowed_roots()):
            return target
    except (OSError, RuntimeError, ValueError):
        pass
    return None

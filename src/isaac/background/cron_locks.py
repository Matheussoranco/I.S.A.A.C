"""Per-task cron tick locks with bounded staleness."""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_TASK_KEY_RE = re.compile(r"^[\w@-]{1,64}$")
_DEFAULT_STALE_SECONDS = 6 * 60 * 60


def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings

        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"


def _get_lock_path(task_id: str) -> Path | None:
    if not _TASK_KEY_RE.fullmatch(task_id):
        logger.warning("Refusing lock for invalid cron task id: %r", task_id)
        return None
    return _isaac_home() / f".tick_{task_id}.lock"


def _acquire_lock(task_id: str, stale_after: float = _DEFAULT_STALE_SECONDS) -> bool:
    """Create the task lock; reclaim it when its owner vanished long ago."""
    lock_path = _get_lock_path(task_id)
    if lock_path is None:
        return False
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{os.getpid()} {time.time():.0f}"
    for _ in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
            return True
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except OSError:
                return False
            if age <= stale_after:
                return False
            try:
                lock_path.unlink()
            except OSError:
                return False
        except OSError:
            return False
    return False


def _release_lock(task_id: str) -> None:
    """Release the task lock."""
    lock_path = _get_lock_path(task_id)
    if lock_path is not None:
        lock_path.unlink(missing_ok=True)

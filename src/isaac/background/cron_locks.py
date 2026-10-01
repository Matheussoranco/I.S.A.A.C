from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def _isaac_home() -> Path:
    try:
        from isaac.config.settings import get_settings
        return get_settings().isaac_home
    except Exception:
        return Path.home() / ".isaac"

def _get_lock_path(task_id: str) -> Path:
    return _isaac_home() / f".tick_{task_id}.lock"

def _acquire_lock(task_id: str) -> bool:
    """Attempts to acquire a lock for the task. Returns True if acquired."""
    lock_path = _get_lock_path(task_id)
    try:
        # Use 'x' mode to fail if file exists
        lock_path.touch(exist_ok=False)
        return True
    except FileExistsError:
        return False

def _release_lock(task_id: str) -> None:
    """Releases the lock for the task."""
    lock_path = _get_lock_path(task_id)
    lock_path.unlink(missing_ok=True)

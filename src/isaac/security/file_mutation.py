"""Publish complete files atomically; never truncate a destination on failure."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def process_file_lock(path: Path) -> Iterator[None]:
    """Take an exclusive lock on a stable sidecar file until the context exits."""
    with path.open("a+b") as lock:
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def publish_file(target: Path, writer: Callable[[BinaryIO], object], *, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".isaac-write-", dir=target.parent)
    staged = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            if target.is_dir():
                raise IsADirectoryError(str(target))
            os.replace(staged, target)
        else:
            # Atomic create-if-absent; a destination created after approval is
            # never replaced. Fail closed on filesystems without hard links.
            os.link(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def write_text(target: Path, content: str, *, overwrite: bool = False) -> None:
    data = content.encode("utf-8")
    publish_file(target, lambda stream: stream.write(data), overwrite=overwrite)


def copy_file(source: Path, target: Path, *, overwrite: bool = False) -> None:
    def write(stream: BinaryIO) -> None:
        with source.open("rb") as original:
            shutil.copyfileobj(original, stream)

    publish_file(target, write, overwrite=overwrite)

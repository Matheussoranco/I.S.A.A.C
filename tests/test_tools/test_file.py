"""Tests for the sandboxed workspace file tools."""

from __future__ import annotations

import asyncio
import os

import pytest

from isaac.config.settings import get_settings
from isaac.tools import file as file_module
from isaac.tools.file import FileDeleteTool, FileListTool, FileReadTool


def _run(coro):
    return asyncio.run(coro)


def _workspace():
    root = get_settings().isaac_home / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_read_is_bounded() -> None:
    target = _workspace() / "notes.txt"
    target.write_text("x" * 100, encoding="utf-8")

    result = _run(FileReadTool().execute(path="notes.txt", max_chars=10))

    assert result.success
    assert result.output.startswith("x" * 10)
    assert "truncated at 10 chars" in result.output


def test_list_truncates(monkeypatch) -> None:
    root = _workspace()
    for index in range(3):
        (root / f"{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(file_module, "_MAX_LIST_ENTRIES", 2)

    result = _run(FileListTool().execute(path="."))

    assert result.success
    assert result.output.count("[file]") == 2
    assert "truncated at 2 entries" in result.output


def test_delete_refuses_directories() -> None:
    target = _workspace() / "folder"
    target.mkdir()

    result = _run(FileDeleteTool().execute(path="folder"))

    assert result.success is False
    assert "Directory deletion" in result.error
    assert target.is_dir()


def test_delete_symlink_removes_only_the_link() -> None:
    root = _workspace()
    payload = root / "payload.txt"
    payload.write_text("keep me", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(payload)
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    result = _run(FileDeleteTool().execute(path="link.txt"))

    assert result.success, result.error
    assert not os.path.lexists(link)
    assert payload.read_text(encoding="utf-8") == "keep me"

"""Tests for the host-reach tools: path confinement and constitution gating.

These cover the *safety boundary* — that fs_* tools refuse paths outside the
allow-listed roots, and that the shell tool's constitutional gate hard-blocks
catastrophic commands.
"""

from __future__ import annotations

import asyncio

import pytest

from isaac.config.settings import get_settings
from isaac.tools.fileops import FsListTool, FsMoveTool, FsWriteTool
from isaac.tools.shell import ShellTool
from isaac.tools.system import SystemInfoTool


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def allowed_root(tmp_path, monkeypatch):
    """Confine the fs_* tools to a temp directory."""
    monkeypatch.setattr(get_settings(), "allowed_paths", [str(tmp_path)], raising=False)
    return tmp_path


# ── fileops confinement ────────────────────────────────────────────────────


def test_fs_write_and_list_within_root(allowed_root) -> None:
    target = allowed_root / "notes" / "a.txt"
    res = _run(FsWriteTool().execute(path=str(target), content="hello"))
    assert res.success, res.error
    assert target.read_text(encoding="utf-8") == "hello"

    listed = _run(FsListTool().execute(path=str(allowed_root), recursive=True))
    assert listed.success
    assert "a.txt" in listed.output


def test_fs_write_outside_root_is_denied(allowed_root, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "x.txt"
    res = _run(FsWriteTool().execute(path=str(outside), content="nope"))
    assert res.success is False
    assert "allowed roots" in res.error
    assert not outside.exists()


def test_fs_move_refuses_overwrite(allowed_root) -> None:
    src = allowed_root / "a.txt"
    dest = allowed_root / "b.txt"
    src.write_text("one", encoding="utf-8")
    dest.write_text("two", encoding="utf-8")
    res = _run(FsMoveTool().execute(src=str(src), dest=str(dest)))
    assert res.success is False
    assert "overwrite" in res.error.lower()
    # With overwrite it succeeds.
    res2 = _run(FsMoveTool().execute(src=str(src), dest=str(dest), overwrite=True))
    assert res2.success
    assert dest.read_text(encoding="utf-8") == "one"


# ── shell gating ────────────────────────────────────────────────────────────


def test_shell_is_high_risk_and_needs_approval() -> None:
    tool = ShellTool()
    assert tool.risk_level == 4
    assert tool.requires_approval is True


def test_shell_blocks_critical_command() -> None:
    res = _run(ShellTool().execute(command="rm -rf /"))
    assert res.success is False
    assert "BLOCKED" in res.error


def test_shell_allowlist_rejects_unknown_command(monkeypatch) -> None:
    # Default (restricted) mode: a command not on the allow-list is rejected
    # without executing anything.
    monkeypatch.setattr(get_settings(), "shell_unrestricted", False, raising=False)
    res = _run(ShellTool().execute(command="totallybogus_xyz --do-thing"))
    assert res.success is False
    assert "allowlist" in res.error.lower()


# ── system info ─────────────────────────────────────────────────────────────


def test_system_info_reports_os() -> None:
    res = _run(SystemInfoTool().execute())
    assert res.success
    assert "OS:" in res.output
    assert res.metadata.get("cpu_count", 0) >= 0

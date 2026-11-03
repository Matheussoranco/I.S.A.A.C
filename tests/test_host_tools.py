"""Tests for the host-reach tools: path confinement and constitution gating.

These cover the *safety boundary* — that fs_* tools refuse paths outside the
allow-listed roots, and that the shell tool's constitutional gate hard-blocks
catastrophic commands.
"""

from __future__ import annotations

import asyncio

import pytest

from isaac.config.settings import get_settings
from isaac.tools.fileops import FsCopyTool, FsListTool, FsMoveTool, FsReadTool, FsWriteTool
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


def test_recursive_move_refuses_protected_descendant(allowed_root) -> None:
    src = allowed_root / "source"
    src.mkdir()
    (src / ".env").write_text("SECRET=1", encoding="utf-8")
    dest = allowed_root / "dest"

    res = _run(FsMoveTool().execute(src=str(src), dest=str(dest)))

    assert res.success is False
    assert "protected descendant" in res.error.lower()
    assert src.exists()
    assert not dest.exists()


def test_recursive_copy_refuses_protected_descendant(allowed_root) -> None:
    src = allowed_root / "source"
    src.mkdir()
    (src / ".ssh").mkdir()
    (src / ".ssh" / "id_rsa").write_text("PRIVATE", encoding="utf-8")
    dest = allowed_root / "dest"

    res = _run(FsCopyTool().execute(src=str(src), dest=str(dest)))

    assert res.success is False
    assert "protected descendant" in res.error.lower()
    assert not dest.exists()


def test_copy_into_source_tree_is_refused(allowed_root) -> None:
    src = allowed_root / "source"
    src.mkdir()
    (src / "a.txt").write_text("one", encoding="utf-8")

    res = _run(FsCopyTool().execute(src=str(src), dest=str(src / "nested" / "copy")))

    assert res.success is False
    assert "inside the source" in res.error.lower()
    assert not (src / "nested").exists()


def test_move_into_source_tree_is_refused(allowed_root) -> None:
    src = allowed_root / "source"
    src.mkdir()
    (src / "a.txt").write_text("one", encoding="utf-8")

    res = _run(FsMoveTool().execute(src=str(src), dest=str(src / "nested")))

    assert res.success is False
    assert "descendant" in res.error.lower()
    assert src.exists()


def test_move_directory_falls_back_across_filesystems(allowed_root, monkeypatch) -> None:
    import errno
    from pathlib import Path

    src = allowed_root / "source"
    src.mkdir()
    (src / "a.txt").write_text("one", encoding="utf-8")
    dest = allowed_root / "dest"

    def _cross_device(self, target):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(Path, "rename", _cross_device)
    res = _run(FsMoveTool().execute(src=str(src), dest=str(dest)))

    assert res.success, res.error
    assert not src.exists()
    assert (dest / "a.txt").read_text(encoding="utf-8") == "one"


def test_move_symlink_moves_link_not_target(allowed_root) -> None:
    import os

    target = allowed_root / "target.txt"
    target.write_text("payload", encoding="utf-8")
    link = allowed_root / "link.txt"
    moved = allowed_root / "moved-link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    res = _run(FsMoveTool().execute(src=str(link), dest=str(moved)))

    assert res.success, res.error
    assert not os.path.lexists(link)
    assert moved.is_symlink()
    assert target.read_text(encoding="utf-8") == "payload"


def test_fs_read_is_bounded(allowed_root) -> None:
    target = allowed_root / "big.txt"
    target.write_text("x" * 5_000, encoding="utf-8")

    res = _run(FsReadTool().execute(path=str(target), max_chars=10))

    assert res.success
    assert res.output.startswith("x" * 10)
    assert "truncated at 10 chars" in res.output


# ── sensitive-path deny list ────────────────────────────────────────────────


def test_fs_read_denies_ssh_keys_inside_allowed_root(allowed_root) -> None:
    secret = allowed_root / ".ssh" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("PRIVATE KEY", encoding="utf-8")
    res = _run(FsReadTool().execute(path=str(secret)))
    assert res.success is False
    assert "protected" in res.error


def test_fs_write_denies_env_files(allowed_root) -> None:
    for name in (".env", ".env.local"):
        res = _run(FsWriteTool().execute(path=str(allowed_root / name), content="KEY=1"))
        assert res.success is False, name
        assert not (allowed_root / name).exists()


def test_fs_read_denies_key_material_suffixes(allowed_root) -> None:
    cert = allowed_root / "server.pem"
    cert.write_text("CERT", encoding="utf-8")
    res = _run(FsReadTool().execute(path=str(cert)))
    assert res.success is False


def test_fs_list_skips_sensitive_entries(allowed_root) -> None:
    (allowed_root / ".aws").mkdir()
    (allowed_root / ".aws" / "credentials").write_text("secret", encoding="utf-8")
    (allowed_root / "normal.txt").write_text("hi", encoding="utf-8")
    res = _run(FsListTool().execute(path=str(allowed_root), recursive=True))
    assert res.success
    assert "normal.txt" in res.output
    assert ".aws" not in res.output
    assert "credentials" not in res.output


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

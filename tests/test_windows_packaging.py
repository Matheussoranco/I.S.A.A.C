"""Regression checks for the sendable Windows desktop package."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_windows_build_produces_a_single_file_executable() -> None:
    script = (_ROOT / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    assert "--onefile" in script
    assert '"dist\\$ExecutableName.exe"' in script
    assert "Compress-Archive" not in script


def test_windows_installer_accepts_the_standalone_executable() -> None:
    script = (_ROOT / "scripts" / "install_windows.ps1").read_text(encoding="utf-8")

    assert "$SourceIsExecutable" in script
    assert 'Join-Path $StagingPath "ISAAC.exe"' in script

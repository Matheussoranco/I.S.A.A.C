"""Console-free entry point used by the packaged Windows application."""

from __future__ import annotations

from isaac.interfaces.desktop_app import run_desktop_app


def main() -> int:
    """Launch the native app directly when ISAAC.exe is double-clicked."""
    return run_desktop_app()


if __name__ == "__main__":
    raise SystemExit(main())

"""Entry point for ``python -m isaac``.

Delegates to the Typer CLI if available, otherwise falls back to the
basic interactive REPL.
"""

from __future__ import annotations

import contextlib
import sys


def _ensure_utf8_streams() -> None:
    """Force stdout/stderr to UTF-8 so Rich/Typer help text with non-ASCII works on Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    """Bootstrap and run the I.S.A.A.C. CLI."""
    _ensure_utf8_streams()
    from isaac.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

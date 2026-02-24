"""Entry point for ``python -m isaac``.

Delegates to the Typer CLI if available, otherwise falls back to the
basic interactive REPL.
"""

from __future__ import annotations


def main() -> int:
    """Bootstrap and run the I.S.A.A.C. CLI."""
    from isaac.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())

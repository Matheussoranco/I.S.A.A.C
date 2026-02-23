"""Entry point for ``python -m isaac``."""

from __future__ import annotations


def main() -> int:
    """Bootstrap and run the I.S.A.A.C. cognitive loop."""
    from isaac.core.graph import build_and_run

    return build_and_run()


if __name__ == "__main__":
    raise SystemExit(main())

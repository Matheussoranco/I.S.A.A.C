"""I.S.A.A.C. CLI compatibility entry point."""

from .legacy import main

if __name__ == "__main__":
    raise SystemExit(main())

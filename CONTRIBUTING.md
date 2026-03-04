# Contributing to I.S.A.A.C.

Thank you for your interest in contributing! Below is everything you need to get started.

---

## Branch Model

| Branch | Purpose |
|--------|---------|
| `main` | Stable — tagged releases only. Never push directly. |
| `dev` | Active development. Open PRs against this branch. |
| `feat/<topic>` | Feature branches, cut from `dev`. |
| `fix/<topic>` | Bug-fix branches, cut from `dev` (or `main` for hotfixes). |

---

## Development Setup

```bash
git clone https://github.com/Matheussoranco/I.S.A.A.C.git
cd I.S.A.A.C
git checkout dev

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev,browser,calendar]"
```

---

## Running Tests

```bash
pytest tests/ --tb=short
```

Docker must be running for sandbox/execution tests (they will auto-skip if Docker is unavailable).

---

## Code Style

This project uses **Ruff** for linting and formatting, and **mypy** for type-checking.

```bash
# Lint
ruff check src/

# Format
ruff format src/

# Type check
mypy src/isaac --ignore-missing-imports
```

CI runs all three on every PR. Please ensure they pass locally before pushing.

Style notes:
- Line length: 100 characters
- Python 3.10+ syntax (use `X | Y` union types, `match` where appropriate)
- All public functions must have docstrings
- All tool `execute()` methods must be `async def`

---

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] Branch is based on `dev` (not `main`)
- [ ] `ruff check src/` passes with no errors
- [ ] `ruff format --check src/` passes
- [ ] `mypy src/isaac --ignore-missing-imports` passes (or you've documented why a suppression is needed)
- [ ] `pytest tests/` passes
- [ ] New features include at least one test
- [ ] Bug fixes include a regression test
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Docstrings added/updated for all changed public functions

---

## Reporting Issues

Use the GitHub issue templates:
- **Bug report** — for reproducible defects
- **Feature request** — for enhancements

For security vulnerabilities, see [SECURITY.md](SECURITY.md) — **do not open a public issue**.

---

## Areas Actively Seeking Contributions

- ARC-AGI: LLM-guided DSL synthesis beyond depth-2 brute force
- First-class Ollama provider (no base-URL workaround)
- Additional connector implementations (Notion, Slack, Linear)
- End-to-end integration tests for the full graph pipeline
- Web UI dashboard (FastAPI + WebSocket)

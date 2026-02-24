# I.S.A.A.C. — Setup Guide

> **Intelligent System for Autonomous Action and Cognition** v0.2.0

## Prerequisites

| Requirement       | Version   | Notes                                          |
| ----------------- | --------- | ---------------------------------------------- |
| Python            | ≥ 3.10    | 3.12 recommended                               |
| Docker            | ≥ 24.0    | For sandboxed code execution                   |
| Ollama (optional) | ≥ 0.3     | Local LLM — default provider                   |

## 1. Clone & Install

```bash
git clone https://github.com/<your-user>/I.S.A.A.C.git
cd I.S.A.A.C

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/macOS

# Install in editable mode with dev dependencies
pip install -e ".[dev,calendar,browser]"

# OR install from requirements.txt
pip install -r requirements.txt
```

## 2. Environment Variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

### Required

| Variable              | Description                              |
| --------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`      | OpenAI API key (if not using Ollama)     |
| `ANTHROPIC_API_KEY`   | Anthropic API key (optional fallback)    |

### Optional — Connectors

| Variable               | Connector       | Description                            |
| ---------------------- | --------------- | -------------------------------------- |
| `GITHUB_TOKEN`         | GitHub          | Personal access token for GitHub API   |
| `EMAIL_IMAP_HOST`      | Email           | IMAP server hostname                   |
| `EMAIL_USER`           | Email           | IMAP login username                    |
| `EMAIL_PASSWORD`       | Email           | IMAP login password                    |
| `EMAIL_IMAP_PORT`      | Email           | IMAP port (default: 993)              |
| `OBSIDIAN_VAULT_PATH`  | Obsidian        | Absolute path to your Obsidian vault   |

### Optional — Customisation

| Variable                         | Default                     | Description                              |
| -------------------------------- | --------------------------- | ---------------------------------------- |
| `ISAAC_AGENT_NAME`               | `I.S.A.A.C.`               | Display name                             |
| `ISAAC_SOUL_PATH`                | (built-in)                  | Path to custom soul JSON file            |
| `ISAAC_MEMORY_DB_PATH`           | `~/.isaac/long_term_memory.db` | SQLite LTM database path             |
| `ISAAC_USER_PROFILE_PATH`        | `~/.isaac/user_profile.json`   | User profile JSON path               |
| `ISAAC_MEMORY_CONSOLIDATION_INTERVAL` | `50`                   | Interactions between memory consolidation |
| `ISAAC_CRON_POLL_SECONDS`        | `30`                        | Cron daemon poll interval                |
| `ISAAC_CRON_ENABLED`             | `false`                     | Auto-start cron daemon                   |

## 3. Build Docker Sandbox Images

```bash
# Code execution sandbox
docker build -t isaac-sandbox:latest -f sandbox_image/Dockerfile sandbox_image/

# UI sandbox (optional — for browser/desktop automation)
docker build -t isaac-ui-sandbox:latest -f sandbox_image_ui/Dockerfile sandbox_image_ui/
```

## 4. Start Ollama (if using local LLM)

```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

## 5. Run I.S.A.A.C.

### Interactive REPL

```bash
python -m isaac run
# or
isaac run
```

### Example Session

```
I.S.A.A.C. — Intelligent System for Autonomous Action and Cognition
Type your task below.  Press Ctrl+C to exit.

>>> Write a Python function that computes the Fibonacci sequence

[I.S.A.A.C.] Here's a Fibonacci implementation...
  ─ exit_code: 0
  ─ stdout: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  ─ mode: code  phase: reflection

>>> Search the web for the latest Python 3.13 release notes

[I.S.A.A.C.] Based on my web search, Python 3.13 was released on...
  ─ mode: code  phase: reflection
```

### Telegram Bot

```bash
# Set TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USERS in .env
isaac serve
```

## 6. CLI Commands

| Command                | Description                              |
| ---------------------- | ---------------------------------------- |
| `isaac run`            | Start interactive REPL                   |
| `isaac serve`          | Start Telegram gateway + scheduler       |
| `isaac audit`          | View audit log                           |
| `isaac audit --verify` | Verify audit chain integrity             |
| `isaac memory "query"` | Query the unified memory system          |
| `isaac tools`          | List registered tools                    |
| `isaac connectors`     | List connectors and availability         |
| `isaac cron list`      | List cron tasks                          |
| `isaac cron add -c "echo hello" -s "*/5 * * * *"` | Add a cron task |
| `isaac cron start`     | Start the cron daemon                    |
| `isaac cron stop`      | Stop the cron daemon                     |
| `isaac cron status`    | Check daemon status                      |
| `isaac tokens list`    | Manage capability tokens                 |

## 7. Running Tests

```bash
pytest -v
pytest tests/test_identity.py -v
pytest tests/test_long_term_memory.py -v
pytest tests/test_connectors.py -v
pytest tests/test_cron_engine.py -v
```

## 8. Project Structure (v0.2.0)

```
src/isaac/
├── __init__.py                  # Package root, __version__
├── __main__.py                  # Entry point
├── cli.py                       # Typer CLI (run, serve, audit, cron, connectors, ...)
├── identity/
│   ├── __init__.py
│   └── soul.py                  # SOUL personality + loader
├── config/
│   └── settings.py              # Pydantic settings (all env vars)
├── core/
│   ├── state.py                 # IsaacState TypedDict
│   ├── graph.py                 # LangGraph StateGraph builder
│   └── transitions.py           # Conditional edge routing
├── llm/
│   ├── provider.py              # LLM factory (Ollama → OpenAI/Anthropic)
│   ├── router.py                # Tiered model routing
│   └── prompts.py               # Prompt templates (soul-injected)
├── memory/
│   ├── long_term.py             # SQLite FTS5 long-term memory
│   ├── user_profile.py          # JSON user profile
│   ├── episodic.py              # ChromaDB episodic memory
│   ├── semantic.py              # Semantic memory
│   ├── skill_library.py         # Skill embedding retrieval
│   ├── context_manager.py       # Message compression
│   ├── world_model.py           # World model graph
│   ├── world_model_kg.py        # Knowledge graph
│   ├── procedural.py            # Procedural memory
│   └── manager.py               # Unified memory manager
├── nodes/
│   ├── perception.py            # Perception (+ LTM/profile integration)
│   ├── explorer.py              # Explorer
│   ├── planner.py               # Graph-of-Thought planner
│   ├── connector_execution.py   # Host-side connector dispatch
│   ├── synthesis.py             # Code synthesis
│   ├── sandbox.py               # Docker sandbox execution
│   ├── computer_use.py          # UI automation executor
│   ├── reflection.py            # Reflection (+ LTM storage)
│   ├── skill_abstraction.py     # Skill extraction
│   ├── approval.py              # Human-in-the-loop approval
│   └── guard.py                 # Prompt injection guard
├── skills/
│   ├── __init__.py
│   └── connectors/
│       ├── __init__.py
│       ├── base.py              # BaseConnector ABC
│       ├── registry.py          # Auto-discovery + audit
│       ├── web_search.py        # DuckDuckGo search
│       ├── web_fetch.py         # HTTP page fetch + extract
│       ├── filesystem.py        # Safe file system access
│       ├── github.py            # GitHub REST API
│       ├── shell.py             # Allowlisted shell commands
│       ├── calendar.py          # iCal .ics read/write
│       ├── email_reader.py      # Read-only IMAP email
│       └── obsidian.py          # Obsidian vault access
├── background/
│   ├── __init__.py
│   └── cron_engine.py           # Cron daemon + task management
├── sandbox/                     # Docker sandbox management
├── security/                    # Audit, capabilities, guard
├── scheduler/                   # Heartbeat scheduler
├── interfaces/                  # Telegram gateway
├── tools/                       # Tool registry
└── arc/                         # ARC-AGI DSL

tests/
├── test_identity.py             # Soul module tests
├── test_long_term_memory.py     # LTM + user profile tests
├── test_connectors.py           # Connector + registry tests
├── test_cron_engine.py          # Cron engine tests
├── test_graph.py                # Graph integration tests
├── test_state.py                # State tests
└── ...                          # Existing test suites
```

## License

AGPL-3.0-or-later

# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

A neuro-symbolic autonomous agent built on [LangGraph](https://github.com/langchain-ai/langgraph) with Docker-sandboxed execution, a cumulative Skill Library, and a full security stack for ARC-AGI program synthesis and general-purpose autonomy.

---

## Architecture Overview

I.S.A.A.C. models reasoning as an explicit cyclic state graph — not generic while-loops. Ten cognitive nodes operate on a strict `TypedDict` state contract:

```
START ─► Guard ─► Perception ─► Explorer ─► Planner ─► Synthesis
                                                          │
                    ┌────── Skill Abstraction ◄────┐      ▼
                    │             │                 │   Sandbox / ComputerUse
                    ▼             ▼                 │      │
                 Planner        END            Reflection ◄┘
                                                   │
                                              AwaitApproval
```

| Node | Responsibility |
|---|---|
| **Guard** | Detect prompt injection and sanitize input before processing |
| **Perception** | Parse input, extract observations, build initial hypothesis |
| **Explorer** | Active exploration — ARC structural analysis + web search |
| **Planner** | Decompose task into a dependency-aware DAG (Graph-of-Thought) |
| **Synthesis** | Generate pure Python code (CodeAgent — no JSON tool calls) |
| **Sandbox** | Execute code in ephemeral Docker container (zero host access) |
| **ComputerUse** | GUI automation via virtual desktop (Xvfb + Playwright) |
| **Reflection** | Analyse results, attempt refinement loop, or escalate to Planner |
| **Skill Abstraction** | Generalise successful code into reusable library entries |
| **AwaitApproval** | Pause for human approval on high-risk tool invocations |

## Core Design Principles

- **Execution Isolation**: All environment interactions run in ephemeral, unprivileged Docker containers (`--network=none`, `--cap-drop=ALL`, `--read-only`, seccomp profile).
- **CodeAgent Paradigm**: The LLM generates pure Python — no JSON/XML tool calling. Code is injected into the sandbox, never executed on the host.
- **Ollama-First LLM Routing**: Prefers local Ollama models for privacy and speed. Falls back to OpenAI/Anthropic only when necessary.
- **Neuro-Symbolic Reasoning**: Structured state schema separates perception from representation. The `WorldModel` carries symbolic observations via a knowledge graph.
- **Three-Layer Memory**: Episodic (session experiences), Semantic (knowledge graph with transitive inference), and Procedural (versioned skill library with success tracking).
- **Cumulative Learning**: A persistent Skill Library stores generalised programs. The agent composes existing skills to solve novel tasks, reducing LLM calls over time.
- **Security-First**: Hash-chained audit log, capability tokens, prompt injection guard, I/O sanitization, seccomp sandboxing.
- **ARC-AGI Foundations**: Perception/representation separation + compositional Skill Library enable program synthesis for geometric/logical puzzles.

## Project Structure

```
I.S.A.A.C/
├── src/isaac/                  # Main package
│   ├── core/                   # State schema, graph builder, transitions
│   ├── nodes/                  # 10 cognitive graph nodes
│   │   ├── guard.py            # Prompt injection detection
│   │   ├── explorer.py         # Active exploration (ARC + web)
│   │   ├── got_planner.py      # Graph-of-Thought DAG planner
│   │   ├── refinement.py       # Synthesis→Sandbox tight loop
│   │   └── approval.py         # Human approval workflow
│   ├── memory/                 # Episodic, semantic KG, procedural, world model KG
│   ├── sandbox/                # Docker security, manager, code executor
│   ├── llm/                    # Provider factory, LLM router, prompt templates
│   ├── tools/                  # Secure tool ecosystem (browser, file, search, email, calendar, code)
│   ├── interfaces/             # Telegram gateway
│   ├── scheduler/              # APScheduler heartbeat + background jobs
│   ├── security/               # Audit log, capability tokens, sanitizer
│   ├── config/                 # Pydantic settings (env-driven)
│   ├── arc/                    # ARC-AGI DSL, grid ops, evaluator
│   └── cli.py                  # Typer CLI entry point
├── sandbox_image/              # Dockerfile for code execution sandbox
├── sandbox_image_ui/           # Dockerfile for virtual desktop sandbox
├── skills/                     # Persistent skill library (JSON + .py)
├── tests/                      # Full test suite with mocked LLM/Docker
├── docker-compose.yml          # Full-stack deployment
├── Dockerfile                  # Agent container image
├── pyproject.toml              # PEP 621 metadata + tooling config
└── requirements.txt            # Flat dependency list
```

## Quick Start

### Prerequisites

- Python ≥ 3.11
- Docker Engine running
- An API key for OpenAI or Anthropic (optional — Ollama works offline)
- [Ollama](https://ollama.ai/) (recommended for local inference)

### Setup

```bash
# Clone
git clone https://github.com/Matheussoranco/I.S.A.A.C.git
cd I.S.A.A.C

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API key and model preferences

# Build the sandbox images
docker build -t isaac-sandbox:latest sandbox_image/
docker build -t isaac-ui-sandbox:latest sandbox_image_ui/
```

### Run

```bash
# Interactive REPL mode
python -m isaac run

# Telegram bot + heartbeat scheduler
python -m isaac serve

# View audit log
python -m isaac audit show --last 20

# Verify audit chain integrity
python -m isaac audit verify

# List registered tools
python -m isaac tools

# Query memory
python -m isaac memory query "search term"
```

### Docker Compose

```bash
# Build all images
docker compose build

# Start the agent (with optional Ollama)
docker compose --profile ollama up -d

# View logs
docker compose logs -f isaac
```

### Run Tests

```bash
pytest
```

## State Schema

The `IsaacState` TypedDict flows through all graph nodes:

| Field | Type | Reducer | Purpose |
|---|---|---|---|
| `messages` | `list[BaseMessage]` | append | Conversation history |
| `world_model` | `WorldModel` | replace | Environment snapshot |
| `hypothesis` | `str` | replace | Current reasoning hypothesis |
| `plan` | `list[PlanStep]` | replace | Dynamic task decomposition |
| `code_buffer` | `str` | replace | Synthesised Python code |
| `execution_logs` | `list[ExecutionResult]` | append | Sandbox stdout/stderr/exit |
| `skill_candidate` | `SkillCandidate \| None` | replace | Code pending library commit |
| `errors` | `list[ErrorEntry]` | append | Failure stack (prevents loops) |
| `iteration` | `int` | replace | Cycle counter (hard-capped) |
| `current_phase` | `str` | replace | Active node name |

## Sandbox Security

| Constraint | Value |
|---|---|
| Network | `none` (total isolation) |
| Memory | 256 MB hard limit |
| CPU | 1 core |
| PIDs | 64 max |
| Capabilities | All dropped |
| Root FS | Read-only |
| User | `nobody` (65534) |
| Timeout | 30s (application-level `SIGKILL`) |

## License

[GNU AGPL v3](LICENSE)
# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

A neuro-symbolic autonomous agent built on [LangGraph](https://github.com/langchain-ai/langgraph) with Docker-sandboxed execution and a cumulative Skill Library for ARC-AGI program synthesis.

---

## Architecture Overview

I.S.A.A.C. models reasoning as an explicit cyclic state graph — not generic while-loops. Six cognitive nodes operate on a strict `TypedDict` state contract:

```
START ─► Perception ─► Planner ─► Synthesis ─► Sandbox ─► Reflection
              ▲                                                │
              │          ┌──── Skill Abstraction ◄─────────────┤
              │          │            │                         │
              │          ▼            ▼                         ▼
              └──── Planner        END                       END
```

| Node | Responsibility |
|---|---|
| **Perception** | Parse input, extract observations, build initial hypothesis |
| **Planner** | Decompose task into dependency-aware steps (Graph-of-Thought) |
| **Synthesis** | Generate pure Python code (CodeAgent — no JSON tool calls) |
| **Sandbox** | Execute code in ephemeral Docker container (zero host access) |
| **Reflection** | Analyse results, diagnose failures, revise hypothesis |
| **Skill Abstraction** | Generalise successful code into reusable library entries |

## Core Design Principles

- **Execution Isolation**: All environment interactions run in ephemeral, unprivileged Docker containers (`--network=none`, `--cap-drop=ALL`, `--read-only`).
- **CodeAgent Paradigm**: The LLM generates pure Python — no JSON/XML tool calling. Code is injected into the sandbox, never executed on the host.
- **Neuro-Symbolic Reasoning**: Structured state schema separates perception from representation. The `WorldModel` carries symbolic observations, not raw data.
- **Cumulative Learning**: A persistent Skill Library stores generalised programs. The agent composes existing skills to solve novel tasks, reducing LLM calls over time.
- **ARC-AGI Foundations**: Perception/representation separation + compositional Skill Library enable program synthesis for geometric/logical puzzles.

## Project Structure

```
I.S.A.A.C/
├── src/isaac/                  # Main package
│   ├── core/                   # State schema, graph builder, transitions
│   ├── nodes/                  # 6 cognitive graph nodes
│   ├── memory/                 # Episodic memory, world model, skill library
│   ├── sandbox/                # Docker security, manager, code executor
│   ├── llm/                    # Provider factory, prompt templates
│   └── config/                 # Pydantic settings (env-driven)
├── sandbox_image/              # Docker build context for sandbox containers
├── skills/                     # Persistent skill library (JSON + .py)
├── tests/                      # Full test suite with mocked LLM/Docker
├── pyproject.toml              # PEP 621 metadata + tooling config
└── requirements.txt            # Flat dependency list
```

## Quick Start

### Prerequisites

- Python ≥ 3.11
- Docker Engine running
- An API key for OpenAI or Anthropic

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

# Build the sandbox image
docker build -t isaac-sandbox:latest sandbox_image/
```

### Run

```bash
python -m isaac
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
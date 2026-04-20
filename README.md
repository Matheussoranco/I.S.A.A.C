# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

[![CI](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml/badge.svg)](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.3.0--beta-blue)](https://github.com/Matheussoranco/I.S.A.A.C/releases/tag/v0.3.0)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A **multimodal, self-improving, local-first** autonomous agent built on
[LangGraph](https://github.com/langchain-ai/langgraph) — Docker-sandboxed
execution, voice + vision input, a cumulative Skill Library, telemetry-driven
self-curation, and a hardened security stack.

---

## Highlights

| Capability | What it does |
|---|---|
| **Local-first LLMs** | First-class Ollama, llama.cpp, and any OpenAI-compatible endpoint. Cloud (OpenAI/Anthropic) only as fallback. |
| **Voice I/O** | Whisper (STT) ↔ Piper / Coqui / pyttsx3 (TTS) with VAD-driven hands-free mode. |
| **Vision** | Local VLMs via Ollama (`llava`, `qwen2.5-vl`). Image / screen-capture input. |
| **Self-improving** | Per-node telemetry, A/B prompt evolution, skill auto-curation, periodic self-critique. |
| **Sandboxed code** | Ephemeral Docker containers, no network, dropped capabilities, seccomp profile. |
| **5-layer memory** | Long-term (SQLite FTS5), Episodic (ChromaDB), Semantic (KG), Procedural (skills), WorldModel KG. |
| **Connectors** | GitHub, IMAP/SMTP, CalDAV, Obsidian, web fetch/search, allowlisted shell. |

## Architecture

I.S.A.A.C. models reasoning as an explicit cyclic state graph — not generic
while-loops.  Cognitive nodes operate on a strict `TypedDict` state contract.

```
START ─► Guard ─► Perception ─► (DirectResponse | Explorer)
                                       │
                                       ▼
                                    Planner
                                       │
                              ConnectorExecution
                                       │
                                    Synthesis
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ mode=ui                  │ mode=code/hybrid │
                ▼                          ▼                  │
          ComputerUse                   Sandbox               │
                └──────────────┬───────────┘                  │
                               ▼                              │
                          Reflection                          │
                               │                              │
            ┌──────────────────┼─────────────────────┐        │
            ▼                  ▼                     ▼        │
     SkillAbstraction      Planner                  END       │
                                                              │
                  AwaitApproval (inserted dynamically) ◄──────┘
```

| Node | Responsibility |
|---|---|
| **Guard** | Detect prompt injection and sanitize input |
| **Perception** | Parse text + image + audio, build hypothesis, set task mode |
| **DirectResponse** | Fast-path for greetings / Q&A — skips planning entirely |
| **Explorer** | Active exploration (ARC structural + web search) |
| **Planner** | Decompose into a dependency-aware DAG (Graph-of-Thought) |
| **ConnectorExecution** | Host-side connector dispatch (web, email, fs, ...) |
| **Synthesis** | Generate pure Python (CodeAgent — no JSON tool calls) |
| **Sandbox** | Run code in ephemeral Docker (no network, no caps) |
| **ComputerUse** | GUI automation in virtual desktop (Xvfb + Playwright) |
| **Reflection** | Analyse, refine, or escalate to Planner |
| **SkillAbstraction** | Generalise successful code into reusable Library entries |
| **AwaitApproval** | Pause for human approval on high-risk tools |

### Multimodal subsystem (new in 0.3.0)

```
src/isaac/multimodal/
├── voice/
│   ├── stt.py            ← Whisper (faster-whisper / openai-whisper)
│   ├── tts.py            ← Piper / Coqui / pyttsx3 auto-selection
│   └── audio_io.py       ← Mic capture, VAD, playback
├── vision/
│   ├── vision_lm.py      ← Image+text VLM wrapper
│   └── screen_capture.py ← mss / PIL.ImageGrab
└── input.py              ← Unified text + image + audio → HumanMessage
```

### Self-improvement engine (new in 0.3.0)

```
src/isaac/improvement/
├── performance.py     ← SQLite-backed per-node + per-skill telemetry
├── skill_curation.py  ← promote / deprecate / quarantine skills
├── prompt_evolution.py ← A/B test prompt variants (epsilon-greedy)
├── self_critique.py   ← LLM-driven meta-reflection
└── engine.py          ← Orchestrator + scheduler hook
```

Every cognitive node is wrapped in a telemetry decorator (`core/telemetry.py`)
so per-run duration / success / error patterns flow into the tracker
**for free** — no node code changes required.

### LLM provider stack (refactored in 0.3.0)

```
src/isaac/llm/
├── providers/
│   ├── ollama.py        ← first-class local
│   ├── llamacpp.py      ← local llama.cpp HTTP
│   ├── openai_compat.py ← LM Studio, vLLM, LiteLLM, ...
│   ├── openai.py        ← cloud
│   └── anthropic.py     ← cloud
├── multimodal_router.py ← (modality × complexity) routing with health checks
└── router.py            ← legacy complexity router (kept for compat)
```

## Core Design Principles

- **Local-first** — every default points at a local backend. Cloud APIs are
  optional fallbacks, never required.
- **Modality-aware routing** — text, vision, and audio each get their own
  routing table; the router picks the best healthy backend per request.
- **Self-improving** — the agent measures itself and acts on the data:
  weak skills get deprecated, good prompt variants get more traffic.
- **Execution Isolation** — all environment interactions in ephemeral
  unprivileged Docker containers (`--network=none`, `--cap-drop=ALL`,
  `--read-only`, seccomp profile).
- **CodeAgent Paradigm** — the LLM generates pure Python; no JSON/XML tool
  calling. Code is injected into the sandbox, never executed on host.
- **Neuro-Symbolic Reasoning** — structured state schema separates perception
  from representation. The `WorldModel` carries symbolic observations via
  a knowledge graph.
- **Five-Layer Memory** — Episodic, Semantic, WorldModelKG, SkillLibrary,
  unified ContextManager.
- **Cumulative Learning** — persistent Skill Library composes existing
  skills to solve novel tasks, reducing LLM calls over time.
- **Security-First** — hash-chained audit log, capability tokens, prompt
  injection guard, I/O sanitization, seccomp sandboxing.

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Docker Engine running
- [Ollama](https://ollama.ai/) (recommended for local inference)
- *Optional:* faster-whisper + Piper for voice; mss + Pillow for vision

### Setup

```bash
git clone https://github.com/Matheussoranco/I.S.A.A.C.git
cd I.S.A.A.C

python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .venv\Scripts\activate            # Windows

# Core install
pip install -e ".[dev]"

# Add multimodal extras (vision + voice)
pip install -e ".[multimodal]"

# Configure
cp .env.example .env
# Edit .env — at minimum set ISAAC_OLLAMA_BASE_URL / ISAAC_MODEL_NAME

# Build sandbox images
docker build -t isaac-sandbox:latest sandbox_image/
docker build -t isaac-ui-sandbox:latest sandbox_image_ui/

# Pull a local model + a VLM
ollama pull qwen2.5-coder:7b
ollama pull llava:7b
```

### Run

```bash
# Rich text REPL (default)
isaac run

# Voice REPL — push-to-talk
isaac voice

# Voice REPL — hands-free (continuous listening + VAD)
isaac voice --hands-free

# Ask a question about an image
isaac vision /path/to/screenshot.png --prompt "What's in this UI?"

# Run one self-improvement cycle on demand
isaac improve

# Print the last critique report alongside the cycle
isaac improve --report

# List all providers + locally-installed Ollama models
isaac models

# Telegram bot + heartbeat scheduler
isaac serve

# Audit / memory / connectors / cron / tokens — see SETUP.md
isaac audit --last 20
isaac memory "search term"
isaac connectors
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
| `errors` | `list[ErrorEntry]` | append | Failure stack |
| `iteration` | `int` | replace | Cycle counter (hard-capped) |
| `current_phase` | `str` | replace | Active node name |
| `task_mode` | `TaskMode` | replace | `"code"` \| `"computer_use"` \| `"hybrid"` |
| `ui_actions` | `list[UIAction]` | append | Pending GUI actions |
| `ui_results` | `list[UIActionResult]` | append | Screenshot+outcome |
| `pending_approvals` | `list[PendingApproval]` | append | High-risk actions awaiting sign-off |

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

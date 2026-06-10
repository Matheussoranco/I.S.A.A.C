# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

[![CI](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml/badge.svg)](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.1.0-blue)](https://github.com/Matheussoranco/I.S.A.A.C/releases/tag/v1.1.0)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A **multimodal, self-improving, local-first** autonomous agent built on
[LangGraph](https://github.com/langchain-ai/langgraph) — Docker-sandboxed
execution, voice + vision input, a cumulative Skill Library, telemetry-driven
self-curation, and a hardened security stack.

> **1.0 framing:** stable CLI/API, tested safety boundary, local-first by
> default. What is *not* yet proven (benchmark numbers, live-model e2e runs)
> is stated plainly in [LIMITATIONS.md](LIMITATIONS.md).

---

## Highlights

| Capability | What it does |
|---|---|
| **Specialist team** | A manager *orchestrator* decomposes a goal and dispatches it to focused local-first mini-agents — coder, file-organizer, researcher, designer, OS-operator, analyst, critic — running independent subtasks in parallel. |
| **Host reach** | Constitution-gated `shell`, real-filesystem `fs_*` tools (organise your actual files, confined to `allowed_paths`), and read-only `system_info` — so it can do nearly any task on the PC. |
| **User personas** | Build, store, and activate custom agent identities (`isaac persona new`); the whole team speaks with your chosen voice. |
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

### Routing matrix `(modality × complexity)`

The `MultimodalRouter` resolves every LLM call to a concrete `(provider, model)` pair via a 3 × 3 table. Local providers are health-checked (cached 60 s) before dispatch; on miss the router walks a fallback chain.

|              | **fast** (perception, classification) | **default** (synthesis, planning) | **strong** (reflection, critique)                       |
| ------------ | ------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| **text**     | `ollama / qwen2.5:3b`                 | `ollama / qwen2.5-coder:7b`       | `ollama / qwen2.5:14b` → `anthropic / claude-haiku-4-5` |
| **vision**   | `ollama / llava:7b`                   | `ollama / llava:7b`               | `ollama / qwen2.5-vl` → `openai / gpt-4o`               |
| **audio**    | faster-whisper (`tiny` / `base`)      | faster-whisper (`small`)          | faster-whisper (`large-v3`)                             |

Cloud providers (`openai`, `anthropic`) are **never required** — they only enter the chain when listed as a fallback **and** the user supplied `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. Audio routing transcribes locally and re-routes the text through the **text** row.

### Self-improvement lifecycle

```
       ┌─────────────────────────────────────────────────────┐
       │              every cognitive node                   │
       │              (telemetry decorator)                  │
       └────────────────────────┬────────────────────────────┘
                                ▼
                ┌──────────────────────────────┐
                │ performance.PerformanceTracker│
                │   • node_runs                 │
                │   • skill_runs                │
                │   • prompt_runs               │  ← SQLite
                └──────────────┬────────────────┘
                               ▼
       ┌────────────────────── engine.run_cycle ──────────────────────┐
       │                                                              │
       │   1. SkillCurator       — promote / deprecate / quarantine   │
       │   2. PromptEvolution    — ε-greedy variant selection         │
       │   3. SelfCritique       — strong-tier LLM meta-reflection    │
       │   4. PerformanceTracker — prune > 90-day rows                │
       │                                                              │
       └──────────────────────────────────────────────────────────────┘
                               │
                               ▼
                  ImprovementResult ──► audit log + scheduler
```

Trigger paths:
- **Manual** — `isaac improve [--report]` runs one cycle synchronously and prints the critique summary
- **Scheduled** — when `ISAAC_IMPROVEMENT_ENABLED=true`, APScheduler fires `improvement_job` every `ISAAC_IMPROVEMENT_INTERVAL_MINUTES` (10 ≤ x ≤ 10080)
- **Telemetry-only** — when improvement is disabled, the tracker still records every node/skill run so a later cycle has data to act on

### Five-layer memory (unchanged in 0.3.0, summarized here)

| Layer            | Backend                            | Purpose                                                  |
| ---------------- | ---------------------------------- | -------------------------------------------------------- |
| **LongTerm**     | SQLite + FTS5                      | Full-text-searchable canonical archive                   |
| **Episodic**     | Ring buffer + ChromaDB             | Recent turns, vector recall                              |
| **Semantic**     | NetworkX KG + SQLite + ChromaDB    | Concept graph, relation queries                          |
| **Procedural**   | `SkillLibrary` (JSON + embeddings) | Reusable Python snippets generalized from past successes |
| **WorldModelKG** | NetworkX DiGraph + SQLite          | Symbolic observations carried inside `IsaacState`        |

`MemoryManager.recall()` produces a single combined prompt string from all five layers; ChromaDB-dependent layers degrade gracefully (exact-match fallback) when the vector store is unavailable.

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
# Preflight — verify Python, settings, Ollama, Docker, and optional extras
isaac doctor

# Autonomous tool-use agent (Claude-Code style) — give it a task and it
# browses, searches, runs code, and writes files until it's done.
isaac agent "Find the current stable Python release and save it to version.txt"

# Restrict the toolbox, allow more steps, and auto-approve high-risk tools
isaac agent "Summarise today's top Hacker News post" --tools browser,web_search -n 20

# Inspect past runs (every agent run is traced to SQLite)
isaac trace                 # list recent runs
isaac trace <run_id>        # replay one run's event stream

# Specialist team — a manager decomposes the goal and dispatches it to the
# right specialists (researcher → designer → coder …), in parallel where it can.
isaac team "Research the 3 best local vector DBs and write a comparison to compare.md"
isaac specialists                       # list the team and each one's tools

# Personas — give I.S.A.A.C. a custom identity (and persist it via ISAAC_SOUL_PATH)
isaac persona examples                  # install bundled 'atlas' / 'sage'
isaac persona new                       # interactively build your own
isaac persona activate --slug atlas

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

## Evaluation (`isaac eval`)

Capability claims are measured, not asserted. The eval harness loads a JSONL
task suite, runs each task through the agent (or the specialist team), scores
the answers with deterministic programmatic checkers — no LLM judging — and
records every run (suite hash, model, provider, git revision, per-task
results) to a SQLite DB so scores are reproducible and comparable.

```bash
# Run the bundled golden suite (33 tasks: reasoning, coding, analysis,
# file-org, writing, research, orchestration, safety)
isaac eval evals/golden_v1.jsonl

# Slice it
isaac eval evals/golden_v1.jsonl --limit 5
isaac eval evals/golden_v1.jsonl --task code-001

# Compare recorded runs (model A vs model B, version N vs N+1)
isaac eval --report
```

Suites are plain JSONL — one task per line with a prompt and checks
(`contains`, `regex`, `numeric`, `file_exists`, `file_contains`, ...), plus
optional seeded workspace files and a tool allow-list. See
[`evals/golden_v1.jsonl`](evals/golden_v1.jsonl) for the format.

### Recorded results

| Date | Model (driver) | Suite | Score |
|---|---|---|---|
| 2026-06-10 | `qwen3-coder:480b-cloud` (Ollama Cloud, via AgentLoop) | `golden_v1` · hash `da9b7c08c5bd342a` · run `d5d463aabba3` | **31/33 (93.9%)** |

Per category: reasoning 8/9 · coding 4/4 · analysis 4/4 · research 5/5 ·
text 4/4 · writing 2/2 · file-org 2/2 · orchestration 1/2 · safety 1/1
(the credential-exfiltration probe was refused, as designed). The two
failures: one date-arithmetic task and one orchestration answer that
skipped the required numbered-list format.

Reproduce: `ISAAC_MODEL_NAME=qwen3-coder:480b-cloud isaac eval evals/golden_v1.jsonl`

> Per [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md) §4, "SOTA" claims
> additionally require a *public* benchmark (GAIA, SWE-bench, ...) against a
> named comparison system — the golden suite is an internal capability bar,
> not a comparative one. That distinction stands.

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

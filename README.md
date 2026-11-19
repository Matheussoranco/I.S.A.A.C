# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

[![CI](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml/badge.svg)](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-1.6.2-blue)](https://github.com/Matheussoranco/I.S.A.A.C/releases)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

A **multimodal, self-improving, local-first** autonomous agent built on
[LangGraph](https://github.com/langchain-ai/langgraph) — Docker-sandboxed
execution, voice + vision input, a cumulative Skill Library, telemetry-driven
self-curation, and a hardened security stack.

> **1.0 framing:** stable CLI/API, tested safety boundary, local-first by
> default. What is *not* yet proven (benchmark numbers, live-model e2e runs)
> is stated plainly in [LIMITATIONS.md](LIMITATIONS.md).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Cognitive Graph (Detailed)](#cognitive-graph-detailed)
- [Multimodal Subsystem](#multimodal-subsystem)
- [Self-Improvement Engine](#self-improvement-engine)
- [LLM Provider Stack](#llm-provider-stack)
- [Routing Matrix](#routing-matrix)
- [Five-Layer Memory](#five-layer-memory)
- [Kanban & Dispatcher](#kanban--dispatcher)
- [Evaluation Harness](#evaluation-harness)
- [State Schema](#state-schema)
- [Sandbox Security Model](#sandbox-security-model)
- [Security & Audit](#security--audit)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

### Prerequisites

- Python ≥ 3.12
- Docker Engine running
- [Ollama](https://ollama.com/download) with the default model pulled — `ollama pull qwen3.6`
- *Optional:* an `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` if you'd rather drive a cloud model
- *Optional:* faster-whisper + Piper for voice; mss + Pillow for vision

### Ready-to-run Windows application

Build, install and launch the standalone Windows application:

```powershell
.\scripts\build_windows.ps1
.\scripts\install_windows.ps1
```

The build produces a single `dist\ISAAC-<version>-Windows-x64.exe` containing
the Python runtime, native DLLs, and UI resources. Python does not need to be
installed on the destination PC. The installer copies that executable to
`%LOCALAPPDATA%\Programs\ISAAC`, creates Start Menu and Desktop shortcuts, and
keeps the previous installation during an update.

In the app, click the model name to choose a local or cloud profile. Cloud API
keys entered there are saved in Windows Credential Manager and are never
returned to the web interface or chat history. The `Computador` mode uses
OpenAI's native computer tool only for the OpenAI profile; every actionable
batch stays behind an approval card.

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

# Configure — optional. The defaults already point at a local Ollama daemon
# running qwen3.6, so no API key is needed.
cp .env.example .env

# Build sandbox images
docker build -t isaac-sandbox:latest sandbox_image/
docker build -t isaac-ui-sandbox:latest sandbox_image_ui/

# Pull the default local model (+ a VLM for vision)
ollama pull qwen3.6
ollama pull llava:7b
```

Prefer a cloud model? Nothing is removed — select it explicitly:

```bash
export ISAAC_LLM_PROVIDER=anthropic
export ISAAC_MODEL_NAME=claude-opus-4-8
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
# Preflight — verify Python, settings, Ollama, Docker, and optional extras
isaac doctor

# Native Windows app — Codex-style chat, activity, approvals, browser/desktop
# preview, and a visible agent cursor.
pip install -e ".[desktop]"
isaac desktop

# The same interface can also run in a normal browser tab.
isaac ui

# Autonomous tool-use agent (Claude-Code style) — give it a task and it
# browses, searches, runs code, and writes files until it's done.
isaac agent "Find the current stable Python release and save it to version.txt"

# Restrict the toolbox, allow more steps, and auto-approve high-risk tools
isaac agent "Summarise today's top Hacker News post" --tools browser,web_search -n 20

# Inspect past runs (metadata-only SQLite traces by default)
isaac trace                 # list recent runs
isaac trace <run_id>        # replay one run's event stream

# Pick a model for the GPU you actually have — each preset pins a model *and*
# the loop settings that model needs (see docs/MODELS.md).
isaac models list
isaac models recommend      # reads your VRAM; stays local when a local rung fits
isaac models use good       # prints the .env block to paste

# Measure how reliably a model emits well-formed tool calls (20 prompts)
isaac eval-toolcalls --model nemotron-3-nano:4b

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
isaac providers

# Telegram bot + heartbeat scheduler
isaac serve

# Audit / memory / connectors / cron / tokens — see SETUP.md
isaac audit --last 20
isaac memory "search term"
isaac connectors
```

---

## Architecture Overview

I.S.A.A.C. is architected as a **LangGraph state machine** where every cognitive operation is an explicit node in a directed cyclic graph. This is not a generic while-loop agent; each node has a strict `TypedDict` contract (`IsaacState`), deterministic reducers, and auditable telemetry.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        I.S.A.A.C. SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   USER INTERFACE │     │   AGENT CORE     │     │   EXECUTION      │
│                  │     │                  │     │   LAYER          │
│  ┌────────────┐  │     │  ┌────────────┐  │     │  ┌────────────┐  │
│  │ Desktop    │  │     │  │ Cognitive  │  │     │  │ Sandbox    │  │
│  │ App (Tauri)│◄─┼────►│  │ Graph      │  │     │  │ (Docker)   │  │
│  │ Web UI     │  │     │  │ (LangGraph)│  │     │  │ Computer   │  │
│  │ CLI/REPL   │  │     │  └─────┬──────┘  │     │  │ Use (Xvfb) │  │
│  │ Voice/Vision│  │     │       │         │     │  └─────┬──────┘  │
│  └────────────┘  │     └───────┼─────────┘     │       │         │
└──────────────────┘             │               └───────┼─────────┘
                                 │                       │
                    ┌────────────▼────────────┐  ┌────────▼────────┐
                    │   PROVIDER & MODEL      │  │   MEMORY        │
                    │   ROUTING               │  │   LAYERS        │
                    │                         │  │               │
                    │  MultimodalRouter       │  │  ┌──────────┐  │
                    │  (modality × complexity)│  │  │LongTerm  │  │
                    │                         │  │  │(SQLite)  │  │
                    │  ┌──────────────────┐   │  │  ├──────────┤  │
                    │  │ Providers        │   │  │  │Episodic  │  │
                    │  │ • ollama         │   │  │  │(ChromaDB)│  │
                    │  │ • llamacpp       │   │  │  ├──────────┤  │
                    │  │ • openai_compat  │   │  │  │Semantic  │  │
                    │  │ • openai         │   │  │  │(KG+Vec)  │  │
                    │  │ • anthropic      │   │  │  ├──────────┤  │
                    │  └──────────────────┘   │  │  │Procedural│  │
                    │                         │  │  │(Skills)  │  │
                    │  Health checks (60s)    │  │  ├──────────┤  │
                    │  Fallback chains        │  │  │WorldModel│  │
                    └─────────────────────────┘  │  │(KG)      │  │
                                                 └──┴──────────┘
```

---

## Cognitive Graph (Detailed)

I.S.A.A.C. models reasoning as an explicit **cyclic state graph** — not generic while-loops. Cognitive nodes operate on a strict `TypedDict` state contract (`IsaacState`).

```
START ──► Guard ──► Perception ──► (DirectResponse | Explorer)
                                        │
                                        ▼
                                     Planner
                                        │
                              ConnectorExecution
                                        │
                                    Synthesis
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
          ComputerUse (mode=ui)                          Sandbox (code/hybrid)
                └───────────────────────┬───────────────────────┘
                                        ▼
                                   Reflection
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
       SkillAbstraction             Planner                     END
```

### Node Responsibilities

| Node | Responsibility | Key Details |
|------|---------------|-------------|
| **Guard** | Detect prompt injection and sanitize input | Regex + LLM-based classification; fast-path reject |
| **Perception** | Parse text + image + audio, build hypothesis, set task mode | Multimodal fusion; outputs `hypothesis` + `task_mode` |
| **DirectResponse** | Fast-path for greetings / Q&A — skips planning entirely | Bypasses graph for trivial interactions |
| **Explorer** | Active exploration (ARC structural + web search) | Iterative deepening; bounded by `ISAAC_EXPLORER_DEPTH` |
| **Planner** | Decompose into a dependency-aware DAG (Graph-of-Thought) | Outputs `plan: list[PlanStep]` with explicit deps |
| **ConnectorExecution** | Host-side connector dispatch (web, email, fs, ...) | Sandbox-isolated; capability-token gated |
| **Synthesis** | Generate pure Python (CodeAgent — no JSON tool calls) | Code-only output; injected into sandbox |
| **Sandbox** | Run code in ephemeral Docker (no network, no caps) | `network=none`, `cap-drop=ALL`, seccomp, 30s timeout |
| **ComputerUse** | GUI automation in virtual desktop (Xvfb + Playwright) | Screenshot → action → screenshot loop; approval-gated |
| **Reflection** | Analyse, refine, or escalate to Planner | Success/failure classification; `should_continue` flag |
| **SkillAbstraction** | Generalise successful code into reusable Library entries | Creates `SkillCandidate` with signature + tests |

### State Transitions (Reducers)

Every field in `IsaacState` has a declared reducer (`append`, `replace`, `merge`). LangGraph's checkpointing provides **exact replay** and **time-travel debugging**.

---

## Multimodal Subsystem

```
src/isaac/multimodal/
├── voice/
│   ├── stt.py            ← Whisper (faster-whisper / openai-whisper)
│   ├── tts.py            ← edge-tts / OpenAI / ElevenLabs / pyttsx3 / Kokoro provider selection
│   └── audio_io.py       ← Mic capture, VAD (silero), playback
├── vision/
│   ├── vision_lm.py      ← Image+text VLM wrapper (Ollama llava, qwen2.5-vl)
│   └── screen_capture.py ← mss / PIL.ImageGrab (multi-monitor)
└── input.py              ← Unified text + image + audio → HumanMessage
```

### Voice Pipeline

```
Audio Input (mic) ──► VAD (silero) ──► faster-whisper ──► Text
                                                            │
                                              ┌─────────────┴─────────────┐
                                              ▼                           ▼
                                         STT Result                     HumanMessage
                                              │                           │
                                              ▼                           ▼
                                         Cognitive Graph            Conversation History
                                              │
                                              ▼
                                         LLM Response ──► TTS Provider ──► Audio Output
                                              (edge-tts / OpenAI / ElevenLabs / pyttsx3 / Kokoro)
```

**Modes:**
- **Push-to-talk** (`isaac voice`) — manual trigger, single utterance
- **Hands-free** (`isaac voice --hands-free`) — continuous VAD-driven listening

### Vision Pipeline

```
Image Input (file / screen / clipboard) ──► VLM (Ollama llava/qwen2.5-vl) ──► Structured Analysis
                                                                      │
                                                          ┌───────────┴───────────┐
                                                          ▼                       ▼
                                                 Text Description          Bounding Boxes / UI Elements
                                                          │                       │
                                                          └───────────┬───────────┘
                                                                      ▼
                                                             Cognitive Graph (Perception)
```

**Supported inputs:** local file paths, clipboard, screen capture (mss), URLs

---

## Self-Improvement Engine

```
src/isaac/improvement/
├── performance.py     ← SQLite-backed per-node + per-skill telemetry
├── skill_curation.py  ← promote / deprecate / quarantine skills
├── prompt_evolution.py ← A/B test prompt variants (epsilon-greedy)
├── self_critique.py   ← LLM-driven meta-reflection (strong-tier model)
└── engine.py          ← Orchestrator + APScheduler hook
```

### Telemetry Decorator

Every cognitive node is wrapped in a telemetry decorator (`core/telemetry.py`) so per-run duration / success / error patterns flow into the tracker **for free** — no node code changes required.

```python
@telemetry("node_name")
async def node_fn(state: IsaacState) -> IsaacState:
    ...
```

### Self-Improvement Lifecycle

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

**Trigger paths:**
- **Manual** — `isaac improve [--report]` runs one cycle synchronously and prints the critique summary
- **Scheduled** — when `ISAAC_IMPROVEMENT_ENABLED=true`, APScheduler fires `improvement_job` every `ISAAC_IMPROVEMENT_INTERVAL_MINUTES` (10 ≤ x ≤ 10080)
- **Telemetry-only** — when improvement is disabled, the tracker still records every node/skill run so a later cycle has data to act on

---

## LLM Provider Stack (refactored in 0.3.0)

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

### Provider Interface

All providers implement a unified async interface:

```python
class LLMProvider(Protocol):
    async def complete(self, messages: list[BaseMessage], **kwargs) -> AIMessage: ...
    async def stream(self, messages: list[BaseMessage], **kwargs) -> AsyncIterator[str]: ...
    async def health_check(self) -> bool: ...
    def get_models(self) -> list[str]: ...
```

### Routing Matrix `(modality × complexity)`

The `MultimodalRouter` resolves every LLM call to a concrete `(provider, model)` pair via a 3 × 3 table. Local providers are health-checked (cached 60 s) before dispatch; on miss the router walks a fallback chain.

Defaults are what a fresh install actually uses — **no API keys, no cloud account**:

|              | **fast** (perception, classification) | **default** (synthesis, planning) | **strong** (reflection, critique) |
| ------------ | ------------------------------------- | --------------------------------- | --------------------------------- |
| **text**     | `ollama / qwen3.6`                    | `ollama / qwen3.6`                | `ollama / qwen3.6`                |
| **vision**   | `ollama / llava:7b`                   | `ollama / llava:7b`               | `ollama / llava:7b`               |
| **audio**    | faster-whisper (`tiny` / `base`)      | faster-whisper (`small`)          | faster-whisper (`large-v3`)       |

Override any cell with `ISAAC_FAST_MODEL` / `ISAAC_MODEL_NAME` / `ISAAC_STRONG_MODEL` (and `ISAAC_VISION_MODEL` for the vision row).

Cloud providers (`openai`, `anthropic`) are **never required and never automatic** — they enter the chain only when you select one (`ISAAC_LLM_PROVIDER=anthropic`) or name one in `ISAAC_LLM_FALLBACK_PROVIDER`, **and** supply the matching `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. If Ollama is down or the model was never pulled, I.S.A.A.C. raises an error naming the exact command (`ollama pull qwen3.6`) rather than quietly billing a cloud API. Audio routing transcribes locally and re-routes the text through the **text** row.

---

## Five-Layer Memory

| Layer            | Backend                            | Purpose                                                  |
| ---------------- | ---------------------------------- | -------------------------------------------------------- |
| **LongTerm**     | SQLite + FTS5                      | Full-text-searchable canonical archive                   |
| **Episodic**     | Ring buffer + ChromaDB             | Recent turns, vector recall                              |
| **Semantic**     | NetworkX KG + SQLite + ChromaDB    | Concept graph, relation queries                          |
| **Procedural**   | `SkillLibrary` (JSON + embeddings) | Reusable Python snippets generalized from past successes |
| **WorldModelKG** | NetworkX DiGraph + SQLite          | Symbolic observations carried inside `IsaacState`        |

`MemoryManager.recall()` produces a single combined prompt string from all five layers; ChromaDB-dependent layers degrade gracefully (exact-match fallback) when the vector store is unavailable.

### Memory Flow

```
User Input ──► Perception ──► MemoryManager.recall(query)
                                                │
                                ┌───────────────┼───────────────┐
                                ▼               ▼               ▼
                         LongTerm           Episodic         Semantic
                          (FTS5)            (ChromaDB)       (KG+Vec)
                                │               │               │
                                └───────────────┼───────────────┘
                                                ▼
                                    Combined Context String
                                                │
                                                ▼
                                    Cognitive Graph (Planner, etc.)
```

---

## Kanban & Dispatcher

### Kanban (SQLite-backed Task Boards)

I.S.A.A.C. includes a built-in Kanban system for task management with:
- Boards, columns, tasks with priorities, assignees, due dates, tags
- Task dependencies (DAG) and blocking
- Worker claims with stale detection and reclamation
- Comments on tasks
- Board statistics

**Tables:** `board`, `column`, `task`, `task_dep`, `claim`, `comment`

### Dispatcher (Background Worker Assignment)

The dispatcher runs as a background task that:
1. Periodically reclaims stale claims
2. Finds ready tasks that are not claimed
3. Assigns them to available workers based on profile
4. Delivers tasks to workers via a gateway (HTTP, message queue, etc.)

**Worker Profiles:** `coder`, `researcher`, `file_organizer`, `designer`, `os_operator`, `analyst`, `critic`

---

## Evaluation Harness

Capability claims are **measured, not asserted**. The eval harness loads a JSONL task suite, runs each task through the agent (or the specialist team), scores the answers with **deterministic programmatic checkers** — no LLM judging — and records every run (suite hash, model, provider, git revision, per-task results) to a SQLite DB so scores are reproducible and comparable.

### Golden Suite (33 tasks)

```bash
# Run the bundled golden suite (reasoning, coding, analysis, file-org, writing, research, orchestration, safety)
isaac eval evals/golden_v1.jsonl

# Slice it
isaac eval evals/golden_v1.jsonl --limit 5
isaac eval evals/golden_v1.jsonl --task code-001

# Compare recorded runs (model A vs model B, version N vs N+1)
isaac eval --report
```

### Public Benchmarks

```bash
# GAIA Level 1 (public benchmark, official quasi-exact-match scoring)
isaac eval --format gaia --download
isaac eval ~/.isaac/datasets/gaia/2023/validation --format gaia --level 1

# ARC-AGI-1 (public benchmark, ungated) — LLM-free symbolic solver
isaac eval --format arc --download
isaac eval ~/.isaac/datasets/arc-agi-1/evaluation --format arc
```

### Recorded Results

| Date | Model (driver) | Suite | Score |
|---|---|---|---|
| 2026-06-10 | `qwen3-coder:480b-cloud` (Ollama Cloud) | `golden_v1` | **31/33 (93.9%)** |
| 2026-07-03 | Symbolic solver (no LLM) | ARC-AGI-1 (400 tasks) | **2/400 (0.5%)** |
| 2026-07-05 | `nemotron-3-nano:4b` (local, RTX 3050 6GB) | GAIA Level 1 (53 tasks) | **8/53 (15.1%)** |

See [LIMITATIONS.md](LIMITATIONS.md) for full methodology and honest positioning.

---

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

---

## Sandbox Security Model

| Constraint | Value |
|---|---|
| Network | `none` (total isolation) |
| Memory | 256 MB hard limit |
| CPU | 1 core |
| PIDs | 64 max |
| Capabilities | All dropped (`--cap-drop=ALL`) |
| Root FS | Read-only (`--read-only`) |
| User | `nobody` (65534) |
| Timeout | 30s (application-level `SIGKILL`) |
| Seccomp | Custom profile (blocks `ptrace`, `bpf`, `clone`, etc.) |

**Docker command template:**

```bash
docker run --rm   --network=none   --cap-drop=ALL   --read-only   --memory=256m   --cpus=1   --pids-limit=64   --user=65534:65534   --security-opt seccomp=/etc/seccomp/isaac.json   -v /tmp/isaac_sandbox:/workspace:rw   isaac-sandbox:latest   python /workspace/user_code.py
```

---

## Security & Audit

### Hash-Chained Audit Log

Every consequential action is recorded in an append-only SQLite log with SHA-256 chaining:

```sql
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT,
    curr_hash TEXT NOT NULL
);
```

Verification: `isaac audit --verify` recomputes the chain and flags tampering.

### Capability Tokens

Short-lived, scoped tokens for connector access:

```bash
isaac token create --scope "github:read,fs:write:/home/user/projects" --ttl 3600
```

### Prompt Injection Guard

The `Guard` node classifies every user input before it reaches the cognitive graph:

- Regex-based fast patterns
- LLM-based semantic classification (strong-tier model)
- Structured output: `{"is_injection": bool, "confidence": float, "reason": str}`

### Credential Path Denial

No connector or tool receives credentials via prompt. All secrets are:
- Stored in platform keyring (Windows Credential Manager, macOS Keychain, libsecret)
- Injected at runtime via environment or injected files
- Never logged, never returned to LLM

---

## Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ISAAC_LLM_PROVIDER` | `ollama` | Provider: `ollama`, `llamacpp`, `openai_compat`, `openai`, `anthropic` |
| `ISAAC_MODEL_NAME` | `qwen3.6` | Default text model |
| `ISAAC_FAST_MODEL` | `qwen3.6` | Fast-tier model |
| `ISAAC_STRONG_MODEL` | `qwen3.6` | Strong-tier model |
| `ISAAC_VISION_MODEL` | `llava:7b` | Vision model |
| `ISAAC_LLM_FALLBACK_PROVIDER` | (empty) | Fallback provider name |
| `ISAAC_LLM_FALLBACK_MODEL` | (empty) | Fallback model |
| `OPENAI_API_KEY` | (empty) | OpenAI API key |
| `ANTHROPIC_API_KEY` | (empty) | Anthropic API key |
| `ISAAC_HOME` | `~/.isaac` | Data directory |
| `ISAAC_IMPROVEMENT_ENABLED` | `false` | Enable scheduled self-improvement |
| `ISAAC_IMPROVEMENT_INTERVAL_MINUTES` | `1440` | Improvement cycle interval (10–10080) |
| `ISAAC_SOUL_PATH` | `~/.isaac/soul.yaml` | Persona persistence |
| `ISAAC_OBSIDIAN_VAULT_PATH` | (empty) | Obsidian vault for connector |
| `ISAAC_GITHUB_TOKEN` | (empty) | GitHub PAT for connector |
| `ISAAC_IMAP_HOST` / `ISAAC_SMTP_HOST` | (empty) | Email connector |

See [`.env.example`](.env.example) for complete list.

### Model Presets (`docs/MODELS.md`)

Presets pin a model **and** the loop settings that model needs:

```bash
isaac models list              # show all presets
isaac models recommend         # reads your VRAM; stays local when a local rung fits
isaac models use good          # prints the .env block to paste
```

---

## CLI Reference

| Command | Description |
|---|---|
| `isaac doctor` | Preflight — verify Python, settings, Ollama, Docker, optional extras |
| `isaac desktop` | Native Windows app (Tauri) |
| `isaac ui` | Browser-based UI |
| `isaac agent "task"` | Autonomous tool-use agent |
| `isaac team "goal"` | Specialist team (manager + specialists) |
| `isaac run` | Rich text REPL |
| `isaac voice [--hands-free]` | Voice REPL |
| `isaac vision <path> --prompt "..."` | Vision query |
| `isaac improve [--report]` | Run self-improvement cycle |
| `isaac trace [run_id]` | Inspect past runs |
| `isaac models [list/recommend/use]` | Model preset management |
| `isaac eval <suite> [--format gaia\|arc]` | Evaluation harness |
| `isaac eval-toolcalls --model <m>` | Tool-calling reliability test |
| `isaac specialists` | List specialist team |
| `isaac persona [new/activate/examples]` | Persona management |
| `isaac providers` | List providers + local models |
| `isaac serve` | Telegram bot + heartbeat |
| `isaac audit [--last N] [--verify]` | Audit log |
| `isaac memory "query"` | Search memory |
| `isaac connectors` | Manage connectors |
| `isaac kanban <subcommand>` | Task board management |

---

## Directory Structure

```
I.S.A.A.C/
├── .github/workflows/       # CI/CD pipelines (CI, nightly-eval, docker)
├── evals/                   # Evaluation suites (golden_v1.jsonl)
├── sandbox_image/           # Code sandbox Dockerfile
├── sandbox_image_ui/        # Computer-use sandbox Dockerfile
├── scripts/                 # Build/install scripts (Windows, Linux)
├── src/
│   └── isaac/
│       ├── agents/          # Agent loops (AgentLoop, TeamLoop)
│       ├── cli/             # Typer CLI commands
│       ├── config/          # Settings, profiles, .env loading
│       ├── connectors/      # External integrations (GitHub, Email, Obsidian, ...)
│       ├── core/            # State, telemetry, graph building
│       ├── eval/            # Evaluation harness, checkers, GAIA/ARC adapters
│       ├── improvement/     # Self-improvement engine
│       ├── kanban/          # Kanban board + dispatcher
│       ├── llm/             # Provider stack + routing
│       ├── memory/          # Five-layer memory
│       ├── multimodal/      # Voice, vision, screen capture
│       ├── nodes/           # Cognitive graph nodes
│       ├── sandbox/         # Docker sandbox executor
│       ├── security/        # Audit, tokens, injection guard
│       ├── skills/          # Skill library + built-in skills
│       └── soul/            # Persona system
├── tests/                   # pytest suite (1278 tests)
├── .env.example             # Template configuration
├── LIMITATIONS.md           # Honest capability positioning
├── SETUP.md                 # Detailed setup guide
├── ROADMAP-1.0.md           # 1.0 release roadmap
└── MODELS.md                # Model presets + VRAM thresholds
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Branch model (`main` ← `dev` ← `feat/*` / `fix/*`)
- Development setup
- Code style (Ruff, mypy, pytest)
- Pull request checklist
- Areas actively seeking contributions

### Running Tests

```bash
pytest tests/ --tb=short
```

CI runs on every PR: `ruff check`, `ruff format --check`, `mypy src/isaac --ignore-missing-imports`, `pytest tests/`.

---

## License

I.S.A.A.C. is licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for copyright information. Previously published releases retain the license they shipped with.

---

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) — state graph framework
- [Ollama](https://ollama.com/) — local LLM serving
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local STT
- [edge-tts](https://github.com/rany2/edge-tts) — local TTS
- [Playwright](https://playwright.dev/) — browser automation
- [NetworkX](https://networkx.org/) — knowledge graphs
- [ChromaDB](https://www.trychroma.com/) — vector storage
- [Tauri](https://tauri.app/) — desktop app framework

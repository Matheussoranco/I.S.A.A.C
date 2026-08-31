# I.S.A.A.C.

**Intelligent System for Autonomous Action and Cognition**

[![CI](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml/badge.svg)](https://github.com/Matheussoranco/I.S.A.A.C/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-1.6.2-blue)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)
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
| **Native agent app** | Windows desktop window with a Codex-style chat, persistent conversations, selectable provider/model/reasoning, live task activity, approval/cancel controls, browser or desktop preview, and visible agent cursor. |
| **Real computer-use loop** | With OpenAI + `gpt-5.6-sol`, the app implements the screenshot/action loop directly through the Responses API. It executes bounded mouse/keyboard batches only after approval and immediately returns the resulting full-resolution screenshot to the model. Other providers retain the generic tool loop. |
| **Specialist team** | A manager *orchestrator* decomposes a goal and dispatches it to focused local-first mini-agents — coder, file-organizer, researcher, designer, OS-operator, analyst, critic — running independent subtasks in parallel. |
| **Host reach** | Constitution-gated `shell`, real-filesystem `fs_*` tools (organise your actual files, confined to `allowed_paths`), and read-only `system_info` — so it can do nearly any task on the PC. |
| **User personas** | Build, store, and activate custom agent identities (`isaac persona new`); the whole team speaks with your chosen voice. |
| **Local-first LLMs** | Ollama + `qwen3.6` **by default** — zero API keys, nothing leaves the machine. llama.cpp and any OpenAI-compatible endpoint are first-class too; cloud (OpenAI/Anthropic) stays fully supported but strictly opt-in. |
| **Capable on small models** | Grammar/JSON-constrained tool calling (Ollama `format`, llama.cpp GBNF) plus salvage + Reflexion retry for malformed calls. Runs agents on models with *no* native tool support: `gemma3:1b` goes from 0/20 to 20/20 well-formed calls. Measured in [`docs/MODELS.md`](docs/MODELS.md). |
| **Test-time compute** | Self-consistency and best-of-N with cheap local verifiers on hard steps, escalating and exiting early exactly as the ARC solver does. |
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

Defaults are what a fresh install actually uses — **no API keys, no cloud account**:

|              | **fast** (perception, classification) | **default** (synthesis, planning) | **strong** (reflection, critique) |
| ------------ | ------------------------------------- | --------------------------------- | --------------------------------- |
| **text**     | `ollama / qwen3.6`                    | `ollama / qwen3.6`                | `ollama / qwen3.6`                |
| **vision**   | `ollama / llava:7b`                   | `ollama / llava:7b`               | `ollama / llava:7b`               |
| **audio**    | faster-whisper (`tiny` / `base`)      | faster-whisper (`small`)          | faster-whisper (`large-v3`)       |

Override any cell with `ISAAC_FAST_MODEL` / `ISAAC_MODEL_NAME` / `ISAAC_STRONG_MODEL` (and `ISAAC_VISION_MODEL` for the vision row).

Cloud providers (`openai`, `anthropic`) are **never required and never automatic** — they enter the chain only when you select one (`ISAAC_LLM_PROVIDER=anthropic`) or name one in `ISAAC_LLM_FALLBACK_PROVIDER`, **and** supply the matching `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`. If Ollama is down or the model was never pulled, I.S.A.A.C. raises an error naming the exact command (`ollama pull qwen3.6`) rather than quietly billing a cloud API. Audio routing transcribes locally and re-routes the text through the **text** row.

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

# Inspect past runs (every agent run is traced to SQLite)
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

# GAIA Level 1 (public benchmark, official quasi-exact-match scoring).
# One-time: accept the terms at huggingface.co/datasets/gaia-benchmark/GAIA
# and authenticate (hf auth login / HF_TOKEN) — the dataset is gated.
isaac eval --format gaia --download
isaac eval ~/.isaac/datasets/gaia/2023/validation --format gaia --level 1

# ARC-AGI-1 (public benchmark, ungated). Downloads the official dataset and
# runs the bundled symbolic synthesis solver — no LLM or API key required,
# fully deterministic. This is also the nightly-CI regression benchmark.
isaac eval --format arc --download
isaac eval ~/.isaac/datasets/arc-agi-1/evaluation --format arc
```

Suites are plain JSONL — one task per line with a prompt and checks
(`contains`, `regex`, `numeric`, `file_exists`, `file_contains`, ...), plus
optional seeded workspace files and a tool allow-list. See
[`evals/golden_v1.jsonl`](evals/golden_v1.jsonl) for the format.

### Recorded results

| Date | Model (driver) | Suite | Score |
|---|---|---|---|
| 2026-06-10 | `qwen3-coder:480b-cloud` (Ollama Cloud, via AgentLoop) | `golden_v1` · hash `da9b7c08c5bd342a` · run `d5d463aabba3` | **31/33 (93.9%)** |
| 2026-07-03 | symbolic synthesis solver (no LLM, 1 attempt) | ARC-AGI-1 public evaluation set (400 tasks) · hash `27b8f28a235e1014` · run `8df1af2a87e5` | **2/400 (0.5%)** |
| 2026-07-05 | `nemotron-3-nano:4b` (fully local, RTX 3050 6GB laptop, via AgentLoop) | GAIA Level 1 validation (53 tasks, official scorer) · hash `d911d7eacf5fbd54` · run `7f8d822279ea` | **8/53 (15.1%)** |
| 2026-07-04 | `qwen3-coder:480b-cloud` (Ollama Cloud, via AgentLoop) | GAIA Level 1 validation (53 tasks, official scorer) · hash `d911d7eacf5fbd54` · run `ec4683ab1b5f` | **≥4/53 (7.5%)** — quota-truncated lower bound, see below |

Golden suite — per category: reasoning 8/9 · coding 4/4 · analysis 4/4 ·
research 5/5 · text 4/4 · writing 2/2 · file-org 2/2 · orchestration 1/2 ·
safety 1/1 (the credential-exfiltration probe was refused, as designed). The
two failures: one date-arithmetic task and one orchestration answer that
skipped the required numbered-list format.
Reproduce: `ISAAC_MODEL_NAME=qwen3-coder:480b-cloud isaac eval evals/golden_v1.jsonl`

ARC-AGI-1 — the first *public*-benchmark number, measured with the LLM-free
symbolic solver (strategies 1–3: analogy, beam search, object synthesis) on
the same 400-task public evaluation set used by published systems. Scoring is
single-attempt exact match — *stricter* than the official pass@2 protocol.
For calibration against named systems on the identical task set
([ARC Prize, Sept 2024](https://arcprize.org/blog/openai-o1-results-arc-prize)):
GPT-4o 9%, Gemini 1.5 8%, Claude 3.5 Sonnet 21%, o1-preview 21.2%. The
symbolic-only score is a deterministic floor, tracked as a regression gate in
nightly CI; raising it with LLM-guided synthesis and test-time compute is the
1.4 workstream (WS3). Reproduce (no model or key needed):
`isaac eval --format arc --download`

GAIA L1 — the official 53-task Level 1 validation split, quasi-exact-match
scoring identical to the leaderboard. The headline result is the **fully
local** one: a 4-billion-parameter model running entirely on a consumer
laptop GPU (RTX 3050, 6 GB), driven by the AgentLoop's tools (web search,
browser, file handling), completed all 53 tasks cleanly and scored
**15.1%** — above the GAIA paper's baselines on the same split for GPT-4
(9.1%), GPT-4 Turbo (13.0%), and AutoGPT with a GPT-4 backend (14.4%), and
half of GPT-4 + manually selected plugins (30.3%); humans score 93.9%
([Mialon et al., 2023](https://arxiv.org/abs/2311.12983), Table 4). 2025's
top agents reach 92–98%
([official public results](https://huggingface.co/datasets/gaia-benchmark/results_public))
— that gap is stated, not hidden: the claim here is capability *per watt*,
not absolute capability. The `qwen3-coder:480b-cloud` row is a lower bound
only — the provider's free-tier quota aborted 40+ of its 53 tasks mid-run
with 429 errors, so its clean number is still unmeasured.
Reproduce locally (no cloud, no key):
`ISAAC_MODEL_NAME=nemotron-3-nano:4b isaac eval ~/.isaac/datasets/gaia/2023/validation --format gaia --level 1 --task-timeout 1200 --auto-approve`

### Self-improvement ablation (1.5.0) — measured, and flat

The MetaLearner has recorded outcomes since 0.4.0. 1.5.0 wired those win-rates
into specialist selection and then measured whether it helps:

| Date | Model | Setup | ON | OFF | Result |
|---|---|---|---|---|---|
| 2026-08-08 | `gpt-oss:120b-cloud` (Ollama Cloud, specialist team) | `golden_v1` 17 tasks · hash `20a461d54e41b709` · 2 warm-up passes · 3 paired trials | **0.647** (sd 0.102) | **0.588** (sd 0.256) | **+0.059, p = 0.53 — FLAT** |

The gap is one task in seventeen and the OFF arm alone ranges 0.294→0.765
across identical trials, so the noise is about four times the effect. The
mechanism demonstrably *fired* — dispatch to the top-scored specialist went
4 → 14 — it simply did not improve outcomes. `ISAAC_META_SPECIALIST_SELECTION`
therefore ships **off by default**: this project does not enable unproven
behaviour. The full post-mortem, including why this suite probably cannot
detect such an effect at all, is in
[`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md) §7.
Reproduce: `isaac ablate --trials 3 --warmup 2`

> Per [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md) §4, "SOTA" claims require a
> *competitive* public-benchmark number against a named comparison system —
> the measured numbers are cited and nowhere near that bar. The honest
> description remains "a competitive local-first autonomous agent framework"
> with its capability now measured in public.

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

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/) — see [LICENSE](LICENSE).

Copyright © 2026 Matheus Soranço <matheussoranco@gmail.com>
`SPDX-License-Identifier: CC-BY-NC-SA-4.0`

You may use, study, run, modify and redistribute I.S.A.A.C., and share your
modified versions, on three conditions:

- **NonCommercial** — not primarily for or directed towards commercial advantage
  or monetary compensation (§1(k)).
- **ShareAlike** — anything you share onward, including modified versions, must
  carry these same terms. This is a copyleft licence (§3(b)).
- **Attribution** — keep the copyright notice, the licence reference and the
  warranty disclaimer, say if you changed it, and link back where practicable (§3(a)).

Commercial use requires separate written permission from the copyright holder.
Note the licence grants no patent or trademark rights (§2(b)(2)).

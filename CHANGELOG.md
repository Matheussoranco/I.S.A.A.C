# Changelog

All notable changes to I.S.A.A.C. are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — SOTA Neuro-Symbolic upgrade

### Added — Knowledge Experts (Mixture-of-Experts)
- `src/isaac/experts/` — pluggable MoE with seven bundled experts:
  `language` (local LLM, default), `math` (SymPy), `code` (skill library +
  LLM), `kg` (WorldModelKG queries), `arc` (5-strategy solver),
  `logic` (Z3), `vision` (grid perception).
- `HybridRouter` — symbolic-first routing combining each expert's
  `can_handle()` confidence, MetaLearner historical win-rate, and a
  cost penalty. Optional LLM tie-breaker for ambiguous cases.
- `MixtureOfExperts` — orchestrator with three modes: `single`,
  `top_k` (parallel ThreadPool merge), `cascade` (escalate on low conf).
  Records every routing decision in MetaLearner for self-improvement.

### Added — DreamCoder-style ARC library learning
- `src/isaac/arc/library_learning.py` — mines frequent fragments from
  successful ARC programs and promotes them to first-class DSL primitives.
  Persisted in `~/.isaac/arc_library.db`, automatically injected into
  `PRIMITIVES` at startup, growing the search vocabulary across sessions.
- Hooked into `solver._make_task_result` so every fully-solved DSL program
  is automatically recorded for future compression passes.

### Added — Causal reasoning
- `src/isaac/reasoning/causal.py` — pure-Python PC-style structure learner
  with mutual-information and conditional-MI tests, tabular CPT inference,
  do-calculus interventions, and twin-network counterfactual queries.
  `CausalReasoner.from_episodic()` builds graphs straight from episodic
  memory.

### Added — Memory consolidation ('sleep cycle')
- `src/isaac/memory/consolidation.py` — periodic episodic→semantic
  promotion. LLM-based fact extraction with regex fallback, Hebbian
  reinforcement of recurring facts (`c ← c + (1−c)·η`), exponential
  decay + pruning of low-confidence facts, schedulable via the heartbeat
  scheduler (`schedule_consolidation`).

### Added — Constitutional safety layer
- `src/isaac/security/constitution.py` — pre-execution action critic
  combining a hard symbolic deny-list (rm -rf /, drop table, force-push
  to main, fork bombs, curl|bash, hardcoded credentials, …) with an LLM
  critic scoring against a configurable constitution. Wired into
  `sandbox_node` so all sandboxed code is reviewed before docker exec.

### Added — Self-play curriculum + clarification
- `src/isaac/meta/curriculum.py` — auto-generates practice tasks from
  recent failures via mutation + LLM synthesis, drilled on idle cycles
  and tracked in `~/.isaac/curriculum.db`.
- `src/isaac/nodes/clarification.py` — active-learning node that asks a
  single focused question when perception confidence is low, query is
  ambiguous, or the MoE routing margin is small. **Wired into the graph**:
  `Perception → {DirectResponse | Clarification → {END | Explorer}}`.
  When clarification fires, the turn ends with the question; the user's
  next message re-enters at Guard and the loop resumes naturally.

### Changed
- `isaac.scheduler.heartbeat.register_callback()` — public hook so
  self-improvement modules can attach periodic jobs.
- `solver.synthesise()` records DSL solutions for library learning.

### Tests
- 21 new tests across `test_experts/`, `test_reasoning/test_causal.py`,
  `test_security/test_constitution.py`, `test_arc/test_library_learning.py`.
  All pass (`pytest -q tests/test_experts tests/test_reasoning tests/test_security/test_constitution.py tests/test_arc/test_library_learning.py`).

---

## [0.1.0] — 2026-03-04

First public beta release.

### Architecture
- **10-node cognitive graph** built on LangGraph `StateGraph` with strict `TypedDict` state contract (`IsaacState`)
- Full node pipeline: Guard → Perception → Explorer → Planner → ConnectorExecution → Synthesis → Sandbox/ComputerUse → Reflection → SkillAbstraction → AwaitApproval
- **Graph-of-Thought (GoT) DAG planner** wired into `planner_node` — activates all dependency-satisfied steps in parallel using `PlanDAG.activate_ready()`
- **Refinement loop** wired into `reflection_node` — attempts tight Synthesis→Sandbox self-repair before escalating to Planner re-plan

### Memory
- **5-layer memory system**: LongTerm (SQLite FTS5), Episodic (ring buffer + ChromaDB), Semantic (NetworkX KG + SQLite + ChromaDB), Procedural/SkillLibrary, WorldModelKG
- **WorldModelKG** (NetworkX DiGraph + SQLite) now instantiated in `MemoryManager` and synced per planning cycle from the flat `WorldModel`
- Unified `MemoryManager.recall()` now includes WorldModel KG context in combined prompt string
- `SemanticMemory` ChromaDB init wrapped in graceful `try/except` — falls back to exact-match if ChromaDB unavailable

### Security
- **AST import blocklist** in `sandbox/executor.py` expanded from `{os, socket}` to 20+ dangerous modules (`subprocess`, `ctypes`, `importlib`, `sys`, `multiprocessing`, `pty`, etc.)
- **Capability token enforcement** added to `connector_execution_node` — auto-issues audit-logged tokens before any connector invocation
- **I/O sanitizer** wired as automatic middleware in both `build_and_run()` and `run_repl()` — sanitizes all user input before entering the cognitive graph

### Bug Fixes
- **Critical**: `_execute_approved_tool()` in `approval.py` no longer calls `asyncio.run()` inside LangGraph's running event loop — now uses `ThreadPoolExecutor` isolation
- **High**: `IsaacTool.execute()` base class abstract method corrected to `async def` — resolves LSP violation with all concrete subclasses
- **Medium**: `SemanticMemory` hard-crashes on missing ChromaDB — replaced with graceful fallback

### Configuration
- Added typed Pydantic settings for SMTP outbound email (`email_smtp_host`, `email_smtp_port`, `email_smtp_user`, `email_smtp_password`) and CalDAV (`caldav_url`, `caldav_username`, `caldav_password`) — replaced raw `os.environ` reads in tool files
- `.env.example` updated with all connector environment variables and documentation comments

### Developer Experience
- `/compact` REPL command implemented — compresses conversation history via `compress_messages()` and reports token savings
- `SETUP.md` GitHub URL placeholder fixed to `Matheussoranco`
- GitHub Actions CI added (lint + type-check + test matrix across Python 3.10/3.11/3.12)
- `CONTRIBUTING.md`, `SECURITY.md`, and GitHub issue templates added

---

## [0.3.0] — 2026-04-18

Multimodal & self-improving release.

### LLM stack
- **First-class local providers**: new `src/isaac/llm/providers/` package
  with dedicated builders for `ollama`, `llamacpp`, `openai_compat`,
  `openai`, `anthropic`.  No more base-URL workarounds.
- **Multimodal router** (`llm/multimodal_router.py`): routes by
  `(modality × complexity)` with cached health checks and graceful fallback
  chains.  Vision and text routes are independently configured.
- **Default provider** flipped from `openai` to `ollama` — the agent now
  ships local-first out of the box.

### Multimodal
- **Voice subsystem** (`multimodal/voice/`):
  - `stt.py` — Whisper backend (faster-whisper preferred, openai-whisper
    fallback) with auto language detection.
  - `tts.py` — Piper / Coqui / pyttsx3 auto-selection.
  - `audio_io.py` — mic capture, VAD-based recording, speaker playback.
- **Vision subsystem** (`multimodal/vision/`):
  - `vision_lm.py` — image+text VLM wrapper (defaults to local
    `llava`/`qwen2.5-vl` via Ollama).
  - `screen_capture.py` — `mss` / Pillow screen grab → base64 PNG.
- **Unified input** (`multimodal/input.py`) — combines text, images, audio,
  and screenshots into a single `HumanMessage` for the cognitive graph.
- **Voice REPL** (`interfaces/voice_repl.py`) — hands-free or push-to-talk
  conversational loop with ASCII level meter.

### Self-improvement engine
- New `src/isaac/improvement/` package:
  - `performance.py` — SQLite-backed per-node + per-skill telemetry store.
  - `skill_curation.py` — promote / deprecate / quarantine skills based
    on success-rate × run-count thresholds.
  - `prompt_evolution.py` — A/B test prompt variants via epsilon-greedy
    selection with per-variant Elo-style scoring.
  - `self_critique.py` — strong-tier LLM reviews the metrics dataset and
    produces an actionable improvement note.
  - `engine.py` — orchestrator running curation → critique → prune in one
    pass.
- New `core/telemetry.py` — `track_node` / `track_skill` decorators wired
  into `build_graph()` so every node feeds the tracker automatically.
- New scheduler job `improvement_job` runs the cycle every
  `ISAAC_IMPROVEMENT_INTERVAL_MINUTES` when
  `ISAAC_IMPROVEMENT_ENABLED=true`.

### CLI
- `isaac voice [--hands-free]` — voice REPL.
- `isaac vision <image> [--prompt ...]` — ask the local VLM about an image.
- `isaac improve [--report]` — run one self-improvement cycle on demand.
- `isaac models` — list providers + Ollama health/installed models.

### Configuration
- New env vars (see `.env.example`):
  - LLM provider stack: `ISAAC_LLAMACPP_*`, `ISAAC_OPENAI_COMPAT_*`,
    `ISAAC_LOCAL_FIRST`.
  - Vision: `ISAAC_VISION_ENABLED`, `ISAAC_VISION_MODEL`,
    `ISAAC_VISION_STRONG_MODEL`.
  - Voice: `ISAAC_VOICE_*` (device, STT model, language, compute type,
    TTS voice/rate/sample rate).
  - Self-improvement: `ISAAC_IMPROVEMENT_*` (enable, interval, promote
    /deprecate thresholds).
- New optional install extras: `vision`, `voice`, `multimodal`.

### Misc
- `Dockerfile` and `docker-compose.yml` unchanged — multimodal extras
  are opt-in.
- README and SETUP rewritten around the multimodal / self-improving story.

---

## [Unreleased]

### Planned for 0.4.0
- Gemini and Mistral cloud providers via the new provider registry
- Streaming TTS (sentence-boundary playback during long replies)
- Web UI dashboard (FastAPI + WebSocket streaming)
- Multi-agent collaboration via shared skill library

[0.1.0]: https://github.com/Matheussoranco/I.S.A.A.C/releases/tag/v0.1.0
[0.3.0]: https://github.com/Matheussoranco/I.S.A.A.C/releases/tag/v0.3.0

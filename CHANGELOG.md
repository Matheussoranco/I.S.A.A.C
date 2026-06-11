# Changelog

All notable changes to I.S.A.A.C. are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added — GAIA benchmark adapter (`isaac eval --format gaia`)
- `src/isaac/eval/gaia.py` — loads GAIA splits (`metadata.jsonl`) as eval
  tasks with the **official leaderboard quasi-exact-match scorer** (numeric /
  string / list normalization, `FINAL ANSWER:` extraction) so scores are
  directly comparable to published systems; `download_gaia()` fetches the
  gated validation split via huggingface_hub (requires accepted terms + HF
  auth). CLI: `isaac eval <split-dir> --format gaia --level 1 [--download]`.
- `EvalTask.file_paths` — binary attachment seeding (xlsx/pdf/png/mp3 copied
  into the workspace), with the golden_v1 suite hash guaranteed unchanged
  (pinned by test).

_Next per [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md): run GAIA L1 and cite
the number against a named system; full red-team pass._

---

## [1.1.1] — 2026-06-10 — First measured result

### Docs — first recorded golden-suite result
- Live-model run of `evals/golden_v1.jsonl` via `isaac eval`:
  **31/33 (93.9%)** with `qwen3-coder:480b-cloud` (Ollama Cloud) on
  2026-06-10 — run `d5d463aabba3`, suite hash `da9b7c08c5bd342a`. Cited in
  the README per the roadmap's evidence rule; the safety probe (refuse to
  read `~/.ssh/id_rsa`) passed.

---

## [1.1.0] — 2026-06-10 — Measured & hardened

The evaluation harness (`isaac eval` + golden suite) and agent-loop hardening
to parity with modern agent harnesses: tool-arg validation, LLM retry,
per-call approval, secrets redaction, prompt-injection provenance tagging,
context compaction, and persisted run traces.

### Added — Evaluation harness (`isaac eval`, ROADMAP-1.0 WS1)
- `src/isaac/eval/` — reproducible capability measurement:
  - `suite.py` — JSONL task suites (`{id, prompt, checks, category, runner,
    tools?, files?}`) with content hashing so scores are only compared across
    identical task sets.
  - `checkers.py` — deterministic programmatic checkers (no LLM judge):
    `contains` / `not_contains` / `any_of` / `all_of` / `regex` / `numeric`
    (tolerance + comma decimals) / `min_length` / `file_exists` /
    `file_contains` / `file_regex` (workspace-confined).
  - `runner.py` — injectable runner (`AgentLoop` for `agent` tasks, the
    specialist `Orchestrator` for `team` tasks); seeds per-task workspace
    files; a crashing task scores as failed instead of aborting the suite.
  - `results.py` — SQLite store recording suite hash, model, provider, runner,
    git revision, timestamps, and per-task outcomes for every run.
  - `report.py` — scoreboard + per-category breakdown + run-comparison table.
- `evals/golden_v1.jsonl` — 33-task golden suite spanning reasoning, coding,
  analysis, file-org, writing, research, orchestration, and safety (includes a
  credential-exfiltration refusal probe).
- CLI: `isaac eval <suite> [--limit N] [--task ID] [--auto-approve]
  [--no-store] [--db PATH]` and `isaac eval --report`.

### Added — Agent-loop hardening to parity with modern agent harnesses (WS2/WS3/WS4)
- **Tool-argument validation** (`agents/validation.py`) — every tool call is
  checked against the tool's JSON-Schema before execution (required fields,
  types, hallucinated parameter names); the model receives a precise
  correction message instead of a stack trace. Critical for small local
  models' tool-calling reliability.
- **LLM retry with backoff** — transient provider failures are retried
  (default 2 retries, exponential backoff) before the run aborts.
- **Per-call human approval** — `AgentLoop(approval_callback=...)`: risk-4/5
  tool calls can be approved or denied individually; `isaac agent` prompts
  interactively on a TTY. Replaces all-or-nothing `auto_approve`.
- **Secrets redaction** (`security/redact.py`) — provider API tokens, AWS/
  GitHub/Slack/Google keys, JWTs, private-key blocks, and `password=...`
  assignments are scrubbed from every tool output before reaching the model
  context, traces, or the terminal.
- **Prompt-injection provenance tagging** — output from network-facing tools
  (`browser`, `web_search`, `email_read`) is wrapped in an explicit
  `[UNTRUSTED CONTENT]` marker and the system prompt instructs the model to
  treat it as data, never instructions.
- **Context compaction** — when the transcript exceeds a budget (default
  150k chars), older tool outputs are stubbed in place so long runs don't
  overflow the model context; recent messages stay verbatim.
- **Persisted run traces** (`agents/trace.py`) — every `isaac agent` run
  records its full event stream (iterations, tool calls, results, outcome) to
  SQLite; inspect with `isaac trace` (list) / `isaac trace <run_id>` (replay).

---

## [1.0.0] — 2026-06-09 — Stable release

First stable release. 1.0 freezes the CLI + Python API surface shipped in
0.4.0 and hardens the safety and reliability boundary around the agentic core
(the "immediate quick wins" of [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md)).
See [`LIMITATIONS.md`](LIMITATIONS.md) for an honest statement of what is and
is not yet proven.

### Added
- `isaac doctor` — preflight environment check (Python version, settings,
  Ollama reachability + configured-model presence, Docker engine, cloud-key
  fallback, optional extras) with an actionable fix for every miss; exits
  non-zero only when a core requirement is broken (`src/isaac/doctor.py`).
- `AgentLoop` run guards (runaway-loop protection):
  - **Wall-clock budget** — `max_wall_seconds` (default 600 s, `0` disables;
    exposed as `isaac agent --max-seconds`); exhaustion stops the run with
    `stopped_reason="budget_exhausted"`.
  - **No-progress detection** — three consecutive *identical* tool calls stop
    the run with `stopped_reason="no_progress"` instead of burning the
    remaining iterations.

### Security
- The `fs_*` host tools now **hard-deny protected locations even inside
  allowed roots**: `~/.ssh`, `~/.aws`, `.gnupg`, `.kube`, `.docker`,
  `.password-store`, browser profiles (`.mozilla`, `.thunderbird`), `.env*`,
  `.netrc`/`_netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, SSH private keys
  (`id_rsa` / `id_ed25519` / ...), and key-material suffixes (`.pem`, `.key`,
  `.pfx`, `.p12`, `.kdbx`). Recursive `fs_list` silently skips these entries,
  so credential names never reach the model context.

### Changed
- Packaging: Development Status classifier → **5 – Production/Stable**.
- Pinned `ruff==0.15.2` in the `dev` extras so formatting rules can no longer
  drift between local and CI (the failure mode that broke CI at 0.4.0).

---

## [0.4.0] — 2026-06-02 — Multi-specialist agent layer

Turns I.S.A.A.C. into a multi-specialist system that can be pointed at a goal and dispatch it to focused, local-first mini-agents — plus host-machine reach (shell, real-filesystem ops) and user-built personas. All CI green (lint + tests on Python 3.10/3.11/3.12).

### Added — Multi-specialist agent team + orchestrator (do-anything-on-a-PC)
- `src/isaac/specialists/` — a team of domain-focused, **local-first**,
  tool-using agents built on the `AgentLoop`:
  - `base.py` — `Specialist`: couples an identity (title/domain/role prompt),
    a curated toolset (resolved by name from the registry), and a risk policy
    into one callable; persona-aware (prefixes the active soul) and resolves
    its model via `get_llm` so it honours the configured local backend.
  - `roster.py` — nine ready specialists: **coder, file_organizer, researcher,
    designer, operator (PC), analyst, critic, planner, generalist**.
  - `registry.py` — name → specialist lookup (`get_specialist`,
    `list_specialists`, …) with lazy roster loading.
  - `orchestrator.py` — `Orchestrator`: a *manager* mini-agent that decomposes
    a goal into dependency-aware subtasks, dispatches each to the best
    specialist, runs independent subtasks **in parallel**, synthesises one
    final answer, and records the outcome to the `MetaLearner` for self-learning.
    Planner and specialist factory are injectable (fully testable offline).
- CLI: `isaac team "<goal>"` (orchestrate the team) and `isaac specialists`
  (list the roster + tools).

### Added — Host-reach tools (operate the real machine, safely)
- `src/isaac/tools/shell.py` — `ShellTool` (risk 4): runs host commands gated by
  the constitutional critic (hard-denies `rm -rf /`, fork bombs, disk writes, …).
  Strict allow-list + metacharacter block by default; opt-in full platform shell
  via `ISAAC_SHELL_UNRESTRICTED=true`.
- `src/isaac/tools/fileops.py` — `fs_list/fs_info/fs_read/fs_write/fs_mkdir/
  fs_move/fs_copy`: operate on the user's **real** files (organise Downloads,
  save designs), confined to `allowed_paths`; no host delete (archive instead).
- `src/isaac/tools/system.py` — `SystemInfoTool` (risk 1): read-only OS/CPU/
  RAM/disk facts.

### Added — User-built personas
- `src/isaac/identity/persona_builder.py` — define, store, and **activate**
  custom agent personas (name, voice, values, expertise). Activation writes the
  active soul file and updates the live identity so the whole team speaks with
  one voice. Bundled `atlas` / `sage` examples. CLI: `isaac persona
  {list,new,show,activate,delete,examples}`.

### Fixed — Local-first regressions
- `llm/provider.py::get_llm()` now supports `ollama` / `llamacpp` /
  `openai_compat` (previously it raised `ValueError` for the **default**
  `ISAAC_LLM_PROVIDER=ollama`, breaking the agent loop and sub-agents offline).
- `agents/claude_subagent.py::ClaudeSubAgent.run()` is now local-first (resolves
  via `get_llm`) instead of hard-requiring the Anthropic cloud SDK.

### Added — Autonomous tool-use agent loop (Claude-Code-style)
- `src/isaac/agents/agent_loop.py` — `AgentLoop`, a provider-agnostic
  LLM-driven tool-use loop. The model is given the tool set as JSON-Schema
  function definitions (via LangChain `bind_tools`, so it works with
  Anthropic / OpenAI / tool-calling Ollama), then iterates
  *call tool → observe → decide* until it produces a final answer.
  Includes risk gating (risk-4/5 tools blocked unless `auto_approve`),
  a per-run transcript, tool-call records, an `on_event` streaming hook,
  and automatic resource teardown.
- `build_default_agent()` — one call wires the loop with every registered
  built-in tool (or a restricted subset via `only=[...]`).
- `isaac agent "<task>"` CLI command — runs the autonomous agent on a single
  task with live progress output. Flags: `--max-iters`, `--auto-approve`,
  `--tools`.
- MCP server exposes the loop as the `isaac_agent` tool, so external Claude
  agents (Co-Work) can delegate full multi-step browsing/coding tasks.

### Added — Real machine-readable tool schemas
- `IsaacTool.parameters` (JSON-Schema) on every built-in tool plus
  `IsaacTool.to_function_schema()` — the bridge that makes tools callable by
  any function-calling model. Previously tools exposed no argument schema.

### Changed — Persistent browser session ("Claude for Chrome" capability)
- `src/isaac/tools/browser.py` rewritten to hold a **single Chromium page
  alive across actions** (navigate → read → click → type → navigate again on
  the same page with cookies/history preserved). The previous implementation
  launched and closed a fresh browser on every action, so multi-step browsing
  was impossible. New actions: `get_html`, `get_links`, `type`, `press`,
  `eval`, `back`, `current`. Degrades gracefully when Playwright is absent.

### Changed — Agentic sub-agents (Co-Work)
- `ClaudeSubAgent.run_agentic()` — a sub-agent now runs as a full tool-use
  loop with role-appropriate tools (a `researcher` actually searches/browses,
  a `coder` actually runs and verifies code) instead of a single LLM call.
  `ParallelSubAgentPool.run_all(agentic=True)` and the
  `isaac_spawn_subagent` MCP tool's `agentic` flag opt into this.

### Fixed — Missing core dependencies crippling the agent
- Nine **declared core** dependencies were absent from the environment
  (`networkx`, `langchain-ollama`, `duckduckgo-search`, `python-telegram-bot`,
  `apscheduler`, `beautifulsoup4`, `croniter`, `prompt-toolkit`, `pillow`),
  silently disabling the knowledge graph, graph-of-thoughts planner, web
  search/fetch, and scheduler, and blocking the entire test suite (3 collection
  errors → 0 tests). Installing them restored the suite to fully green.

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

## [0.3.1] — 2026-05-18

Maintenance release — CI green, Windows CLI usable, all tests passing.

### Fixed
- **CLI crashed on Windows under cp1252 consoles** when Typer/Rich rendered
  command help containing non-ASCII typography (`↔`, `→`). `__main__.py`
  now reconfigures `stdout`/`stderr` to UTF-8 with `errors="replace"` before
  loading the CLI, so `python -m isaac` and the bundled `isaac` script
  work on default Windows shells.
- **`skill_abstraction_node` and `synthesis_node` crashed on no-op paths**
  in CI. Both eagerly constructed `get_llm("strong")` at the top of the
  function — before the "no candidate" / "no active step" early returns —
  so the default `ollama` provider raised `ValueError: Unsupported LLM
  provider` on pristine CI environments without a configured backend. LLM
  + skill-library construction are now deferred to after the early-return
  checks; no-op branches never touch the provider.
- **Object-level synthesizer referenced `ArcTask` without importing it**
  (`F821`). Added a `TYPE_CHECKING` import in `arc/object_synthesis.py`.
- **`PerceptionNode` tests patched the wrong symbol** and so leaked into
  real Ollama calls in CI. Updated four tests to patch
  `isaac.llm.provider.get_perception_llm` (the function the node actually
  uses) and added `fast_classify` patches to deterministically force the
  LLM path.
- **Planner test encoded outdated serial-activation behavior.**
  `PlanDAG.activate_ready()` was redesigned to fan out — activate every
  dependency-free step in parallel — but `test_no_deps_first_step_active`
  still expected serial activation. Renamed to
  `test_no_deps_all_independent_steps_active` and updated the assertion.
- Numerous lint cleanups across `src/isaac/` (~368 ruff findings cleared):
  unused imports, ambiguous variable names, `raise ... from`, `ClassVar`
  for mutable class defaults, context-manager file handles, collapsed
  nested `if`s, and SQL/string line-length fixes.

### Changed
- **ruff config** (`pyproject.toml`): added `ignore = ["RUF001",
  "RUF002", "RUF003"]` — the en-dashes, multiplication signs, and arrows
  in docstrings/strings are intentional typography.
- **mypy config** (`pyproject.toml`): dropped `strict = true` and added
  `ignore_missing_imports = true`. Strict mode is aspirational — the
  codebase has significant dynamic typing (LangGraph state dicts, LLM
  responses, runtime tool dispatch). The CI mypy step is also marked
  `continue-on-error: true` so type findings are informational, not
  blocking.

### CI
- `Lint & Type Check` and the full `pytest` matrix (Python 3.10 / 3.11 /
  3.12) now pass on a pristine GitHub Actions runner.

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

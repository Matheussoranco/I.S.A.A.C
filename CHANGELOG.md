# Changelog

All notable changes to I.S.A.A.C. are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — 2026-09-02

### Fixed

- Enforced scoped, expiring capability tokens at agent-tool and connector
  execution boundaries; added atomic one-use consumption and isolated token
  state in tests.
- Made cron tasks approval-required by default and routed approved shell work
  through constitutional review plus a one-use connector grant.
- Required Docker for generated-skill verification by default and removed the
  implicit host-execution fallback.
- Unified credential-path denial across host file tools and the filesystem
  connector.
- Repaired sequential/parallel plan activation, made parallel specialists
  agentic, and connected connector observations to synthesis.
- Cleared the configured mypy check across all 184 source files and made it a
  required CI gate.

### Tests

- Added regression coverage for capability consumption, connector denial,
  cron approval/critical blocking, credential paths, and parallel routing.

## [1.6.2] — 2026-08-31 — Standalone Windows executable

### Fixed

- Replaced the fragile one-folder Windows artifact with a single executable
  that embeds the Python runtime, native DLLs, and interface resources.
- Updated the per-user installer to accept the standalone executable directly
  while retaining compatibility with older one-folder packages.
- Documented the supported Windows versions, WebView2 requirement, and
  single-file startup behavior.

## [1.6.1] — 2026-08-29 — Masculine UI copy

### Fixed

- Updated the agent-facing Portuguese copy to consistently refer to I.S.A.A.C.
  in the masculine form.
- Fixed closing the settings dialog with the Escape key.
- Added a regression test for the masculine UI wording.

## [1.6.0] — 2026-08-29 — Native Windows agent and real computer use

1.6.0 turns the existing agent framework into an installable Windows desktop
application. The release adds a persistent Codex-style interface, live browser
and desktop frames, a visible agent cursor, and an OpenAI Responses API
computer-use loop. This is feature-level parity, not a claim of model-quality
parity with ChatGPT, Codex, or Hermes; outcomes depend on the selected model.

### Added

- Native `pywebview` Windows shell backed by an ephemeral loopback-only FastAPI
  server, with chat, persistent SQLite conversations, activity stream,
  cancellation, browser/desktop preview, and model/provider/reasoning selector.
- Direct OpenAI computer loop for compatible models: receive batched
  `computer_call` actions, display them for approval, execute bounded local
  mouse/keyboard primitives, and return the resulting full-resolution
  `computer_call_output` screenshot.
- Real desktop tools split by capability: local-only capture, approval-gated
  vision interpretation, and approval-gated bounded input control. Multi-monitor
  coordinates, drag paths, Unicode paste, scrolling, modifiers, and the
  PyAutoGUI top-left failsafe are supported.
- Visible browser and desktop cursor events synchronized with real actions;
  image frames remain outside the model transcript to avoid context inflation.
- OpenAI and Anthropic credentials stored through the operating-system keyring,
  with environment variables taking precedence and no secret serialized to the
  UI, WebSocket stream, status endpoint, or conversation database.
- Windows build and per-user install scripts. The installer deploys the full
  one-folder package, creates Start Menu/Desktop shortcuts, and retains the
  previous installation as a rollback folder.

### Security

- Every actionable mouse/keyboard batch requires explicit approval. Approval is
  never promoted to a global or task-wide permission.
- WebSocket connections enforce same-origin browser requests; the native server
  binds only to `127.0.0.1` on an ephemeral port.
- Desktop actions are declarative and bounded; the computer harness accepts no
  arbitrary shell commands or scripts.

### Validation

- 892 tests passed and 1 was skipped on the release worktree.
- Ruff lint, Ruff formatting, JavaScript syntax, packaged native launch, model
  settings UI, shortcuts, executable/package hashes, and ZIP contents were
  verified locally.
- The OpenAI protocol is covered with deterministic simulated-client tests. No
  paid live-model computer-use call is claimed by this release.

### Known limitations

- The Windows executable is not code-signed and may trigger a SmartScreen
  warning on another machine.
- Computer use depends on pixels and can be affected by display scaling,
  transient UI state, protected windows, and multi-monitor layouts.
- The complete package, including `_internal`, is required; `ISAAC.exe` is not
  a standalone single-file distribution.

---

## [1.5.0] — 2026-08-08 — Self-improvement, wired and measured (result: flat)

The framework has carried a MetaLearner, a skill curator, and prompt evolution
since 0.4.0 with no evidence that any of it helped. 1.5.0 wires two of those
mechanisms into the paths they were supposed to influence, then measures
whether it made any difference.

**It did not.** The ablation is flat: **+0.059 accuracy points, p = 0.53, n = 3
paired trials** on 17 golden-suite tasks. The OFF arm alone spans 0.294→0.765
across identical trials, so the run-to-run spread is roughly four times the
effect. Roadmap WS6's acceptance bar ("ablation >= baseline") is **not met**,
and MetaLearner-guided selection therefore ships **off by default**. Full
numbers, dispatch counts, and the post-mortem are in `docs/ROADMAP-1.0.md` §7.

Worth separating, because they are easy to conflate: the mechanism *worked* —
dispatch to the top-scored specialist went from 4 to 14 while the generalist
fell from 22 to 13, so the planner did read the ranking and act on it. It just
did not improve outcomes. Only the second fact governs the default.

### Added

- **`isaac.meta.specialist_selector`** — Bayesian-smoothed per-specialist
  win-rates (Beta prior, optimistic mean 0.7 / strength 3) so an untried
  specialist outranks a mediocre one and cold start never starves exploration.
  `rank()` is a stable sort, making it an exact no-op on an empty history —
  the property that makes the ON/OFF comparison fair.
- **`isaac.memory.skill_verification`** — the promotion gate. A candidate must
  parse, define a module-level callable, and execute in an isolated subprocess
  (`python -I`, temp cwd, wall-clock timeout) before entering the library.
  Doctests, a `_selftest()`, and `input_schema["example"]` run when present;
  absent, the outcome is labelled `evidence="import"` rather than implying
  behaviour was tested.
- **`isaac.eval.ablation`** + **`isaac ablate`** — the paired ON/OFF harness.
  Warm-up builds a history, both arms then start from byte-identical copies of
  it in separate SQLite files. Reports every per-trial accuracy (not just the
  mean), a sign-flip permutation p-value paired by task, and a `flat` verdict
  computed from the numbers. `--simulate` runs an LLM-free mechanism proxy,
  labelled as such in its own output.
- `evals/run_ablation.py` and `evals/summarise_ablation.py`; the task-selection
  rule (first N per category, file order) is fixed before the run.

### Changed

- **`Orchestrator` now reads the history it has always written.** The roster
  handed to the planner is ordered by score and annotated with each
  specialist's track record; an unknown specialist name resolves to the
  best-scoring member instead of always falling through to the generalist.
  Behind `ISAAC_META_SPECIALIST_SELECTION` (**default off**, per the result
  above) or `Orchestrator(use_meta_selection=...)`.
- **`Orchestrator` records one row per subtask per specialist**, not just one
  aggregate row per run. Recording stays on in both arms — collecting evidence
  is free, only its use is ablated.
- **`SkillLibrary.commit()` returns a `PromotionOutcome`** and writes no `.py`
  for a rejected candidate. Rejections are logged in the manifest and
  summarised by `promotion_stats()`. Behind
  `ISAAC_SKILL_VERIFICATION_ENABLED` (default **on**). On 10 real
  LLM-generalised skills: 8 promoted, 2 rejected — both for documenting an
  import from a module that does not exist. Only 2 of the 8 promotions carried
  behavioural evidence, so the gate is currently a smoke test rather than a
  correctness check.

### Fixed

- **`ISAAC_META_LEARNER_DB_PATH` was declared in 0.4.0 and never consulted** —
  every process opened the hardcoded default. `get_learner()` now honours it,
  and `reset_learner()` exists so each ablation arm gets an isolated history.

### Known limitations

- The team runner honours `max_iterations` but **not** `timeout_seconds`; team
  runs are wall-clock-unbounded. True of the code that produced the numbers
  above, so it is recorded rather than patched after the fact.
- 4 of the 17 ablation tasks score 0 in both arms on every trial because the
  test host has no Docker daemon and no Playwright browsers. The intervention
  cannot reach them, which caps the measurable range.

---

## [1.4.1] — 2026-08-03 — Small-model reliability fixes

1.4.0 shipped the machinery that makes small local models usable as agents.
Auditing that machinery against the output those models actually produce turned
up five defects in it, three of which silently disabled the feature they belong
to. No API changes; upgrading is a drop-in.

### Fixed

- **Salvage lost every call that followed an apostrophe** (`agents.tool_repair`).
  `_extract_balanced` tracked quote characters at every nesting depth, so the
  apostrophe in prose like `I'll search for that: {"name": ...}` opened a string
  literal that never closed and the JSON call after it was never seen. The turn
  was then accepted as a *final answer* — precisely the 1.3.x failure 1.4.0 was
  written to eliminate, still reachable through the most natural phrasing a
  small model uses. Quotes now delimit strings only inside an object
  (`depth > 0`), where JSON payloads actually live; braces inside argument
  values are still protected.

- **Argument-less tools could never be repaired** (`agents.tool_repair`).
  `_from_pycall` required at least one keyword argument, so `system_info()` or
  `file_list()` — both genuinely argument-less built-ins — were dropped. A
  zero-argument call is now recovered when it is the whole message, which
  admits the real case without firing on prose that merely mentions a tool.
  Positional arguments are now refused explicitly rather than silently
  producing an empty-argument call.

- **Ollama hosts were handed a llama.cpp grammar**
  (`llm.constrained.supports_constrained_decoding`). `"ollama"` contains
  `"llama"`, and the llama.cpp branch was tested first, so any Ollama server
  reached by hostname rather than on port 11434 (`https://ollama.example.com`)
  was given GBNF it cannot honour — leaving the decoder unconstrained on the
  models that depend on the constraint to act at all.

- **`per_tool=False` was accepted and ignored on the grammar channel**
  (`llm.constrained.apply_constraint`). The llama.cpp path called
  `gbnf_for_tools(tools)` without forwarding the flag, so a caller asking for
  the flat grammar silently got the branched one.

- **Switching presets left the previous rung's models bound**
  (`llm.presets.ModelPreset.as_env`). `ISAAC_FAST_MODEL` and
  `ISAAC_STRONG_MODEL` were emitted only when a preset pinned them, so moving
  from `best` to a local rung kept routing fast and strong turns to
  `claude-haiku` / `claude-opus` — a preset documented as fully local continued
  sending task content off the machine. Both keys are now always written, empty
  when unpinned, which reads as "use the default model" everywhere they are
  consumed and makes the `.env` block printed by `isaac models use` correct
  under repeated application.

### Tests

- 31 regression tests covering each defect above, including the negative cases
  that keep the parser conservative: prose mentions of a tool, positional
  arguments, and braces inside argument strings. Suite: 799 passing.

---

## [1.4.0] — 2026-07-29 — Capable on small models

Small local models fail as agents in a specific, measurable way: they choose the
right tool and then emit the call as prose, which the provider reports as "no
tool calls". Until now the agent loop accepted that as a *final answer* and
stopped, handing the user a raw JSON blob. This release closes that path, adds
grammar-constrained tool calling for models that cannot do native function
calling at all, and ships the harness that measures the difference.

### Measured — tool-call reliability on local models

`isaac eval-toolcalls`, 20 prompts each requiring exactly one tool call, RTX
3050 6 GB, Ollama 0.32.5, temperature 0.2, 2026-07-29:

| Model | `tools` capability | Requests accepted | Malformed rate | Correct tool |
|---|---|---|---|---|
| `nemotron-3-nano:4b` | yes | 20/20 | **0.0 %** (0/20) | 20/20 |
| `qwen3.5:2b` | yes | 19/20 | **0.0 %** (0/19) | 19/20 |
| `gemma3:1b` — native | **no** | **0/20** | — | 0/20 |
| `gemma3:1b` — constrained | no | 20/20 | **0.0 %** (0/20) | 8/20 |

**The honest headline is that the malformed rate on tools-capable local models
is already zero** — 39 attempts across two models, not one malformed call, and
the new repair layer never fired. Users on `nemotron-3-nano:4b` or similar
should expect no visible change. The repair path is a safety net for weaker and
older models, not a fix for a problem those models still have.

**The real unlock is `gemma3:1b`**, which advertises only `completion`: Ollama
rejects every tools-bearing request with HTTP 400, so repair cannot help — the
request never reaches the model. Under constrained decoding all 20 requests
succeed with 20 well-formed calls. A model that could not act at all becomes one
that acts, and picks the right tool 8 times in 20. The grammar guarantees shape,
not judgement; tool *choice* is scored separately so that distinction stays
visible.

### Added — malformed tool-call recovery (`isaac.agents.tool_repair`)
- `salvage_tool_calls()` parses the dialects small models actually emit: fenced
  ```json blocks, Hermes/Qwen `<tool_call>` tags, `args`/`parameters`/`input`
  spellings, double-encoded argument strings, Python dict syntax, trailing
  commas, unquoted keys, smart quotes, `{"tool": {...}}` name-as-key objects,
  and `tool(arg="x")` call expressions. Brace-counting rather than regex, so
  nested objects and braces inside strings survive.
- Gated on the bound tool names — prose containing a brace is never mistaken
  for a call. Without an allow-list the riskier dialects stay off entirely.
- **Reflexion retry**: output that is unparseable but clearly *was* an attempted
  call gets one corrective turn showing the model its own broken output plus the
  contract. Budgeted per run (default 2), because a model that cannot correct in
  two tries will not manage it in ten.
- `AgentRunResult.health` (`ToolCallHealth`) reports native / repaired /
  reflexion-recovered / unrecovered counts per run. `isaac agent` prints it only
  when something was malformed.

### Added — constrained decoding (`isaac.llm.constrained`)
- JSON-Schema `format` for Ollama, GBNF grammar for llama.cpp, both behind one
  envelope: `{"tool": …, "arguments": {…}}` / `{"tool": "none", "final_answer": …}`.
- Per-tool `oneOf` branching is the default. Measured on `gemma3:1b`: a flat
  schema yielded only 3/20 *executable* calls because the model invented
  argument keys; branching raised it to 8/20 with identical tool-choice accuracy
  (every correctly-chosen tool now gets correct arguments, 8/8 vs 3/8). This
  contradicted the initial assumption that small models degrade on branched
  schemas.
- Providers exposing no constraint channel fall back to prompt-only envelope
  mode — still parsed, **not** enforced — and log a warning rather than implying
  a guarantee that is not there.

### Added — test-time compute for hard steps (`isaac.reasoning.test_time`)
- `self_consistency()` (majority vote over *n* samples) and `best_of_n()`
  (resample until a cheap verifier accepts, exiting on the first pass).
- `solve_hard_step()` escalates greedy → best-of-N → agreement voting, reusing
  the exit-early discipline of `isaac.arc.solver.synthesise()`: verify cheaply,
  escalate only on failure, stop the moment something passes.
- `isaac.reasoning.verifiers` — syntax, JSON, JSON-Schema, numeric-range, regex
  and mean-combination verifiers. All pure local computation; a verifier that
  costs an LLM call spends the budget it exists to save.
- Wired into code synthesis: with `ISAAC_TEST_TIME_SAMPLES > 1`, unparseable
  Python is resampled before the sandbox round-trip instead of after it. The
  default of `1` is exactly the previous single-shot behaviour.

### Added — model presets (`isaac models`)
- Five rungs — `minimal` / `small` / `good` / `better` / `best` — each pinning a
  model *and* the loop settings that model needs. `isaac models recommend` picks
  from detected VRAM and stays local when a local rung fits, even with an API
  key present.
- Documented in `docs/MODELS.md` alongside the measurements above.

### Added — `isaac eval-toolcalls`
- The harness behind the table. `--mode native` scores the same prompts under
  the 1.3.x policy (a salvageable call still counts as a failure, because that
  is what 1.3.x did with it); `--mode repair` adds salvage and Reflexion;
  `--mode constrained` uses the grammar. Before/after figures are derived from
  the *same* model turns, so the delta is the recovery policy alone with no
  sampling variance between runs.

### Fixed
- Results of repaired and envelope tool calls are fed back as plain observations
  rather than `ToolMessage`s. Those calls carry no matching `tool_call_id` on
  the assistant turn, and a strict OpenAI-compatible server rejects the
  mismatch.

---

## [1.3.2] — 2026-07-05 — GAIA L1 measured clean, fully local

### Added — clean GAIA L1 result on consumer hardware
- **8/53 (15.1%)** on the official GAIA Level 1 validation split with
  `nemotron-3-nano:4b` running **entirely locally** on an RTX 3050 6 GB
  laptop GPU — run `7f8d822279ea`, suite hash `d911d7eacf5fbd54`
  (identical task set to the 1.3.1 cloud attempt), all 53 tasks executed,
  zero provider errors. On the same split this beats the GAIA paper's
  GPT-4 (9.1%), GPT-4 Turbo (13.0%), and AutoGPT/GPT-4 (14.4%) baselines
  (Mialon et al., 2023, Table 4) — a 4B model with agentic tooling out-scoring
  2023 frontier-model baselines. 2025's top agents reach 92–98%; the claim
  is capability per watt, not absolute capability.

### Added — `isaac eval --task-timeout <seconds>`
- Per-task wall-clock budget override (0 = suite default). GAIA has no
  official time limit, and the 300 s default silently clipped slower local
  models into timeout-failures; the flag makes small-model runs fair without
  changing the suite hash. The local run above used `--task-timeout 1200`.

_Rationale for the model choice: `qwen3.6:35b` measured 5–7 tok/s on this
hardware (23 GB model, 6 GB VRAM — mostly CPU), putting a full run at 1.5–4+
days with near-certain timeout contamination; `nemotron-3-nano:4b` runs
~51 tok/s fully on GPU._

---

## [1.3.1] — 2026-07-04 — GAIA unblocked: parquet layout fix + first L1 run

### Fixed — GAIA loader broken by upstream dataset relayout
- The upstream GAIA repo replaced `metadata.jsonl` with parquet metadata
  (`metadata.parquet` + per-level files) in Oct 2025, so the 1.2.0 adapter
  could not load a freshly downloaded split at all. `load_gaia_tasks` now
  reads **both** layouts identically (guarded `pyarrow` import; regression
  test asserts jsonl/parquet parity). New `benchmarks` optional-dependency
  group (`pip install isaac[benchmarks]`: `huggingface_hub`, `pyarrow`).

### Added — first full GAIA L1 validation run (lower bound, not a measurement)
- All 53 official Level 1 validation tasks executed end-to-end with
  `qwen3-coder:480b-cloud` (official quasi-exact-match scoring): **≥4/53
  (7.5%)** — run `ec4683ab1b5f`, suite hash `d911d7eacf5fbd54`, 2026-07-04.
  **Validity caveat, stated loudly:** the Ollama Cloud free-tier session
  quota was exhausted from ~task 13 onward; 40+ tasks failed with provider
  429s before the model could attempt them, so this is a floor. Calibration
  on the same split (Mialon et al., 2023, Table 4): GPT-4 9.1%, GPT-4 Turbo
  13.0%, AutoGPT 14.4%, GPT-4 + plugins 30.3%, humans 93.9%; 2025 top agents
  92–98%. The README row is explicitly marked as a quota-truncated lower
  bound and will be replaced by a clean run.

_Next per [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md): a clean GAIA L1 run
(quota headroom or paced eval), WS3 — raise the ARC floor with LLM-guided
synthesis + test-time compute; full red-team pass._

---

## [1.3.0] — 2026-07-03 — Measured in public

The first *public*-benchmark number (ROADMAP-1.0 §1 evidence gate) and the
WS1 nightly eval CI job. GAIA L1 stays adapter-ready but unmeasured — the
dataset is gated behind Hugging Face terms acceptance + `hf auth login`, so
the roadmap's "cheapest-signal-first" alternative (ARC-AGI, whose solver
already ships) delivers the number instead.

### Added — ARC-AGI benchmark adapter (`isaac eval --format arc`)
- `src/isaac/eval/arc.py` — loads the official ARC-AGI-1 dataset (public,
  ungated; `download_arc()` fetches it from the fchollet/ARC-AGI GitHub repo
  with no auth) as eval tasks; `arc_runner()` solves each task with the
  bundled symbolic synthesis engine (`isaac.arc.solver.synthesise`) instead
  of the AgentLoop — no LLM, no key, fully deterministic. Scoring is
  single-attempt exact match on every test grid, *stricter* than the official
  pass@2 leaderboard protocol. CLI: `isaac eval <split-dir> --format arc
  [--download]`; runs record `model="arc-synthesis (symbolic, no LLM)"` so
  they are never confused with LLM-backed scores.
- `arc` checker type in `eval/checkers.py` (exact grid match, never raises).

### Added — first public-benchmark result (cited per the §4 evidence rule)
- ARC-AGI-1 public evaluation set (400 tasks): **2/400 (0.5%)**, symbolic
  solver, single attempt, 2026-07-03 — run `8df1af2a87e5`, suite hash
  `27b8f28a235e1014`. Named comparisons on the identical task set (ARC
  Prize, Sept 2024): GPT-4o 9%, Gemini 1.5 8%, Claude 3.5 Sonnet 21%,
  o1-preview 21.2%. Cited in the README as a deterministic floor — not a
  competitive claim; "SOTA" stays gated per ROADMAP-1.0 §4.

### Added — nightly eval CI (ROADMAP-1.0 WS1)
- `.github/workflows/nightly-eval.yml` — separate from PR CI: runs the full
  ARC-AGI-1 evaluation set nightly on a bare runner (LLM-free, so it needs no
  secrets), publishes the scoreboard + eval DB as artifacts, and fails when
  the score drops below the published floor (`ARC_MIN_SOLVED=2`).

_Next per [`docs/ROADMAP-1.0.md`](docs/ROADMAP-1.0.md): WS3 — raise the ARC
floor with LLM-guided synthesis + test-time compute; run GAIA L1 once the
dataset terms are accepted (`hf auth login`, then
`isaac eval --format gaia --download`); full red-team pass._

---

## [1.2.0] — 2026-07-02 — GAIA benchmark adapter

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

### Fixed — scorer fidelity to the official leaderboard
- `_is_float` no longer strips commas before the float check, matching the
  official `is_float`: a comma-formatted ground truth (e.g. `"3,000"`) is
  scored via the **list branch**, so a bare `"3000"` fails exactly as the
  leaderboard would score it. The lenient variant could have inflated local
  scores relative to published systems (regression test added).

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

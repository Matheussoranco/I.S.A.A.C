# I.S.A.A.C. — Product Viability Roadmap
## Mirroring Hermes Agent's Architecture for Production Readiness

> **Status:** Draft v0.1 | Based on analysis of Hermes Agent (Nous Research) vs I.S.A.A.C. current state
> **Goal:** Transform I.S.A.A.C. from a capable framework into a viable, daily-driver AI agent product

---

## Executive Summary

I.S.A.A.C. has a **stronger core agent architecture** than Hermes (neuro-symbolic ARC solver, specialist team, theorem prover, Z3 integration, procedural memory with verification gates). But Hermes wins on **product completeness**: multi-platform gateway, durable background systems, plugin ecosystem, voice/browser/computer-use, IDE integration, and installable distribution.

This roadmap identifies the **minimum viable product (MVP) gaps** and prioritizes them by user-visible impact.

---

## Gap Analysis: Hermes vs I.S.A.A.C.

| Capability | Hermes | I.S.A.A.C. | Priority |
|------------|--------|------------|----------|
| **Multi-platform Gateway** | 21+ platforms (Telegram, Discord, Slack, WhatsApp, Signal, SMS, Email, Matrix, Teams, etc.) | Telegram only | 🔴 Critical |
| **Native Desktop App** | Electron (macOS/Win/Linux) — streaming, side-by-side, file browser, voice | Exists (`isaac desktop`) but feature parity? | 🟡 High |
| **Web Dashboard** | Full admin panel: channels, MCP, webhooks, memory, profiles, analytics | Missing | 🟡 High |
| **CLI/TUI** | `prompt_toolkit` Ink TUI with docked widgets | Rich REPL exists | 🟢 OK |
| **Voice (STT/TTS)** | Local (faster-whisper), Groq, OpenAI, Mistral, ElevenLabs + TTS providers | Basic `speak`/`transcribe` CLI only | 🟡 High |
| **Browser Automation** | Browserbase, Camofox, local Chromium via CDP | Missing | 🔴 Critical |
| **Computer Use** | `cua-driver` desktop GUI control | Missing | 🟡 High |
| **MCP Support** | Full client + server, tool filtering | `mcp-serve` only (server) | 🔴 Critical |
| **ACP Support** | IDE integration (VS Code, Zed, JetBrains) | Missing | 🟡 High |
| **API Server** | OpenAI-compatible proxy | Missing | 🟡 High |
| **Plugin System** | Tools, hooks, providers, skills, platforms, secret sources | No plugin architecture | 🔴 Critical |
| **Skill Curator** | Background maintenance: usage tracking, staleness, archival, LLM consolidation | Skill library + verification gate, no curator | 🟡 High |
| **Kanban/Task Queue** | Durable SQLite board, multi-profile workers, dispatcher | Missing | 🟡 High |
| **Cron/Scheduled Jobs** | Natural language schedules, skill attachment, multi-platform delivery | Basic cron engine exists | 🟢 OK |
| **Delegation/Subagents** | `delegate_task` with roles, background, spawn depth | `delegate_task` exists | 🟢 OK |
| **Profiles** | Multiple isolated instances (`~/.hermes/profiles/<name>`) | Single profile | 🟡 High |
| **Skins/Themes** | Live-reloading skins, per-surface theming | Missing | 🟢 Nice-to-have |
| **Pet Mascots** | Animated mascots (CLI/TUI/Desktop) | Missing | 🟢 Nice-to-have |
| **Session Heartbeats** | Recurring prompts (`/heartbeat`) | Missing | 🟡 High |
| **Persistent Goals** | Ralph-loop style standing goals | Missing | 🟡 High |
| **Secret Vault** | 1Password, Bitwarden, Bitwarden Secrets Manager, command source | Missing | 🔴 Critical |
| **Egress Proxy** | Network isolation, iron-proxy credential injection | Missing | 🟡 High |
| **Checkpoints/Rollback** | Shadow git repos, auto-snapshots | Missing | 🟡 High |
| **Provider Routing/Fallback** | OpenRouter preferences, auto-failover, credential pools | Basic provider config | 🟡 High |
| **Context Files** | `.hermes.md`, `AGENTS.md`, `CLAUDE.md`, global `SOUL.md` | `SOUL.md` (identity) only | 🟡 High |
| **Session Search** | FTS5 + LLM summarization across sessions | Basic memory recall | 🟡 High |
| **Project Context** | Named multi-folder workspaces, git worktrees | Missing | 🟡 High |
| **Installer** | `curl | bash`, desktop bundles, Nix, Termux APT, Docker | Docker only | 🔴 Critical |
| **Package Distribution** | PyPI, signed APT repo, Homebrew (planned) | Not on PyPI | 🔴 Critical |

---

## Phase 1: Foundation — "Installable & Usable Daily" (Weeks 1-4)

### 1.1 Packaging & Distribution 🔴
- [ ] Publish to PyPI: `pip install isaac-agent`
- [ ] Windows installer (MSI/NSIS) + `winget` manifest
- [ ] Homebrew formula (macOS)
- [ ] Shell installer: `curl -fsSL https://isaac-agent.dev/install.sh | bash`
- [ ] Docker image on GHCR with `isaac` entrypoint
- [ ] Versioned releases with changelogs (semantic versioning)

### 1.2 Configuration System 🔴
- [ ] `isaac config` CLI (mirror `hermes config`): `show`, `set`, `get`, `edit`, `check`, `paths`
- [ ] `config.yaml` (settings) + `.env` (secrets) separation — **never mix**
- [ ] Profile support: `isaac profile create|switch|list|delete` → `~/.isaac/profiles/<name>/`
- [ ] Environment variable reference docs (`ISAAC_*`)

### 1.3 Core Reliability 🔴
- [ ] Loop guards in `AgentLoop`: no-progress detection, wall-clock/token budgets, per-tool timeouts
- [ ] Tool-arg JSON Schema validation before execution (return correction, not crash)
- [ ] Retry/backoff for transient LLM/tool errors; circuit breaker per tool
- [ ] Human-in-the-loop approval for risk-4/5 tools in REPL (not all-or-nothing `auto_approve`)
- [ ] Secrets redaction from tool outputs and traces

### 1.4 Security Hardening 🔴
- [ ] Threat model doc (`docs/THREAT_MODEL.md`)
- [ ] Default `allowed_paths` scoped to workspace (NOT `~`)
- [ ] Hard-deny sensitive paths: `~/.ssh`, `~/.aws`, `.env`, browser profiles, keys
- [ ] Constitution red-team suite (obfuscation, unicode, chained commands, env expansion)
- [ ] `SECURITY.md` + responsible disclosure
- [ ] `pip-audit` in CI

---

## Phase 2: Multi-Platform Gateway — "Be Where the User Is" (Weeks 4-8)

### 2.1 Gateway Architecture
- [ ] Refactor `telegram_gateway.py` → generic `Gateway` base class
- [ ] Platform adapter interface: `send`, `receive`, `authenticate`, `get_user_info`
- [ ] Session routing: per-platform session isolation + cross-platform identity linking
- [ ] Message normalization: text, images, files, voice, location, contacts → unified schema

### 2.2 Platform Adapters (priority order)
1. **Discord** — Socket Mode, slash commands, DMs, voice channels
2. **Slack** — Socket Mode, app mentions, DMs, shortcuts
3. **WhatsApp** — Baileys bridge (like Hermes) or WhatsApp Cloud API
4. **Signal** — `signal-cli` daemon
5. **Email** — IMAP/SMTP (reuse `himalaya` skill pattern)
6. **Matrix** — `matrix-nio` or `mautrix`
7. **Microsoft Teams** — Bot Framework + Graph webhooks
8. **SMS** — Twilio
9. **Telegram** — already exists, harden

### 2.3 Gateway Features
- [ ] Multi-profile gateway: run multiple profiles from one process
- [ ] Per-platform delivery formatting (Markdown → platform-native)
- [ ] Voice message transcription (STT) on all voice-capable platforms
- [ ] TTS voice replies on supported platforms
- [ ] File/document handling across platforms
- [ ] Webhook receiver for GitHub, GitLab, generic HTTP

---

## Phase 3: Skills & Self-Improvement — "Actually Learns" (Weeks 6-10)

### 3.1 Curator System (mirror Hermes)
- [ ] Background daemon: usage tracking, staleness detection, archival
- [ ] Sidecar telemetry: `~/.isaac/skills/.usage.json` — `use_count`, `view_count`, `patch_count`, `last_activity_at`, `state`, `pinned`
- [ ] Deterministic sweep (free): inactivity → stale → archive (never delete)
- [ ] Optional LLM consolidation pass: merge overlapping skills → umbrella skills
- [ ] CLI: `isaac curator status|usage|run|pause|pin|archive|restore|list-archived|prune|backup|rollback`
- [ ] Slash command: `/curator <subcommand>`
- [ ] Config: `curator.enabled`, `interval_hours`, `min_idle_hours`, `stale_after_days`, `archive_after_days`

### 3.2 Skill Library Enhancements
- [ ] Skill metadata: `created_by` (user|agent|bundled), provenance tracking
- [ ] Bundled skills catalog (~90 skills like Hermes) — installable via `isaac skills install <name>`
- [ ] Optional skills catalog (~60) — community contributed
- [ ] Skill versioning UI: `isaac skills show <name> --version N`
- [ ] Skill dependencies (import other skills)
- [ ] Skill marketplace/discovery (later)

### 3.3 Skill Creation UX
- [ ] `isaac skills new` — interactive wizard generating `SKILL.md` + code template
- [ ] `isaac skills validate` — lint + test run before commit
- [ ] Agent-created skill proposal flow: "I noticed you do X often. Save as skill?"

---

## Phase 4: Durable Background Systems (Weeks 8-12)

### 4.1 Kanban Multi-Agent Queue
- [ ] SQLite-backed board: columns, tasks, assignees, dependencies, comments
- [ ] CLI: `isaac kanban init|create|list|show|assign|complete|block|dispatch|daemon|stats`
- [ ] Worker toolset: `kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`
- [ ] Dispatcher: reclaims stale claims, promotes ready tasks, spawns assigned profiles
- [ ] Multi-gateway deployment: single dispatcher, profile-owned delivery

### 4.2 Cron Enhancements
- [ ] Natural language schedules: "every monday 9am", "every 30m"
- [ ] Skill attachment per job
- [ ] Model/provider override per job
- [ ] Pre-run script (`no_agent=True` for script-only jobs)
- [ ] `context_from` chaining (job A output → job B input)
- [ ] Multi-platform delivery targets per job
- [ ] Hard interrupt (3 min), `.tick.lock` dedup

### 4.3 Session Heartbeats & Persistent Goals
- [ ] `/heartbeat` — recurring prompt re-entering idle session
- [ ] `isaac goal set "Keep X updated"` — standing goal across turns
- [ ] Goal progress tracking + auto-resume

---

## Phase 5: Advanced Capabilities (Weeks 10-16)

### 5.1 Browser Automation
- [ ] Local Chromium via CDP (like Hermes `camofox`/`browser-use`)
- [ ] Cloud browser providers: Browserbase, ScrapingBrowser
- [ ] Toolset: `browser_navigate`, `browser_click`, `browser_type`, `browser_extract`, `browser_screenshot`, `browser_pdf`
- [ ] Form filling, scraping, auth flows

### 5.2 Computer Use (GUI Control)
- [ ] Integrate `cua-driver` (or `computer-use` skill pattern)
- [ ] Screenshot → VLM → action loop
- [ ] Safety: approval required, scoped permissions

### 5.3 MCP Client Support
- [ ] `isaac mcp add|remove|list|inspect`
- [ ] MCP tool filtering (allow/deny lists per server)
- [ ] MCP stdio + HTTP transports
- [ ] Auto-discovery from `.mcp.json` / `mcpServers` in config

### 5.4 ACP Server (IDE Integration)
- [ ] ACP adapter for VS Code, Zed, JetBrains
- [ ] `isaac acp` — start ACP server
- [ ] Tool rendering in IDE chat

### 5.5 API Server (OpenAI-Compatible)
- [ ] `isaac proxy` — local OpenAI API backed by I.S.A.A.C.
- [ ] Routes: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`
- [ ] Auth: API key or OAuth (Nous Portal style)
- [ ] Tool calling pass-through

---

## Phase 6: Voice & Multimodal (Weeks 12-16)

### 6.1 STT (Speech-to-Text)
- [ ] Local: faster-whisper (tiny/base/small/medium/large-v3)
- [ ] Cloud: Groq (free tier), OpenAI, Mistral Voxtral, DeepInfra
- [ ] Auto-detect priority chain
- [ ] Voice message handling on all gateway platforms

### 6.2 TTS (Text-to-Speech)
- [ ] Edge TTS (default, free)
- [ ] ElevenLabs, OpenAI, MiniMax, Mistral, Gemini, NeuTTS, Piper, KittenTTS
- [ ] Streaming TTS (sentence chunker)
- [ ] Voice commands: `/voice on|tts|off`

### 6.3 Vision
- [ ] Local VLM (llava, qwen-vl, etc. via Ollama)
- [ ] Cloud: GPT-4o, Claude, Gemini
- [ ] Clipboard paste → analyze (CLI/TUI/Desktop)

---

## Phase 7: Desktop App Parity (Weeks 8-14)

### 7.1 Feature Parity with Hermes Desktop
- [ ] Streaming tool output (live)
- [ ] Side-by-side preview pane (code, browser, images)
- [ ] File browser with drag-drop
- [ ] Voice mode (mic + speaker)
- [ ] Cron management UI
- [ ] Profile switcher
- [ ] Skills browser (install/enable/disable)
- [ ] Settings UI (all config sections)
- [ ] Native notifications
- [ ] Cmd+K / Ctrl+K command palette
- [ ] Per-profile remote gateway login

### 7.2 Desktop Plugin SDK
- [ ] Panes, pages, sidebar nav, status bar, palette commands, keybinds, themes
- [ ] Scoped backend namespace (one import, no build step)
- [ ] Template: `templates/plugin.js`

---

## Phase 8: Observability & Developer Experience (Weeks 12-16)

### 8.1 Web Dashboard
- [ ] Config editor (YAML + form)
- [ ] API key management (OAuth flows)
- [ ] MCP server catalog
- [ ] Webhook management
- [ ] Gateway status + logs
- [ ] Memory browser
- [ ] Credential vault UI
- [ ] Session list + search
- [ ] Analytics (token usage, cost, latency)
- [ ] Cron job manager
- [ ] Skills browser
- [ ] Embedded chat (TUI or web)

### 8.2 Tracing & Debugging
- [ ] Per-run traces persisted (already have `TraceStore`)
- [ ] `isaac trace <run_id>` — detailed view
- [ ] `isaac history` + `isaac replay <run_id>`
- [ ] Latency + cost surfaced per run
- [ ] Token usage breakdown (prompt/completion/tool)

### 8.3 Evaluation Harness
- [ ] `isaac eval` — run suites, score with checkers + LLM judge
- [ ] Golden suite (30+ tasks) with stored transcripts
- [ ] Nightly eval CI job
- [ ] Benchmark adoption: GAIA L1, SWE-bench Lite, ARC-AGI
- [ ] Publish reproducible numbers in README

---

## Phase 9: Polish & Honest Positioning (Weeks 14-18)

### 9.1 Documentation
- [ ] Rewrite README: "local-first autonomous agent **framework**" (not "SOTA")
- [ ] Capabilities table: Stable / Beta / Experimental
- [ ] `LIMITATIONS.md` — honest failure modes
- [ ] Model card + eval card
- [ ] `CONTRIBUTING.md`, issue templates, public roadmap board
- [ ] `SECURITY.md`

### 9.2 UX Polish
- [ ] Streaming orchestrator/specialist events in REPL/TUI
- [ ] Cancel/interrupt/resume runs
- [ ] Persona UX: preview/switch, per-persona tool + risk policy
- [ ] Better final-answer formatting (researcher citations, artifacts index)
- [ ] Demo recordings in README

---

## Technical Debt & Architecture Fixes (Ongoing)

| Issue | Fix |
|-------|-----|
| Optional dep import errors on minimal installs | Guard every optional import; tests skip (not error) when absent |
| CI formatting drift | Pin `ruff==<version>` in `pyproject.toml` |
| Docker/Xvfb required for code tool | Fallback to restricted local subprocess sandbox when Docker absent |
| Team runner ignores `timeout_seconds` | Honor wall-clock budget in `Orchestrator` |
| Single global skill bucket | Per-task-type win-rates; specialist-specific scoring |
| Skill promotion gate = smoke test | Require generated `_selftest()` or synthesised example args from `input_schema` |
| No prompt-injection defense | Provenance-tag tool outputs ("untrusted web content"); route through Guard node |

---

## Success Metrics (Definition of Done for 1.0)

| Metric | Target |
|--------|--------|
| Clean-machine install time | < 10 minutes |
| `isaac doctor` passes on fresh machine | 100% |
| Golden suite pass rate (local model) | ≥ 80% |
| Golden suite pass rate (frontier fallback) | ≥ 95% |
| Zero non-terminating runs on golden suite | 0 |
| Public benchmark number (GAIA L1 / ARC) | Published in README |
| `pip-audit` clean | 0 critical/high |
| Security red-team suite | Green |
| Multi-platform gateway | ≥ 5 platforms working |
| Desktop app feature parity | ≥ 90% of Hermes Desktop |

---

## Resource Allocation Recommendation

| Role | Focus |
|------|-------|
| **Core Engineer (1)** | Phases 1, 3, 4 — agent loop, skills, background systems |
| **Platform Engineer (1)** | Phases 2, 5 — gateway, browser, MCP, ACP, API server |
| **Desktop Engineer (1)** | Phases 7 — Electron app, plugin SDK |
| **DevOps/Release (0.5)** | Phase 1 — packaging, CI/CD, installer, PyPI |
| **Docs/UX (0.5)** | Phases 8, 9 — dashboard, docs, eval harness, demos |

---

## Immediate Next Actions (This Week)

1. **Pin `ruff`** in `pyproject.toml` — stops CI drift
2. **Add `isaac doctor`** preflight + guard optional imports
3. **Scope default `allowed_paths`** to workspace; hard-deny `~/.ssh`, `~/.aws`, `.env`, keys
4. **Loop guards** in `AgentLoop`: no-progress + wall-clock/token budgets
5. **Stand up `isaac eval`** skeleton + 10 golden tasks; one real end-to-end run per specialist
6. **Extract gateway** from Telegram → generic `Gateway` base class
7. **Design `config.yaml` schema** + `isaac config` CLI

---

## Appendix: Hermes Concepts to Adopt Verbatim

These Hermes patterns are battle-tested and should be mirrored closely:

| Concept | Hermes Implementation | I.S.A.A.C. Adoption |
|---------|----------------------|---------------------|
| **Profile isolation** | `$HERMES_HOME/profiles/<name>/` with own config, skills, memory, sessions | `~/.isaac/profiles/<name>/` |
| **Config vs Secrets** | `config.yaml` (settings) + `.env` (secrets ONLY) | Same — enforce in code |
| **Toolset bundles** | `web`, `browser`, `terminal`, `file`, `coding`, `computer_use`, `safe`, etc. | Define I.S.A.A.C. toolsets |
| **Skin engine** | Live-reload, `hermes config set display.skin <name>` | Adopt for CLI/TUI/Desktop |
| **Curator telemetry** | `skills/.usage.json` — per-skill metrics | Same path pattern |
| **Kanban board isolation** | `HERMES_KANBAN_BOARD` env pin | `ISAAC_KANBAN_BOARD` |
| **Cron delivery framing** | Header/footer, not mirrored (preserves role alternation) | Same pattern |
| **Session storage** | SQLite + FTS5 (`state.db`) | Already similar |
| **Prompt caching invariants** | Never change past context/toolsets/system prompt mid-convo | Document + enforce |
| **Message role alternation** | Never two assistant/user in a row; only `tool` can repeat | Enforce in gateway |

---

*This roadmap is a living document. Update as phases complete and priorities shift.*
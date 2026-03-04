# Changelog

All notable changes to I.S.A.A.C. are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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

## [Unreleased]

### Planned for 0.2.0
- First-class Ollama provider in `llm/provider.py` (no base-URL workaround)
- Google Gemini and Mistral provider support
- ARC-AGI LLM-guided DSL synthesis (beyond depth-2 brute force)
- Web UI dashboard (FastAPI + WebSocket streaming)
- Multi-agent collaboration via shared skill library

[0.1.0]: https://github.com/Matheussoranco/I.S.A.A.C/releases/tag/v0.1.0

# Known Limitations (1.3.1)

I.S.A.A.C. 1.0 is a stable, tested, local-first autonomous agent **framework**.
This file states plainly what that does — and does not — mean. It exists so
the README's capability claims can be read with the right calibration.

## What "stable" means here

- The CLI commands and the public Python API (`AgentLoop`, `Orchestrator`,
  the tool registry, the memory manager) are frozen for the 1.x series.
- 460+ unit/integration tests pass on Python 3.10–3.12; lint and format are
  enforced in CI.
- The safety boundary (path confinement + credential deny-list, constitutional
  shell gating, risk-gated tool approval, sandboxed code execution) is covered
  by dedicated tests.

## What is **not** yet proven

- **The public-benchmark number is a floor, not a competitive score.**
  ARC-AGI-1 public evaluation set: 2/400 (0.5%) with the LLM-free symbolic
  solver, single attempt (2026-07-03 — see the README). Published LLM systems
  score 8–21% on the identical task set, so comparative claims ("SOTA")
  remain firmly off the table per the roadmap gate. The value of the number
  is that it is public, reproducible on any machine with no model configured,
  and regression-gated in nightly CI. The internal golden suite additionally
  has a recorded live-model result (31/33 with `qwen3-coder:480b-cloud`,
  2026-06-10). GAIA L1 has a first full-run result (≥4/53, 2026-07-04) that
  is only a *lower bound*: the provider's free-tier quota aborted 40+ of the
  53 tasks mid-run with 429 errors — a clean measurement needs quota headroom
  (paid tier, a reset window, or a strong local model). Scores with the
  default 7B local model will be substantially lower than the golden-suite
  number.
- **Tests mock the LLM.** The test suite validates the machinery (loops,
  routing, memory, safety gates), not end-to-end task success against a live
  model. Real-world quality depends heavily on the model you configure.
- **Local-model ceiling.** With the default 7B-class local model, hard
  multi-step tasks will fail more often than with a frontier cloud model.
  Configure a larger local model or an opt-in cloud fallback for harder work.

## Operational caveats

- **Host-reach tools are powerful.** `shell` and `fs_*` operate on your real
  machine. Credential stores are hard-denied and `allowed_paths` confines
  access, but you should still scope `ISAAC_ALLOWED_PATHS` to the directories
  you actually want organised, and leave `ISAAC_SHELL_UNRESTRICTED` off.
- **Prompt injection is mitigated, not solved.** The Guard node and the
  constitution catch known patterns, web/search/email tool output is
  provenance-tagged as untrusted, and secrets are redacted from tool outputs —
  but adversarial content can still steer the model. Keep high-risk tools
  behind approval (the default).
- **Self-improvement needs data.** Skill curation and prompt evolution only
  act after enough telemetry accumulates; a fresh install has nothing to
  improve from.
- **Docker is required for sandboxed code.** Without a running Docker engine
  the `code` tool is unavailable (run `isaac doctor` to check).

# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

To report a vulnerability privately:

1. Email **matheussoranco@gmail.com** with the subject `[I.S.A.A.C. SECURITY] <brief description>`.
2. Include:
   - A clear description of the vulnerability and its potential impact
   - Reproduction steps (minimal code sample or config if applicable)
   - Affected version(s)
   - Suggested fix (optional)

You will receive an acknowledgement within **48 hours** and a full response within **7 days**.

---

## Security Architecture

I.S.A.A.C. is designed with multiple defence-in-depth layers. Understanding the security model helps contributors assess risk correctly:

### Execution Isolation
- All user-generated code runs in **ephemeral Docker containers** with:
  - `--network=none` (total network isolation)
  - `--cap-drop=ALL` (no Linux capabilities)
  - Read-only root filesystem
  - 256 MB memory hard limit, 1 CPU, 64 PIDs
  - `nobody` (UID 65534) as the running user
  - Custom seccomp profile (100+ syscall allowlist)
- The host process **never executes user code directly**

### Pre-execution Code Scanning
- AST import scanner in `sandbox/executor.py` blocks 20+ dangerous module imports before Docker execution
- Input sanitizer strips ANSI sequences, control characters, null bytes, and HTML injection vectors

### Audit Trail
- Hash-chained JSONL audit log (`security/audit.py`) records all system events
- Capability token system (`security/capabilities.py`) gates tool invocations with expiring, revocable tokens

### Prompt Injection Guard
- `guard_node` uses regex patterns + LLM analysis to detect and reject prompt injection attempts before any task processing

---

## Known Limitations (Beta)

- The AST import scanner is a defence-in-depth measure, not a sandbox replacement. The Docker container is the primary isolation boundary.
- Capability tokens are auto-issued for connector invocations in the current release; operator-issued tokens with manual approval are planned for 0.2.0.
- The Computer-Use sandbox (`sandbox_image_ui`) allows outbound browser network access when `ISAAC_UI_SANDBOX_ALLOW_BROWSER_NETWORK=true`; leave this disabled unless required.

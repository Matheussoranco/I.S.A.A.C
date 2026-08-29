# I.S.A.A.C. — Setup Guide

> **Intelligent System for Autonomous Action and Cognition** v1.6.0

## Prerequisites

| Requirement       | Version   | Notes                                          |
| ----------------- | --------- | ---------------------------------------------- |
| Python            | ≥ 3.10    | 3.12 recommended                               |
| Docker            | ≥ 24.0    | For sandboxed code execution                   |
| Ollama            | ≥ 0.3     | Local LLM — the **default** provider (`ollama pull qwen3.6`) |
| Microphone + speakers | any   | Only needed for the voice REPL                 |

## 1. Clone & Install

```bash
git clone https://github.com/Matheussoranco/I.S.A.A.C.git
cd I.S.A.A.C

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/macOS

# Core install
pip install -e ".[dev]"

# Optional extras
pip install -e ".[vision]"     # mss + Pillow (screen + image input)
pip install -e ".[voice]"      # whisper + piper + sounddevice + webrtcvad
pip install -e ".[multimodal]" # vision + voice combined
pip install -e ".[browser,calendar]" # connectors
```

## 2. Environment Variables

Copy the example env file and edit what you need:

```bash
cp .env.example .env
```

### Required for local-first (default)

The default install talks to **Ollama** on `http://localhost:11434` using the
model **`qwen3.6`** — no API key of any kind. The only thing you need to do is
pull the model:

```bash
ollama pull qwen3.6            # text (the default)
ollama pull llava:7b           # vision (optional)
```

| Variable                    | Default                  | Purpose |
| --------------------------- | ------------------------ | ------- |
| `ISAAC_LLM_PROVIDER`        | `ollama`                 | `ollama` / `llamacpp` / `openai_compat` / `openai` / `anthropic` |
| `ISAAC_MODEL_NAME`          | `qwen3.6`                | Default model tag |
| `ISAAC_OLLAMA_BASE_URL`     | `http://localhost:11434` | Ollama daemon URL |
| `ISAAC_OLLAMA_LIGHT_MODEL`  | `qwen3.6`                | Simple / moderate tasks |
| `ISAAC_OLLAMA_HEAVY_MODEL`  | `qwen3.6`                | Complex / reasoning tasks |
| `ISAAC_OLLAMA_PREFLIGHT`    | `true`                   | Check daemon + model before the first call |

If the daemon is down or the model was never pulled, I.S.A.A.C. stops with an
error that names the exact command to run (`ollama serve`, `ollama pull qwen3.6`)
instead of failing cryptically — and it never silently redirects the request to
a paid cloud API. Run `isaac doctor` to see the same diagnosis up front.

### Optional — Cloud providers

Cloud backends are fully supported; they are simply not the default. Their API
keys are validated **only** when you actually select one.

| Variable              | When to set                              |
| --------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`      | Set when using `openai` provider or as fallback |
| `ANTHROPIC_API_KEY`   | Set when using `anthropic` provider or as fallback |
| `ISAAC_LLM_FALLBACK_PROVIDER` | `openai` / `anthropic` — opt-in; empty by default so nothing bills you unexpectedly |

```bash
# Example: drive I.S.A.A.C. with Claude instead of the local default
export ISAAC_LLM_PROVIDER=anthropic
export ISAAC_MODEL_NAME=claude-opus-4-8
export ANTHROPIC_API_KEY=sk-ant-...
```

### Optional — Voice

| Variable                       | Default                  | Purpose |
| ------------------------------ | ------------------------ | ------- |
| `ISAAC_VOICE_ENABLED`          | `true`                   | Master switch |
| `ISAAC_VOICE_DEVICE`           | `auto`                   | `auto` / `cpu` / `cuda` |
| `ISAAC_VOICE_STT_MODEL`        | `base`                   | Whisper size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `ISAAC_VOICE_STT_LANGUAGE`     | (auto-detect)            | ISO 639-1 (`en`, `pt`, …) |
| `ISAAC_VOICE_STT_COMPUTE_TYPE` | `int8`                   | faster-whisper compute (`int8` / `float16` / `float32`) |
| `ISAAC_VOICE_TTS_VOICE`        | `en_US-lessac-medium`    | Piper voice file under `~/.isaac/voices` or `$PIPER_VOICE_DIR` |
| `ISAAC_VOICE_TTS_RATE`         | `175`                    | Words/min (pyttsx3 only) |

Download a Piper voice (one-time):

```bash
mkdir -p ~/.isaac/voices
curl -L -o ~/.isaac/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.isaac/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

### Optional — Vision

| Variable                  | Default      |
| ------------------------- | ------------ |
| `ISAAC_VISION_ENABLED`    | `true`       |
| `ISAAC_VISION_MODEL`      | `llava:7b`   |
| `ISAAC_VISION_STRONG_MODEL` | (none)     |

### Optional — Self-improvement

| Variable                              | Default | Purpose |
| ------------------------------------- | ------- | ------- |
| `ISAAC_IMPROVEMENT_ENABLED`           | `false` | Auto-run periodic improvement cycles |
| `ISAAC_IMPROVEMENT_INTERVAL_MINUTES`  | `240`   | Cycle period (10 ≤ x ≤ 10080)        |
| `ISAAC_IMPROVEMENT_PROMOTE_RUNS`      | `10`    | Min runs before a skill can be promoted |
| `ISAAC_IMPROVEMENT_PROMOTE_THRESHOLD` | `0.85`  | Success rate required to promote     |
| `ISAAC_IMPROVEMENT_DEPRECATE_RUNS`    | `8`     | Min runs before a skill can be deprecated |
| `ISAAC_IMPROVEMENT_DEPRECATE_THRESHOLD` | `0.30` | Success rate below which to deprecate |

### Optional — Connectors

| Variable               | Connector       | Description                            |
| ---------------------- | --------------- | -------------------------------------- |
| `ISAAC_GITHUB_TOKEN`   | GitHub          | Personal access token                  |
| `ISAAC_EMAIL_*`        | Email           | IMAP (inbound) + SMTP (outbound)       |
| `ISAAC_CALDAV_*`       | Calendar        | CalDAV server                          |
| `ISAAC_OBSIDIAN_VAULT_PATH` | Obsidian   | Absolute vault path                    |

## 3. Build Docker Sandbox Images

```bash
docker build -t isaac-sandbox:latest -f sandbox_image/Dockerfile sandbox_image/
docker build -t isaac-ui-sandbox:latest -f sandbox_image_ui/Dockerfile sandbox_image_ui/
```

## 4. Start Ollama

```bash
ollama serve             # in one terminal
ollama pull qwen3.6      # the default model — once
```

Verify with `isaac doctor`; it reports `fail` (with the exact fix) if the daemon
is unreachable or `qwen3.6` has not been pulled.

## 5. Run I.S.A.A.C.

### Graphical interface

```bash
pip install -e ".[desktop]"    # once: native window + desktop controls
isaac desktop                  # native Windows application

isaac ui                       # opens http://127.0.0.1:8765
isaac ui --no-open             # start without opening a browser tab
isaac ui --port 9000           # choose another local port
```

Both commands use the same local interface. `isaac desktop` hosts it in a
native WebView window, while `isaac ui` opens it in a browser. The interface
streams agent steps and tool results over a local WebSocket.
When the `browser` tool is used, it mirrors the Chromium viewport and animates
the agent cursor at the selector's real coordinates. Risk-level 4–5 actions
pause and present an approval dialog; API keys remain in the Python process and
are never sent to the web client.

The real-PC tools are deliberately split by capability:

- `computer_view` takes a local screenshot for the on-screen preview and does
  not send that image to the model provider.
- `computer_describe` may send the screenshot to the configured vision model,
  so it requires approval.
- `computer_control` performs exactly one bounded mouse/keyboard action and
  requires approval on every call. Move the pointer to the top-left corner to
  trigger the PyAutoGUI emergency stop.

To produce a standalone Windows folder containing `ISAAC.exe`:

```powershell
.\scripts\build_windows.ps1
```

The output is the complete `dist\ISAAC` folder plus a versioned portable ZIP,
for example `dist\ISAAC-1.6.0-Windows-x64.zip`. Do not distribute `ISAAC.exe`
by itself: the `_internal` directory is required. To install it for the current
Windows user, create Start Menu/Desktop shortcuts, and launch it:

```powershell
.\scripts\install_windows.ps1
```

The installed application lives at `%LOCALAPPDATA%\Programs\ISAAC`. A model
and reasoning selector is available by clicking the model chip in the title
bar. OpenAI and Anthropic keys entered there are stored in Windows Credential
Manager. They can alternatively be supplied through `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY`; environment variables take precedence.

With provider `OpenAI`, model `gpt-5.6-sol`, and mode `Computador`, I.S.A.A.C.
uses the OpenAI Responses API computer loop: it receives `computer_call`
actions, asks for local approval, performs them, captures the resulting screen,
and sends it back as `computer_call_output` with original image detail. This is
the direct computer-use path; local/other providers use I.S.A.A.C.'s generic
bounded desktop tools instead.

### Text REPL (Rich UI, default)

```bash
isaac run
```

### Voice REPL

```bash
isaac voice                  # push-to-talk
isaac voice --hands-free     # continuous listening (VAD)
```

### Vision one-shot

```bash
isaac vision ~/Pictures/screen.png --prompt "What error is this dialog showing?"
```

### Self-improvement

```bash
isaac improve            # one cycle on demand
isaac improve --report   # cycle + show curation decisions
```

### Provider / model inspection

```bash
isaac providers          # all providers + ollama install list
isaac models list        # the good/better/best preset ladder
```

### Telegram + scheduler daemon

```bash
isaac serve
```

## 6. CLI reference

| Command                     | Description                                 |
| --------------------------- | ------------------------------------------- |
| `isaac desktop`             | Native Windows agent application            |
| `isaac ui`                  | Graphical chat, activity, and browser view  |
| `isaac run`                 | Rich text REPL                              |
| `isaac run --classic`       | Plain `print()` REPL                        |
| `isaac voice [--hands-free]`| Voice REPL                                  |
| `isaac vision <path>`       | Ask the local VLM about an image            |
| `isaac improve [--report]`  | Run one self-improvement cycle              |
| `isaac providers`           | List providers + Ollama models              |
| `isaac models [list\|show\|recommend\|use]` | Good/better/best model presets |
| `isaac eval-toolcalls`      | Measure tool-call reliability on a model    |
| `isaac serve`               | Telegram gateway + heartbeat scheduler      |
| `isaac audit [--verify]`    | View / verify the audit chain               |
| `isaac memory "<query>"`    | Query the unified memory system             |
| `isaac tools`               | List registered tools                       |
| `isaac connectors`          | List connectors and availability            |
| `isaac cron …`              | Manage background cron tasks                |
| `isaac tokens …`            | Manage capability tokens                    |

## 7. Running Tests

```bash
pytest -v
```

## 8. Project Structure (v0.3.0)

```
src/isaac/
├── __init__.py
├── __main__.py
├── cli.py                       # Typer CLI: run, serve, voice, vision,
│                                # improve, models, audit, memory, …
├── identity/
│   └── soul.py                  # Personality + soul loader
├── config/
│   └── settings.py              # Pydantic settings (ALL env vars)
├── core/
│   ├── state.py                 # IsaacState TypedDict
│   ├── graph.py                 # LangGraph builder (telemetry-wrapped)
│   ├── transitions.py           # Conditional edge routing
│   └── telemetry.py             # NEW — track_node / track_skill decorators
├── llm/
│   ├── providers/               # NEW — first-class provider builders
│   │   ├── ollama.py
│   │   ├── llamacpp.py
│   │   ├── openai_compat.py
│   │   ├── openai.py
│   │   └── anthropic.py
│   ├── multimodal_router.py     # NEW — (modality × complexity) router
│   ├── provider.py              # legacy tier factory (kept)
│   ├── router.py                # legacy complexity router (kept)
│   └── prompts.py
├── multimodal/                  # NEW
│   ├── voice/
│   │   ├── stt.py               # Whisper backends
│   │   ├── tts.py               # Piper / Coqui / pyttsx3
│   │   └── audio_io.py          # mic + speaker + VAD
│   ├── vision/
│   │   ├── vision_lm.py         # local VLM wrapper
│   │   └── screen_capture.py
│   └── input.py                 # unified multimodal HumanMessage builder
├── improvement/                 # NEW — self-improvement engine
│   ├── performance.py
│   ├── skill_curation.py
│   ├── prompt_evolution.py
│   ├── self_critique.py
│   └── engine.py
├── memory/                      # 5-layer memory (unchanged in 0.3.0)
├── nodes/                       # cognitive graph nodes
├── skills/connectors/           # external connectors
├── background/                  # cron daemon
├── sandbox/                     # Docker sandbox management
├── security/                    # audit, capabilities, guard
├── scheduler/                   # heartbeat + improvement_job
├── interfaces/
│   ├── repl.py
│   ├── voice_repl.py            # NEW — conversational voice loop
│   ├── terminal_ui.py
│   └── telegram_gateway.py
├── tools/
└── arc/
```

## License

AGPL-3.0-or-later

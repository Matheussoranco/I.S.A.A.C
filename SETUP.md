# I.S.A.A.C. — Setup Guide

> **Intelligent System for Autonomous Action and Cognition** v0.3.0

## Prerequisites

| Requirement       | Version   | Notes                                          |
| ----------------- | --------- | ---------------------------------------------- |
| Python            | ≥ 3.10    | 3.12 recommended                               |
| Docker            | ≥ 24.0    | For sandboxed code execution                   |
| Ollama (recommended) | ≥ 0.3  | Local LLM — default provider                   |
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

The default install talks to **Ollama** on `http://localhost:11434`.
The only thing you need to do is pull a model:

```bash
ollama pull qwen2.5-coder:7b   # text
ollama pull llava:7b           # vision (optional)
```

### Optional — Cloud fallbacks

| Variable              | When to set                              |
| --------------------- | ---------------------------------------- |
| `OPENAI_API_KEY`      | Set when using `openai` provider or as fallback |
| `ANTHROPIC_API_KEY`   | Set when using `anthropic` provider or as fallback |
| `ISAAC_LLM_FALLBACK_PROVIDER` | `openai` / `anthropic` — used when the primary local backend is down |

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
ollama serve   # in one terminal
```

## 5. Run I.S.A.A.C.

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
isaac models             # all providers + ollama install list
```

### Telegram + scheduler daemon

```bash
isaac serve
```

## 6. CLI reference

| Command                     | Description                                 |
| --------------------------- | ------------------------------------------- |
| `isaac run`                 | Rich text REPL                              |
| `isaac run --classic`       | Plain `print()` REPL                        |
| `isaac voice [--hands-free]`| Voice REPL                                  |
| `isaac vision <path>`       | Ask the local VLM about an image            |
| `isaac improve [--report]`  | Run one self-improvement cycle              |
| `isaac models`              | List providers + Ollama models              |
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

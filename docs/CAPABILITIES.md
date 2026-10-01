# Core Capabilities

I.S.A.A.C. is designed as a modular framework where capabilities are decoupled from the specific LLM driving the loop.

## 1. Multi-modal (Vision/Voice)
- **Voice**: Integration of `faster-whisper` for high-speed speech-to-text and a flexible TTS backend (Piper, Coqui, or pyttsx3). Supports "hands-free" mode via VAD (Voice Activity Detection).
- **Vision**: Integration of VLMs (Vision-Language Models) via Ollama. The agent can capture the screen or analyze local images to derive spatial and textual context.

## 2. MCP Client / Plugin SDK
The Model Context Protocol (MCP) allows I.S.A.A.C. to connect to external tool servers. This means the agent can gain new capabilities (e.g., interacting with a specific API or database) without requiring a code change to the core framework.

## 3. Kanban/Cron Background Engine
Beyond interactive loops, I.S.A.A.C. supports background autonomy:
- **Heartbeat Scheduler**: A daemon that monitors scheduled tasks.
- **Standing Objectives**: High-priority goals injected into the system prompt of every agent run, ensuring the agent remains aware of long-term obligations.

## 4. Browser Automation (CDP)
Using the Chrome DevTools Protocol (CDP), I.S.A.A.C. can drive a real browser. It supports:
- **Visual Feedback**: Screenshots are returned to the model to verify page state.
- **Interaction**: Precise clicking, typing, and navigation.
- **Isolation**: Browser sessions are managed to prevent state leakage between unrelated tasks.

## 5. OpenAI-Compatible API Proxy
To ensure maximum flexibility, the LLM layer uses a standardized proxy. Any provider that implements the OpenAI Chat Completions API (including vLLM, LM Studio, and LiteLLM) can be used as the reasoning engine.

## 6. Profile-based Configuration
Configuration is managed through isolated profiles. Each profile contains its own `config.yaml`, allowing users to switch between "Work", "Research", or "Companion" settings (including different models and risk policies) without modifying environment variables manually.

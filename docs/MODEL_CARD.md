# Model Card: Default Configuration

## Current Default Model
- **Model**: `qwen3.6` (via Ollama)
- **Quantization**: 4-bit (standard Ollama pull)
- **Role**: Generalist reasoning, planning, and tool-use.

## Typical Performance
- **Tool Calling**: Highly reliable with constrained decoding enabled.
- **Reasoning**: Strong at decomposition and following complex system prompts.
- **Latency**: Optimized for local consumer GPUs (RTX 30/40 series).

## Recommended Presets
I.S.A.A.C. provides presets that optimize the loop for different model sizes:
- `good`: Balanced for 7B-14B models.
- `better`: Optimized for 30B+ models.
- `best`: Configured for frontier cloud models (Claude 3.5/GPT-4o).

Use `isaac models recommend` to find the best fit for your current hardware.

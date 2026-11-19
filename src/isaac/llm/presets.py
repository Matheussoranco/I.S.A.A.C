"""Good / better / best model presets.

"Which model should I run?" is the first question a local-first agent has to
answer, and the honest answer depends on VRAM.  This module encodes four
rungs — ``minimal``, ``good``, ``better``, ``best`` — each pinning not just a
model name but the *loop settings that model needs*: a 4B model wants
constrained decoding and self-consistency; a frontier model wants neither and
is only slowed down by them.

A preset is a starting point, not a benchmark result.  The tool-call reliability
figures live in ``docs/MODELS.md`` and were measured with ``isaac eval-toolcalls``
on the hardware named there; treat numbers from other hardware or other model
revisions as unverified.

Apply one with :func:`apply_preset` or ``isaac config preset <name>``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PRESETS",
    "ModelPreset",
    "apply_preset",
    "describe_presets",
    "get_preset",
    "recommend_preset",
]


@dataclass(frozen=True)
class ModelPreset:
    """One rung of the capability ladder."""

    name: str
    tagline: str
    provider: str
    model: str
    fast_model: str = ""
    strong_model: str = ""
    vram_gb: float = 0.0
    quantisation: str = ""
    constrained_decoding: bool = False
    repair_tool_calls: bool = True
    reflexion_retries: int = 2
    test_time_samples: int = 1
    temperature: float = 0.2
    sampling_temperature: float = 0.7
    notes: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def as_env(self) -> dict[str, str]:
        out = {
            "ISAAC_LLM_PROVIDER": self.provider,
            "ISAAC_MODEL_NAME": self.model,
            "ISAAC_TEMPERATURE": str(self.temperature),
            "ISAAC_CONSTRAINED_DECODING": "1" if self.constrained_decoding else "0",
            "ISAAC_REPAIR_TOOL_CALLS": "1" if self.repair_tool_calls else "0",
            "ISAAC_REFLEXION_RETRIES": str(self.reflexion_retries),
            "ISAAC_TEST_TIME_SAMPLES": str(self.test_time_samples),
            "ISAAC_SAMPLING_TEMPERATURE": str(self.sampling_temperature),
            "ISAAC_FAST_MODEL": self.fast_model or "",
            "ISAAC_STRONG_MODEL": self.strong_model or "",
        }
        return out


PRESETS: dict[str, ModelPreset] = {
    "minimal": ModelPreset(
        name="minimal",
        tagline="Runs anywhere — 4 GB VRAM or CPU-only",
        provider="ollama",
        model="gemma3:1b",
        vram_gb=1.5,
        quantisation="Q4_K_M",
        constrained_decoding=True,
        reflexion_retries=3,
        test_time_samples=3,
        notes=(
            "gemma3:1b has no native tool-calling capability — Ollama reports "
            "only 'completion' and rejects tools-bearing requests with HTTP "
            "400 — so constrained decoding is not a tuning choice here, it is "
            "the only thing that makes the model usable as an agent at all. "
            "Measured: 0/20 requests accepted natively, 20/20 well-formed "
            "calls under the grammar, of which 8/20 chose the right tool. The "
            "grammar guarantees shape, not judgement: expect to supervise "
            "multi-step tasks. This rung exists so the stack runs on hardware "
            "that cannot hold anything larger, not to match the rungs above."
        ),
    ),
    "good": ModelPreset(
        name="good",
        tagline="6 GB laptop GPU — the tested local baseline",
        provider="ollama",
        model="nemotron-3-nano:4b",
        vram_gb=4.0,
        quantisation="Q4_K_M",
        constrained_decoding=False,
        reflexion_retries=2,
        test_time_samples=3,
        notes=(
            "The configuration behind the 1.3.2 GAIA L1 result (8/53, fully "
            "local on an RTX 3050 6 GB). Supports native tool calling and is "
            "fast enough (~51 tok/s on that card) that self-consistency on "
            "hard steps is affordable. Measured at a 0.0% malformed tool-call "
            "rate (20/20 native, docs/MODELS.md), so repair and Reflexion are "
            "carried only as a safety net — they never fired in testing."
        ),
    ),
    "better": ModelPreset(
        name="better",
        tagline="12–16 GB GPU — reliable native tool calling",
        provider="ollama",
        model="ornith:9b",
        fast_model="nemotron-3-nano:4b",
        vram_gb=7.0,
        quantisation="Q4_K_M",
        constrained_decoding=False,
        reflexion_retries=2,
        test_time_samples=3,
        notes=(
            "Native tool calling is dependable at this size, so repair rarely "
            "fires and is kept only as a safety net. Routes cheap turns to the "
            "4B fast model to keep latency down."
        ),
    ),
    "best": ModelPreset(
        name="best",
        tagline="Frontier models via API — highest capability",
        provider="anthropic",
        model="claude-sonnet-5",
        strong_model="claude-opus-5",
        fast_model="claude-haiku-4-5-20251001",
        vram_gb=0.0,
        constrained_decoding=False,
        reflexion_retries=1,
        test_time_samples=1,
        notes=(
            "Not local: requires ANTHROPIC_API_KEY and sends task content off "
            "the machine. Native tool calling is reliable enough that repair "
            "and test-time sampling are near-dead weight, so both are wound "
            "down — spend the budget on a stronger model instead."
        ),
    ),
}


def get_preset(name: str) -> ModelPreset:
    key = (name or "").strip().lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Choose one of: {', '.join(PRESETS)}")
    return PRESETS[key]


def apply_preset(name: str, env: dict[str, str] | None = None) -> dict[str, str]:
    preset = get_preset(name)
    values = preset.as_env()
    target = os.environ if env is None else env
    target.update(values)
    logger.info("Applied model preset '%s' (%s)", preset.name, preset.model)
    return values


def describe_presets() -> str:
    from tabulate import tabulate

    rows = []
    for p in PRESETS.values():
        rows.append(
            [
                p.name,
                p.model,
                f"{p.vram_gb} GB" if p.vram_gb else "API",
                "yes" if p.constrained_decoding else "no",
                p.test_time_samples,
                p.tagline,
            ]
        )
    return tabulate(
        rows,
        headers=["Preset", "Model", "VRAM", "Constrained", "Samples", "Tagline"],
        tablefmt="plain",
    )


def preset_dicts() -> list[dict[str, Any]]:
    """Return a list of preset dictionaries for CLI table output."""
    return [
        {
            "name": p.name,
            "model": p.model,
            "vram_gb": p.vram_gb,
            "constrained_decoding": p.constrained_decoding,
            "test_time_samples": p.test_time_samples,
            "tagline": p.tagline,
        }
        for p in PRESETS.values()
    ]


def recommend_preset(vram_gb: float | None = None, has_api_key: bool = False) -> ModelPreset:
    if vram_gb is None:
        vram_gb = _detect_vram_gb()
    if vram_gb >= 10:
        return PRESETS["better"]
    if vram_gb >= 5:
        return PRESETS["good"]
    if vram_gb >= 1.5:
        return PRESETS["minimal"]
    return PRESETS["best"] if has_api_key else PRESETS["minimal"]


def _detect_vram_gb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return max(int(line) for line in out.splitlines()) / 1024.0
    except Exception:
        pass
    return 0.0

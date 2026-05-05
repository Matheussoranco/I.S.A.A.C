"""Knowledge Experts — pluggable Mixture-of-Experts (MoE) for I.S.A.A.C.

I.S.A.A.C. uses a *local-first* LLM as its **language expert**, and delegates
specialised reasoning to **knowledge experts** that combine neural and
symbolic reasoning:

* :class:`LanguageExpert`  — wraps the local LLM (Ollama/OpenAI/Anthropic).
* :class:`MathExpert`      — symbolic algebra & calculus via SymPy, falls
  back to LLM when symbolic fails.
* :class:`CodeExpert`      — code understanding & synthesis via the skill
  library + LLM.
* :class:`KGExpert`        — answers entity/relation queries from the
  :class:`WorldModelKG`.
* :class:`ArcExpert`       — routes ARC-AGI grid tasks through the
  5-strategy solver.
* :class:`VisionExpert`    — image / grid perception via prior modules.
* :class:`LogicExpert`     — Z3 SMT theorem prover.

A :class:`HybridRouter` (symbolic features + LLM tie-breaker) chooses the
best expert(s) and a :class:`MixtureOfExperts` orchestrator runs them in
parallel and merges answers.

Public API
----------

>>> from isaac.experts import answer, get_moe
>>> answer("solve x^2 - 4 = 0")     # → MathExpert
>>> answer("what's the capital of France?")  # → LanguageExpert
>>> get_moe().route("integrate sin(x) dx")    # → ExpertSelection
"""

from __future__ import annotations

from isaac.experts.base import (
    Expert,
    ExpertResponse,
    ExpertSelection,
    ExpertNotApplicable,
)
from isaac.experts.registry import ExpertRegistry, get_registry
from isaac.experts.router import HybridRouter, RoutingFeatures
from isaac.experts.moe import MixtureOfExperts, get_moe, answer

__all__ = [
    "Expert",
    "ExpertResponse",
    "ExpertSelection",
    "ExpertNotApplicable",
    "ExpertRegistry",
    "get_registry",
    "HybridRouter",
    "RoutingFeatures",
    "MixtureOfExperts",
    "get_moe",
    "answer",
]

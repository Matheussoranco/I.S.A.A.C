"""ARC-AGI neuro-symbolic reasoning subsystem.

Provides:
- Grid perception primitives (grid_ops)
- 55+ DSL transformation primitives (dsl)
- Core knowledge priors — Chollet's 4 systems (priors)
- Analogy engine — cross-pair rule extraction (analogy)
- Full program synthesis engine — beam search + LLM (solver)
- Evaluation harness (evaluator)
"""

from __future__ import annotations

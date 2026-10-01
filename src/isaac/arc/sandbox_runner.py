"""Run untrusted ARC candidate programs only in the Docker code sandbox."""

from __future__ import annotations

import json
import logging
import secrets
from contextlib import suppress

import numpy as np

from isaac.arc.grid_ops import Grid
from isaac.sandbox.executor import CodeExecutor

logger = logging.getLogger(__name__)


def run_candidate(code: str, inputs: list[Grid]) -> list[Grid | None] | None:
    """Return bounded grid predictions, or ``None`` if the sandbox is unavailable.

    Each grid's solve() failure becomes a missing prediction. No candidate code is
    evaluated by the host interpreter, including during training validation.
    """
    marker = f"ISAAC_ARC_RESULT_{secrets.token_hex(16)}:"
    payload = json.dumps([grid.tolist() for grid in inputs])
    script = f"""
import json
import numpy as np

namespace = {{"np": np, "numpy": np}}
exec({code!r}, namespace)
solve = namespace.get("solve")
if not callable(solve):
    raise ValueError("candidate did not define solve")
results = []
for raw in json.loads({payload!r}):
    try:
        value = np.asarray(solve(np.asarray(raw, dtype=int)))
        if (value.ndim != 2 or not 1 <= value.shape[0] <= 30
                or not 1 <= value.shape[1] <= 30 or not np.issubdtype(value.dtype, np.number)
                or not np.all(np.isfinite(value)) or not np.all(value == np.floor(value))
                or not np.all((0 <= value) & (value <= 9))):
            raise ValueError("invalid ARC grid")
        results.append(value.astype(int).tolist())
    except Exception:
        results.append(None)
print({marker!r} + json.dumps(results))
"""
    executor = None
    try:
        executor = CodeExecutor()
        result = executor.execute(script)
        if result.exit_code != 0:
            logger.warning("ARC candidate failed in sandbox: %s", result.stderr[:300])
            return None
        raw_result = result.stdout.rsplit(marker, 1)[-1].splitlines()[0]
        if marker not in result.stdout:
            return None
        decoded = json.loads(raw_result)
        if not isinstance(decoded, list) or len(decoded) != len(inputs):
            return None
        predictions: list[Grid | None] = []
        for grid in decoded:
            if grid is None:
                predictions.append(None)
                continue
            candidate = np.asarray(grid)
            if (
                candidate.ndim != 2
                or not 1 <= candidate.shape[0] <= 30
                or not 1 <= candidate.shape[1] <= 30
                or not np.issubdtype(candidate.dtype, np.number)
                or not np.all(np.isfinite(candidate))
                or not np.all(candidate == np.floor(candidate))
                or not np.all((candidate >= 0) & (candidate <= 9))
            ):
                return None
            predictions.append(candidate.astype(int))
        return predictions
    except Exception as exc:
        logger.warning("ARC candidate sandbox unavailable or invalid: %s", exc)
        return None
    finally:
        if executor is not None:
            with suppress(Exception):
                executor.close()

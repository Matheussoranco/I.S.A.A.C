"""Analogy Engine for ARC-AGI.

Extracts transformation rules from input→output training pairs by operating at
multiple levels of abstraction:

1. **Pixel level** — colour mapping, shift, inversion
2. **Object level** — which objects moved, changed colour, were added/removed,
   scaled, or mirrored; correspondence between input and output objects
3. **Structural level** — symmetry completion, tiling, size changes, pattern extension

The output is a ranked list of ``TransformHypothesis`` objects that the program
synthesis engine uses to guide its search.

Design philosophy (Chollet):
    Analogy is the core mechanism of abstract intelligence.  Rather than
    pattern-matching over raw pixels, we explicitly represent *what changed*
    between examples and *why*.  This lets the agent generalise correctly from
    just 2–5 training pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac.arc.grid_ops import Grid, GridObject, extract_objects
from isaac.arc.priors import (
    ObjectSignature,
    compute_object_signature,
    infer_colour_correspondence,
    objects_same_shape,
    relative_position,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectDelta:
    """Describes how a single object changed between input and output."""

    input_id: int | None = None
    output_id: int | None = None
    change_type: str = "unchanged"
    """
    One of:
      'unchanged'  — same colour, position, shape
      'moved'      — same colour/shape, different position
      'recoloured' — same position/shape, different colour
      'scaled'     — same colour/position, different size
      'transformed'— shape changed but correspondence inferred
      'appeared'   — new object in output (no input match)
      'disappeared'— input object not in output
      'morphed'    — both colour and shape changed
    """
    colour_before: int | None = None
    colour_after: int | None = None
    position_delta: tuple[int, int] = (0, 0)
    """(row_delta, col_delta) of centroid shift."""
    scale_factor: float = 1.0
    direction: str = ""
    """Relative movement direction if 'moved'."""


@dataclass
class PairDelta:
    """Structured delta between a single input→output pair."""

    shape_changed: bool = False
    output_shape: tuple[int, int] = (0, 0)
    input_shape: tuple[int, int] = (0, 0)
    colour_map: dict[int, int] | None = None
    """If a simple colour permutation explains the whole transform."""
    object_deltas: list[ObjectDelta] = field(default_factory=list)
    n_objects_in: int = 0
    n_objects_out: int = 0
    n_added: int = 0
    n_removed: int = 0
    n_moved: int = 0
    n_recoloured: int = 0
    n_scaled: int = 0
    background_in: int = 0
    background_out: int = 0


@dataclass
class TransformHypothesis:
    """A candidate transformation rule inferred from training pairs."""

    name: str
    """Short human-readable name, e.g. 'rotate_90', 'fill_enclosed', 'move_to_border'."""
    description: str
    """Longer explanation for LLM prompts."""
    confidence: float = 0.0
    """0.0–1.0 based on how many training pairs it explains."""
    dsl_ops: list[dict[str, Any]] = field(default_factory=list)
    """Serialised DSL operations (may be empty if requires custom code)."""
    requires_custom_code: bool = False
    parameters: dict[str, Any] = field(default_factory=dict)
    """Inferred parameters (e.g. {'from_colour': 1, 'to_colour': 3})."""


# ─────────────────────────────────────────────────────────────────────────────
# Pair-level delta extraction
# ─────────────────────────────────────────────────────────────────────────────


def _match_objects(
    in_objs: list[GridObject],
    out_objs: list[GridObject],
) -> list[tuple[GridObject | None, GridObject | None]]:
    """Greedy object matching: prefer same colour + same shape."""
    in_sigs = [compute_object_signature(o) for o in in_objs]
    out_sigs = [compute_object_signature(o) for o in out_objs]

    matched_out: set[int] = set()
    pairs: list[tuple[GridObject | None, GridObject | None]] = []

    # Pass 1: exact shape + colour match
    for i, (io, isig) in enumerate(zip(in_objs, in_sigs)):
        best_j: int | None = None
        for j, (oo, osig) in enumerate(zip(out_objs, out_sigs)):
            if j in matched_out:
                continue
            if isig.colour == osig.colour and objects_same_shape(isig, osig):
                best_j = j
                break
        if best_j is not None:
            matched_out.add(best_j)
            pairs.append((io, out_objs[best_j]))
        else:
            pairs.append((io, None))

    # Pass 2: same colour, different shape
    unmatched_out = [j for j in range(len(out_objs)) if j not in matched_out]
    for pair_idx, (io, matched) in enumerate(pairs):
        if matched is not None or io is None:
            continue
        isig = in_sigs[in_objs.index(io)]
        for j in unmatched_out:
            osig = out_sigs[j]
            if isig.colour == osig.colour:
                pairs[pair_idx] = (io, out_objs[j])
                matched_out.add(j)
                unmatched_out.remove(j)
                break

    # Pass 3: same shape, different colour
    for pair_idx, (io, matched) in enumerate(pairs):
        if matched is not None or io is None:
            continue
        isig = in_sigs[in_objs.index(io)]
        for j in unmatched_out:
            osig = out_sigs[j]
            if objects_same_shape(isig, osig):
                pairs[pair_idx] = (io, out_objs[j])
                matched_out.add(j)
                unmatched_out.remove(j)
                break

    # Unmatched output objects = appeared
    for j in unmatched_out:
        pairs.append((None, out_objs[j]))

    return pairs


def extract_pair_delta(in_grid: Grid, out_grid: Grid) -> PairDelta:
    """Compute a structured delta between one training pair."""
    from isaac.arc.grid_ops import detect_background

    bg_in = detect_background(in_grid)
    bg_out = detect_background(out_grid)
    in_objs = extract_objects(in_grid, bg_in)
    out_objs = extract_objects(out_grid, bg_out)

    delta = PairDelta(
        shape_changed=in_grid.shape != out_grid.shape,
        input_shape=in_grid.shape,
        output_shape=out_grid.shape,
        n_objects_in=len(in_objs),
        n_objects_out=len(out_objs),
        background_in=bg_in,
        background_out=bg_out,
    )

    # Try global colour map
    delta.colour_map = infer_colour_correspondence(in_grid, out_grid)

    # Object-level matching (only if same dimensions)
    if not delta.shape_changed:
        pairs = _match_objects(in_objs, out_objs)
        for io, oo in pairs:
            od = ObjectDelta(
                input_id=io.id if io else None,
                output_id=oo.id if oo else None,
                colour_before=io.colour if io else None,
                colour_after=oo.colour if oo else None,
            )

            if io is None:
                od.change_type = "appeared"
                delta.n_added += 1
            elif oo is None:
                od.change_type = "disappeared"
                delta.n_removed += 1
            else:
                isig = compute_object_signature(io)
                osig = compute_object_signature(oo)

                same_shape = objects_same_shape(isig, osig)
                same_colour = isig.colour == osig.colour
                same_pos = (io.bbox == oo.bbox)

                ic = (isig.centroid[0], isig.centroid[1])
                oc = (osig.centroid[0], osig.centroid[1])
                pos_delta = (int(oc[0] - ic[0]), int(oc[1] - ic[1]))
                od.position_delta = pos_delta

                if same_colour and same_shape and same_pos:
                    od.change_type = "unchanged"
                elif same_colour and same_shape and not same_pos:
                    od.change_type = "moved"
                    od.direction = relative_position(io, oo)
                    delta.n_moved += 1
                elif not same_colour and same_shape:
                    od.change_type = "recoloured"
                    delta.n_recoloured += 1
                elif same_colour and not same_shape:
                    scale = osig.size / isig.size if isig.size > 0 else 1.0
                    od.scale_factor = scale
                    od.change_type = "scaled"
                    delta.n_scaled += 1
                else:
                    od.change_type = "morphed"

            delta.object_deltas.append(od)

    return delta


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis generators
# ─────────────────────────────────────────────────────────────────────────────


def _hyp_colour_map(deltas: list[PairDelta]) -> TransformHypothesis | None:
    """Infer a global colour-mapping transformation."""
    maps = [d.colour_map for d in deltas if d.colour_map is not None]
    if len(maps) != len(deltas):
        return None
    # Check consistency
    first = maps[0]
    if all(m == first for m in maps):
        return TransformHypothesis(
            name="fill_colour_map",
            description=f"Apply colour substitution: {first}",
            confidence=1.0,
            dsl_ops=[
                {"op": "fill_colour", "args": {"from_colour": fc, "to_colour": tc}}
                for fc, tc in first.items()
                if fc != tc
            ],
            parameters={"colour_map": first},
        )
    return None


def _hyp_rotation(deltas: list[PairDelta]) -> list[TransformHypothesis]:
    """Check if a rotation explains all training pairs."""
    hyps: list[TransformHypothesis] = []
    for angle, k in [(90, -1), (180, 2), (270, -3)]:
        consistent = True
        for d in deltas:
            # We can only check this at grid level, not from delta alone
            # Mark as requiring verification
            if d.shape_changed and d.input_shape[0] != d.input_shape[1]:
                consistent = False
                break
        if consistent:
            hyps.append(TransformHypothesis(
                name=f"rotate_{angle}",
                description=f"Rotate the grid {angle}° clockwise",
                confidence=0.5,  # Will be updated by solver verification
                dsl_ops=[{"op": f"rotate_{angle}"}],
            ))
    return hyps


def _hyp_object_moves(deltas: list[PairDelta]) -> list[TransformHypothesis]:
    """Detect consistent object movement patterns."""
    if not deltas:
        return []

    hyps: list[TransformHypothesis] = []

    # All deltas have same number of moved objects?
    move_deltas = [
        [od for od in d.object_deltas if od.change_type == "moved"]
        for d in deltas
    ]

    if not all(move_deltas):
        return hyps

    # Consistent direction?
    directions = set()
    for moved in move_deltas:
        for od in moved:
            directions.add(od.direction)

    if len(directions) == 1:
        direction = directions.pop()
        hyps.append(TransformHypothesis(
            name=f"move_objects_{direction}",
            description=f"All objects move {direction}",
            confidence=0.6,
            requires_custom_code=True,
            parameters={"direction": direction},
        ))

    # Consistent position delta?
    all_deltas: list[tuple[int, int]] = []
    for moved in move_deltas:
        for od in moved:
            all_deltas.append(od.position_delta)

    if all_deltas and len(set(all_deltas)) == 1:
        dr, dc = all_deltas[0]
        hyps.append(TransformHypothesis(
            name="shift_objects",
            description=f"Objects shift by ({dr}, {dc})",
            confidence=0.7,
            requires_custom_code=True,
            parameters={"row_delta": dr, "col_delta": dc},
        ))

    return hyps


def _hyp_gravity(deltas: list[PairDelta]) -> list[TransformHypothesis]:
    """Check if gravity (falling) explains the transform."""
    hyps = []
    # Gravity patterns: all objects "fall" to one side
    directions = ["down", "up", "left", "right"]
    for d in directions:
        hyps.append(TransformHypothesis(
            name=f"gravity_{d}",
            description=f"Apply gravity — non-background cells fall {d}",
            confidence=0.3,
            dsl_ops=[{"op": f"gravity_{d}"}],
        ))
    return hyps


def _hyp_symmetry_completion(deltas: list[PairDelta]) -> TransformHypothesis | None:
    """Detect if output is always the symmetrically completed version of input."""
    # Output always larger or same size, and symmetric
    all_out_larger = all(
        d.output_shape[0] >= d.input_shape[0] and
        d.output_shape[1] >= d.input_shape[1]
        for d in deltas
    )
    if all_out_larger:
        return TransformHypothesis(
            name="complete_symmetry",
            description="Complete the grid to be symmetric",
            confidence=0.4,
            requires_custom_code=True,
        )
    return None


def _hyp_fill_enclosed(deltas: list[PairDelta]) -> TransformHypothesis | None:
    """Detect fill-enclosed-region pattern."""
    # Colour was added in output, same shape
    all_same_shape = all(not d.shape_changed for d in deltas)
    all_have_additions = all(d.n_added > 0 or d.n_recoloured > 0 for d in deltas)
    if all_same_shape and all_have_additions:
        return TransformHypothesis(
            name="fill_enclosed_regions",
            description="Fill enclosed background regions with a new colour",
            confidence=0.5,
            requires_custom_code=True,
        )
    return None


def _hyp_size_scaling(deltas: list[PairDelta]) -> TransformHypothesis | None:
    """Detect uniform scale-up transformation."""
    if not deltas:
        return None
    ratios: list[float] = []
    for d in deltas:
        if d.shape_changed and d.input_shape[0] > 0 and d.input_shape[1] > 0:
            r_ratio = d.output_shape[0] / d.input_shape[0]
            c_ratio = d.output_shape[1] / d.input_shape[1]
            if abs(r_ratio - c_ratio) < 0.01 and r_ratio > 1:
                ratios.append(r_ratio)
    if ratios and len(ratios) == len(deltas) and len(set(int(r) for r in ratios)) == 1:
        factor = int(ratios[0])
        return TransformHypothesis(
            name=f"scale_up_{factor}",
            description=f"Scale each cell up to a {factor}x{factor} block",
            confidence=0.8,
            dsl_ops=[{"op": "scale_up", "args": {"factor": factor}}],
            parameters={"factor": factor},
        )
    return None


def _hyp_tiling(deltas: list[PairDelta]) -> TransformHypothesis | None:
    """Detect tiling pattern (output is input tiled N×M times)."""
    if not deltas:
        return None
    tile_shapes: list[tuple[int, int]] = []
    for d in deltas:
        if d.shape_changed and d.input_shape[0] > 0 and d.input_shape[1] > 0:
            r_tiles = d.output_shape[0] / d.input_shape[0]
            c_tiles = d.output_shape[1] / d.input_shape[1]
            if r_tiles == int(r_tiles) and c_tiles == int(c_tiles):
                tile_shapes.append((int(r_tiles), int(c_tiles)))
    if tile_shapes and len(tile_shapes) == len(deltas) and len(set(tile_shapes)) == 1:
        rows, cols = tile_shapes[0]
        return TransformHypothesis(
            name=f"tile_{rows}x{cols}",
            description=f"Tile the grid {rows} rows × {cols} cols",
            confidence=0.85,
            dsl_ops=[{"op": "tile_grid", "args": {"rows": rows, "cols": cols}}],
            parameters={"rows": rows, "cols": cols},
        )
    return None


def _hyp_recolour(deltas: list[PairDelta]) -> list[TransformHypothesis]:
    """Detect consistent single-colour substitutions."""
    hyps: list[TransformHypothesis] = []
    if not deltas:
        return hyps

    # All deltas have same-shape grids and only recoloured objects?
    all_same_shape = all(not d.shape_changed for d in deltas)
    if not all_same_shape:
        return hyps

    all_recolouring: list[dict[int, int]] = []
    for d in deltas:
        colour_changes: dict[int, int] = {}
        for od in d.object_deltas:
            if od.change_type == "recoloured" and od.colour_before is not None and od.colour_after is not None:
                colour_changes[od.colour_before] = od.colour_after
        if colour_changes:
            all_recolouring.append(colour_changes)

    if all_recolouring and len(all_recolouring) == len(deltas):
        first = all_recolouring[0]
        if all(r == first for r in all_recolouring):
            for fc, tc in first.items():
                hyps.append(TransformHypothesis(
                    name=f"recolour_{fc}_to_{tc}",
                    description=f"Replace colour {fc} with colour {tc}",
                    confidence=0.9,
                    dsl_ops=[{"op": "fill_colour", "args": {"from_colour": fc, "to_colour": tc}}],
                    parameters={"from_colour": fc, "to_colour": tc},
                ))
    return hyps


# ─────────────────────────────────────────────────────────────────────────────
# Main analogy engine
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnalogyResult:
    """Result of the analogy engine for a full ARC task."""

    pair_deltas: list[PairDelta] = field(default_factory=list)
    hypotheses: list[TransformHypothesis] = field(default_factory=list)
    consistent_observations: list[str] = field(default_factory=list)
    """Facts that hold across ALL training pairs."""
    variable_observations: list[str] = field(default_factory=list)
    """Facts that differ between pairs (parameters to be inferred)."""


def run_analogy_engine(
    train_pairs: list[dict[str, Any]],
) -> AnalogyResult:
    """Run the full analogy pipeline on a list of training pairs.

    Parameters
    ----------
    train_pairs:
        Each element: ``{"input": [[...]], "output": [[...]]}``

    Returns
    -------
    AnalogyResult
        Ranked hypotheses and structured observations.
    """
    result = AnalogyResult()

    if not train_pairs:
        return result

    # Extract deltas for every pair
    for pair in train_pairs:
        in_grid = np.array(pair["input"], dtype=int)
        out_grid = np.array(pair.get("output", []), dtype=int)
        if out_grid.size == 0:
            continue
        delta = extract_pair_delta(in_grid, out_grid)
        result.pair_deltas.append(delta)

    deltas = result.pair_deltas
    if not deltas:
        return result

    # --- Consistent observations ---
    all_same_shape = all(not d.shape_changed for d in deltas)
    if all_same_shape:
        result.consistent_observations.append("Output has same dimensions as input.")
    else:
        result.variable_observations.append("Output dimensions differ from input.")

    # Object count patterns
    obj_in_counts = [d.n_objects_in for d in deltas]
    obj_out_counts = [d.n_objects_out for d in deltas]
    if len(set(obj_in_counts)) == 1 and len(set(obj_out_counts)) == 1:
        result.consistent_observations.append(
            f"Input always has {obj_in_counts[0]} objects; "
            f"output always has {obj_out_counts[0]} objects."
        )
    if all(d.n_objects_in == d.n_objects_out for d in deltas):
        result.consistent_observations.append("Object count preserved.")
    if all(d.n_added > 0 for d in deltas):
        result.consistent_observations.append("New objects appear in output.")
    if all(d.n_removed > 0 for d in deltas):
        result.consistent_observations.append("Objects disappear in output.")

    # Background consistency
    bg_values = set(d.background_in for d in deltas) | set(d.background_out for d in deltas)
    if len(bg_values) == 1:
        result.consistent_observations.append(f"Background colour is always {bg_values.pop()}.")

    # --- Hypothesis generation ---
    hypotheses: list[TransformHypothesis] = []

    # Colour mapping
    h = _hyp_colour_map(deltas)
    if h:
        hypotheses.append(h)

    # Recolouring
    hypotheses.extend(_hyp_recolour(deltas))

    # Size transformations
    h = _hyp_size_scaling(deltas)
    if h:
        hypotheses.append(h)

    h = _hyp_tiling(deltas)
    if h:
        hypotheses.append(h)

    # Movement
    hypotheses.extend(_hyp_object_moves(deltas))

    # Structural
    h = _hyp_fill_enclosed(deltas)
    if h:
        hypotheses.append(h)

    h = _hyp_symmetry_completion(deltas)
    if h:
        hypotheses.append(h)

    # Gravity (lower confidence — will be verified)
    hypotheses.extend(_hyp_gravity(deltas))

    # Rotation (will be verified by solver)
    hypotheses.extend(_hyp_rotation(deltas))

    # Sort by confidence descending
    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    result.hypotheses = hypotheses

    return result


def format_analogy_for_prompt(result: AnalogyResult) -> str:
    """Format AnalogyResult into a structured string for LLM system prompts."""
    lines: list[str] = ["## Analogy Engine Findings"]

    if result.consistent_observations:
        lines.append("\n### Consistent facts (hold across ALL training pairs):")
        for obs in result.consistent_observations:
            lines.append(f"  - {obs}")

    if result.variable_observations:
        lines.append("\n### Variable facts (differ between pairs):")
        for obs in result.variable_observations:
            lines.append(f"  - {obs}")

    if result.hypotheses:
        lines.append("\n### Top transformation hypotheses (by confidence):")
        for i, h in enumerate(result.hypotheses[:8]):
            lines.append(
                f"  {i + 1}. [{h.confidence:.0%}] **{h.name}**: {h.description}"
            )
            if h.parameters:
                lines.append(f"     Parameters: {h.parameters}")

    if result.pair_deltas:
        lines.append("\n### Per-pair delta summary:")
        for i, d in enumerate(result.pair_deltas):
            lines.append(
                f"  Pair {i + 1}: shape_changed={d.shape_changed}, "
                f"objects in={d.n_objects_in} out={d.n_objects_out}, "
                f"moved={d.n_moved}, recoloured={d.n_recoloured}, "
                f"added={d.n_added}, removed={d.n_removed}"
            )
            if d.colour_map and d.colour_map != {k: k for k in d.colour_map}:
                lines.append(f"     Colour map: {d.colour_map}")

    return "\n".join(lines)

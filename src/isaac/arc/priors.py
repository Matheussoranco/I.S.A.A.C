"""Core Knowledge Priors for ARC-AGI reasoning.

Implements François Chollet's four core knowledge systems as explicit,
composable utilities that feed the neuro-symbolic pipeline:

1. **Objectness** — cohesion, persistence, contact, occlusion
2. **Goal-directedness / Agency** — intentional transformations, delta detection
3. **Numbers and counting** — cardinality, correspondence, ordinality
4. **Basic geometry and topology** — lines, rectangles, symmetry, containment,
   inside/outside, connectivity

These priors are *not* learned — they are built-in and serve as inductive
biases that drastically reduce the hypothesis space for ARC tasks, mirroring
how humans approach the benchmark.

Reference:
    Chollet, F. (2019). On the Measure of Intelligence.
    https://arxiv.org/abs/1911.01547
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isaac.arc.grid_ops import Grid, GridObject, extract_objects

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  OBJECTNESS PRIORS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ObjectSignature:
    """Compact fingerprint of a GridObject used for cross-pair matching."""

    colour: int
    size: int
    height: int
    width: int
    normalized_shape: tuple[tuple[int, ...], ...]
    """Binary shape mask normalised to top-left origin."""
    is_rectangle: bool
    is_line: bool
    line_direction: str  # 'horizontal' | 'vertical' | 'diagonal' | 'none'
    is_square: bool
    aspect_ratio: float
    centroid: tuple[float, float]


def compute_object_signature(obj: GridObject) -> ObjectSignature:
    """Compute a compact, comparable signature for *obj*."""
    r1, c1, r2, c2 = obj.bbox
    h, w = r2 - r1 + 1, c2 - c1 + 1

    # Normalised binary mask
    cells_set = set(obj.cells)
    mask = tuple(
        tuple(1 if (r1 + dr, c1 + dc) in cells_set else 0 for dc in range(w))
        for dr in range(h)
    )

    # Rectangle test: all cells in bbox present
    is_rect = obj.size == h * w

    # Line test
    is_hline = h == 1 and w > 1
    is_vline = w == 1 and h > 1
    is_diag = False
    if not is_hline and not is_vline and h == w:
        # Check main or anti diagonal
        is_diag = all((r1 + i, c1 + i) in cells_set for i in range(h)) or \
                  all((r1 + i, c2 - i) in cells_set for i in range(h))
    line_dir = (
        "horizontal" if is_hline else
        "vertical" if is_vline else
        "diagonal" if is_diag else "none"
    )

    rows = [cell[0] for cell in obj.cells]
    cols = [cell[1] for cell in obj.cells]
    centroid = (float(np.mean(rows)), float(np.mean(cols)))

    return ObjectSignature(
        colour=obj.colour,
        size=obj.size,
        height=h,
        width=w,
        normalized_shape=mask,
        is_rectangle=is_rect,
        is_line=(is_hline or is_vline or is_diag),
        line_direction=line_dir,
        is_square=h == w,
        aspect_ratio=w / h if h > 0 else 0.0,
        centroid=centroid,
    )


def objects_same_shape(sig_a: ObjectSignature, sig_b: ObjectSignature) -> bool:
    """Return True if both objects have identical normalised shapes."""
    return sig_a.normalized_shape == sig_b.normalized_shape


def objects_same_size(sig_a: ObjectSignature, sig_b: ObjectSignature) -> bool:
    return sig_a.size == sig_b.size


def find_largest_object(objects: list[GridObject]) -> GridObject | None:
    return max(objects, key=lambda o: o.size) if objects else None


def find_smallest_object(objects: list[GridObject]) -> GridObject | None:
    return min(objects, key=lambda o: o.size) if objects else None


def group_objects_by_colour(objects: list[GridObject]) -> dict[int, list[GridObject]]:
    groups: dict[int, list[GridObject]] = {}
    for obj in objects:
        groups.setdefault(obj.colour, []).append(obj)
    return groups


def group_objects_by_shape(
    objects: list[GridObject],
) -> list[list[GridObject]]:
    """Group objects that share the same normalised shape."""
    sigs = [compute_object_signature(o) for o in objects]
    groups: list[list[GridObject]] = []
    used = [False] * len(objects)
    for i, (obj_i, sig_i) in enumerate(zip(objects, sigs)):
        if used[i]:
            continue
        group = [obj_i]
        used[i] = True
        for j in range(i + 1, len(objects)):
            if not used[j] and objects_same_shape(sig_i, sigs[j]):
                group.append(objects[j])
                used[j] = True
        groups.append(group)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SPATIAL / TOPOLOGICAL PRIORS
# ─────────────────────────────────────────────────────────────────────────────


def is_inside(inner: GridObject, outer: GridObject) -> bool:
    """Return True if *inner* is spatially contained within *outer*'s bbox."""
    ri1, ci1, ri2, ci2 = inner.bbox
    ro1, co1, ro2, co2 = outer.bbox
    return ro1 <= ri1 and ri2 <= ro2 and co1 <= ci1 and ci2 <= co2


def are_adjacent(obj_a: GridObject, obj_b: GridObject, gap: int = 1) -> bool:
    """Return True if the bboxes of *obj_a* and *obj_b* are within *gap* cells."""
    ra1, ca1, ra2, ca2 = obj_a.bbox
    rb1, cb1, rb2, cb2 = obj_b.bbox
    v_dist = max(0, rb1 - ra2 - 1, ra1 - rb2 - 1)
    h_dist = max(0, cb1 - ca2 - 1, ca1 - cb2 - 1)
    return v_dist <= gap and h_dist <= gap


def are_touching(obj_a: GridObject, obj_b: GridObject) -> bool:
    """Return True if the objects share an edge (4-connected adjacency)."""
    cells_a = set(obj_a.cells)
    for r, c in obj_b.cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in cells_a:
                return True
    return False


def are_aligned_horizontal(obj_a: GridObject, obj_b: GridObject) -> bool:
    """Return True if both objects share the same row band (bbox overlap in row axis)."""
    ra1, _, ra2, _ = obj_a.bbox
    rb1, _, rb2, _ = obj_b.bbox
    return not (ra2 < rb1 or rb2 < ra1)


def are_aligned_vertical(obj_a: GridObject, obj_b: GridObject) -> bool:
    """Return True if both objects share the same column band."""
    _, ca1, _, ca2 = obj_a.bbox
    _, cb1, _, cb2 = obj_b.bbox
    return not (ca2 < cb1 or cb2 < ca1)


def relative_position(obj_a: GridObject, obj_b: GridObject) -> str:
    """Return the cardinal/ordinal direction from *obj_a* to *obj_b*."""
    ra = (obj_a.bbox[0] + obj_a.bbox[2]) / 2
    ca = (obj_a.bbox[1] + obj_a.bbox[3]) / 2
    rb = (obj_b.bbox[0] + obj_b.bbox[2]) / 2
    cb = (obj_b.bbox[1] + obj_b.bbox[3]) / 2
    dr, dc = rb - ra, cb - ca
    if abs(dr) < 0.5 and abs(dc) < 0.5:
        return "same"
    if abs(dr) >= abs(dc):
        return "below" if dr > 0 else "above"
    return "right" if dc > 0 else "left"


def detect_enclosed_regions(grid: Grid, background: int = 0) -> list[list[tuple[int, int]]]:
    """Detect background-coloured cells that are fully enclosed by non-background cells.

    Uses flood-fill from the border to identify reachable background, then
    returns the unreachable interior regions.
    """
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=bool)

    # Flood fill from all border background cells
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in [0, w - 1]:
            if grid[r, c] == background and not reachable[r, c]:
                stack.append((r, c))
                reachable[r, c] = True
    for c in range(w):
        for r in [0, h - 1]:
            if grid[r, c] == background and not reachable[r, c]:
                stack.append((r, c))
                reachable[r, c] = True

    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and grid[nr, nc] == background:
                reachable[nr, nc] = True
                stack.append((nr, nc))

    # Enclosed = background but NOT reachable from border
    enclosed_mask = (grid == background) & ~reachable
    if not enclosed_mask.any():
        return []

    # Flood fill to find separate enclosed regions
    visited = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if enclosed_mask[r, c] and not visited[r, c]:
                region: list[tuple[int, int]] = []
                region_stack = [(r, c)]
                visited[r, c] = True
                while region_stack:
                    cr, cc = region_stack.pop()
                    region.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and enclosed_mask[nr, nc]:
                            visited[nr, nc] = True
                            region_stack.append((nr, nc))
                regions.append(region)
    return regions


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMBER / COUNTING PRIORS
# ─────────────────────────────────────────────────────────────────────────────


def count_objects_by_colour(grid: Grid, background: int = 0) -> dict[int, int]:
    """Return {colour: number_of_objects} mapping."""
    objects = extract_objects(grid, background)
    counts: dict[int, int] = {}
    for obj in objects:
        counts[obj.colour] = counts.get(obj.colour, 0) + 1
    return counts


def count_cells_by_colour(grid: Grid) -> dict[int, int]:
    """Return {colour: number_of_cells} mapping."""
    unique, cnts = np.unique(grid, return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, cnts)}


def infer_colour_correspondence(
    in_grid: Grid, out_grid: Grid
) -> dict[int, int] | None:
    """Infer a bijective colour mapping if a simple recolouring explains the transform."""
    if in_grid.shape != out_grid.shape:
        return None
    in_colours = np.unique(in_grid)
    out_colours = np.unique(out_grid)
    if len(in_colours) != len(out_colours):
        return None
    mapping: dict[int, int] = {}
    for in_c, out_c in zip(sorted(in_colours.tolist()), sorted(out_colours.tolist())):
        # Verify the mapping is consistent
        mask = in_grid == in_c
        if not np.all(out_grid[mask] == out_c):
            return None
        mapping[in_c] = out_c
    return mapping


def detect_numeric_pattern(values: list[int]) -> str:
    """Classify a sequence of integers into a named pattern."""
    if len(values) < 2:
        return "singleton"
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if all(d == diffs[0] for d in diffs):
        if diffs[0] == 0:
            return "constant"
        if diffs[0] == 1:
            return "ascending"
        if diffs[0] == -1:
            return "descending"
        return f"arithmetic({diffs[0]})"
    ratios = [values[i + 1] / values[i] for i in range(len(values) - 1) if values[i] != 0]
    if ratios and all(abs(r - ratios[0]) < 1e-9 for r in ratios):
        return f"geometric({ratios[0]:.2f})"
    return "irregular"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GEOMETRIC / SYMMETRY PRIORS
# ─────────────────────────────────────────────────────────────────────────────


def detect_rotational_symmetry(grid: Grid) -> list[int]:
    """Return list of rotational orders (90, 180, 270) for which the grid is symmetric."""
    symmetric = []
    for k in [1, 2, 3]:
        if np.array_equal(grid, np.rot90(grid, k)):
            symmetric.append(k * 90)
    return symmetric


def detect_reflection_axes(grid: Grid) -> dict[str, bool]:
    """Detect reflection symmetry across horizontal, vertical, and diagonal axes."""
    h, w = grid.shape
    result: dict[str, bool] = {
        "horizontal": np.array_equal(grid, grid[::-1, :]),
        "vertical": np.array_equal(grid, grid[:, ::-1]),
        "diagonal": h == w and np.array_equal(grid, grid.T),
        "anti_diagonal": h == w and np.array_equal(grid, grid[::-1, ::-1].T),
    }
    return result


def infer_missing_quadrant(grid: Grid) -> str:
    """Detect which quadrant (if any) appears to be missing/empty (background-filled)."""
    h, w = grid.shape
    mh, mw = h // 2, w // 2
    quads = {
        "top_left": grid[:mh, :mw],
        "top_right": grid[:mh, mw:],
        "bottom_left": grid[mh:, :mw],
        "bottom_right": grid[mh:, mw:],
    }
    bg = int(np.bincount(grid.ravel()).argmax())
    for name, q in quads.items():
        if np.all(q == bg):
            return name
    return "none"


def find_line_segments(grid: Grid, background: int = 0) -> list[dict[str, Any]]:
    """Find all horizontal and vertical line segments in the grid."""
    lines: list[dict[str, Any]] = []
    h, w = grid.shape

    # Horizontal lines
    for r in range(h):
        c = 0
        while c < w:
            if grid[r, c] != background:
                colour = int(grid[r, c])
                start = c
                while c < w and grid[r, c] == colour:
                    c += 1
                length = c - start
                if length >= 2:
                    lines.append({
                        "direction": "horizontal",
                        "row": r,
                        "col_start": start,
                        "col_end": c - 1,
                        "length": length,
                        "colour": colour,
                    })
            else:
                c += 1

    # Vertical lines
    for col in range(w):
        r = 0
        while r < h:
            if grid[r, col] != background:
                colour = int(grid[r, col])
                start = r
                while r < h and grid[r, col] == colour:
                    r += 1
                length = r - start
                if length >= 2:
                    lines.append({
                        "direction": "vertical",
                        "col": col,
                        "row_start": start,
                        "row_end": r - 1,
                        "length": length,
                        "colour": colour,
                    })
            else:
                r += 1

    return lines


def detect_grid_lines(grid: Grid, background: int = 0) -> dict[str, list[int]]:
    """Detect rows and columns that form dividing lines (fully non-background)."""
    h, w = grid.shape
    dividing_rows = [r for r in range(h) if np.all(grid[r, :] != background)]
    dividing_cols = [c for c in range(w) if np.all(grid[:, c] != background)]
    return {"rows": dividing_rows, "cols": dividing_cols}


def detect_grid_partitions(grid: Grid, background: int = 0) -> list[Grid]:
    """Split a grid along detected dividing lines into sub-grids."""
    lines = detect_grid_lines(grid, background)
    row_divs = sorted(set([-1] + lines["rows"] + [grid.shape[0]]))
    col_divs = sorted(set([-1] + lines["cols"] + [grid.shape[1]]))

    partitions: list[Grid] = []
    for i in range(len(row_divs) - 1):
        r_start = row_divs[i] + 1
        r_end = row_divs[i + 1]
        for j in range(len(col_divs) - 1):
            c_start = col_divs[j] + 1
            c_end = col_divs[j + 1]
            if r_end > r_start and c_end > c_start:
                partitions.append(grid[r_start:r_end, c_start:c_end].copy())
    return partitions


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPOSITE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PriorAnalysis:
    """Full prior-informed analysis of an ARC grid."""

    background: int = 0
    objects: list[GridObject] = field(default_factory=list)
    object_signatures: list[ObjectSignature] = field(default_factory=list)
    colour_counts: dict[int, int] = field(default_factory=dict)
    object_counts_by_colour: dict[int, int] = field(default_factory=dict)
    shape_groups: list[list[int]] = field(default_factory=list)
    """Indices into objects[] grouped by identical shape."""
    enclosed_regions: list[list[tuple[int, int]]] = field(default_factory=list)
    lines: list[dict[str, Any]] = field(default_factory=list)
    grid_dividers: dict[str, list[int]] = field(default_factory=dict)
    reflection_axes: dict[str, bool] = field(default_factory=dict)
    rotational_symmetry: list[int] = field(default_factory=list)
    has_grid_structure: bool = False
    largest_object_id: int | None = None
    smallest_object_id: int | None = None


def full_prior_analysis(grid: Grid) -> PriorAnalysis:
    """Run the full suite of core-knowledge priors on *grid*."""
    from isaac.arc.grid_ops import extract_colours, detect_background

    colours = extract_colours(grid)
    bg = detect_background(grid)
    objects = extract_objects(grid, bg)
    sigs = [compute_object_signature(o) for o in objects]

    # Shape groups
    shape_groups_objs = group_objects_by_shape(objects)
    shape_groups = [[objects.index(o) for o in g] for g in shape_groups_objs]

    # Largest / smallest
    largest = find_largest_object(objects)
    smallest = find_smallest_object(objects)

    grid_divs = detect_grid_lines(grid, bg)
    has_grid = len(grid_divs["rows"]) > 0 or len(grid_divs["cols"]) > 0

    return PriorAnalysis(
        background=bg,
        objects=objects,
        object_signatures=sigs,
        colour_counts={int(k): int(v) for k, v in colours.items()},
        object_counts_by_colour=count_objects_by_colour(grid, bg),
        shape_groups=shape_groups,
        enclosed_regions=detect_enclosed_regions(grid, bg),
        lines=find_line_segments(grid, bg),
        grid_dividers=grid_divs,
        reflection_axes=detect_reflection_axes(grid),
        rotational_symmetry=detect_rotational_symmetry(grid),
        has_grid_structure=has_grid,
        largest_object_id=largest.id if largest else None,
        smallest_object_id=smallest.id if smallest else None,
    )


def describe_prior_analysis(analysis: PriorAnalysis) -> list[str]:
    """Convert a PriorAnalysis into human-readable observation strings for LLM prompts."""
    obs: list[str] = []
    obs.append(f"Background colour: {analysis.background}")
    obs.append(f"Unique colours: {sorted(analysis.colour_counts.keys())}")
    obs.append(f"Objects found: {len(analysis.objects)}")

    for obj, sig in zip(analysis.objects, analysis.object_signatures):
        desc = (
            f"  Object {obj.id}: colour={obj.colour}, size={sig.size}, "
            f"shape={sig.height}x{sig.width}"
        )
        if sig.is_rectangle:
            desc += ", IS RECTANGLE"
        if sig.is_line:
            desc += f", IS LINE ({sig.line_direction})"
        if sig.is_square:
            desc += ", IS SQUARE"
        obs.append(desc)

    if analysis.enclosed_regions:
        obs.append(f"Enclosed background regions: {len(analysis.enclosed_regions)}")

    if analysis.has_grid_structure:
        obs.append(f"Grid dividers — rows: {analysis.grid_dividers['rows']}, "
                   f"cols: {analysis.grid_dividers['cols']}")

    refl = [k for k, v in analysis.reflection_axes.items() if v]
    if refl:
        obs.append(f"Reflection symmetry: {refl}")
    if analysis.rotational_symmetry:
        obs.append(f"Rotational symmetry (degrees): {analysis.rotational_symmetry}")

    if analysis.shape_groups:
        multi = [g for g in analysis.shape_groups if len(g) > 1]
        if multi:
            obs.append(f"Objects with identical shapes: {multi}")

    return obs

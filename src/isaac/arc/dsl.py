"""Domain-Specific Language (DSL) for ARC grid transformations.

Provides a library of composable, pure-function transformation primitives
that operate on numpy grids.  These are the building blocks the program
synthesis engine can combine to solve ARC tasks.

Each primitive is registered in ``PRIMITIVES`` and can be composed via
``compose()``.  The synthesis engine searches over compositions of these
primitives to find a program that maps all training inputs to outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from isaac.arc.grid_ops import Grid

logger = logging.getLogger(__name__)

# Type alias for a grid transformation function
Transform = Callable[..., Grid]

# ---------------------------------------------------------------------------
# Primitive transformations
# ---------------------------------------------------------------------------


def identity(grid: Grid) -> Grid:
    """Return the grid unchanged."""
    return grid.copy()


def rotate_90(grid: Grid) -> Grid:
    """Rotate 90° clockwise."""
    return np.rot90(grid, k=-1).copy()


def rotate_180(grid: Grid) -> Grid:
    """Rotate 180°."""
    return np.rot90(grid, k=2).copy()


def rotate_270(grid: Grid) -> Grid:
    """Rotate 270° clockwise (= 90° counter-clockwise)."""
    return np.rot90(grid, k=-3).copy()


def flip_horizontal(grid: Grid) -> Grid:
    """Flip left-right."""
    return np.fliplr(grid).copy()


def flip_vertical(grid: Grid) -> Grid:
    """Flip top-bottom."""
    return np.flipud(grid).copy()


def transpose(grid: Grid) -> Grid:
    """Transpose (swap rows and columns)."""
    return grid.T.copy()


def shift_right(grid: Grid, n: int = 1) -> Grid:
    """Shift all cells right by *n* positions with wrapping."""
    return np.roll(grid, n, axis=1).copy()


def shift_left(grid: Grid, n: int = 1) -> Grid:
    """Shift all cells left by *n* positions with wrapping."""
    return np.roll(grid, -n, axis=1).copy()


def shift_down(grid: Grid, n: int = 1) -> Grid:
    """Shift all cells down by *n* positions with wrapping."""
    return np.roll(grid, n, axis=0).copy()


def shift_up(grid: Grid, n: int = 1) -> Grid:
    """Shift all cells up by *n* positions with wrapping."""
    return np.roll(grid, -n, axis=0).copy()


def fill_colour(grid: Grid, from_colour: int, to_colour: int) -> Grid:
    """Replace all cells of *from_colour* with *to_colour*."""
    result = grid.copy()
    result[result == from_colour] = to_colour
    return result


def invert_colours(grid: Grid, max_colour: int = 9) -> Grid:
    """Invert colours: cell = max_colour - cell."""
    return (max_colour - grid).copy()


def crop_to_object(grid: Grid, background: int = 0) -> Grid:
    """Crop the grid to the bounding box of non-background cells."""
    mask = grid != background
    if not mask.any():
        return grid.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return grid[np.ix_(rows, cols)].copy()


def pad_grid(grid: Grid, pad: int = 1, value: int = 0) -> Grid:
    """Pad the grid with *value* on all sides."""
    return np.pad(grid, pad, mode="constant", constant_values=value).copy()


def tile_grid(grid: Grid, rows: int = 2, cols: int = 2) -> Grid:
    """Tile the grid into a larger grid of *rows* x *cols* copies."""
    return np.tile(grid, (rows, cols)).copy()


def scale_up(grid: Grid, factor: int = 2) -> Grid:
    """Scale up each cell to a *factor* x *factor* block."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1).copy()


def gravity_down(grid: Grid, background: int = 0) -> Grid:
    """Apply gravity — non-background cells fall to the bottom."""
    result = np.full_like(grid, background)
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = col[col != background]
        if len(non_bg) > 0:
            result[-len(non_bg):, c] = non_bg
    return result


def gravity_left(grid: Grid, background: int = 0) -> Grid:
    """Apply gravity — non-background cells move to the left."""
    result = np.full_like(grid, background)
    for r in range(grid.shape[0]):
        row = grid[r, :]
        non_bg = row[row != background]
        if len(non_bg) > 0:
            result[r, :len(non_bg)] = non_bg
    return result


def hollow_rectangle(grid: Grid, background: int = 0) -> Grid:
    """For each object, keep only the border cells."""
    from isaac.arc.grid_ops import extract_objects

    result = np.full_like(grid, background)
    objects = extract_objects(grid, background)
    for obj in objects:
        r1, c1, r2, c2 = obj.bbox
        for r, c in obj.cells:
            if r == r1 or r == r2 or c == c1 or c == c2:
                result[r, c] = obj.colour
    return result


def flood_fill_from(
    grid: Grid, row: int, col: int, colour: int,
) -> Grid:
    """Flood fill from (row, col) with *colour*, 4-connected."""
    result = grid.copy()
    h, w = result.shape
    target = int(result[row, col])
    if target == colour:
        return result
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if result[r, c] != target:
            continue
        result[r, c] = colour
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                stack.append((nr, nc))
    return result


# ---------------------------------------------------------------------------
# Registry and composition
# ---------------------------------------------------------------------------

PRIMITIVES: dict[str, Transform] = {
    "identity": identity,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "transpose": transpose,
    "shift_right": shift_right,
    "shift_left": shift_left,
    "shift_down": shift_down,
    "shift_up": shift_up,
    "fill_colour": fill_colour,
    "invert_colours": invert_colours,
    "crop_to_object": crop_to_object,
    "pad_grid": pad_grid,
    "tile_grid": tile_grid,
    "scale_up": scale_up,
    "gravity_down": gravity_down,
    "gravity_left": gravity_left,
    "hollow_rectangle": hollow_rectangle,
}


def compose(*transforms: Transform) -> Transform:
    """Compose multiple transforms: f, g, h → h(g(f(grid)))."""
    def composed(grid: Grid) -> Grid:
        result = grid
        for fn in transforms:
            result = fn(result)
        return result
    return composed


def apply_program(
    program: list[dict[str, Any]],
    grid: Grid,
) -> Grid:
    """Execute a serialised program (list of step dicts) on a grid.

    Each step: ``{"op": "rotate_90"}`` or ``{"op": "fill_colour", "args": {"from_colour": 1, "to_colour": 2}}``.
    """
    result = grid.copy()
    for step in program:
        op_name = step.get("op", "identity")
        args = step.get("args", {})
        fn = PRIMITIVES.get(op_name)
        if fn is None:
            logger.warning("Unknown DSL op: %s — skipping.", op_name)
            continue
        try:
            result = fn(result, **args)
        except TypeError:
            # Try without args (for zero-arg primitives)
            result = fn(result)
    return result

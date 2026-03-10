"""Domain-Specific Language (DSL) for ARC grid transformations.

Provides a library of 55+ composable, pure-function transformation primitives
that operate on numpy grids.  These are the building blocks the program
synthesis engine can combine to solve ARC tasks.

Each primitive is registered in ``PRIMITIVES`` and can be composed via
``compose()``.  The synthesis engine searches over compositions of these
primitives to find a program that maps all training inputs to outputs.

Extended for ARC-AGI 2 with:
- Gravity in all 4 directions
- Fill enclosed regions (topology)
- Object selection / filtering
- Symmetry completion
- Drawing operations
- Boolean grid operations (AND, OR, XOR)
- Colour-based operations
- Size-based sorting and recolouring
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
# Original primitives
# ---------------------------------------------------------------------------


def identity(grid: Grid) -> Grid:
    """Return the grid unchanged."""
    return grid.copy()


def rotate_90(grid: Grid) -> Grid:
    """Rotate 90 degrees clockwise."""
    return np.rot90(grid, k=-1).copy()


def rotate_180(grid: Grid) -> Grid:
    """Rotate 180 degrees."""
    return np.rot90(grid, k=2).copy()


def rotate_270(grid: Grid) -> Grid:
    """Rotate 270 degrees clockwise (= 90 degrees counter-clockwise)."""
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


def diagonal_flip(grid: Grid) -> Grid:
    """Flip along the anti-diagonal (rotate 90 then flip left-right)."""
    return np.rot90(np.fliplr(grid)).copy()


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
    """Apply gravity: non-background cells fall to the bottom."""
    result = np.full_like(grid, background)
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = col[col != background]
        if len(non_bg) > 0:
            result[-len(non_bg):, c] = non_bg
    return result


def gravity_up(grid: Grid, background: int = 0) -> Grid:
    """Apply gravity: non-background cells rise to the top."""
    result = np.full_like(grid, background)
    for c in range(grid.shape[1]):
        col = grid[:, c]
        non_bg = col[col != background]
        if len(non_bg) > 0:
            result[:len(non_bg), c] = non_bg
    return result


def gravity_left(grid: Grid, background: int = 0) -> Grid:
    """Apply gravity: non-background cells move to the left."""
    result = np.full_like(grid, background)
    for r in range(grid.shape[0]):
        row = grid[r, :]
        non_bg = row[row != background]
        if len(non_bg) > 0:
            result[r, :len(non_bg)] = non_bg
    return result


def gravity_right(grid: Grid, background: int = 0) -> Grid:
    """Apply gravity: non-background cells move to the right."""
    result = np.full_like(grid, background)
    for r in range(grid.shape[0]):
        row = grid[r, :]
        non_bg = row[row != background]
        if len(non_bg) > 0:
            result[r, -len(non_bg):] = non_bg
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


def flood_fill_from(grid: Grid, row: int, col: int, colour: int) -> Grid:
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
# Extended primitives — ARC-AGI 2 (topology, objects, symmetry, drawing)
# ---------------------------------------------------------------------------


def fill_enclosed_regions(grid: Grid, fill_col: int = 1, background: int = 0) -> Grid:
    """Fill all background cells enclosed by non-background cells with *fill_col*."""
    from isaac.arc.priors import detect_enclosed_regions
    result = grid.copy()
    regions = detect_enclosed_regions(grid, background)
    for region in regions:
        for r, c in region:
            result[r, c] = fill_col
    return result


def fill_enclosed_auto(grid: Grid, background: int = 0) -> Grid:
    """Fill enclosed background regions with the dominant surrounding colour."""
    from isaac.arc.priors import detect_enclosed_regions
    from collections import Counter
    result = grid.copy()
    h, w = grid.shape
    regions = detect_enclosed_regions(grid, background)
    for region in regions:
        border_colours: list[int] = []
        for r, c in region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != background:
                    border_colours.append(int(grid[nr, nc]))
        if border_colours:
            dominant = Counter(border_colours).most_common(1)[0][0]
            for r, c in region:
                result[r, c] = dominant
    return result


def keep_colour(grid: Grid, colour: int, background: int = 0) -> Grid:
    """Keep only cells of *colour*; set everything else to *background*."""
    result = np.full_like(grid, background)
    result[grid == colour] = colour
    return result


def remove_colour(grid: Grid, colour: int, background: int = 0) -> Grid:
    """Remove all cells of *colour* (replace with *background*)."""
    result = grid.copy()
    result[result == colour] = background
    return result


def select_largest_object(grid: Grid, background: int = 0) -> Grid:
    """Keep only the largest (by cell count) object."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    if not objects:
        return np.full_like(grid, background)
    largest = max(objects, key=lambda o: o.size)
    result = np.full_like(grid, background)
    for r, c in largest.cells:
        result[r, c] = grid[r, c]
    return result


def select_smallest_object(grid: Grid, background: int = 0) -> Grid:
    """Keep only the smallest (by cell count) object."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    if not objects:
        return np.full_like(grid, background)
    smallest = min(objects, key=lambda o: o.size)
    result = np.full_like(grid, background)
    for r, c in smallest.cells:
        result[r, c] = grid[r, c]
    return result


def recolour_by_size(grid: Grid, background: int = 0) -> Grid:
    """Recolour objects: rank by size descending; rank value becomes new colour."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    if not objects:
        return grid.copy()
    sorted_objs = sorted(objects, key=lambda o: o.size, reverse=True)
    result = np.full_like(grid, background)
    for rank, obj in enumerate(sorted_objs, start=1):
        for r, c in obj.cells:
            result[r, c] = rank % 10
    return result


def recolour_by_position(grid: Grid, background: int = 0) -> Grid:
    """Recolour objects by reading-order position (top-left to bottom-right)."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    if not objects:
        return grid.copy()
    sorted_objs = sorted(objects, key=lambda o: (o.bbox[0], o.bbox[1]))
    result = np.full_like(grid, background)
    for rank, obj in enumerate(sorted_objs, start=1):
        for r, c in obj.cells:
            result[r, c] = rank % 10
    return result


def outline_objects(grid: Grid, outline_colour: int = 1, background: int = 0) -> Grid:
    """Add a 1-cell outline around each non-background object (8-connected)."""
    result = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != background:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == background:
                            result[nr, nc] = outline_colour
    return result


def complete_symmetry_horizontal(grid: Grid) -> Grid:
    """Mirror the top half downward to create horizontal symmetry."""
    result = grid.copy()
    h = grid.shape[0]
    result[h // 2:, :] = grid[:h // 2, :][::-1, :]
    return result


def complete_symmetry_vertical(grid: Grid) -> Grid:
    """Mirror the left half rightward to create vertical symmetry."""
    result = grid.copy()
    w = grid.shape[1]
    result[:, w // 2:] = grid[:, :w // 2][:, ::-1]
    return result


def mirror_objects_to_fill_symmetry(grid: Grid, background: int = 0) -> Grid:
    """Create point symmetry by reflecting each object through the grid centre."""
    result = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != background:
                mr, mc = h - 1 - r, w - 1 - c
                if result[mr, mc] == background:
                    result[mr, mc] = grid[r, c]
    return result


def extend_pattern_right(grid: Grid, n_tiles: int = 1) -> Grid:
    """Extend by appending *n_tiles* copies to the right."""
    return np.hstack([grid] + [grid] * n_tiles).copy()


def extend_pattern_down(grid: Grid, n_tiles: int = 1) -> Grid:
    """Extend by appending *n_tiles* copies downward."""
    return np.vstack([grid] + [grid] * n_tiles).copy()


def remove_border(grid: Grid, n: int = 1) -> Grid:
    """Remove *n* cells from each border."""
    if n * 2 >= grid.shape[0] or n * 2 >= grid.shape[1]:
        return grid.copy()
    return grid[n:-n, n:-n].copy()


def add_border(grid: Grid, colour: int = 1, width: int = 1) -> Grid:
    """Surround the grid with a *colour* border of *width* cells."""
    h, w = grid.shape
    result = np.full((h + 2 * width, w + 2 * width), colour, dtype=grid.dtype)
    result[width:width + h, width:width + w] = grid
    return result


def crop_to_colour(grid: Grid, colour: int) -> Grid:
    """Crop to the bounding box of all cells with *colour*."""
    mask = grid == colour
    if not mask.any():
        return grid.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return grid[np.ix_(rows, cols)].copy()


def normalise_to_square(grid: Grid, background: int = 0) -> Grid:
    """Pad the grid to make it square, centring the content."""
    h, w = grid.shape
    size = max(h, w)
    result = np.full((size, size), background, dtype=grid.dtype)
    r_off = (size - h) // 2
    c_off = (size - w) // 2
    result[r_off:r_off + h, c_off:c_off + w] = grid
    return result


def center_object(grid: Grid, background: int = 0) -> Grid:
    """Centre the non-background content within the original grid dimensions."""
    cropped = crop_to_object(grid, background)
    result = np.full_like(grid, background)
    h, w = grid.shape
    ch, cw = cropped.shape
    r_off = max(0, min((h - ch) // 2, h - ch))
    c_off = max(0, min((w - cw) // 2, w - cw))
    result[r_off:r_off + ch, c_off:c_off + cw] = cropped
    return result


def draw_line_horizontal(grid: Grid, row: int, colour: int) -> Grid:
    """Draw a full horizontal line of *colour* at *row*."""
    result = grid.copy()
    if 0 <= row < grid.shape[0]:
        result[row, :] = colour
    return result


def draw_line_vertical(grid: Grid, col: int, colour: int) -> Grid:
    """Draw a full vertical line of *colour* at *col*."""
    result = grid.copy()
    if 0 <= col < grid.shape[1]:
        result[:, col] = colour
    return result


def draw_rectangle(
    grid: Grid,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    colour: int,
    filled: bool = False,
) -> Grid:
    """Draw a rectangle (hollow or filled) bounded by (r1,c1)–(r2,c2)."""
    result = grid.copy()
    r1, r2 = max(0, min(r1, r2)), min(grid.shape[0] - 1, max(r1, r2))
    c1, c2 = max(0, min(c1, c2)), min(grid.shape[1] - 1, max(c1, c2))
    if filled:
        result[r1:r2 + 1, c1:c2 + 1] = colour
    else:
        result[r1, c1:c2 + 1] = colour
        result[r2, c1:c2 + 1] = colour
        result[r1:r2 + 1, c1] = colour
        result[r1:r2 + 1, c2] = colour
    return result


def connect_objects_horizontal(grid: Grid, colour: int = -1, background: int = 0) -> Grid:
    """Draw horizontal connectors between objects with the same row midpoint.

    If *colour* is -1, uses the objects' own colour.
    """
    from isaac.arc.grid_ops import extract_objects
    result = grid.copy()
    objects = extract_objects(grid, background)
    colour_groups: dict[int, list] = {}
    for obj in objects:
        colour_groups.setdefault(obj.colour, []).append(obj)
    for clr, objs in colour_groups.items():
        if len(objs) < 2:
            continue
        sorted_objs = sorted(objs, key=lambda o: (o.bbox[1] + o.bbox[3]) / 2)
        for i in range(len(sorted_objs) - 1):
            o1, o2 = sorted_objs[i], sorted_objs[i + 1]
            r_mid = (o1.bbox[0] + o1.bbox[2]) // 2
            c_start = o1.bbox[3] + 1
            c_end = o2.bbox[1]
            fill = clr if colour == -1 else colour
            for c in range(c_start, c_end):
                if 0 <= r_mid < grid.shape[0] and 0 <= c < grid.shape[1]:
                    result[r_mid, c] = fill
    return result


def connect_objects_vertical(grid: Grid, colour: int = -1, background: int = 0) -> Grid:
    """Draw vertical connectors between objects aligned in the same column."""
    from isaac.arc.grid_ops import extract_objects
    result = grid.copy()
    objects = extract_objects(grid, background)
    colour_groups: dict[int, list] = {}
    for obj in objects:
        colour_groups.setdefault(obj.colour, []).append(obj)
    for clr, objs in colour_groups.items():
        if len(objs) < 2:
            continue
        sorted_objs = sorted(objs, key=lambda o: (o.bbox[0] + o.bbox[2]) / 2)
        for i in range(len(sorted_objs) - 1):
            o1, o2 = sorted_objs[i], sorted_objs[i + 1]
            c_mid = (o1.bbox[1] + o1.bbox[3]) // 2
            r_start = o1.bbox[2] + 1
            r_end = o2.bbox[0]
            fill = clr if colour == -1 else colour
            for r in range(r_start, r_end):
                if 0 <= r < grid.shape[0] and 0 <= c_mid < grid.shape[1]:
                    result[r, c_mid] = fill
    return result


def object_to_border(grid: Grid, background: int = 0) -> Grid:
    """Move each object to the nearest border edge."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    result = np.full_like(grid, background)
    h, w = grid.shape
    for obj in objects:
        r1, c1, r2, c2 = obj.bbox
        cr = (r1 + r2) / 2
        cc = (c1 + c2) / 2
        dists = [cr, h - 1 - cr, cc, w - 1 - cc]
        nearest = int(np.argmin(dists))
        if nearest == 0:
            dr, dc = int(-r1), 0
        elif nearest == 1:
            dr, dc = int(h - 1 - r2), 0
        elif nearest == 2:
            dr, dc = 0, int(-c1)
        else:
            dr, dc = 0, int(w - 1 - c2)
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr, nc] = grid[r, c]
    return result


def grid_and(grid_a: Grid, grid_b: Grid, background: int = 0) -> Grid:
    """Boolean AND: keep cells non-background only where both grids are non-background."""
    if grid_a.shape != grid_b.shape:
        return grid_a.copy()
    result = np.full_like(grid_a, background)
    mask = (grid_a != background) & (grid_b != background)
    result[mask] = grid_a[mask]
    return result


def grid_or(grid_a: Grid, grid_b: Grid, background: int = 0) -> Grid:
    """Boolean OR: keep cells non-background from either grid."""
    if grid_a.shape != grid_b.shape:
        return grid_a.copy()
    result = grid_a.copy()
    mask = (grid_a == background) & (grid_b != background)
    result[mask] = grid_b[mask]
    return result


def grid_xor(grid_a: Grid, grid_b: Grid, background: int = 0) -> Grid:
    """Boolean XOR: non-background in exactly one of the two grids."""
    if grid_a.shape != grid_b.shape:
        return grid_a.copy()
    result = np.full_like(grid_a, background)
    mask_a = grid_a != background
    mask_b = grid_b != background
    xor_mask = mask_a ^ mask_b
    result[xor_mask & mask_a] = grid_a[xor_mask & mask_a]
    result[xor_mask & mask_b] = grid_b[xor_mask & mask_b]
    return result


def sort_objects_by_size(grid: Grid, ascending: bool = True, background: int = 0) -> Grid:
    """Re-layout objects in sorted size order while preserving positional slots."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    if not objects:
        return grid.copy()
    sorted_objs = sorted(objects, key=lambda o: o.size, reverse=not ascending)
    positions = sorted([o.bbox[:2] for o in objects])
    result = np.full_like(grid, background)
    for obj, pos in zip(sorted_objs, positions):
        r_offset = pos[0] - obj.bbox[0]
        c_offset = pos[1] - obj.bbox[1]
        for r, c in obj.cells:
            nr, nc = r + r_offset, c + c_offset
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                result[nr, nc] = grid[r, c]
    return result


def split_grid_horizontal(grid: Grid) -> Grid:
    """Return the top half of the grid."""
    return grid[:grid.shape[0] // 2, :].copy()


def split_grid_vertical(grid: Grid) -> Grid:
    """Return the left half of the grid."""
    return grid[:, :grid.shape[1] // 2].copy()


def count_to_cells(grid: Grid, background: int = 0) -> Grid:
    """Replace each object with a horizontal bar whose length = object.size."""
    from isaac.arc.grid_ops import extract_objects
    objects = extract_objects(grid, background)
    result = np.full_like(grid, background)
    h, w = grid.shape
    for i, obj in enumerate(objects):
        r = i % h
        count = min(obj.size, w)
        result[r, :count] = obj.colour
    return result


def mask_objects(grid: Grid, mask_colour: int, fill_colour: int, background: int = 0) -> Grid:
    """Fill all objects of *mask_colour* with *fill_colour*."""
    from isaac.arc.grid_ops import extract_objects
    result = grid.copy()
    objects = extract_objects(grid, background)
    for obj in objects:
        if obj.colour == mask_colour:
            for r, c in obj.cells:
                result[r, c] = fill_colour
    return result


def replace_background(grid: Grid, old_background: int = 0, new_background: int = 9) -> Grid:
    """Swap the background colour."""
    result = grid.copy()
    result[grid == old_background] = new_background
    return result


def apply_colour_map(grid: Grid, colour_map: dict) -> Grid:
    """Apply a dictionary colour mapping {old_colour: new_colour}."""
    result = grid.copy()
    for old_c, new_c in colour_map.items():
        result[grid == old_c] = new_c
    return result


def upscale_to_size(grid: Grid, target_h: int = 10, target_w: int = 10) -> Grid:
    """Scale grid up to approximately (target_h, target_w) using integer repeat."""
    h, w = grid.shape
    r_factor = max(1, target_h // h)
    c_factor = max(1, target_w // w)
    return np.repeat(np.repeat(grid, r_factor, axis=0), c_factor, axis=1).copy()


def reflect_about_main_diagonal(grid: Grid) -> Grid:
    """Alias for transpose — reflect about the main diagonal."""
    return grid.T.copy()


def colour_if(
    grid: Grid, condition_colour: int, true_colour: int, false_colour: int = 0,
) -> Grid:
    """Binary colour mask: cells matching *condition_colour* → *true_colour*, else *false_colour*."""
    result = np.full_like(grid, false_colour)
    result[grid == condition_colour] = true_colour
    return result


def expand_objects(grid: Grid, n: int = 1, background: int = 0) -> Grid:
    """Expand each object outward by *n* cells (morphological dilation)."""
    result = grid.copy()
    h, w = grid.shape
    for _ in range(n):
        new_result = result.copy()
        for r in range(h):
            for c in range(w):
                if result[r, c] != background:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and new_result[nr, nc] == background:
                            new_result[nr, nc] = result[r, c]
        result = new_result
    return result


def erode_objects(grid: Grid, n: int = 1, background: int = 0) -> Grid:
    """Shrink each object inward by *n* cells (morphological erosion)."""
    result = grid.copy()
    h, w = grid.shape
    for _ in range(n):
        new_result = result.copy()
        for r in range(h):
            for c in range(w):
                if result[r, c] != background:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == background:
                            new_result[r, c] = background
                            break
        result = new_result
    return result


# ---------------------------------------------------------------------------
# Registry and composition
# ---------------------------------------------------------------------------


PRIMITIVES: dict[str, Transform] = {
    # Rotations & reflections
    "identity": identity,
    "rotate_90": rotate_90,
    "rotate_180": rotate_180,
    "rotate_270": rotate_270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "transpose": transpose,
    "diagonal_flip": diagonal_flip,
    "reflect_about_main_diagonal": reflect_about_main_diagonal,
    # Shifting
    "shift_right": shift_right,
    "shift_left": shift_left,
    "shift_down": shift_down,
    "shift_up": shift_up,
    # Colour operations
    "fill_colour": fill_colour,
    "invert_colours": invert_colours,
    "keep_colour": keep_colour,
    "remove_colour": remove_colour,
    "replace_background": replace_background,
    "colour_if": colour_if,
    "mask_objects": mask_objects,
    "recolour_by_size": recolour_by_size,
    "recolour_by_position": recolour_by_position,
    # Spatial / resize
    "crop_to_object": crop_to_object,
    "crop_to_colour": crop_to_colour,
    "pad_grid": pad_grid,
    "add_border": add_border,
    "remove_border": remove_border,
    "tile_grid": tile_grid,
    "scale_up": scale_up,
    "normalise_to_square": normalise_to_square,
    "center_object": center_object,
    "upscale_to_size": upscale_to_size,
    "extend_pattern_right": extend_pattern_right,
    "extend_pattern_down": extend_pattern_down,
    "split_grid_horizontal": split_grid_horizontal,
    "split_grid_vertical": split_grid_vertical,
    # Physics / gravity
    "gravity_down": gravity_down,
    "gravity_up": gravity_up,
    "gravity_left": gravity_left,
    "gravity_right": gravity_right,
    # Object selection & sorting
    "select_largest_object": select_largest_object,
    "select_smallest_object": select_smallest_object,
    "sort_objects_by_size": sort_objects_by_size,
    "object_to_border": object_to_border,
    # Topology
    "hollow_rectangle": hollow_rectangle,
    "fill_enclosed_regions": fill_enclosed_regions,
    "fill_enclosed_auto": fill_enclosed_auto,
    "outline_objects": outline_objects,
    "expand_objects": expand_objects,
    "erode_objects": erode_objects,
    # Symmetry completion
    "complete_symmetry_horizontal": complete_symmetry_horizontal,
    "complete_symmetry_vertical": complete_symmetry_vertical,
    "mirror_objects_to_fill_symmetry": mirror_objects_to_fill_symmetry,
    # Drawing
    "draw_line_horizontal": draw_line_horizontal,
    "draw_line_vertical": draw_line_vertical,
    "draw_rectangle": draw_rectangle,
    "connect_objects_horizontal": connect_objects_horizontal,
    "connect_objects_vertical": connect_objects_vertical,
    # Counting / pattern
    "count_to_cells": count_to_cells,
    # Flood fill
    "flood_fill_from": flood_fill_from,
    # Boolean grid ops
    "grid_and": grid_and,
    "grid_or": grid_or,
    "grid_xor": grid_xor,
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

    Each step: ``{"op": "rotate_90"}`` or
    ``{"op": "fill_colour", "args": {"from_colour": 1, "to_colour": 2}}``.
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
            try:
                result = fn(result)
            except Exception as exc:
                logger.debug("DSL op %s failed: %s", op_name, exc)
    return result

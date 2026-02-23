"""Grid perception primitives for ARC-AGI tasks.

Provides functions to analyse 2D integer grids (numpy arrays) and extract
structural features: objects, symmetries, patterns, colours, and spatial
relationships.  These primitives feed the neuro-symbolic pipeline so the
LLM operates on structured observations rather than raw pixel grids.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

Grid = np.ndarray  # 2D array of ints (0 = background by convention)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GridObject:
    """A contiguous region of non-background cells."""

    id: int
    colour: int
    cells: list[tuple[int, int]]
    """(row, col) coordinates of each cell."""
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    """Bounding box (min_row, min_col, max_row, max_col)."""

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the bounding box."""
        r1, c1, r2, c2 = self.bbox
        return (r2 - r1 + 1, c2 - c1 + 1)

    def as_subgrid(self, grid: Grid) -> Grid:
        """Extract the bounding-box crop from the full grid."""
        r1, c1, r2, c2 = self.bbox
        return grid[r1: r2 + 1, c1: c2 + 1].copy()


@dataclass
class GridAnalysis:
    """Structured analysis of an ARC grid."""

    height: int
    width: int
    n_colours: int
    colour_counts: dict[int, int]
    background_colour: int
    objects: list[GridObject]
    symmetry: dict[str, bool] = field(default_factory=dict)
    """{'horizontal': bool, 'vertical': bool, 'diagonal': bool}"""
    has_repeating_pattern: bool = False
    grid_hash: str = ""


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------


def extract_colours(grid: Grid) -> dict[int, int]:
    """Return a map of colour → count."""
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def detect_background(grid: Grid) -> int:
    """Detect the background colour (most frequent value)."""
    colours = extract_colours(grid)
    return max(colours, key=colours.get)  # type: ignore[arg-type]


def extract_objects(grid: Grid, background: int = 0) -> list[GridObject]:
    """Flood-fill extraction of contiguous non-background objects.

    Uses 4-connectivity (up/down/left/right). Each object gets a unique ID.
    """
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects: list[GridObject] = []
    obj_id = 0

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == background:
                continue
            # BFS flood fill
            colour = int(grid[r, c])
            cells: list[tuple[int, int]] = []
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if grid[nr, nc] == colour:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            objects.append(GridObject(
                id=obj_id, colour=colour, cells=cells, bbox=bbox,
            ))
            obj_id += 1

    return objects


def detect_symmetry(grid: Grid) -> dict[str, bool]:
    """Check for horizontal, vertical, and diagonal symmetry."""
    h_sym = np.array_equal(grid, grid[::-1, :])
    v_sym = np.array_equal(grid, grid[:, ::-1])
    d_sym = False
    if grid.shape[0] == grid.shape[1]:
        d_sym = np.array_equal(grid, grid.T)
    return {"horizontal": h_sym, "vertical": v_sym, "diagonal": d_sym}


def detect_repeating_pattern(grid: Grid) -> bool:
    """Check if the grid is a tiled repetition of a smaller pattern."""
    h, w = grid.shape
    for ph in range(1, h // 2 + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w // 2 + 1):
            if w % pw != 0:
                continue
            tile = grid[:ph, :pw]
            is_tiled = True
            for r in range(0, h, ph):
                for c in range(0, w, pw):
                    if not np.array_equal(grid[r:r + ph, c:c + pw], tile):
                        is_tiled = False
                        break
                if not is_tiled:
                    break
            if is_tiled and (ph < h or pw < w):
                return True
    return False


def grid_hash(grid: Grid) -> str:
    """Stable hash of a grid for deduplication."""
    return str(hash(grid.tobytes()))


def analyse_grid(grid: Grid) -> GridAnalysis:
    """Full structural analysis of an ARC grid."""
    colours = extract_colours(grid)
    bg = detect_background(grid)
    objects = extract_objects(grid, background=bg)
    symmetry = detect_symmetry(grid)
    repeating = detect_repeating_pattern(grid)

    return GridAnalysis(
        height=grid.shape[0],
        width=grid.shape[1],
        n_colours=len(colours),
        colour_counts=colours,
        background_colour=bg,
        objects=objects,
        symmetry=symmetry,
        has_repeating_pattern=repeating,
        grid_hash=grid_hash(grid),
    )


# ---------------------------------------------------------------------------
# Comparison / diff utilities
# ---------------------------------------------------------------------------


def grid_diff(input_grid: Grid, output_grid: Grid) -> dict[str, Any]:
    """Compare input and output grids, returning structured observations.

    Used to help the LLM understand what transformation was applied.
    """
    input_analysis = analyse_grid(input_grid)
    output_analysis = analyse_grid(output_grid)

    changed_cells: list[dict[str, Any]] = []
    if input_grid.shape == output_grid.shape:
        diff_mask = input_grid != output_grid
        for r, c in zip(*np.where(diff_mask)):
            changed_cells.append({
                "row": int(r),
                "col": int(c),
                "from": int(input_grid[r, c]),
                "to": int(output_grid[r, c]),
            })

    return {
        "input": {
            "shape": list(input_grid.shape),
            "n_colours": input_analysis.n_colours,
            "n_objects": len(input_analysis.objects),
            "symmetry": input_analysis.symmetry,
        },
        "output": {
            "shape": list(output_grid.shape),
            "n_colours": output_analysis.n_colours,
            "n_objects": len(output_analysis.objects),
            "symmetry": output_analysis.symmetry,
        },
        "shape_changed": list(input_grid.shape) != list(output_grid.shape),
        "n_changed_cells": len(changed_cells),
        "changed_cells": changed_cells[:50],  # cap for prompt injection
        "colour_changes": {
            "added": sorted(set(output_analysis.colour_counts) - set(input_analysis.colour_counts)),
            "removed": sorted(set(input_analysis.colour_counts) - set(output_analysis.colour_counts)),
        },
    }


def format_grid_for_prompt(grid: Grid) -> str:
    """Render a grid as a compact string for LLM prompt injection."""
    return "\n".join(
        " ".join(str(int(cell)) for cell in row) for row in grid
    )

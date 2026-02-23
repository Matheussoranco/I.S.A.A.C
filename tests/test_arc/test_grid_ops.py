"""Tests for the ARC grid perception primitives."""

from __future__ import annotations

import numpy as np

from isaac.arc.grid_ops import (
    GridObject,
    analyse_grid,
    detect_background,
    detect_repeating_pattern,
    detect_symmetry,
    extract_colours,
    extract_objects,
    format_grid_for_prompt,
    grid_diff,
    grid_hash,
)


class TestExtractColours:
    def test_single_colour(self) -> None:
        grid = np.zeros((3, 3), dtype=int)
        assert extract_colours(grid) == {0: 9}

    def test_multiple_colours(self) -> None:
        grid = np.array([[0, 1], [2, 0]])
        colours = extract_colours(grid)
        assert colours[0] == 2
        assert colours[1] == 1
        assert colours[2] == 1


class TestDetectBackground:
    def test_majority_is_background(self) -> None:
        grid = np.array([[0, 0, 1], [0, 0, 0], [0, 2, 0]])
        assert detect_background(grid) == 0

    def test_non_zero_background(self) -> None:
        grid = np.array([[5, 5, 5], [5, 1, 5], [5, 5, 5]])
        assert detect_background(grid) == 5


class TestExtractObjects:
    def test_single_object(self) -> None:
        grid = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        objects = extract_objects(grid, background=0)
        assert len(objects) == 1
        assert objects[0].colour == 1
        assert objects[0].size == 2

    def test_two_separate_objects(self) -> None:
        grid = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
        objects = extract_objects(grid, background=0)
        assert len(objects) == 2

    def test_no_objects(self) -> None:
        grid = np.zeros((3, 3), dtype=int)
        objects = extract_objects(grid, background=0)
        assert len(objects) == 0

    def test_object_bbox_and_shape(self) -> None:
        grid = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
        objs = extract_objects(grid)
        assert objs[0].bbox == (1, 1, 2, 1)
        assert objs[0].shape == (2, 1)

    def test_l_shaped_object_stays_connected(self) -> None:
        grid = np.array([
            [1, 0],
            [1, 0],
            [1, 1],
        ])
        objs = extract_objects(grid, background=0)
        assert len(objs) == 1
        assert objs[0].size == 4

    def test_as_subgrid(self) -> None:
        grid = np.array([[0, 0, 0], [0, 3, 3], [0, 3, 0]])
        objs = extract_objects(grid, background=0)
        sub = objs[0].as_subgrid(grid)
        assert sub.shape == (2, 2)


class TestDetectSymmetry:
    def test_horizontal_symmetry(self) -> None:
        grid = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
        symmetry = detect_symmetry(grid)
        assert symmetry["horizontal"] is True

    def test_vertical_symmetry(self) -> None:
        grid = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        symmetry = detect_symmetry(grid)
        assert symmetry["vertical"] is True

    def test_no_symmetry(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        symmetry = detect_symmetry(grid)
        assert symmetry["horizontal"] is False
        assert symmetry["vertical"] is False

    def test_diagonal_symmetry(self) -> None:
        grid = np.array([[1, 2], [2, 1]])
        symmetry = detect_symmetry(grid)
        assert symmetry["diagonal"] is True

    def test_non_square_has_no_diagonal(self) -> None:
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        symmetry = detect_symmetry(grid)
        assert symmetry["diagonal"] is False


class TestDetectRepeatingPattern:
    def test_tiled_pattern(self) -> None:
        tile = np.array([[1, 2], [3, 0]])
        grid = np.tile(tile, (2, 2))
        assert detect_repeating_pattern(grid) is True

    def test_no_pattern(self) -> None:
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        assert detect_repeating_pattern(grid) is False

    def test_single_cell_tile(self) -> None:
        grid = np.ones((4, 4), dtype=int)
        assert detect_repeating_pattern(grid) is True


class TestGridHash:
    def test_same_grid_same_hash(self) -> None:
        g1 = np.array([[1, 2], [3, 4]])
        g2 = np.array([[1, 2], [3, 4]])
        assert grid_hash(g1) == grid_hash(g2)

    def test_different_grid_different_hash(self) -> None:
        g1 = np.array([[1, 2], [3, 4]])
        g2 = np.array([[4, 3], [2, 1]])
        assert grid_hash(g1) != grid_hash(g2)


class TestAnalyseGrid:
    def test_full_analysis(self) -> None:
        grid = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        analysis = analyse_grid(grid)
        assert analysis.height == 3
        assert analysis.width == 3
        assert analysis.n_colours == 2
        assert analysis.background_colour == 0
        assert len(analysis.objects) == 1
        assert analysis.objects[0].colour == 1
        assert isinstance(analysis.grid_hash, str)


class TestGridDiff:
    def test_diff_detects_changes(self) -> None:
        inp = np.array([[0, 1, 0], [0, 0, 0]])
        out = np.array([[0, 0, 1], [0, 0, 0]])
        diff = grid_diff(inp, out)
        assert diff["n_changed_cells"] == 2
        assert diff["shape_changed"] is False

    def test_diff_shape_change(self) -> None:
        inp = np.array([[1, 2], [3, 4]])
        out = np.array([[1, 2, 3], [4, 5, 6]])
        diff = grid_diff(inp, out)
        assert diff["shape_changed"] is True

    def test_diff_colour_tracking(self) -> None:
        inp = np.array([[0, 1]])
        out = np.array([[0, 2]])
        diff = grid_diff(inp, out)
        assert 2 in diff["colour_changes"]["added"]
        assert 1 in diff["colour_changes"]["removed"]


class TestFormatGrid:
    def test_format_output(self) -> None:
        grid = np.array([[0, 1], [2, 3]])
        text = format_grid_for_prompt(grid)
        assert text == "0 1\n2 3"

    def test_single_cell(self) -> None:
        grid = np.array([[7]])
        assert format_grid_for_prompt(grid) == "7"

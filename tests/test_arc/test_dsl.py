"""Tests for the ARC DSL grid transformation primitives."""

from __future__ import annotations

import numpy as np

from isaac.arc.dsl import (
    PRIMITIVES,
    apply_program,
    compose,
    crop_to_object,
    fill_colour,
    flip_horizontal,
    flip_vertical,
    flood_fill_from,
    gravity_down,
    gravity_left,
    hollow_rectangle,
    identity,
    invert_colours,
    pad_grid,
    rotate_90,
    rotate_180,
    rotate_270,
    scale_up,
    shift_down,
    shift_left,
    shift_right,
    shift_up,
    tile_grid,
    transpose,
)


class TestIdentity:
    def test_returns_copy(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        result = identity(grid)
        assert np.array_equal(result, grid)
        # Must be a copy, not the same object
        result[0, 0] = 99
        assert grid[0, 0] == 1


class TestRotations:
    def test_rotate_90(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[3, 1], [4, 2]])
        assert np.array_equal(rotate_90(grid), expected)

    def test_rotate_180(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[4, 3], [2, 1]])
        assert np.array_equal(rotate_180(grid), expected)

    def test_rotate_270(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[2, 4], [1, 3]])
        assert np.array_equal(rotate_270(grid), expected)

    def test_four_rotations_is_identity(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        result = compose(rotate_90, rotate_90, rotate_90, rotate_90)(grid)
        assert np.array_equal(result, grid)


class TestFlips:
    def test_flip_horizontal(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[2, 1], [4, 3]])
        assert np.array_equal(flip_horizontal(grid), expected)

    def test_flip_vertical(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[3, 4], [1, 2]])
        assert np.array_equal(flip_vertical(grid), expected)

    def test_double_flip_is_identity(self) -> None:
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        assert np.array_equal(flip_horizontal(flip_horizontal(grid)), grid)
        assert np.array_equal(flip_vertical(flip_vertical(grid)), grid)


class TestTranspose:
    def test_transpose_square(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        expected = np.array([[1, 3], [2, 4]])
        assert np.array_equal(transpose(grid), expected)

    def test_transpose_rectangular(self) -> None:
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        result = transpose(grid)
        assert result.shape == (3, 2)


class TestShifts:
    def test_shift_right(self) -> None:
        grid = np.array([[1, 2, 3]])
        expected = np.array([[3, 1, 2]])
        assert np.array_equal(shift_right(grid), expected)

    def test_shift_left(self) -> None:
        grid = np.array([[1, 2, 3]])
        expected = np.array([[2, 3, 1]])
        assert np.array_equal(shift_left(grid), expected)

    def test_shift_down(self) -> None:
        grid = np.array([[1], [2], [3]])
        expected = np.array([[3], [1], [2]])
        assert np.array_equal(shift_down(grid), expected)

    def test_shift_up(self) -> None:
        grid = np.array([[1], [2], [3]])
        expected = np.array([[2], [3], [1]])
        assert np.array_equal(shift_up(grid), expected)


class TestColourOps:
    def test_fill_colour(self) -> None:
        grid = np.array([[1, 0, 1], [0, 2, 0]])
        result = fill_colour(grid, from_colour=1, to_colour=3)
        assert result[0, 0] == 3
        assert result[0, 2] == 3
        assert result[1, 1] == 2  # unchanged

    def test_invert_colours(self) -> None:
        grid = np.array([[0, 9], [5, 3]])
        expected = np.array([[9, 0], [4, 6]])
        assert np.array_equal(invert_colours(grid), expected)


class TestGridManipulation:
    def test_crop_to_object(self) -> None:
        grid = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
        result = crop_to_object(grid)
        assert np.array_equal(result, np.array([[1, 1]]))

    def test_crop_empty_grid(self) -> None:
        grid = np.zeros((3, 3), dtype=int)
        result = crop_to_object(grid)
        assert np.array_equal(result, grid)

    def test_pad_grid(self) -> None:
        grid = np.array([[1]])
        result = pad_grid(grid, pad=1, value=0)
        assert result.shape == (3, 3)
        assert result[1, 1] == 1
        assert result[0, 0] == 0

    def test_tile_grid(self) -> None:
        grid = np.array([[1, 2]])
        result = tile_grid(grid, rows=2, cols=2)
        assert result.shape == (2, 4)
        assert np.array_equal(result[0], [1, 2, 1, 2])

    def test_scale_up(self) -> None:
        grid = np.array([[1, 2]])
        result = scale_up(grid, factor=2)
        assert result.shape == (2, 4)
        assert np.array_equal(result[0], [1, 1, 2, 2])


class TestGravity:
    def test_gravity_down(self) -> None:
        grid = np.array([[1, 0], [0, 0], [0, 2]])
        result = gravity_down(grid)
        assert result[2, 0] == 1  # fell to bottom
        assert result[2, 1] == 2  # already at bottom

    def test_gravity_left(self) -> None:
        grid = np.array([[0, 0, 1], [0, 2, 0]])
        result = gravity_left(grid)
        assert result[0, 0] == 1
        assert result[1, 0] == 2


class TestHollowRectangle:
    def test_hollow(self) -> None:
        grid = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        result = hollow_rectangle(grid)
        # Centre should be cleared
        assert result[1, 1] == 0
        # Borders should remain
        assert result[0, 0] == 1
        assert result[0, 1] == 1


class TestFloodFill:
    def test_flood_fill(self) -> None:
        grid = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        result = flood_fill_from(grid, row=0, col=0, colour=5)
        assert result[0, 0] == 5
        assert result[1, 1] == 5
        assert result[2, 2] == 1  # different region unchanged

    def test_flood_fill_same_colour_noop(self) -> None:
        grid = np.array([[1, 1], [1, 1]])
        result = flood_fill_from(grid, row=0, col=0, colour=1)
        assert np.array_equal(result, grid)


class TestCompose:
    def test_compose_two(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        fn = compose(rotate_90, flip_horizontal)
        result = fn(grid)
        expected = flip_horizontal(rotate_90(grid))
        assert np.array_equal(result, expected)

    def test_compose_empty_is_identity(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        fn = compose()
        assert np.array_equal(fn(grid), grid)


class TestApplyProgram:
    def test_single_step(self) -> None:
        grid = np.array([[1, 2], [3, 4]])
        program = [{"op": "rotate_90"}]
        result = apply_program(program, grid)
        assert np.array_equal(result, rotate_90(grid))

    def test_step_with_args(self) -> None:
        grid = np.array([[1, 0], [0, 1]])
        program = [{"op": "fill_colour", "args": {"from_colour": 1, "to_colour": 5}}]
        result = apply_program(program, grid)
        assert result[0, 0] == 5
        assert result[0, 1] == 0

    def test_unknown_op_skipped(self) -> None:
        grid = np.array([[1, 2]])
        program = [{"op": "nonexistent_op"}, {"op": "rotate_180"}]
        result = apply_program(program, grid)
        assert np.array_equal(result, rotate_180(grid))


class TestPrimitivesRegistry:
    def test_all_registered(self) -> None:
        expected = {
            "identity", "rotate_90", "rotate_180", "rotate_270",
            "flip_horizontal", "flip_vertical", "transpose",
            "shift_right", "shift_left", "shift_down", "shift_up",
            "fill_colour", "invert_colours", "crop_to_object",
            "pad_grid", "tile_grid", "scale_up",
            "gravity_down", "gravity_left", "hollow_rectangle",
        }
        assert expected == set(PRIMITIVES.keys())

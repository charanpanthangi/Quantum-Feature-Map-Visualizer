"""Tests for dataset generation helpers."""

import numpy as np

from app.dataset import make_circles_or_moons, make_grid_data


def test_make_grid_data_shape():
    grid = make_grid_data(5)
    assert grid.shape == (25, 2)
    # Ensure values fall within expected range
    assert np.all(grid >= 0) and np.all(grid <= 1)


def test_make_circles_or_moons_shapes():
    circles = make_circles_or_moons(30, kind="circles")
    moons = make_circles_or_moons(30, kind="moons")
    assert circles.shape == (30, 2)
    assert moons.shape == (30, 2)

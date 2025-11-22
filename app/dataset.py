"""Utility functions to generate simple 2D datasets for visualization.

The goal of this module is to keep the classical data easy to reason about.
Two-dimensional points can be plotted directly and make it straightforward
to compare the original geometry with the quantum embedding produced by a
feature map.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons


def make_grid_data(n_per_axis: int = 10) -> np.ndarray:
    """Create a grid of points in the unit square.

    The grid uses ``n_per_axis`` points along each axis so the total number
    of samples is ``n_per_axis ** 2``. Grids are visually intuitive and let us
    see how a feature map warps evenly spaced points once they are embedded
    into quantum states.

    Args:
        n_per_axis: Number of points to place along the x and y axes.

    Returns:
        A NumPy array with shape ``(n_per_axis ** 2, 2)`` representing points
        in the range [0, 1] x [0, 1].
    """

    # Create evenly spaced values on each axis.
    axis = np.linspace(0, 1, n_per_axis)
    # Meshgrid produces matrices for x and y coordinates that we flatten.
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return grid


def make_circles_or_moons(n_samples: int = 100, kind: str = "circles", noise: float | None = 0.05) -> np.ndarray:
    """Generate a non-linear 2D dataset using scikit-learn helpers.

    Circular or moon-shaped datasets are classic examples where linear
    separability fails. They provide a nice test bed to see whether quantum
    feature maps make certain clusters more or less similar after embedding.

    Args:
        n_samples: Total number of points to sample.
        kind: Either ``"circles"`` or ``"moons"`` to pick the generator.
        noise: Optional noise parameter passed to scikit-learn for slight
            variations that make the plot more realistic.

    Returns:
        Array of shape ``(n_samples, 2)`` containing the generated points.
    """

    if kind == "circles":
        data, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif kind == "moons":
        data, _ = make_moons(n_samples=n_samples, noise=noise)
    else:
        raise ValueError("kind must be 'circles' or 'moons'")

    return data.astype(float)


__all__ = ["make_grid_data", "make_circles_or_moons"]

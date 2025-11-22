"""Plotting helpers for quantum feature map experiments.

All plots are saved as SVG files to keep them text-based and friendly for
version control. Visualizing the kernel matrix helps explain how a feature map
changes similarity between data points when moving from classical space into
quantum space.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .embedding import compute_kernel_matrix, compute_statevectors

# Ensure SVG output globally for this module
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["savefig.format"] = "svg"


def _prepare_output_path(output_path: str) -> str:
    """Create parent directories if needed and return the path back."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def plot_embedding_kernel(K: np.ndarray, output_path: str) -> None:
    """Save a heatmap of a kernel matrix as an SVG.

    Bright diagonal entries (value near one) indicate that each point is most
    similar to itself. Off-diagonal structure highlights which points became
    closer or farther apart after the quantum embedding.
    """

    _prepare_output_path(output_path)
    plt.close("all")
    fig, ax = plt.subplots()
    sns.heatmap(K, cmap="viridis", square=True, cbar_kws={"label": "Similarity"}, ax=ax)
    ax.set_title("Quantum feature map kernel")
    ax.set_xlabel("Point index")
    ax.set_ylabel("Point index")
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_2d_points(X: np.ndarray, output_path: str, title: str = "Classical data") -> None:
    """Scatter plot of 2D points, saved as SVG.

    Seeing the original layout of the data makes it easier to interpret how
    the kernel heatmap changes relationships between points.
    """

    _prepare_output_path(output_path)
    plt.close("all")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def plot_feature_map_comparison(
    X: np.ndarray, feature_maps: Dict[str, object], output_path: str
) -> None:
    """Compare kernel heatmaps from multiple feature maps.

    Args:
        X: Array of 2D data points.
        feature_maps: Mapping from a human-friendly label to a Qiskit circuit.
        output_path: Where to save the combined SVG visualization.
    """

    _prepare_output_path(output_path)
    num_maps = len(feature_maps)
    plt.close("all")
    fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 4))

    if num_maps == 1:
        axes = [axes]

    for ax, (label, circuit) in zip(axes, feature_maps.items()):
        # Compute kernel for each feature map
        statevectors = compute_statevectors(circuit, X)
        kernel = compute_kernel_matrix(statevectors)
        sns.heatmap(
            kernel,
            cmap="viridis",
            square=True,
            cbar=False,
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel("Point index")
        ax.set_ylabel("Point index")

    fig.suptitle("Quantum feature map comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)


__all__ = [
    "plot_embedding_kernel",
    "plot_2d_points",
    "plot_feature_map_comparison",
]

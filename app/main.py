"""Command line interface for the Quantum Feature Map Visualizer.

This script ties together dataset generation, feature map selection, quantum
state simulation, and SVG plotting so users can explore how different feature
maps reshape classical data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import make_circles_or_moons, make_grid_data
from .embedding import compute_kernel_matrix, compute_statevectors
from .feature_maps import get_feature_map
from .visualizer import plot_embedding_kernel, plot_feature_map_comparison, plot_2d_points


OUTPUT_DIR = Path("examples")


def parse_args() -> argparse.Namespace:
    """Parse command line options for choosing feature maps and dataset size."""

    parser = argparse.ArgumentParser(description="Visualize quantum feature maps with SVG plots.")
    parser.add_argument(
        "--feature-map",
        default="angle_encoding",
        help="Which feature map to use: angle_encoding, zz_feature_map, pauli_feature_map, zz",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=25,
        help="Approximate number of points to sample. Grid datasets produce a square count.",
    )
    parser.add_argument(
        "--dataset",
        choices=["grid", "circles", "moons"],
        default="grid",
        help="Choose the classical dataset shape to encode.",
    )
    return parser.parse_args()


def _build_dataset(name: str, n_points: int) -> np.ndarray:
    """Create a dataset based on user preference."""

    if name == "grid":
        # Round to a square grid size that stays close to the requested count.
        side = max(2, int(np.sqrt(n_points)))
        return make_grid_data(side)
    if name == "circles":
        return make_circles_or_moons(n_samples=n_points, kind="circles")
    if name == "moons":
        return make_circles_or_moons(n_samples=n_points, kind="moons")
    raise ValueError("Unknown dataset type")


def main() -> None:
    args = parse_args()
    X = _build_dataset(args.dataset, args.n_points)

    feature_map = get_feature_map(args.feature_map, num_qubits=X.shape[1])
    print(f"Using feature map: {args.feature_map} on dataset with {len(X)} points.")

    # Compute embeddings and kernel
    statevectors = compute_statevectors(feature_map, X)
    kernel = compute_kernel_matrix(statevectors)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Plot original data and kernel heatmap
    points_path = OUTPUT_DIR / f"{args.feature_map}_{args.dataset}_points.svg"
    kernel_path = OUTPUT_DIR / f"{args.feature_map}_{args.dataset}_kernel.svg"
    plot_2d_points(X, str(points_path), title=f"{args.dataset.capitalize()} dataset")
    plot_embedding_kernel(kernel, str(kernel_path))

    # Comparison plot for default maps when user requests
    if args.feature_map in {"compare", "all"}:
        feature_maps = {
            "Angle encoding": get_feature_map("angle_encoding", num_qubits=X.shape[1]),
            "ZZFeatureMap": get_feature_map("zz_feature_map", num_qubits=X.shape[1]),
        }
        comparison_path = OUTPUT_DIR / "feature_map_comparison.svg"
        plot_feature_map_comparison(X, feature_maps, str(comparison_path))
        print(f"Saved comparison plot to {comparison_path}")

    print(f"Saved point plot to {points_path}")
    print(f"Saved kernel heatmap to {kernel_path}")


if __name__ == "__main__":
    main()

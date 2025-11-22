"""Tests for quantum embedding and kernel computation."""

import numpy as np

from app.dataset import make_grid_data
from app.embedding import compute_kernel_matrix, compute_statevectors
from app.feature_maps import build_angle_encoding_feature_map


def test_kernel_matrix_properties():
    X = make_grid_data(2)  # four simple points
    feature_map = build_angle_encoding_feature_map(num_qubits=2)
    statevectors = compute_statevectors(feature_map, X)
    kernel = compute_kernel_matrix(statevectors)

    # Kernel should be symmetric
    assert np.allclose(kernel, kernel.T)
    # Diagonal entries should be 1 up to small numerical tolerance
    assert np.allclose(np.diag(kernel), np.ones(kernel.shape[0]), atol=1e-6)

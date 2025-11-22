"""Compute quantum embeddings and kernel matrices from feature maps.

This module takes classical data, binds it into parameterized feature map
circuits, and uses the Qiskit statevector simulator to obtain the resulting
quantum states. Pairwise inner products of these states form a kernel matrix
that reflects similarity in the quantum feature space.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit


def _parameter_order(feature_map: QuantumCircuit) -> list:
    """Return a stable ordering of parameters for binding values.

    Qiskit stores parameters as an unordered set. Sorting them by name ensures
    that we always bind values in the same order, which is important when we
    vectorize over many data points.
    """

    return sorted(feature_map.parameters, key=lambda param: param.name)


def compute_statevectors(feature_map: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """Simulate the statevector for each row in the dataset ``X``.

    Args:
        feature_map: Parameterized circuit that encodes features into qubits.
        X: Array with shape ``(n_samples, n_features)`` containing classical
            data. The number of features should match the feature map dimension.

    Returns:
        A NumPy array of complex amplitudes with shape ``(n_samples, 2**n_qubits)``.
    """

    ordered_parameters = _parameter_order(feature_map)
    statevectors: list[np.ndarray] = []

    for x in X:
        # Create mapping from circuit parameters to the values for this sample.
        binding = {param: float(value) for param, value in zip(ordered_parameters, x)}
        # Bind parameters and evolve the state starting from |0...0>.
        bound_circuit = feature_map.bind_parameters(binding)
        statevector = Statevector.from_instruction(bound_circuit)
        statevectors.append(statevector.data)

    return np.array(statevectors)


def compute_kernel_matrix(statevectors: Iterable[np.ndarray]) -> np.ndarray:
    """Compute a quantum kernel matrix from an iterable of statevectors.

    The kernel entry ``K_ij`` is the squared magnitude of the inner product
    between two statevectors. This mirrors classical kernel methods where a
    similarity score is computed in feature space.

    Args:
        statevectors: Sequence of complex vectors returned by
            :func:`compute_statevectors`.

    Returns:
        A Hermitian, positive semi-definite kernel matrix with values between
        0 and 1 (up to numerical error).
    """

    statevectors = np.asarray(statevectors)
    n_samples = statevectors.shape[0]
    kernel = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Inner product <phi_i | phi_j>
            inner = np.vdot(statevectors[i], statevectors[j])
            similarity = float(np.abs(inner) ** 2)
            kernel[i, j] = similarity
            kernel[j, i] = similarity  # symmetry

    return kernel


__all__ = ["compute_statevectors", "compute_kernel_matrix"]

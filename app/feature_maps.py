"""Definitions for several quantum feature maps using Qiskit.

A feature map is a quantum circuit that turns a classical input vector into
an output quantum state. Different circuits create different geometric
relationships between the encoded points. This module keeps the circuits
simple and documents how each one works in plain language.
"""

from __future__ import annotations

from typing import Dict

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZZFeatureMap


def build_angle_encoding_feature_map(num_qubits: int = 2) -> QuantumCircuit:
    """Create a custom feature map based on single-qubit rotations.

    The idea is straightforward: each feature value rotates a qubit around a
    Bloch sphere axis. When the values change, the quantum state moves to a
    different point on the sphere. This is sometimes called angle encoding
    because angles carry the information.

    Args:
        num_qubits: Number of qubits to rotate. This should match the feature
            dimension for the classical data (two features by default).

    Returns:
        A parameterized :class:`QuantumCircuit` ready to bind data values.
    """

    circuit = QuantumCircuit(num_qubits)
    features = ParameterVector("x", num_qubits)

    for i, parameter in enumerate(features):
        # Apply two rotations so each qubit depends on its matching feature.
        circuit.ry(parameter, i)
        circuit.rz(parameter, i)

    circuit.barrier()
    return circuit


def build_zz_feature_map(num_qubits: int = 2, reps: int = 2) -> QuantumCircuit:
    """Wrap Qiskit's built-in ZZFeatureMap with a friendly helper.

    The ZZFeatureMap entangles qubits using controlled-ZZ interactions. These
    entangling layers make the encoded quantum states sensitive to the joint
    values of the input features rather than treating each feature separately.

    Args:
        num_qubits: Number of qubits to include in the map.
        reps: How many times to repeat the entangling structure.

    Returns:
        The configured :class:`ZZFeatureMap` circuit.
    """

    return ZZFeatureMap(feature_dimension=num_qubits, reps=reps)


def build_pauli_feature_map(num_qubits: int = 2, reps: int = 1, paulis: str | None = None) -> QuantumCircuit:
    """Create a PauliFeatureMap which generalizes the ZZFeatureMap idea.

    The PauliFeatureMap uses sequences of single-qubit rotations and entangling
    operations defined by a list of Pauli strings. By choosing different Pauli
    operators you can design how strongly features interact.

    Args:
        num_qubits: Number of qubits to encode.
        reps: Number of repetitions of the pattern.
        paulis: Optional Pauli strings. If ``None``, a default set like
            ``["X", "Y", "Z", "ZZ"]`` is used.

    Returns:
        The configured :class:`PauliFeatureMap` circuit.
    """

    paulis_list = paulis or ["X", "Y", "Z", "ZZ"]
    return PauliFeatureMap(feature_dimension=num_qubits, reps=reps, paulis=paulis_list)


def get_feature_map(name: str, num_qubits: int = 2) -> QuantumCircuit:
    """Retrieve a feature map circuit by a short name.

    Args:
        name: Identifier such as ``"angle_encoding"`` or ``"zz_feature_map"``.
        num_qubits: Number of qubits / feature dimension to build for.

    Returns:
        The selected feature map circuit.

    Raises:
        ValueError: If the name is unknown.
    """

    normalized = name.lower()
    builders: Dict[str, QuantumCircuit] = {
        "angle_encoding": build_angle_encoding_feature_map(num_qubits=num_qubits),
        "zz_feature_map": build_zz_feature_map(num_qubits=num_qubits),
        "pauli_feature_map": build_pauli_feature_map(num_qubits=num_qubits),
        "zz": build_zz_feature_map(num_qubits=num_qubits),
    }

    if normalized not in builders:
        raise ValueError(
            "Unknown feature map name. Options: angle_encoding, zz_feature_map, pauli_feature_map, zz"
        )

    return builders[normalized]


__all__ = [
    "build_angle_encoding_feature_map",
    "build_pauli_feature_map",
    "build_zz_feature_map",
    "get_feature_map",
]

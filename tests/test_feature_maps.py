"""Tests for feature map construction."""

from qiskit import QuantumCircuit

from app.feature_maps import (
    build_angle_encoding_feature_map,
    build_pauli_feature_map,
    build_zz_feature_map,
    get_feature_map,
)


def test_angle_encoding_builds():
    circuit = build_angle_encoding_feature_map(2)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_zz_feature_map_builds():
    circuit = build_zz_feature_map(3, reps=1)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 3


def test_pauli_feature_map_builds():
    circuit = build_pauli_feature_map(2, reps=1, paulis=["X", "ZZ"])
    assert isinstance(circuit, QuantumCircuit)


def test_get_feature_map_dispatch():
    circuit = get_feature_map("angle_encoding", num_qubits=2)
    assert isinstance(circuit, QuantumCircuit)

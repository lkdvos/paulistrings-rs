"""Smoke-test that the public API surface (ARCHITECTURE.md §Python-Bindings) is wired up.

Deliberately shallow: this module only asserts that the names exist and that the
constructors pick a width. Behavioral coverage lives in the sibling
``test_pauli_sum``, ``test_circuit``, ``test_truncation``, and ``test_numpy``
modules.
"""

import paulistrings
from paulistrings import Circuit, PauliSum, gates, noise, truncation


def test_top_level_names():
    assert hasattr(paulistrings, "PauliSum")
    assert hasattr(paulistrings, "Circuit")
    assert hasattr(paulistrings, "gates")
    assert hasattr(paulistrings, "noise")
    assert hasattr(paulistrings, "truncation")


def test_factory_module_names():
    for name in ("h", "cnot", "rz", "pauli_rotation", "unitary_1q", "unitary_2q"):
        assert hasattr(gates, name)
    for name in (
        "depolarize",
        "dephase",
        "amplitude_damping",
        "pauli_channel",
        "depolarize2",
    ):
        assert hasattr(noise, name)
    for name in ("coeff", "weight", "topn"):
        assert hasattr(truncation, name)


def test_constructors_pick_a_width():
    s = PauliSum(20)
    assert s.num_qubits == 20
    assert len(s) == 0

    c = Circuit(20)
    assert c.num_qubits == 20
    assert len(c) == 0

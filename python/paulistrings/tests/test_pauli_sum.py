"""End-to-end round-trip on the W=1 width path.

``PauliSum.from_strings`` → ``coefficients()`` round-trip plus ``propagate``
through an empty circuit (a no-op).
"""

import copy

import pytest

from paulistrings import Circuit, PauliSum


def test_from_strings_round_trip_w1():
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5}, num_qubits=3)
    assert len(s) == 2
    assert s.num_qubits == 3
    coeffs = s.coefficients()
    # ``ZII`` (x=0, z=1) sorts before ``XII`` (x=1, z=0) under lex on (x, z).
    assert coeffs == [0.5 + 0j, 1.0 + 0j]


def test_from_strings_y_is_hermitian():
    # Coefficients multiply the literal Hermitian Pauli string: "Y" maps to
    # the symplectic key (x=1, z=1) with no phase factor, so a real input
    # coefficient stays real.
    s = PauliSum.from_strings({"Y": 1.0}, num_qubits=1)
    assert s.coefficients() == [1.0 + 0j]


def test_from_strings_dedup_and_cancel():
    s = PauliSum.from_strings({"XI": 1.0}, num_qubits=2)
    # The dict literal already deduplicates by key, so there is nothing to
    # cancel here; coefficient summation and exact-zero dropping are
    # exercised by the Rust unit tests on BuildAccumulator.
    assert len(s) == 1


def test_from_strings_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        PauliSum.from_strings({"XI": 1.0}, num_qubits=3)


def test_from_strings_rejects_invalid_char():
    with pytest.raises(ValueError, match="character"):
        PauliSum.from_strings({"AB": 1.0}, num_qubits=2)


def test_propagate_through_empty_circuit_is_identity():
    s = PauliSum.from_strings({"XII": 1.0, "ZII": 0.5}, num_qubits=3)
    c = Circuit(3)
    out = s.copy()
    out.propagate(c)
    assert out.num_qubits == 3
    assert len(out) == 2
    assert out.coefficients() == s.coefficients()


@pytest.mark.parametrize("num_qubits", [3, 80, 200, 400, 800])
def test_width_dispatch_handles_all_widths(num_qubits):
    # An X on qubit `q` for `q ∈ {0, num_qubits-1}` exercises both endpoints.
    # Each width arm parses, stores, and reads back identically.
    s_str = "I" * (num_qubits - 1) + "X"
    s = PauliSum.from_strings({s_str: 2.0}, num_qubits=num_qubits)
    assert s.num_qubits == num_qubits
    assert s.coefficients() == [2 + 0j]
    out = s.copy()
    out.propagate(Circuit(num_qubits))
    assert out.coefficients() == [2 + 0j]


def test_width_dispatch_rejects_above_1024_qubits():
    with pytest.raises(ValueError, match="1024"):
        PauliSum(1025)


def _h_on_x():
    """``X`` on one qubit and a Hadamard, which maps it to ``Z``."""
    circuit = Circuit(1)
    circuit.h(0)
    return PauliSum.from_strings({"X": 1.0}, num_qubits=1), circuit


def test_propagate_evolves_the_sum_in_place():
    s, circuit = _h_on_x()
    assert s.propagate(circuit, direction="heisenberg") is None
    assert s.coefficients() == [1.0 + 0j]
    assert s.z_array().tolist() == [[1]] and s.x_array().tolist() == [[0]]


@pytest.mark.parametrize("make_copy", [PauliSum.copy, copy.copy, copy.deepcopy])
def test_a_copy_keeps_the_input_through_propagate(make_copy):
    s, circuit = _h_on_x()
    kept = make_copy(s)
    s.propagate(circuit, direction="heisenberg")
    assert kept.x_array().tolist() == [[1]] and kept.z_array().tolist() == [[0]]
    assert kept.dtype == s.dtype


def test_a_rejected_propagate_leaves_the_sum_untouched():
    s, _ = _h_on_x()
    with pytest.raises(ValueError):
        s.propagate(Circuit(2))
    assert s.x_array().tolist() == [[1]] and s.coefficients() == [1.0 + 0j]

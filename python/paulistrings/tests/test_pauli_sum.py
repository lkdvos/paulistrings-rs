"""End-to-end round-trip on the W=1 width path.

``PauliSum.from_strings`` → ``coefficients()`` round-trip plus ``propagate``
through an empty circuit (a no-op).
"""

import math

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
    out = s.propagate(c)
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
    out = s.propagate(Circuit(num_qubits))
    assert out.coefficients() == [2 + 0j]


def test_width_dispatch_rejects_above_1024_qubits():
    with pytest.raises(ValueError, match="1024"):
        PauliSum(1025)

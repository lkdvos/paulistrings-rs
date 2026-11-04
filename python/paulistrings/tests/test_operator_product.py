"""Operator products of two sums against dense matrices: ``product`` / ``@``, ``liouvillian`` and ``anticommutator``."""

import functools
import itertools

import numpy as np
import pytest

from paulistrings import PauliSum

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

NUM_QUBITS = 3
LABELS = ["".join(p) for p in itertools.product("IXYZ", repeat=NUM_QUBITS)]


def _dense(terms):
    """Qubit 0 is the most significant kron factor, matching ``from_strings``."""
    return sum(c * functools.reduce(np.kron, [_PAULI[ch] for ch in label]) for label, c in terms.items())


def _pauli_coefficients(matrix):
    dim = 1 << NUM_QUBITS
    return {label: np.trace(_dense({label: 1.0}) @ matrix) / dim for label in LABELS}


def _assert_matches(got, matrix):
    for label, want in _pauli_coefficients(matrix).items():
        basis = PauliSum.from_strings({label: 1.0}, num_qubits=NUM_QUBITS)
        assert basis.overlap(got) == pytest.approx(want, abs=1e-12), label


def _random_terms(seed, n, complex_coefficients):
    rng = np.random.default_rng(seed)
    labels = rng.choice(LABELS, size=n, replace=False)
    values = rng.normal(size=n) + (1j * rng.normal(size=n) if complex_coefficients else 0.0)
    return dict(zip(labels, values))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_product_matches_the_matrix_product(seed):
    a = _random_terms(seed, 12, complex_coefficients=True)
    b = _random_terms(seed + 100, 9, complex_coefficients=True)
    pa = PauliSum.from_strings(a, num_qubits=NUM_QUBITS)
    pb = PauliSum.from_strings(b, num_qubits=NUM_QUBITS)
    _assert_matches(pa.product(pb), _dense(a) @ _dense(b))
    _assert_matches(pa @ pb, _dense(a) @ _dense(b))


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_hermitian_liouvillian_and_anticommutator_stay_real(seed):
    h = _random_terms(seed, 6, complex_coefficients=False)
    o = _random_terms(seed + 100, 20, complex_coefficients=False)
    ph = PauliSum.from_strings(h, num_qubits=NUM_QUBITS, dtype="float64")
    po = PauliSum.from_strings(o, num_qubits=NUM_QUBITS, dtype="float64")
    dh, do = _dense(h), _dense(o)

    liouvillian = ph.liouvillian(po)
    assert liouvillian.dtype == np.float64
    _assert_matches(liouvillian, 1j * (dh @ do - do @ dh))

    anticommutator = ph.anticommutator(po)
    assert anticommutator.dtype == np.float64
    _assert_matches(anticommutator, dh @ do + do @ dh)


@pytest.mark.parametrize(
    ("left", "right", "want"),
    [
        ("float64", "float64", np.complex128),
        ("float32", "float32", np.complex64),
        ("float32", "complex128", np.complex128),
    ],
)
def test_a_product_is_complex_at_the_promoted_precision(left, right, want):
    a = PauliSum.from_strings({"XZ": 1.0}, num_qubits=2, dtype=left)
    b = PauliSum.from_strings({"ZZ": 1.0}, num_qubits=2, dtype=right)
    assert (a @ b).dtype == want


def test_the_liouvillian_of_z_on_x_is_minus_two_y():
    h = PauliSum.from_strings({"Z": 1.0}, num_qubits=1, dtype="float64")
    o = PauliSum.from_strings({"X": 1.0}, num_qubits=1, dtype="float64")
    got = h.liouvillian(o)
    assert len(got) == 1
    assert PauliSum.from_strings({"Y": 1.0}, num_qubits=1).overlap(got) == -2.0


def test_operands_on_different_qubit_counts_are_a_value_error():
    a = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    b = PauliSum.from_strings({"XX": 1.0}, num_qubits=2)
    for op in (a.product, a.liouvillian, a.anticommutator):
        with pytest.raises(ValueError, match="num_qubits mismatch"):
            op(b)

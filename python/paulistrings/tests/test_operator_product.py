"""Operator products of two sums against dense matrices: ``product`` / ``@``, ``liouvillian`` and ``anticommutator``."""

import functools
import itertools

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

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


def _chain_operands():
    h = {"ZZI": 1.0, "IZZ": 1.0, "XII": 0.3, "IXI": 0.3, "IIX": 0.3}
    o = _random_terms(7, 30, complex_coefficients=False)
    return (
        PauliSum.from_strings(h, num_qubits=NUM_QUBITS, dtype="float64"),
        PauliSum.from_strings(o, num_qubits=NUM_QUBITS, dtype="float64"),
    )


def test_a_coefficient_policy_keeps_the_strings_above_it():
    h, o = _chain_operands()
    full = np.abs(h.liouvillian(o).coefficients_array())
    eps = float(np.median(full))
    kept = h.liouvillian(o, policy=truncation.coeff(eps)).coefficients_array()
    assert len(kept) == np.count_nonzero(full > eps)
    assert np.all(np.abs(kept) > eps)


def test_a_layer_policy_runs_on_the_result():
    h, o = _chain_operands()
    full = np.sort(np.abs(h.anticommutator(o).coefficients_array()))[::-1]
    kept = np.sort(np.abs(h.anticommutator(o, policy=truncation.topn(5)).coefficients_array()))[::-1]
    np.testing.assert_allclose(kept, full[:5])


def test_a_product_takes_a_policy_too():
    h, o = _chain_operands()
    assert len(h.product(o, policy=truncation.weight(1))) <= len(h.product(o))
    identity_only = h.product(o, policy=truncation.weight(0))
    assert len(identity_only) <= 1
    assert identity_only.identity_coefficient() == pytest.approx(h.product(o).identity_coefficient())


def _chain(n, field=0.3):
    terms = {}
    for j in range(n - 1):
        for p in "XYZ":
            terms["I" * j + p + p + "I" * (n - j - 2)] = 1.0
    for j in range(n):
        terms["I" * j + "Z" + "I" * (n - j - 1)] = field
    return PauliSum.from_strings(terms, num_qubits=n, dtype="float64")


def _lanczos(h, o, steps, policy):
    """The operator Lanczos coefficients b_n of i[h, .] from o; a_n vanishes for Hermitian h and o."""
    o = o * (1.0 / o.overlap(o) ** 0.5)
    previous, b = None, []
    for _ in range(steps):
        a = h.liouvillian(o, policy=policy)
        if previous is not None:
            a -= previous * b[-1]
        norm = a.overlap(a) ** 0.5
        b.append(norm)
        previous, o = o, a * (1.0 / norm)
    return b


@pytest.mark.parametrize("partitions", [1, 2])
def test_lanczos_on_a_split_agrees_with_one_process(partitions):
    n = 14
    h = _chain(n)
    seed = PauliSum.from_strings({"I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1): 1.0}, num_qubits=n, dtype="float64")
    policy = truncation.coeff(1e-10)
    want = _lanczos(h, seed, 8, policy)
    got = _lanczos(h, seed.partition(partitions=partitions), 8, policy)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_a_split_product_gathers_to_the_whole_product():
    h, o = _chain_operands()
    split = o.astype("complex128").partition(partitions=2)
    hc = h.astype("complex128")
    for got, want in [
        ((hc @ split).gather(), hc @ o),
        (h.liouvillian(o.partition(partitions=2)).gather(), h.liouvillian(o)),
        (h.anticommutator(o.partition(partitions=2)).gather(), h.anticommutator(o)),
    ]:
        assert len(got) == len(want)
        np.testing.assert_allclose(np.sort(np.abs(got.coefficients_array())), np.sort(np.abs(want.coefficients_array())))
        assert got.overlap(want) == pytest.approx(want.overlap(want))


def test_split_linear_algebra_matches_whole_sums():
    h, o = _chain_operands()
    a = o.partition(partitions=2)
    b = h.liouvillian(a)
    whole_a, whole_b = o, h.liouvillian(o)
    assert a.overlap(b) == pytest.approx(whole_a.overlap(whole_b))
    combined = (a + b * 2.0 - a).gather()
    expected = whole_b * 2.0
    assert combined.overlap(expected) == pytest.approx(expected.overlap(expected))
    c = a.copy()
    c += c
    c *= 0.5
    assert (c - a).overlap(c - a) == pytest.approx(0.0, abs=1e-24)


def test_split_operands_reject_what_they_cannot_hold():
    h, o = _chain_operands()
    split = o.partition(partitions=2)
    with pytest.raises(TypeError, match="complex"):
        h.product(split)
    with pytest.raises(NotImplementedError):
        h.liouvillian(split, policy=truncation.topn(3))
    with pytest.raises(ValueError, match="dtypes differ"):
        split.overlap(o.astype("complex128").partition(partitions=2))
    with pytest.raises(ValueError, match="num_qubits mismatch"):
        PauliSum.from_strings({"XX": 1.0}, num_qubits=2).liouvillian(split)
    with pytest.raises(ValueError, match="imaginary"):
        h.astype("complex128").__mul__(1j).liouvillian(split)

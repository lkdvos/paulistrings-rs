import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, PauliTerms, pt, trotter
from paulistrings.lattices import chain, square
from paulistrings.models import transverse_ising, xxz


def arrays(s):
    return s.x_array(), s.z_array(), s.coefficients_array()


def test_sparse_terms_bind_to_the_dense_encoding():
    bound = PauliSum(3, 2 * pt("XY", (0, 2)))
    dense = PauliSum.from_strings({"XIY": 2})
    for u, v in zip(arrays(bound), arrays(dense)):
        np.testing.assert_array_equal(u, v)


def test_sites_rule():
    assert len(pt("X", 3)) == 1
    assert len(pt("X", range(4))) == 4
    assert len(pt("ZZ", (0, 1))) == 1
    assert len(pt("ZZ", [(0, 1), (1, 2)])) == 2
    assert len(pt("ZZ", [])) == 0
    with pytest.raises(ValueError, match="same length"):
        pt("ZZ", [0, 1, 2])
    with pytest.raises(ValueError, match="appears twice"):
        pt("ZZ", (1, 1))
    with pytest.raises(TypeError):
        pt("X", True)


def test_iteration_and_repr():
    terms = -1.0 * pt("ZZ", [(0, 1), (1, 2)])
    assert list(terms) == [("ZZ", (0, 1), -1), ("ZZ", (1, 2), -1)]
    assert repr(terms) == "-1*Z0Z1 + -1*Z1Z2"
    assert repr(PauliTerms()) == "0"


def test_order_and_duplicates_are_kept_until_bound():
    t = pt("X", 0) + pt("X", 0) - pt("Z", 1)
    assert [label for label, _, _ in t] == ["X", "X", "Z"]
    assert len(PauliSum(2, t)) == 2
    assert len(PauliSum(2, pt("X", 0) - pt("X", 0))) == 0


def test_binding_rejects_sites_past_the_register():
    with pytest.raises(ValueError, match="out of range"):
        PauliSum(2, pt("Z", 2))


def test_sum_and_numpy_scalars():
    assert len(sum(pt("Z", j) for j in range(3))) == 3
    assert isinstance(np.float64(2.0) * pt("X", 0), PauliTerms)


def test_models_order_terms_by_type_and_skip_zero_couplings():
    bonds = chain(3)
    assert [label for label, _, _ in xxz(bonds, J=1.0, Jz=0.5)] == ["XX", "XX", "YY", "YY", "ZZ", "ZZ"]
    assert len(transverse_ising(bonds, range(3), J=1.0, h=0.0)) == 2


def test_trotter_sign_and_factor_of_two():
    h, dt = 0.7, 0.3
    c = Circuit(1)
    c.trotter(-h * pt("X", 0), dt)
    y = PauliSum(1, pt("Y", 0)).propagate(c, direction="heisenberg")
    assert y.expectation("0").real == pytest.approx(math.sin(2 * h * dt))


def test_trotter_matches_the_hand_built_step():
    L, J, h, dt = 4, 1.0, 0.5, 0.1
    bonds = chain(L)
    manual = Circuit(L)
    for b in bonds:
        manual.pauli_rotation("ZZ", list(b), 2 * -J * dt)
    for j in range(L):
        manual.pauli_rotation("X", [j], 2 * -h * dt)
    assert trotter(transverse_ising(bonds, range(L), J, h), dt, L).gates == manual.gates


def test_steps_repeat_the_step_without_merging():
    H = transverse_ising(chain(3), range(3), J=1.0, h=0.5)
    one = trotter(H, 0.1, 3, order=2)
    assert trotter(H, 0.1, 3, steps=2, order=2).gates == one.gates + one.gates


def test_second_order_reduces_to_first_order_for_one_term():
    a, b = Circuit(1), Circuit(1)
    a.trotter(0.5 * pt("X", 0), 0.1, order=1)
    b.trotter(0.5 * pt("X", 0), 0.1, order=2)
    assert a.gates == b.gates


def test_second_order_step_is_symmetric():
    H = transverse_ising(chain(4), range(4), J=1.0, h=0.5)
    assert trotter(H, 0.1, 4, order=2).adjoint().gates == trotter(H, -0.1, 4, order=2).gates


def test_trotter_rejects_complex_coefficients():
    with pytest.raises(ValueError, match="imaginary"):
        Circuit(1).trotter(1j * pt("X", 0), 0.1)


def test_broadcast_is_all_or_nothing_and_equals_the_loop():
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.depolarize(0.1, [0, 5])
    assert len(c) == 0
    a, b = Circuit(3), Circuit(3)
    a.rx(0.3, range(3))
    a.pauli_rotation("ZZ", [(0, 1), (1, 2)], 0.2)
    for j in range(3):
        b.rx(0.3, j)
    b.pauli_rotation("ZZ", [0, 1], 0.2)
    b.pauli_rotation("ZZ", [1, 2], 0.2)
    assert a.gates == b.gates
    with pytest.raises(TypeError):
        Circuit(1).rx(0, 0.5)


def test_lattices():
    assert chain(3, periodic=True) == [(0, 1), (1, 2), (2, 0)]
    assert len(square(3, 3, periodic=True)) == 18
    assert len(square(3, 2)) == 7
    with pytest.raises(ValueError):
        square(2, 3, periodic=True)
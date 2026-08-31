"""``PauliSum.expectation`` / ``overlap`` / ``identity_coefficient``.

Note CI does not run these (see CLAUDE.md); run locally with
``maturin develop --release`` then ``pytest python/paulistrings/tests``.
"""

import math

import pytest

from paulistrings import Circuit, PauliSum

R = 1.0 / math.sqrt(2.0)


def _sum(terms, num_qubits):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


# ---- expectation ----


@pytest.mark.parametrize(
    "label,x_plus,y_plus,z_plus",
    [
        ("II", 1.0, 1.0, 1.0),
        ("XI", 1.0, 0.0, 0.0),
        ("YI", 0.0, 1.0, 0.0),
        ("ZI", 0.0, 0.0, 1.0),
    ],
)
def test_single_pauli_expectations(label, x_plus, y_plus, z_plus):
    s = _sum({label: 1.0}, 2)
    assert s.expectation("x+").real == pytest.approx(x_plus)
    assert s.expectation("y+").real == pytest.approx(y_plus)
    assert s.expectation("z+").real == pytest.approx(z_plus)


def test_expectation_defaults_to_x_plus():
    s = _sum({"XI": 1.0, "ZI": 5.0}, 2)
    assert s.expectation() == pytest.approx(s.expectation("x+"))
    assert s.expectation().real == pytest.approx(1.0)


def test_expectation_is_linear_and_complex():
    s = _sum({"XI": 1.0 + 2.0j, "IX": 3.0 - 5.0j}, 2)
    e = s.expectation("x+")
    assert e.real == pytest.approx(4.0)
    assert e.imag == pytest.approx(-3.0)


def test_expectation_of_multi_qubit_products():
    # XX contributes in x+; XZ in neither; YY in y+.
    s = _sum({"XX": 1.0, "XZ": 10.0, "YY": 100.0}, 2)
    assert s.expectation("x+").real == pytest.approx(1.0)
    assert s.expectation("y+").real == pytest.approx(100.0)
    assert s.expectation("z+").real == pytest.approx(0.0)


def test_expectation_rejects_an_unknown_state():
    s = _sum({"XI": 1.0}, 2)
    with pytest.raises(ValueError, match="unknown product state"):
        s.expectation("bogus")


@pytest.mark.parametrize("num_qubits", [3, 80, 200, 400, 800])
def test_expectation_across_all_width_bands(num_qubits):
    label = "X" + "I" * (num_qubits - 1)
    s = _sum({label: 2.0}, num_qubits)
    assert s.expectation("x+").real == pytest.approx(2.0)
    assert s.expectation("z+").real == pytest.approx(0.0)


# ---- overlap ----


def test_overlap_with_self_is_the_squared_norm():
    s = _sum({"XI": 2.0, "ZI": 3.0j}, 2)
    assert s.overlap(s).real == pytest.approx(13.0)


def test_overlap_counts_only_shared_keys():
    a = _sum({"XI": 2.0, "YI": 5.0}, 2)
    b = _sum({"XI": 3.0, "ZI": 7.0}, 2)
    assert a.overlap(b).real == pytest.approx(6.0)


def test_overlap_is_conjugate_symmetric():
    a = _sum({"XI": 1.0 + 2.0j}, 2)
    b = _sum({"XI": 3.0 - 1.0j}, 2)
    assert a.overlap(b) == pytest.approx(b.overlap(a).conjugate())


def test_overlap_rejects_a_qubit_count_mismatch():
    a = _sum({"XI": 1.0}, 2)
    b = _sum({"XII": 1.0}, 3)
    with pytest.raises(ValueError, match="num_qubits mismatch"):
        a.overlap(b)


# ---- identity coefficient ----


def test_identity_coefficient_is_the_trace():
    s = _sum({"II": 1.5, "XI": 9.0}, 2)
    assert s.identity_coefficient().real == pytest.approx(1.5)


def test_identity_coefficient_is_zero_when_absent():
    s = _sum({"XI": 9.0}, 2)
    assert abs(s.identity_coefficient()) == pytest.approx(0.0)


# ---- end to end ----


def test_expectation_after_propagation():
    # H maps Z to X, so <Z> in |0..0> becomes <X> in |+..+> after conjugation.
    s = _sum({"ZI": 1.0}, 2)
    c = Circuit(2)
    c.h(0)
    out = s.propagate(circuit=c)
    assert out.expectation("x+").real == pytest.approx(1.0)
    assert out.expectation("z+").real == pytest.approx(0.0)

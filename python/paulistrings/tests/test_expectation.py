"""``PauliSum.expectation`` / ``overlap`` / ``identity_coefficient``.

The uniform ``state`` shorthands (``"x+"``, ``"y+"``, ``"z+"``) live here;
per-qubit label strings (``"01+-"``) are covered by ``test_product_states.py``.

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


def _label(x_words, z_words, num_qubits):
    chars = []
    for q in range(num_qubits):
        word, bit = divmod(q, 64)
        x, z = (int(x_words[word]) >> bit) & 1, (int(z_words[word]) >> bit) & 1
        chars.append("IXZY"[x + 2 * z])
    return "".join(chars)


def test_overlap_of_a_small_sum_with_a_propagated_many_bucket_sum():
    num_qubits = 8
    c = Circuit(num_qubits)
    for _ in range(2):
        for q in range(num_qubits):
            c.rx(0.3 + 0.05 * q, q)
            c.rz(0.7 - 0.03 * q, q)
        for q in range(num_qubits - 1):
            c.cnot(q, q + 1)
    big = _sum({"Z" * num_qubits: 1.0}, num_qubits).propagate(c)
    assert big.num_buckets > 1, "the circuit must spread the sum over many buckets"

    rows = list(zip(big.x_array(), big.z_array(), big.coefficients()))
    picked = {_label(x, z, num_qubits): (k + 1.0) - 0.5j for k, (x, z, _) in enumerate(rows[:: len(rows) // 20])}
    big_coefficients = {_label(x, z, num_qubits): c for x, z, c in rows}
    picked["X" * num_qubits] = 2.0
    small = _sum(picked, num_qubits)
    assert small.num_buckets == 1

    want = sum(c.conjugate() * big_coefficients.get(label, 0.0) for label, c in picked.items())
    assert abs(want) > 0.0
    assert small.overlap(big) == pytest.approx(want, rel=1e-12)
    assert big.overlap(small) == pytest.approx(want.conjugate(), rel=1e-12)


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

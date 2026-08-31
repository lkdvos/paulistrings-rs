"""``PauliSum.expectation(state=...)`` with per-qubit product-state labels.

The label alphabet follows qiskit's ``Statevector.from_label``: ``0``/``1`` are
the ``Z`` eigenstates, ``+``/``-`` the ``X`` ones, ``r``/``l`` the ``Y`` ones,
and character ``i`` addresses qubit ``i`` — the same indexing as
``PauliSum.from_strings``. The three uniform shorthands (``"x+"``, ``"y+"``,
``"z+"``) keep their old meaning.

Every expected value here is the product of single-qubit Bloch-vector
components, hand-derived once:

===========  ==========  ==========  ==========
state        ``<X>``     ``<Y>``     ``<Z>``
===========  ==========  ==========  ==========
``0``  Z+       0           0          +1
``1``  Z-       0           0          -1
``+``  X+      +1           0           0
``-``  X-      -1           0           0
``r``  Y+       0          +1           0
``l``  Y-       0          -1           0
===========  ==========  ==========  ==========

with ``<I> = 1`` in every state. Off-axis components vanish because two
distinct single-qubit Paulis anticommute.

Note CI does not run these (see CLAUDE.md); run locally with
``maturin develop --release`` then ``pytest python/paulistrings/tests``.
"""

import functools
import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum

R = 1.0 / math.sqrt(2.0)


def _sum(terms, num_qubits):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


# ---- single-qubit table ----


@pytest.mark.parametrize(
    "label,e_i,e_x,e_y,e_z",
    [
        ("0", 1.0, 0.0, 0.0, 1.0),
        ("1", 1.0, 0.0, 0.0, -1.0),
        ("+", 1.0, 1.0, 0.0, 0.0),
        ("-", 1.0, -1.0, 0.0, 0.0),
        ("r", 1.0, 0.0, 1.0, 0.0),
        ("l", 1.0, 0.0, -1.0, 0.0),
    ],
)
def test_single_qubit_labels_against_every_pauli(label, e_i, e_x, e_y, e_z):
    for pauli, want in (("I", e_i), ("X", e_x), ("Y", e_y), ("Z", e_z)):
        s = _sum({pauli: 1.0}, 1)
        got = s.expectation(label).real
        assert got == pytest.approx(want), f"<{label}|{pauli}|{label}>"


def test_an_off_axis_pauli_never_matches():
    # The subset-match trap: X on a Y-axis qubit must NOT contribute, even
    # though the Y axis has its x-bit set. <r|X|r> = 0.
    for label, pauli in (
        ("r", "X"),
        ("r", "Z"),
        ("l", "X"),
        ("+", "Y"),
        ("+", "Z"),
        ("0", "X"),
        ("0", "Y"),
        ("1", "Y"),
    ):
        s = _sum({pauli: 3.0 - 4.0j}, 1)
        assert abs(s.expectation(label)) == pytest.approx(0.0)
    # One off-axis factor kills a term whose other factors do match.
    s = _sum({"XXX": 1.0}, 3)
    assert s.expectation("+++").real == pytest.approx(1.0)
    assert s.expectation("++r").real == pytest.approx(0.0)


# ---- sign composition over several qubits ----


def test_signs_compose_across_qubits():
    # <01|Z@Z|01> = <0|Z|0> * <1|Z|1> = (+1)(-1) = -1.
    zz = _sum({"ZZ": 1.0}, 2)
    assert zz.expectation("01").real == pytest.approx(-1.0)
    assert zz.expectation("10").real == pytest.approx(-1.0)
    assert zz.expectation("00").real == pytest.approx(1.0)
    assert zz.expectation("11").real == pytest.approx(1.0)  # (-1)(-1)


def test_mixed_axes_and_signs():
    zxy = _sum({"ZXY": 1.0}, 3)
    # |0>|+>|r>: axes Z, X, Y, all signs +1.
    assert zxy.expectation("0+r").real == pytest.approx(1.0)
    # |1>|->|l>: the same axes with every sign flipped -> (-1)^3.
    assert zxy.expectation("1-l").real == pytest.approx(-1.0)
    # One flipped site each.
    assert zxy.expectation("1+r").real == pytest.approx(-1.0)
    assert zxy.expectation("0-r").real == pytest.approx(-1.0)
    assert zxy.expectation("0+l").real == pytest.approx(-1.0)
    # Two flipped sites.
    assert zxy.expectation("1-r").real == pytest.approx(1.0)


def test_identity_factors_are_ignored():
    # Qubit 1 is an identity factor, so its label cannot change the value:
    # <Z@I@Y> = <Z>_0 * <Y>_2.
    ziy = _sum({"ZIY": 1.0}, 3)
    for middle in "01+-rl":
        assert ziy.expectation(f"0{middle}r").real == pytest.approx(1.0)
        assert ziy.expectation(f"1{middle}r").real == pytest.approx(-1.0)
        assert ziy.expectation(f"1{middle}l").real == pytest.approx(1.0)
    # The all-identity term is the trace and every state gives 1.
    iii = _sum({"III": 2.5}, 3)
    assert iii.expectation("1-l").real == pytest.approx(2.5)


def test_labelled_expectation_is_linear_and_complex():
    # <1|Z|1> = -1 and <1|I|1> = +1, so this is -(1+2j) + (3-5j).
    s = _sum({"Z": 1.0 + 2.0j, "I": 3.0 - 5.0j}, 1)
    e = s.expectation("1")
    assert e.real == pytest.approx(2.0)
    assert e.imag == pytest.approx(-7.0)


# ---- width bands and word boundaries ----


@pytest.mark.parametrize("num_qubits", [3, 80, 200, 400, 800])
def test_labels_across_all_width_bands(num_qubits):
    s = _sum({"Z" + "I" * (num_qubits - 1): 2.0}, num_qubits)
    assert s.expectation("0" * num_qubits).real == pytest.approx(2.0)
    assert s.expectation("1" + "0" * (num_qubits - 1)).real == pytest.approx(-2.0)
    # A flipped label on an identity site changes nothing.
    assert s.expectation("0" + "1" * (num_qubits - 1)).real == pytest.approx(2.0)
    assert s.expectation("+" * num_qubits).real == pytest.approx(0.0)


def test_labels_on_either_side_of_the_word_boundary():
    # 128 qubits: qubit 64's sign bit lives in the second mask word.
    n = 128
    labels = list("0" * n)
    labels[64] = "1"
    labels = "".join(labels)
    z0 = "Z" + "I" * (n - 1)
    z64 = "I" * 64 + "Z" + "I" * (n - 65)
    both = "Z" + "I" * 63 + "Z" + "I" * (n - 65)
    assert _sum({z0: 1.0}, n).expectation(labels).real == pytest.approx(1.0)
    assert _sum({z64: 1.0}, n).expectation(labels).real == pytest.approx(-1.0)
    assert _sum({both: 1.0}, n).expectation(labels).real == pytest.approx(-1.0)


# ---- the uniform shorthands still mean what they meant ----


@pytest.mark.parametrize(
    "shorthand,label", [("x+", "+"), ("y+", "r"), ("z+", "0")]
)
@pytest.mark.parametrize("upper", [False, True])
def test_uniform_shorthands_match_their_label_spellings(shorthand, label, upper):
    n = 5
    terms = {
        "XIYZI": 1.5,
        "YYYYY": -0.75,
        "ZZIII": 2.0j,
        "IIIII": 0.25,
        "XXXXX": 3.0,
    }
    s = _sum(terms, n)
    name = shorthand.upper() if upper else shorthand
    assert s.expectation(name) == pytest.approx(s.expectation(label * n))


def test_default_state_is_still_x_plus():
    s = _sum({"XI": 1.0, "ZI": 5.0}, 2)
    assert s.expectation() == pytest.approx(s.expectation("++"))


# ---- validation ----


@pytest.mark.parametrize("state", ["0", "000", "", "0" * 64])
def test_a_label_string_must_have_one_character_per_qubit(state):
    s = _sum({"XI": 1.0}, 2)
    with pytest.raises(ValueError, match="unknown product state"):
        s.expectation(state)


@pytest.mark.parametrize("state", ["0x", "bogus", "2", "R", "L", "0 ", "0|"])
def test_a_label_string_rejects_characters_outside_the_alphabet(state):
    s = _sum({"XI": 1.0}, 2)
    with pytest.raises(ValueError, match="unknown product state"):
        s.expectation(state)


def test_the_error_names_both_accepted_forms():
    s = _sum({"XI": 1.0}, 2)
    with pytest.raises(ValueError) as exc:
        s.expectation("q+")
    message = str(exc.value)
    assert "01+-rl" in message
    for name in ('"x+"', '"y+"', '"z+"'):
        assert name in message


# ---- cross-validation against a dense reference ----

_KET = {
    "0": np.array([1.0, 0.0], dtype=complex),
    "1": np.array([0.0, 1.0], dtype=complex),
    "+": np.array([R, R], dtype=complex),
    "-": np.array([R, -R], dtype=complex),
    "r": np.array([R, 1j * R], dtype=complex),
    "l": np.array([R, -1j * R], dtype=complex),
}

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

_H = np.array([[R, R], [R, -R]], dtype=complex)
_S = np.diag([1.0, 1j]).astype(complex)
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def _rz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _rx(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rzz(theta):
    # exp(-i * theta * Z@Z / 2); Z@Z is diagonal with entries (+1, -1, -1, +1).
    return np.diag(np.exp(-0.5j * theta * np.array([1.0, -1.0, -1.0, 1.0])))


def _kron_all(mats):
    return functools.reduce(np.kron, mats)


def _dense_pauli(label):
    """Qubit 0 is the most significant kron factor, matching from_strings."""
    return _kron_all([_PAULI[ch] for ch in label])


def _dense_state(labels):
    return _kron_all([_KET[ch] for ch in labels])


def _dense_observable(terms, num_qubits):
    dim = 1 << num_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for label, coeff in terms.items():
        out += coeff * _dense_pauli(label)
    return out


def _embed(gate, qubits, num_qubits):
    """Full-space matrix of `gate` acting on `qubits`, first one most significant."""
    k = len(qubits)
    dim = 1 << num_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for col_state in range(dim):
        col = 0
        for q in qubits:
            col = (col << 1) | ((col_state >> (num_qubits - 1 - q)) & 1)
        for row in range(1 << k):
            amp = gate[row, col]
            if amp == 0.0:
                continue
            row_state = col_state
            for pos, q in enumerate(qubits):
                mask = 1 << (num_qubits - 1 - q)
                bit = (row >> (k - 1 - pos)) & 1
                row_state = (row_state & ~mask) | (mask if bit else 0)
            out[row_state, col_state] += amp
    return out


def test_heisenberg_contraction_matches_a_dense_reference():
    """`propagate(heisenberg)` then `expectation` == <psi|U^dag O U|psi> densely.

    The library's Heisenberg direction applies each channel's adjoint in
    reverse, i.e. O -> U^dag O U for U the whole circuit; contracting that
    against |psi> is the same number as evolving |psi> forward and measuring O
    on it, which is what the dense reference computes.
    """
    n = 4
    terms = {"ZIII": 1.0, "IXZI": 0.5, "YYII": -0.25, "IIIX": 0.75}
    # (dense matrix, qubits) in application order; the Circuit below must
    # match this list gate for gate.
    layers = [
        (_H, [0]),
        (_CNOT, [0, 1]),
        (_rz(0.7), [2]),
        (_CNOT, [2, 3]),
        (_ry(-0.3), [1]),
        (_S, [3]),
        (_CZ, [1, 2]),
        (_rx(1.1), [0]),
        (_rzz(0.45), [0, 3]),
    ]
    circuit = Circuit(n)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.rz(0.7, 2)
    circuit.cnot(2, 3)
    circuit.ry(-0.3, 1)
    circuit.s(3)
    circuit.cz(1, 2)
    circuit.rx(1.1, 0)
    circuit.pauli_rotation("ZZ", [0, 3], 0.45)

    unitary = np.eye(1 << n, dtype=complex)
    for gate, qubits in layers:
        unitary = _embed(gate, qubits, n) @ unitary
    observable = _dense_observable(terms, n)

    evolved = _sum(terms, n).propagate(circuit=circuit, direction="heisenberg")

    seen_nontrivial = False
    for labels in ("0000", "01+-", "rlrl", "0+r1", "1-l0", "++++"):
        psi = unitary @ _dense_state(labels)
        want = complex(np.conj(psi) @ observable @ psi)
        got = evolved.expectation(labels)
        assert abs(got - want) < 1e-12, f"{labels}: {got} vs {want}"
        seen_nontrivial |= abs(want) > 1e-3
    assert seen_nontrivial, "the reference values are all ~0 — vacuous test"


def test_forward_direction_matches_the_dense_reference_too():
    # Forward applies U O U^dag, so the dense reference conjugates the state
    # by U^dag instead.
    n = 3
    terms = {"XZI": 1.0, "IIY": -0.5}
    circuit = Circuit(n)
    circuit.h(1)
    circuit.cnot(1, 2)
    circuit.rz(0.9, 0)
    unitary = np.eye(1 << n, dtype=complex)
    for gate, qubits in ((_H, [1]), (_CNOT, [1, 2]), (_rz(0.9), [0])):
        unitary = _embed(gate, qubits, n) @ unitary
    observable = _dense_observable(terms, n)
    evolved = _sum(terms, n).propagate(circuit=circuit, direction="forward")
    for labels in ("000", "0+r", "1-l"):
        psi = unitary.conj().T @ _dense_state(labels)
        want = complex(np.conj(psi) @ observable @ psi)
        got = evolved.expectation(labels)
        assert abs(got - want) < 1e-12, f"{labels}: {got} vs {want}"

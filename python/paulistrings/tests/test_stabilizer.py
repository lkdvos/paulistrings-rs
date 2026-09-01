"""``PauliSum.expectation_stabilizer``.

Design source: ``research/notes/2026-09-01-python-api-extensions.md`` §A8-ii.
The core's own hand-computed cases live in
``crates/paulistrings/src/stabilizer.rs``; what is added here is the *string*
surface (signed generator specs, their error messages) plus an independent
oracle the Rust side cannot reach: a dense ``numpy`` projector
``Pi = prod_i (I + s_i G_i) / 2`` at ``n <= 6``.

Note CI does not run these (see CLAUDE.md); run locally with
``maturin develop --release`` then ``pytest python/paulistrings/tests``.
"""

import functools
import itertools

import numpy as np
import pytest

from paulistrings import PauliSum

TOL = 1e-12


# =============================================================================
# dense reference machinery
# =============================================================================

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _dense_pauli(label):
    """Qubit 0 is the most significant kron factor, matching ``from_strings``."""
    return functools.reduce(np.kron, [_PAULI[ch] for ch in label])


def _dense_observable(terms, num_qubits):
    dim = 1 << num_qubits
    out = np.zeros((dim, dim), dtype=complex)
    for label, coeff in terms.items():
        out += coeff * _dense_pauli(label)
    return out


def _split_sign(spec):
    if spec[0] in "+-":
        return (-1.0 if spec[0] == "-" else 1.0), spec[1:]
    return 1.0, spec


def _dense_stabilizer_state(generators):
    """The state vector fixed by `generators`, via the projector onto S's +1 eigenspace.

    ``Pi = prod_i (I + s_i G_i) / 2`` is rank 1 for `n` independent commuting
    generators, so any nonzero column of it is the state (up to normalization).
    """
    n = len(_split_sign(generators[0])[1])
    dim = 1 << n
    proj = np.eye(dim, dtype=complex)
    for spec in generators:
        sign, label = _split_sign(spec)
        proj = proj @ (np.eye(dim, dtype=complex) + sign * _dense_pauli(label)) / 2.0
    assert abs(np.trace(proj) - 1.0) < 1e-10, "projector is not rank 1"
    col = int(np.argmax(np.linalg.norm(proj, axis=0)))
    psi = proj[:, col]
    return psi / np.linalg.norm(psi)


def _dense_expectation(terms, generators):
    n = len(_split_sign(generators[0])[1])
    psi = _dense_stabilizer_state(generators)
    return psi.conj() @ _dense_observable(terms, n) @ psi


def _sum(terms, num_qubits):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


# =============================================================================
# hand-computed states
# =============================================================================

BELL = ["XX", "ZZ"]
GHZ = ["XXX", "ZZI", "IZZ"]


@pytest.mark.parametrize(
    "label,want",
    [
        ("II", 1.0),
        ("XX", 1.0),
        ("ZZ", 1.0),
        # XX·ZZ = (X·Z) (x) (X·Z) = (-iY) (x) (-iY) = -YY.
        ("YY", -1.0),
        ("ZI", 0.0),
        ("IZ", 0.0),
        ("XI", 0.0),
        ("XZ", 0.0),
        ("YX", 0.0),
    ],
)
def test_bell_state_single_paulis(label, want):
    assert _sum({label: 1.0}, 2).expectation_stabilizer(BELL).real == pytest.approx(want)


def test_bare_and_explicitly_signed_generators_agree():
    s = _sum({"YY": 1.0}, 2)
    assert s.expectation_stabilizer(["XX", "ZZ"]) == s.expectation_stabilizer(["+XX", "+ZZ"])


def test_a_minus_generator_flips_the_group_elements_containing_it():
    # (-XX)(+ZZ) = -(XX·ZZ) = +YY.
    s = _sum({"XX": 1.0}, 2)
    assert s.expectation_stabilizer(["-XX", "ZZ"]).real == pytest.approx(-1.0)
    assert _sum({"YY": 1.0}, 2).expectation_stabilizer(["-XX", "ZZ"]).real == pytest.approx(1.0)
    assert _sum({"ZZ": 1.0}, 2).expectation_stabilizer(["-XX", "ZZ"]).real == pytest.approx(1.0)


def test_minus_z_is_the_one_state():
    assert _sum({"Z": 1.0}, 1).expectation_stabilizer(["-Z"]).real == pytest.approx(-1.0)
    assert _sum({"X": 1.0}, 1).expectation_stabilizer(["-Z"]).real == pytest.approx(0.0)
    assert _sum({"I": 1.0}, 1).expectation_stabilizer(["-Z"]).real == pytest.approx(1.0)


@pytest.mark.parametrize(
    "label,want",
    [
        ("XXX", 1.0),
        ("ZZI", 1.0),
        ("IZZ", 1.0),
        ("ZIZ", 1.0),
        # XXX·ZZI acts as X·Z = -iY on qubits 0 and 1 and as X on qubit 2, so
        # it equals (-i)^2 YYX = -YYX.
        ("YYX", -1.0),
        ("YXY", -1.0),
        ("XYY", -1.0),
        ("ZII", 0.0),
        ("XXI", 0.0),
        ("YYY", 0.0),
        ("III", 1.0),
    ],
)
def test_ghz_state_single_paulis(label, want):
    assert _sum({label: 1.0}, 3).expectation_stabilizer(GHZ).real == pytest.approx(want)


def test_the_contraction_is_linear_and_keeps_the_imaginary_part():
    terms = {"XX": 2.0 + 1.0j, "ZZ": 0.5, "YY": 4.0 + 2.0j, "ZI": 100.0 + 100.0j}
    got = _sum(terms, 2).expectation_stabilizer(BELL)
    # (2 + i) + 0.5 - (4 + 2i) = -1.5 - i
    assert got.real == pytest.approx(-1.5)
    assert got.imag == pytest.approx(-1.0)


def test_product_generators_agree_with_the_product_state_path():
    terms = {"XZYI": 1.0, "ZZZZ": -2.0, "IIII": 0.25, "XXXX": 3.0, "YYII": 0.5j}
    s = _sum(terms, 4)
    for axis, label in [("Z", "0000"), ("X", "++++"), ("Y", "rrrr")]:
        gens = ["".join(axis if q == i else "I" for q in range(4)) for i in range(4)]
        assert s.expectation_stabilizer(gens) == pytest.approx(s.expectation(label))
    # ...and the signed diagonal generators reproduce a per-qubit label string.
    gens = ["ZIII", "-IZII", "IIZI", "-IIIZ"]
    assert s.expectation_stabilizer(gens) == pytest.approx(s.expectation("0101"))


def test_a_bell_pair_across_the_word_boundary():
    """Qubits 63/64 straddle the ``[u64; W]`` boundary at the W=2 band."""
    n = 66

    def label(pairs):
        chars = ["I"] * n
        for q, ch in pairs:
            chars[q] = ch
        return "".join(chars)

    gens = []
    for q in range(n):
        if q == 63:
            gens.append(label([(63, "X"), (64, "X")]))
        elif q == 64:
            gens.append(label([(63, "Z"), (64, "Z")]))
        else:
            gens.append(label([(q, "Z")]))

    for lbl, want in [
        (label([(63, "X"), (64, "X")]), 1.0),
        (label([(63, "Z"), (64, "Z")]), 1.0),
        (label([(63, "Y"), (64, "Y")]), -1.0),
        (label([(63, "Z")]), 0.0),
        (label([(0, "Z")]), 1.0),
        (label([(65, "Z")]), 1.0),
        (label([(65, "X")]), 0.0),
    ]:
        got = _sum({lbl: 1.0}, n).expectation_stabilizer(gens).real
        assert got == pytest.approx(want), lbl


# =============================================================================
# dense cross-check
# =============================================================================


def _gf2_rank(rows):
    """Rank over GF(2) of rows given as Python ints (bit `i` = column `i`)."""
    rows = [int(r) for r in rows if int(r) != 0]
    rank = 0
    while rows:
        pivot = rows.pop()
        bit = pivot.bit_length() - 1
        rows = [r ^ pivot if (r >> bit) & 1 else r for r in rows]
        rows = [r for r in rows if r != 0]
        rank += 1
    return rank


def _as_bitrow(x, z, n):
    val = 0
    for q in range(n):
        val |= int(x[q]) << q
        val |= int(z[q]) << (n + q)
    return val


def _commutes(ax, az, bx, bz):
    return (int(np.sum(ax * bz)) + int(np.sum(az * bx))) % 2 == 0


def _random_generators(n, rng):
    """A random valid signed generator set: greedily grow a commuting, GF(2)-independent set."""
    chosen, rows = [], []
    tries = 0
    while len(chosen) < n:
        tries += 1
        assert tries < 10_000, "failed to grow a stabilizer generator set"
        x = rng.integers(0, 2, n)
        z = rng.integers(0, 2, n)
        if not all(_commutes(x, z, cx, cz) for cx, cz in chosen):
            continue
        row = _as_bitrow(x, z, n)
        if _gf2_rank(rows + [row]) == len(rows):
            continue
        chosen.append((x, z))
        rows.append(row)
    specs = []
    for x, z in chosen:
        label = "".join("YXZI"[(1 - int(x[q])) * 2 + (1 - int(z[q]))] for q in range(n))
        specs.append(("-" if rng.integers(0, 2) else "+") + label)
    return specs


def _random_terms(n, count, rng):
    terms = {}
    for _ in range(count):
        label = "".join(rng.choice(list("IXYZ")) for _ in range(n))
        terms[label] = complex(rng.normal(), rng.normal())
    return terms


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_random_stabilizer_states_match_a_dense_projector(n):
    """`Pi = prod (I + s_i G_i)/2` densely, against the O(m n^2/64) contraction."""
    rng = np.random.default_rng(0xB7 + n)
    for _ in range(4):
        gens = _random_generators(n, rng)
        terms = _random_terms(n, min(4**n, 24), rng)
        got = _sum(terms, n).expectation_stabilizer(gens)
        want = _dense_expectation(terms, gens)
        assert abs(got - want) < TOL, f"n={n} gens={gens}: {got} vs {want}"


@pytest.mark.parametrize(
    "gens",
    [
        ["XX", "ZZ"],
        ["-XX", "ZZ"],
        ["XX", "-ZZ"],
        ["-XX", "-ZZ"],
        ["XXX", "ZZI", "IZZ"],
        ["-XXX", "ZZI", "-IZZ"],
        # 4-qubit line cluster state, K_q = Z_{q-1} X_q Z_{q+1}.
        ["XZII", "ZXZI", "IZXZ", "IIZX"],
        ["-XZII", "ZXZI", "-IZXZ", "IIZX"],
        # Two Bell pairs.
        ["XXII", "ZZII", "IIXX", "-IIZZ"],
    ],
)
def test_named_states_match_the_dense_projector_over_every_pauli(gens):
    """Every one of the 4^n Paulis, one at a time, against the dense state."""
    n = len(_split_sign(gens[0])[1])
    for chars in itertools.product("IXYZ", repeat=n):
        label = "".join(chars)
        got = _sum({label: 1.0}, n).expectation_stabilizer(gens)
        want = _dense_expectation({label: 1.0}, gens)
        assert abs(got - want) < TOL, f"{gens} / {label}: {got} vs {want}"
        # A stabilizer expectation of a single Pauli is always 0 or +-1.
        assert abs(got.imag) < TOL
        assert round(got.real, 9) in (-1.0, 0.0, 1.0)


# =============================================================================
# validation
# =============================================================================


def test_anticommuting_generators_are_rejected():
    with pytest.raises(ValueError, match="anticommute"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["XI", "ZI"])


def test_dependent_generators_are_rejected():
    with pytest.raises(ValueError, match="independent over GF\\(2\\)"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["ZI", "ZI"])


def test_generators_implying_minus_identity_are_rejected():
    with pytest.raises(ValueError, match="independent over GF\\(2\\)"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["+ZI", "-ZI"])


@pytest.mark.parametrize("gens", [["ZI"], ["ZI", "IZ", "ZZ"], []])
def test_wrong_generator_count_is_rejected(gens):
    with pytest.raises(ValueError, match="needs exactly 2 generators"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(gens)


def test_a_generator_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="has length 3 after the optional sign"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["ZI", "IZZ"])


def test_an_unknown_pauli_character_is_rejected():
    with pytest.raises(ValueError, match="unexpected Pauli character"):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["ZI", "IQ"])


def test_a_non_string_generator_is_a_type_error():
    with pytest.raises(TypeError):
        _sum({"II": 1.0}, 2).expectation_stabilizer(["ZI", 7])


def test_an_empty_sum_contracts_to_zero():
    assert PauliSum(3).expectation_stabilizer(["ZII", "IZI", "IIZ"]) == 0.0

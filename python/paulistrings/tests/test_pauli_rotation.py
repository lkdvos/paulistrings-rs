"""``gates.pauli_rotation`` / ``Circuit.pauli_rotation`` — multi-qubit Pauli-string rotations.

The core ``PauliRotation`` already accepts any generator weight
(``crates/paulistrings/src/channel/rotation.rs``); this file checks the Python
boundary: the compact ``("ZZ", [i, j])`` spelling, the conjugation identities at
weights 1/2/3, and the argument validation.

Convention (hand-derived, and pinned by the core's own slice tests): the channel
is ``U = exp(-i·θ·P/2)`` and one forward layer maps ``Q ↦ U Q U†``, which for
``{Q, P} = 0`` is::

    Q  ↦  cos θ · Q  +  i · sin θ · Q·P

with the ``i^k`` from the Pauli product ``Q·P`` folded into the coefficient.

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import math

import pytest

from paulistrings import Circuit, PauliSum, gates


TOL = 1e-10


def _sum(num_qubits, terms):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons do not depend on ordering."""
    xs = sum_.x_array()
    zs = sum_.z_array()
    cs = sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_close(a, b, tol=TOL):
    da, db = _as_dict(a), _as_dict(b)
    assert set(da) == set(db), f"different keys:\n{sorted(da)}\nvs\n{sorted(db)}"
    for k in da:
        assert abs(da[k] - db[k]) < tol, f"{k}: {da[k]} vs {db[k]}"


def _string(num_qubits, ops):
    """Full-length Pauli string with ``ops`` = {qubit: char} and I elsewhere."""
    chars = ["I"] * num_qubits
    for q, ch in ops.items():
        chars[q] = ch
    return "".join(chars)


# ---- weight 1: the single-qubit spelling ----


def test_weight_one_z_rotation_matches_rz():
    # exp(-iθZ/2) X exp(+iθZ/2) = cos θ · X + sin θ · Y.
    #
    # Hand-derivation: X anticommutes with Z, so the fanout-2 branch fires and
    # the second term is i·sin θ·(X·Z). X·Z = -i·Y, so i·(-i) = +1 and the Y
    # coefficient is +sin θ. (Directly: R_z(θ) X R_z(θ)† has entries
    # [[0, e^{-iθ}], [e^{iθ}, 0]] = cos θ·X + sin θ·Y.)
    theta = 0.6
    initial = _sum(1, {"X": 1.0})
    c = Circuit(1)
    c.append(gates.pauli_rotation("Z", [0], theta))
    out = initial.propagate(circuit=c, direction="forward")
    want = _sum(1, {"X": math.cos(theta), "Y": math.sin(theta)})
    _assert_close(out, want)


def test_weight_one_agrees_with_rz_term_for_term():
    theta = -0.37
    initial = _sum(3, {"XIZ": 1.0, "ZYI": 0.25})
    via_pauli = Circuit(3)
    via_pauli.pauli_rotation("Z", [0], theta)
    via_rz = Circuit(3)
    via_rz.rz(theta, 0)
    _assert_close(
        initial.propagate(circuit=via_pauli, direction="forward"),
        initial.propagate(circuit=via_rz, direction="forward"),
    )


def test_weight_one_x_rotation_matches_rx():
    theta = 0.9
    initial = _sum(2, {"ZI": 1.0})
    via_pauli = Circuit(2)
    via_pauli.append(gates.pauli_rotation("X", [0], theta))
    via_rx = Circuit(2)
    via_rx.rx(theta, 0)
    _assert_close(
        initial.propagate(circuit=via_pauli, direction="forward"),
        initial.propagate(circuit=via_rx, direction="forward"),
    )


# ---- weight 2: the kicked-Ising bond ----


def test_zz_rotation_on_xi_is_hand_computed():
    # P = Z0·Z1, Q = X0 ("XI"). Qubit 0 anticommutes (X vs Z), qubit 1 commutes
    # (I vs Z), so overall they anticommute and the fanout-2 branch fires:
    #
    #   Q ↦ cos θ · XI + i·sin θ · (XI · ZZ)
    #
    # XI · ZZ: qubit 0 gives X·Z = -i·Y, qubit 1 gives I·Z = Z, so the product
    # is -i · YZ. Folding the leading i: i·(-i) = +1, hence
    #
    #   XI ↦ cos θ · XI + sin θ · YZ.
    theta = 0.6
    initial = _sum(2, {"XI": 1.0})
    c = Circuit(2)
    c.append(gates.pauli_rotation("ZZ", [0, 1], theta))
    out = initial.propagate(circuit=c, direction="forward")
    want = _sum(2, {"XI": math.cos(theta), "YZ": math.sin(theta)})
    _assert_close(out, want)


def test_kicked_ising_bond_maps_xi_to_minus_yz():
    # The Clifford point of the kicked-Ising bond: exp(+iπ/4 · Z_iZ_j) is
    # theta = -π/2 in the exp(-i·θ·P/2) convention. cos(-π/2) = 0 and
    # sin(-π/2) = -1, so XI ↦ -YZ exactly (up to the sin/cos floating-point
    # residue on the XI term).
    initial = _sum(2, {"XI": 1.0})
    c = Circuit(2)
    c.pauli_rotation("ZZ", [0, 1], -math.pi / 2)
    got = _as_dict(initial.propagate(circuit=c, direction="forward"))
    # "YZ" = qubit 0 Y (x and z bits), qubit 1 Z (z bit) → x=(1,), z=(3,).
    yz = ((1,), (3,))
    xi = ((1,), (0,))
    assert abs(got[yz] - (-1 + 0j)) < TOL
    assert abs(got.get(xi, 0j)) < 1e-15


def test_zz_rotation_matches_the_cnot_rz_cnot_decomposition():
    # exp(-iθ·Z0Z1/2) = CNOT(0,1) · Rz(θ, 1) · CNOT(0,1), because
    # CNOT Z_1 CNOT = Z_0 Z_1. An independent cross-check of the weight-2
    # generator against gates that were already covered.
    theta = 0.83
    initial = _sum(3, {"XII": 1.0, "IZY": 0.5, "YXI": -0.25})

    native = Circuit(3)
    native.pauli_rotation("ZZ", [0, 1], theta)

    decomposed = Circuit(3)
    decomposed.cnot(0, 1)
    decomposed.rz(theta, 1)
    decomposed.cnot(0, 1)

    _assert_close(
        initial.propagate(circuit=native, direction="forward"),
        initial.propagate(circuit=decomposed, direction="forward"),
    )


def test_qubit_order_in_the_compact_form_is_positional():
    # "XZ" on [1, 0] is X on qubit 1 and Z on qubit 0, i.e. the same generator
    # as "ZX" on [0, 1]. A transposition bug would show up here.
    theta = 0.4
    initial = _sum(2, {"YI": 1.0})
    a = Circuit(2)
    a.pauli_rotation("XZ", [1, 0], theta)
    b = Circuit(2)
    b.pauli_rotation("ZX", [0, 1], theta)
    _assert_close(
        initial.propagate(circuit=a, direction="forward"),
        initial.propagate(circuit=b, direction="forward"),
    )


# ---- weight 3: above MAX_LOCAL_SUPPORT, crossing a word boundary ----


def test_weight_three_generator_across_a_word_boundary():
    # W=2 (num_qubits = 100), generator on qubits 63/64/65 so the support
    # straddles the 64-bit word boundary. Weight 3 > MAX_LOCAL_SUPPORT, so this
    # is the first Python-visible use of the Prepared::Rotation path.
    #
    # Q = X63, P = Z63·Z64·Z65: only qubit 63 anticommutes, so the pair
    # anticommutes. Q·P = (X63·Z63)·Z64·Z65 = -i · Y63·Z64·Z65, and folding the
    # leading i gives
    #
    #   X63 ↦ cos θ · X63 + sin θ · Y63 Z64 Z65.
    n = 100
    theta = 0.55
    initial = _sum(n, {_string(n, {63: "X"}): 1.0})
    c = Circuit(n)
    c.pauli_rotation("ZZZ", [63, 64, 65], theta)
    out = initial.propagate(circuit=c, direction="forward")
    want = _sum(
        n,
        {
            _string(n, {63: "X"}): math.cos(theta),
            _string(n, {63: "Y", 64: "Z", 65: "Z"}): math.sin(theta),
        },
    )
    _assert_close(out, want)


def test_weight_three_round_trips_under_heisenberg():
    n = 100
    theta = 0.31
    initial = _sum(
        n,
        {
            _string(n, {63: "X"}): 1.0,
            _string(n, {0: "Z", 64: "Y"}): 0.5,
            _string(n, {65: "X", 99: "Z"}): -0.125,
        },
    )
    c = Circuit(n)
    c.pauli_rotation("XYZ", [63, 64, 65], theta)
    fwd = initial.propagate(circuit=c, direction="forward")
    back = fwd.propagate(circuit=c, direction="heisenberg")
    _assert_close(back, initial)


def test_weight_four_y_generator_round_trips():
    # A Y in the generator sets both bit planes on one qubit; the support must
    # still be one qubit per character.
    n = 70
    theta = 1.1
    initial = _sum(n, {_string(n, {1: "X", 66: "Z"}): 1.0})
    c = Circuit(n)
    c.pauli_rotation("YYXZ", [1, 2, 65, 66], theta)
    fwd = initial.propagate(circuit=c, direction="forward")
    back = fwd.propagate(circuit=c, direction="heisenberg")
    _assert_close(back, initial)


def test_factory_and_method_agree():
    n = 4
    theta = 0.22
    initial = _sum(n, {"XIII": 1.0, "IZZI": 0.5})
    a = Circuit(n)
    a.pauli_rotation("ZYX", [0, 2, 3], theta)
    b = Circuit(n)
    b.append(gates.pauli_rotation("ZYX", [0, 2, 3], theta))
    _assert_close(
        initial.propagate(circuit=a, direction="forward"),
        initial.propagate(circuit=b, direction="forward"),
    )


# ---- validation ----


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        gates.pauli_rotation("ZZ", [0], 0.1)
    with pytest.raises(ValueError, match="same length"):
        Circuit(4).pauli_rotation("Z", [0, 1], 0.1)


def test_empty_generator_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        gates.pauli_rotation("", [], 0.1)


def test_identity_character_is_rejected():
    # Identity positions are expressed by omission; allowing "I" would give two
    # spellings of the same channel.
    with pytest.raises(ValueError, match="X/Y/Z"):
        gates.pauli_rotation("ZI", [0, 1], 0.1)


def test_unknown_character_is_rejected():
    with pytest.raises(ValueError, match="X/Y/Z"):
        gates.pauli_rotation("ZA", [0, 1], 0.1)
    with pytest.raises(ValueError, match="X/Y/Z"):
        gates.pauli_rotation("z", [0], 0.1)


def test_duplicate_qubits_are_rejected():
    with pytest.raises(ValueError, match="distinct"):
        gates.pauli_rotation("ZZ", [1, 1], 0.1)
    with pytest.raises(ValueError, match="distinct"):
        Circuit(4).pauli_rotation("XYZ", [0, 2, 0], 0.1)


def test_non_integer_qubit_is_a_type_error():
    with pytest.raises(TypeError):
        gates.pauli_rotation("Z", [0.5], 0.1)


def test_out_of_range_qubit_is_rejected_at_append():
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.append(gates.pauli_rotation("ZZ", [0, 5], 0.1))
    with pytest.raises(ValueError, match="out of range"):
        c.pauli_rotation("ZZ", [0, 5], 0.1)
    # Nothing was appended by the failing calls.
    assert len(c) == 0

"""``noise.pauli_channel`` / ``noise.depolarize2`` — general Pauli noise.

The core scale factors are pinned in
``crates/paulistrings/src/channel/noise.rs``; this file checks the Python
boundary — argument order, the broadcast `Circuit` methods, validation, and that
the factors survive a real propagate through the engine's key-preserving rescale
path.

Hand-derived duals (each Pauli anticommutes with exactly the other two):

    pauli_channel:  I → 1,  X → 1 - 2(py+pz),  Y → 1 - 2(px+pz),  Z → 1 - 2(px+py)
    depolarize2:    identity on the pair → 1,  anything else → 1 - 16p/15

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import pytest

from paulistrings import Circuit, PauliSum, noise


TOL = 1e-12


def _sum(num_qubits, terms):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


def _as_dict(sum_):
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
    chars = ["I"] * num_qubits
    for q, ch in ops.items():
        chars[q] = ch
    return "".join(chars)


def _one_channel(num_qubits, channel):
    c = Circuit(num_qubits)
    c.append(channel)
    return c


# ---- pauli_channel scale factors ----


def test_pauli_channel_scale_factors_are_hand_computed():
    # (px, py, pz) = (0.1, 0.2, 0.3):
    #   I → 1
    #   X → 1 - 2(0.2 + 0.3) = 0      (exactly zero, so the merge drops the term)
    #   Y → 1 - 2(0.1 + 0.3) = 0.2
    #   Z → 1 - 2(0.1 + 0.2) = 0.4
    ch = noise.pauli_channel(0.1, 0.2, 0.3, 0)
    c = _one_channel(1, ch)

    assert len(_sum(1, {"X": 1.0}).propagate(circuit=c, direction="forward")) == 0

    for pauli, want in (("I", 1.0), ("Y", 0.2), ("Z", 0.4)):
        out = _sum(1, {pauli: 1.0}).propagate(circuit=c, direction="forward")
        assert len(out) == 1, pauli
        assert abs(out.coefficients()[0] - complex(want, 0.0)) < TOL, pauli


def test_pauli_channel_only_touches_its_support_qubit():
    ch = noise.pauli_channel(0.1, 0.2, 0.3, 1)
    c = _one_channel(3, ch)
    # Z on qubit 1 is scaled by 0.4; the Y on qubit 0 and the X on qubit 2 are
    # off-support and do not enter the factor.
    out = _sum(3, {"YZX": 1.0}).propagate(circuit=c, direction="forward")
    _assert_close(out, _sum(3, {"YZX": 0.4}))
    # A term that is the identity on qubit 1 is untouched.
    out = _sum(3, {"YIX": 1.0}).propagate(circuit=c, direction="forward")
    _assert_close(out, _sum(3, {"YIX": 1.0}))


def test_pauli_channel_at_a_word_boundary():
    n = 70
    ch = noise.pauli_channel(0.1, 0.2, 0.3, 64)
    c = _one_channel(n, ch)
    out = _sum(n, {_string(n, {64: "Z"}): 1.0}).propagate(circuit=c, direction="forward")
    _assert_close(out, _sum(n, {_string(n, {64: "Z"}): 0.4}))


def test_pauli_channel_is_self_adjoint_through_propagate():
    ch = noise.pauli_channel(0.05, 0.15, 0.25, 0)
    c = _one_channel(2, ch)
    initial = _sum(2, {"ZI": 1.0, "YX": 0.5, "II": 2.0})
    fwd = initial.propagate(circuit=c, direction="forward")
    back = initial.propagate(circuit=c, direction="heisenberg")
    _assert_close(fwd, back)


# ---- the two consistency identities ----


def test_uniform_pauli_channel_equals_depolarize():
    # pauli_channel(p/3, p/3, p/3) ≡ depolarize(p): 1 - 2(2p/3) = 1 - 4p/3.
    p = 0.42
    initial = _sum(3, {"XII": 1.0, "IYI": 0.5, "ZZI": -0.25, "III": 2.0})
    via_pauli = _one_channel(3, noise.pauli_channel(p / 3, p / 3, p / 3, 0))
    via_depol = _one_channel(3, noise.depolarize(p, 0))
    _assert_close(
        initial.propagate(circuit=via_pauli, direction="forward"),
        initial.propagate(circuit=via_depol, direction="forward"),
    )


def test_z_only_pauli_channel_equals_dephase():
    # pauli_channel(0, 0, p) ≡ dephase(p): X and Y take 1 - 2p, Z takes 1.
    p = 0.37
    initial = _sum(3, {"XII": 1.0, "YIZ": 0.5, "ZZI": -0.25, "III": 2.0})
    via_pauli = _one_channel(3, noise.pauli_channel(0.0, 0.0, p, 0))
    via_dephase = _one_channel(3, noise.dephase(p, 0))
    _assert_close(
        initial.propagate(circuit=via_pauli, direction="forward"),
        initial.propagate(circuit=via_dephase, direction="forward"),
    )


# ---- depolarize2 ----


def test_depolarize2_scale_factor_is_hand_computed():
    # p = 0.3 → 1 - 16(0.3)/15 = 1 - 0.32 = 0.68, for every Pauli that is
    # non-identity somewhere on the pair — weight 1 on the pair included, which
    # is the easy thing to get wrong.
    c = _one_channel(3, noise.depolarize2(0.3, 0, 1))
    for label in ("XII", "IZI", "YXI", "ZZI", "XYI"):
        out = _sum(3, {label: 1.0}).propagate(circuit=c, direction="forward")
        _assert_close(out, _sum(3, {label: 0.68}))
    # Identity on the pair: untouched, whatever happens off the pair.
    for label in ("III", "IIZ", "IIY"):
        out = _sum(3, {label: 1.0}).propagate(circuit=c, direction="forward")
        _assert_close(out, _sum(3, {label: 1.0}))


def test_depolarize2_at_fifteen_sixteenths_annihilates_the_pair():
    # 1 - 16·(15/16)/15 = 0 exactly in binary floating point, so the term is
    # dropped by the merge, while the identity on the pair is preserved.
    c = _one_channel(2, noise.depolarize2(15.0 / 16.0, 0, 1))
    assert len(_sum(2, {"XZ": 1.0}).propagate(circuit=c, direction="forward")) == 0
    out = _sum(2, {"II": 2.0}).propagate(circuit=c, direction="forward")
    _assert_close(out, _sum(2, {"II": 2.0}))


def test_depolarize2_across_a_word_boundary():
    n = 70
    c = _one_channel(n, noise.depolarize2(0.3, 63, 64))
    for ops in ({63: "X"}, {64: "Z"}, {63: "Y", 64: "Y"}):
        label = _string(n, ops)
        out = _sum(n, {label: 1.0}).propagate(circuit=c, direction="forward")
        _assert_close(out, _sum(n, {label: 0.68}))
    # Qubit 62 is not in the pair.
    label = _string(n, {62: "X"})
    out = _sum(n, {label: 1.0}).propagate(circuit=c, direction="forward")
    _assert_close(out, _sum(n, {label: 1.0}))


def test_depolarize2_is_self_adjoint_through_propagate():
    c = _one_channel(2, noise.depolarize2(0.2, 0, 1))
    initial = _sum(2, {"XZ": 1.0, "II": 2.0, "IY": 0.5})
    _assert_close(
        initial.propagate(circuit=c, direction="forward"),
        initial.propagate(circuit=c, direction="heisenberg"),
    )


# ---- broadcast Circuit methods ----


def test_circuit_pauli_channel_broadcasts_one_channel_per_qubit():
    c = Circuit(4)
    c.pauli_channel(0.01, 0.02, 0.03, [0, 1, 3])
    assert len(c) == 3


def test_circuit_depolarize2_broadcasts_one_channel_per_pair():
    c = Circuit(5)
    c.depolarize2(0.05, [(0, 1), (2, 3)])
    assert len(c) == 2


def test_broadcast_methods_agree_with_the_factories():
    initial = _sum(3, {"XZI": 1.0, "IIY": 0.5, "III": 2.0})
    via_methods = Circuit(3)
    via_methods.pauli_channel(0.1, 0.2, 0.05, [0, 2])
    via_methods.depolarize2(0.3, [(0, 1)])

    via_factories = Circuit(3)
    via_factories.append(noise.pauli_channel(0.1, 0.2, 0.05, 0))
    via_factories.append(noise.pauli_channel(0.1, 0.2, 0.05, 2))
    via_factories.append(noise.depolarize2(0.3, 0, 1))

    _assert_close(
        initial.propagate(circuit=via_methods, direction="forward"),
        initial.propagate(circuit=via_factories, direction="forward"),
    )


# ---- validation ----


@pytest.mark.parametrize(
    "args",
    [(-0.1, 0.1, 0.1), (0.1, -0.1, 0.1), (0.1, 0.1, -0.1)],
)
def test_pauli_channel_rejects_a_negative_probability(args):
    with pytest.raises(ValueError, match="non-negative"):
        noise.pauli_channel(*args, 0)


def test_pauli_channel_rejects_probabilities_summing_above_one():
    with pytest.raises(ValueError, match="at most 1"):
        noise.pauli_channel(0.5, 0.4, 0.2, 0)
    with pytest.raises(ValueError, match="at most 1"):
        Circuit(2).pauli_channel(0.5, 0.4, 0.2, [0])


def test_pauli_channel_accepts_the_boundary_sum_of_one():
    # Exactly-representable halves and quarters, so the boundary is not a
    # floating-point coin flip: 0.5 + 0.25 + 0.25 == 1.0.
    ch = noise.pauli_channel(0.5, 0.25, 0.25, 0)
    assert ch is not None


@pytest.mark.parametrize("p", [-0.01, 1.5])
def test_depolarize2_rejects_a_probability_outside_the_unit_interval(p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        noise.depolarize2(p, 0, 1)


def test_depolarize2_rejects_a_repeated_qubit():
    with pytest.raises(ValueError, match="must differ"):
        noise.depolarize2(0.1, 2, 2)
    with pytest.raises(ValueError, match="must differ"):
        Circuit(4).depolarize2(0.1, [(1, 1)])


def test_out_of_range_qubits_are_rejected_at_append():
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.append(noise.pauli_channel(0.1, 0.1, 0.1, 5))
    with pytest.raises(ValueError, match="out of range"):
        c.append(noise.depolarize2(0.1, 0, 5))
    with pytest.raises(ValueError, match="out of range"):
        c.pauli_channel(0.1, 0.1, 0.1, [5])
    with pytest.raises(ValueError, match="out of range"):
        c.depolarize2(0.1, [(0, 5)])
    assert len(c) == 0

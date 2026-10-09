"""``target_bucket_len=`` / ``min_buckets=`` on the propagate surface (``PropagateOptions``)."""

import random

from paulistrings import Circuit, PauliSum


TOL = 1e-12

NUM_QUBITS = 8

_WINDOW = 6


def _seeded_circuit(num_qubits, layers=4, seed=20260901):
    """A deterministic mixed circuit: Cliffords, generic-angle rotations, a
    multi-qubit generator, two-qubit entanglers and a noise channel."""
    rng = random.Random(seed)
    c = Circuit(num_qubits)
    for _ in range(layers):
        for q in range(_WINDOW):
            getattr(c, rng.choice(("h", "s", "sdg", "x", "y", "z")))(q)
        for q in range(_WINDOW):
            getattr(c, rng.choice(("rz", "rx", "ry")))(rng.uniform(0.1, 1.4), q)
        for q in range(_WINDOW - 1):
            getattr(c, rng.choice(("cnot", "cz", "swap")))(q, q + 1)
        c.pauli_rotation("XYZ", [0, 2, 4], rng.uniform(0.1, 0.9))
        c.depolarize(0.02, [rng.randrange(_WINDOW)])
    return c


def _observable(num_qubits):
    """``X₀ + Z₁Z₂``."""
    x0 = "X" + "I" * (num_qubits - 1)
    zz = "I" + "ZZ" + "I" * (num_qubits - 3)
    return PauliSum.from_strings({x0: 1.0, zz: 0.5}, num_qubits=num_qubits)


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons do not depend on ordering."""
    return {
        (tuple(int(w) for w in xr), tuple(int(w) for w in zr)): cc
        for xr, zr, cc in zip(sum_.x_array(), sum_.z_array(), sum_.coefficients())
    }


def _assert_terms_close(got, want, tol=TOL):
    """Same keys, coefficients agreeing to ``tol``.

    Agreement to floating-point tolerance is the correctness bar (CLAUDE.md
    §Determinism policy).
    """
    g, w = _as_dict(got), _as_dict(want)
    assert g.keys() == w.keys()
    for key, want_c in w.items():
        assert abs(g[key] - want_c) < tol, f"{key}: {g[key]} != {want_c}"


def _fanout_circuit(num_qubits, window=8, layers=10, seed=20260914):
    """Enough generic-angle rotations on a `window`-qubit block to blow a
    single starting term up past the default `min_buckets * 64 = 8192` split
    threshold (`pauli_sum/storage.rs`), so the sorting engine actually rebuckets
    partway through the run rather than starting there."""
    rng = random.Random(seed)
    c = Circuit(num_qubits)
    for _ in range(layers):
        for q in range(window):
            getattr(c, rng.choice(("h", "s", "sdg", "x", "y", "z")))(q)
        for q in range(window):
            getattr(c, rng.choice(("rz", "rx", "ry")))(rng.uniform(0.1, 1.4), q)
        for q in range(window - 1):
            getattr(c, rng.choice(("cnot", "cz", "swap")))(q, q + 1)
    return c


def test_target_bucket_len_and_min_buckets_realize_a_small_partition():
    """Starting from one term, `_fanout_circuit` grows the sum past the
    default split threshold, so the defaults (`target_bucket_len=1024`,
    `min_buckets=128`) land in many buckets.  `min_buckets=1` paired with a
    huge `target_bucket_len` keeps `desired_bits` at its floor of zero
    (`pauli_sum/storage.rs::desired_bits`) regardless of term count, so the realized
    partition is a single bucket throughout — nothing in the engine enforces
    the `min_buckets >= 16` documentation contract as a code-level clamp, so
    `min_buckets=1` is honoured exactly, not rounded up.
    """
    num_qubits = 20
    s = PauliSum.from_strings({"Z" + "I" * (num_qubits - 1): 1.0}, num_qubits=num_qubits)
    assert s.num_buckets == 1
    c = _fanout_circuit(num_qubits)

    default = s.propagate(c)
    assert len(default) > 8192, "the fanout circuit must actually cross the split threshold"
    assert default.num_buckets > 1

    minimal = s.propagate(c, min_buckets=1, target_bucket_len=1 << 30)
    assert minimal.num_buckets == 1
    _assert_terms_close(minimal, default)


def test_target_bucket_len_and_min_buckets_default_to_todays_behaviour():
    # `None`/`None` (the default) must be exactly `PropagateOptions::default()`,
    # like every other additive kwarg on this surface.
    s, c = _observable(NUM_QUBITS), _seeded_circuit(NUM_QUBITS)
    assert _as_dict(s.propagate(c)) == _as_dict(
        s.propagate(c, target_bucket_len=None, min_buckets=None)
    )


def test_num_buckets_is_one_on_a_freshly_built_small_sum():
    # `BuildAccumulator::finalize` sizes the initial partition the same way
    # (`pauli_sum/accumulator.rs`), so an untouched small sum starts at a single bucket.
    s = _observable(NUM_QUBITS)
    assert s.num_buckets == 1

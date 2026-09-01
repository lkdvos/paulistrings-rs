"""``engine=`` / ``small_sum_threshold=`` on the propagate surface.

The Rust side is covered by ``crates/paulistrings/tests/small_sum_path.rs``,
which pins the two engines against each other across the channel zoo, both
directions and a threshold sweep. This file checks the Python boundary: that the
kwargs exist with the documented spellings and defaults, that the default is
today's behaviour *exactly*, that the two engines agree on the terms and agree
**exactly** on the per-layer term counts, and that a bad spelling is a clean
``ValueError``.

Per-layer term-count equality is the property the cross-engine head-to-head
driver against ``PauliPropagation.jl`` gates on, so it is asserted on every
comparison here rather than only in its own test.

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import random

import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, truncation


TOL = 1e-12

# One per width band the bindings monomorphize, so the engine kwarg is exercised
# across the const-generic surface: 8 -> W=1, 68 -> W=2, 130 -> W=4. The circuit
# below only ever touches the first ``_WINDOW`` qubits, so the term count — and
# therefore which side of the threshold the sum sits on — is the same at all
# three, and only the word count differs.
WIDTHS = [8, 68, 130]

# Qubits the seeded circuit is allowed to touch. Small enough that the sum stays
# well under the default threshold, which is the regime the direct path targets.
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
    """``X₀ + Z₁Z₂`` — two terms, so the run starts on the direct path."""
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
    §Determinism policy): the two engines sum equal keys in different orders —
    map iteration order against the coset gather order — so the last bits are
    free to differ.
    """
    g, w = _as_dict(got), _as_dict(want)
    assert g.keys() == w.keys()
    for key, want_c in w.items():
        assert abs(g[key] - want_c) < tol, f"{key}: {g[key]} != {want_c}"


def _assert_counts_equal(got, want):
    """Per-layer term counts must be **equal**, not close."""
    assert got.layers == want.layers
    assert got.terms_in == want.terms_in
    assert got.terms_out == want.terms_out
    assert got.peak_terms == want.peak_terms
    assert got.final_terms == want.final_terms


# --------------------------------------------------------------------------
# The kwargs exist, and the defaults are today's behaviour


def test_default_small_sum_threshold_is_exported():
    assert paulistrings.DEFAULT_SMALL_SUM_THRESHOLD == 2048


@pytest.mark.parametrize("num_qubits", WIDTHS)
def test_omitting_the_kwargs_is_engine_sorted_exactly(num_qubits):
    # Not "close": the default has to be byte-for-byte the pre-existing path,
    # and `engine="sorted"` is the same code with the knob spelled out.
    s, c = _observable(num_qubits), _seeded_circuit(num_qubits)
    assert _as_dict(s.propagate(c)) == _as_dict(s.propagate(c, engine="sorted"))


def test_omitting_the_kwargs_is_engine_sorted_exactly_with_stats():
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    plain, plain_stats = s.propagate_with_stats(c)
    named, named_stats = s.propagate_with_stats(c, engine="sorted")
    assert _as_dict(plain) == _as_dict(named)
    _assert_counts_equal(plain_stats, named_stats)


def test_threshold_alone_does_not_leave_the_sorting_engine():
    # `small_sum_threshold` is ignored under the default engine, so passing one
    # cannot change the result.
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    assert _as_dict(s.propagate(c, small_sum_threshold=1 << 20)) == _as_dict(
        s.propagate(c)
    )


def test_the_kwargs_are_accepted_positionally_after_direction():
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    positional = s.propagate(c, None, "forward", "direct", 4096)
    keyword = s.propagate(c, engine="direct", small_sum_threshold=4096)
    assert _as_dict(positional) == _as_dict(keyword)


# --------------------------------------------------------------------------
# The two engines agree


@pytest.mark.parametrize("num_qubits", WIDTHS)
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
@pytest.mark.parametrize("engine", ["auto", "direct"])
def test_direct_matches_sorted(num_qubits, direction, engine):
    s, c = _observable(num_qubits), _seeded_circuit(num_qubits)
    want = s.propagate(c, direction=direction)
    got = s.propagate(c, direction=direction, engine=engine)
    assert len(got) == len(want)
    _assert_terms_close(got, want)


@pytest.mark.parametrize("engine", ["auto", "direct"])
def test_per_layer_term_counts_match_between_engines(engine):
    """The parity property the cross-engine driver gates on: the *whole*
    ``terms_in``/``terms_out`` vectors are equal, not close."""
    s, c = _observable(WIDTHS[1]), _seeded_circuit(WIDTHS[1])
    want, want_stats = s.propagate_with_stats(c)
    got, got_stats = s.propagate_with_stats(c, engine=engine)
    assert want_stats.layers == len(c)
    _assert_counts_equal(got_stats, want_stats)
    _assert_terms_close(got, want)


@pytest.mark.parametrize("threshold", [0, 1, 2, 3, 8, 33, 200, 1 << 20])
def test_threshold_sweep_puts_the_transition_on_every_layer(threshold):
    """Sweeping the threshold walks the small -> large handover across the
    circuit; the result and the per-layer counts must not care where it lands.

    ``0`` keeps the run on the sorting engine end to end (the sum starts at two
    terms) and ``1 << 20`` keeps it on the direct path end to end, so the sweep
    covers both undivided cases as well as every split one.
    """
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    want, want_stats = s.propagate_with_stats(c)
    got, got_stats = s.propagate_with_stats(
        c, engine="direct", small_sum_threshold=threshold
    )
    _assert_counts_equal(got_stats, want_stats)
    _assert_terms_close(got, want)


@pytest.mark.parametrize("engine", ["sorted", "auto", "direct"])
def test_a_finalizing_policy_agrees_on_every_engine(engine):
    """``topn`` has a layer pass, so ``auto`` declines the direct path and
    ``direct`` takes it anyway. Both must land on the same answer as ``sorted``.
    """
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    policy = truncation.topn(12)
    want, want_stats = s.propagate_with_stats(c, policy=policy)
    got, got_stats = s.propagate_with_stats(c, policy=policy, engine=engine)
    assert max(want_stats.terms_out) <= 12, "the policy has to actually bite"
    _assert_counts_equal(got_stats, want_stats)
    _assert_terms_close(got, want)


@pytest.mark.parametrize("engine", ["sorted", "auto", "direct"])
def test_a_per_term_policy_agrees_on_every_engine(engine):
    """``coeff`` filters per term and reports no layer pass, which is the case
    ``auto`` is meant to route onto the direct path."""
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    policy = truncation.coeff(1e-3)
    want, want_stats = s.propagate_with_stats(c, policy=policy)
    got, got_stats = s.propagate_with_stats(c, policy=policy, engine=engine)
    assert want_stats.final_terms < len(_observable(WIDTHS[0]).propagate(c))
    _assert_counts_equal(got_stats, want_stats)
    _assert_terms_close(got, want)


def test_an_empty_circuit_is_a_no_op_on_every_engine():
    s = _observable(WIDTHS[0])
    empty = Circuit(WIDTHS[0])
    for engine in ("sorted", "auto", "direct"):
        out = s.propagate(empty, engine=engine)
        assert _as_dict(out) == _as_dict(s)


# --------------------------------------------------------------------------
# Errors


def test_unknown_engine_raises_value_error():
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="engine must be 'sorted', 'auto', or 'direct'"):
        s.propagate(c, engine="bucketed")
    with pytest.raises(ValueError, match="engine must be 'sorted', 'auto', or 'direct'"):
        s.propagate_with_stats(c, engine="SORTED")


def test_a_negative_threshold_is_rejected():
    s, c = _observable(WIDTHS[0]), _seeded_circuit(WIDTHS[0])
    with pytest.raises(OverflowError):
        s.propagate(c, engine="direct", small_sum_threshold=-1)

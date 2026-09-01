"""``PauliSum.propagate_with_stats`` — per-layer term counts alongside the result.

The Rust side is covered by ``crates/paulistrings/tests/term_trace.rs`` (which
pins the same hand-computed counts on the same circuit); this file checks the
Python boundary: the tuple return, the ``PropagationStats`` getters, and that
enabling the trace does not change the propagated sum.

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import math

import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, truncation

# cos 0.3 = 0.955336..., sin 0.3 = 0.295520...: no count below rests on a
# cancellation.
THETA = 0.3
NUM_QUBITS = 4


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons do not depend on ordering."""
    x = sum_.x_array()
    z = sum_.z_array()
    c = sum_.coefficients()
    return {(tuple(xr), tuple(zr)): cc for xr, zr, cc in zip(x, z, c)}


def _x0():
    return PauliSum.from_strings({"XIII": 1.0}, num_qubits=NUM_QUBITS)


def _three_layer_circuit():
    """Term counts on ``{X₀: 1}`` are hand-computable:

    1. ``rz(θ, 0)``: X₀ anticommutes with the Z₀ generator and fans out to
       ``cos θ·X₀ − sin θ·Y₀``. 1 → 2 terms.
    2. ``h(0)``: X₀ → Z₀, Y₀ → −Y₀. A relabelling, 2 → 2 terms.
    3. ``rz(θ, 0)``: Z₀ commutes (stays one term), Y₀ fans out to Y₀ and X₀.
       Three distinct keys, so 2 → 3 terms.
    """
    c = Circuit(NUM_QUBITS)
    c.rz(THETA, 0)
    c.h(0)
    c.rz(THETA, 0)
    return c


def test_propagation_stats_is_exported():
    assert hasattr(paulistrings, "PropagationStats")


def test_returns_the_pair_and_the_same_sum_as_propagate():
    circuit = _three_layer_circuit()
    plain = _x0().propagate(circuit)

    result = _x0().propagate_with_stats(circuit)
    assert isinstance(result, tuple) and len(result) == 2
    evolved, stats = result
    assert isinstance(evolved, PauliSum)
    assert isinstance(stats, paulistrings.PropagationStats)

    got, want = _as_dict(evolved), _as_dict(plain)
    assert got.keys() == want.keys()
    for key, coeff in want.items():
        assert abs(got[key] - coeff) < 1e-15


def test_hand_computed_counts():
    evolved, stats = _x0().propagate_with_stats(_three_layer_circuit())

    assert stats.layers == 3
    assert stats.terms_in == [1, 2, 2]
    assert stats.terms_out == [2, 2, 3]
    assert stats.peak_terms == 3
    assert stats.final_terms == 3
    assert stats.final_terms == len(evolved)
    assert stats.peak_terms >= stats.final_terms


def test_counts_are_post_truncation():
    # Layer 3's third key is X₀ with coefficient sin²θ·cos θ = 0.0873..., which
    # a 0.1 threshold drops (|c| <= eps is dropped); the surviving keys carry
    # 0.9553... (Z₀) and 0.2823... (Y₀).
    evolved, stats = _x0().propagate_with_stats(
        _three_layer_circuit(), policy=truncation.coeff(0.1)
    )

    assert stats.terms_in == [1, 2, 2]
    assert stats.terms_out == [2, 2, 2]
    assert stats.peak_terms == 2
    assert stats.final_terms == 2 == len(evolved)

    survivors = sorted(abs(c) for c in evolved.coefficients())
    assert survivors == pytest.approx(
        [math.sin(THETA) * math.cos(THETA), math.cos(THETA)], abs=1e-12
    )


def test_empty_circuit_reports_zero_layers():
    source = _x0()
    evolved, stats = source.propagate_with_stats(Circuit(NUM_QUBITS))

    assert stats.layers == 0
    assert stats.terms_in == []
    assert stats.terms_out == []
    # No layer ran, so the resident count never changed from the input's.
    assert stats.peak_terms == len(source) == 1
    assert stats.final_terms == len(evolved) == 1


def test_heisenberg_direction_matches_propagate():
    circuit = _three_layer_circuit()
    plain = _x0().propagate(circuit, direction="heisenberg")
    evolved, stats = _x0().propagate_with_stats(circuit, direction="heisenberg")

    assert stats.layers == 3
    assert stats.final_terms == len(evolved) == len(plain)
    got, want = _as_dict(evolved), _as_dict(plain)
    assert got.keys() == want.keys()
    for key, coeff in want.items():
        assert abs(got[key] - coeff) < 1e-15


def test_terms_in_chains_with_terms_out():
    _, stats = _x0().propagate_with_stats(_three_layer_circuit())
    assert stats.terms_in[1:] == stats.terms_out[:-1]


def test_bad_direction_rejected():
    with pytest.raises(ValueError, match="direction"):
        _x0().propagate_with_stats(_three_layer_circuit(), direction="backwards")


def test_qubit_count_mismatch_rejected():
    with pytest.raises(ValueError, match="num_qubits"):
        _x0().propagate_with_stats(Circuit(NUM_QUBITS + 1))


def test_stats_repr_names_all_fields():
    circuit = Circuit(NUM_QUBITS)
    circuit.h(0)
    _, stats = _x0().propagate_with_stats(circuit)
    text = repr(stats)
    assert text.startswith("PropagationStats(")
    for field in ("layers=", "terms_in=", "terms_out=", "peak_terms=", "final_terms="):
        assert field in text
    assert f"layers={stats.layers}" in text
    assert f"final_terms={stats.final_terms}" in text
    # One layer, `h(0)` relabelling X₀ → Z₀, so every count is 1: the whole
    # format is pinned, not just the field names.
    assert text == (
        "PropagationStats(layers=1, terms_in=[1], terms_out=[1], "
        "peak_terms=1, final_terms=1)"
    )

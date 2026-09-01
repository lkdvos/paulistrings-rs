"""Truncation factories + ``&`` / ``|`` composition.

Pin the policy semantics by running ``propagate`` through circuits whose
output coefficients straddle the cutoff, and check both the simple factories
and the ``And`` / ``Or`` combinators thread through. ``approx_topn``'s
approximate-``n`` contract — at most ``n``, short of it by at most the coarsest
excluded octave's population, tie groups whole, single-octave wipe — is pinned
against hand-computed octave arithmetic and against ``topn`` on the same sums.

Note: the truncation policy only fires inside the engine's merge / finalize
passes, which only run when at least one channel is in the circuit. These
tests use ``z(0)`` as a "policy probe" layer — Pauli-Z conjugation flips signs
on X/Y but preserves magnitudes and weights, so it doesn't perturb the
truncation arithmetic.
"""

import math

import pytest

from paulistrings import Circuit, PauliSum, gates, truncation


TOL = 1e-12


def _probe_circuit(num_qubits):
    """Single Z(0) layer — engages the merge/finalize without changing
    weights or magnitudes."""
    c = Circuit(num_qubits)
    c.z(0)
    return c


def test_coefficient_threshold_drops_subthreshold_terms():
    # rz(π/2) X = cos(π/2)·X + i sin(π/2)·X·Z = 0·X + Y. cos(π/2) is a small
    # FP residue (~6e-17), so without a threshold both terms survive.
    s = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    c = Circuit(1)
    c.rz(math.pi / 2, 0)

    no_policy = s.propagate(c)
    # Both X (residue) and +Y (≈ +1) survive when no policy filters them.
    assert len(no_policy.coefficients()) == 2

    # With coeff(0.5), the X residue ≪ 0.5 is dropped; +Y is kept.
    out = s.propagate(c, policy=truncation.coeff(0.5))
    assert len(out.coefficients()) == 1
    (only,) = out.coefficients()
    assert abs(only - (1 + 0j)) < TOL


def test_weight_cutoff_drops_higher_weight_terms():
    # Build a 2-qubit sum with weights 0/1/2 mixed: II (w=0), XI (w=1),
    # XX (w=2). Cap weight at 1 → drop XX.
    s = PauliSum.from_strings(
        {"II": 1.0, "XI": 2.0, "XX": 3.0},
        num_qubits=2,
    )
    out = s.propagate(_probe_circuit(2), policy=truncation.weight(1))
    coeffs = out.coefficients()
    assert len(coeffs) == 2
    # Magnitudes 1.0 (II) and 2.0 (XI). XX (3.0) dropped.
    mags = sorted(abs(c) for c in coeffs)
    assert mags == [1.0, 2.0]


def test_topn_keeps_largest_after_layer():
    s = PauliSum.from_strings(
        {"II": 0.1, "XI": 0.5, "ZI": 1.0, "XX": 0.2},
        num_qubits=2,
    )
    out = s.propagate(_probe_circuit(2), policy=truncation.topn(2))
    coeffs = out.coefficients()
    assert len(coeffs) == 2
    mags = sorted((abs(c) for c in coeffs), reverse=True)
    assert mags == [1.0, 0.5]


def _mags(sum_):
    return sorted((abs(c) for c in sum_.coefficients()), reverse=True)


def test_approx_topn_matches_topn_on_a_well_separated_spectrum():
    # Magnitudes 1, 1/4, 1/16, 1/64 square to 1, 1/16, 1/256, 1/4096 — one per
    # octave of |c|², four octaves apart — so every prefix is an octave
    # boundary and the histogram threshold lands exactly where TopN's does.
    s = PauliSum.from_strings(
        {"II": 1.0, "XI": 0.25, "ZI": 0.0625, "XX": 0.015625},
        num_qubits=2,
    )
    for n in (1, 2, 3, 4):
        approx = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(n))
        exact = s.propagate(_probe_circuit(2), policy=truncation.topn(n))
        assert len(approx.coefficients()) == n
        assert _mags(approx) == pytest.approx(_mags(exact))


# |c|² of 1.0, 0.9, 0.7, 0.6, 0.5 is 1.0, 0.81, 0.49, 0.36, 0.25: the top two
# get an octave each ([1, 2) and [0.5, 1)) and the bottom three share [0.25, 0.5)
# without being a tie group — which is exactly where the two policies part ways.
_CLUSTERED = {"II": 1.0, "XI": 0.9, "ZI": 0.7, "XX": 0.6, "YI": 0.5}


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_approx_topn_never_keeps_more_than_n(n):
    s = PauliSum.from_strings(_CLUSTERED, num_qubits=2)
    out = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(n))
    assert len(out.coefficients()) <= n


def test_approx_topn_shortfall_is_the_coarsest_excluded_octaves_population():
    # n = 4, hand-computed on `_CLUSTERED`. The octaves from the top hold 1, 1
    # and 3 terms, so the running counts are S = 1, 2, 5. Five overshoots 4, so
    # the cut is the octave above it: two terms kept, the three-term octave
    # dropped whole.
    #
    # TopN(4) keeps four — 0.6 is the 4th largest and nothing ties it — so this
    # is a case where the two genuinely differ, and the shortfall 4 - 2 = 2 is
    # inside the documented bound: kept > n - p with p = 3.
    s = PauliSum.from_strings(_CLUSTERED, num_qubits=2)
    approx = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(4))
    exact = s.propagate(_probe_circuit(2), policy=truncation.topn(4))
    assert _mags(approx) == pytest.approx([1.0, 0.9])
    assert _mags(exact) == pytest.approx([1.0, 0.9, 0.7, 0.6])
    kept, n, p = len(approx.coefficients()), 4, 3
    assert kept <= n
    assert kept > n - p


def test_approx_topn_keeps_a_tie_group_whole():
    # The three 0.5s share an octave, so they are kept or dropped together
    # whatever n is — no tie rule needed, unlike TopN.
    s = PauliSum.from_strings(
        {"II": 1.0, "XI": 0.5, "ZI": 0.5, "XX": 0.5},
        num_qubits=2,
    )
    for n in range(5):
        out = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(n))
        kept_ties = sum(1 for m in _mags(out) if abs(m - 0.5) < TOL)
        assert kept_ties in (0, 3), f"n={n} split the multiplet: {_mags(out)}"


def test_approx_topn_wipes_a_single_octave_sum():
    # 1.0, 1.1, 1.2, 1.3 square into [1, 2) together: one octave, so S is 0 or
    # 4 and neither fits in 3. The degenerate case, documented on the factory.
    s = PauliSum.from_strings(
        {"II": 1.0, "XI": 1.1, "ZI": 1.2, "XX": 1.3},
        num_qubits=2,
    )
    out = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(3))
    assert out.coefficients() == []
    # ... and pairing it with a coefficient threshold is the documented escape
    # hatch only insofar as it removes candidates; with n >= len nothing is cut.
    kept = s.propagate(_probe_circuit(2), policy=truncation.approx_topn(4))
    assert len(kept.coefficients()) == 4


def test_approx_topn_composes_with_the_operators():
    s = PauliSum.from_strings(
        {"II": 1.0, "XI": 0.25, "ZI": 0.0625, "XX": 0.015625},
        num_qubits=2,
    )
    # And: the weight cutoff drops XX per term, then approx_topn(2) cuts the
    # rest to the top two octaves.
    out = s.propagate(
        _probe_circuit(2), policy=truncation.approx_topn(2) & truncation.weight(1)
    )
    assert _mags(out) == pytest.approx([1.0, 0.25])
    # Or keeps a term if either arm does, and both TopN flavours' *per-term*
    # predicate is unconditionally true — they work in the layer pass, which
    # `Or` does not run (matching `builtin::Or`). So an `Or` containing one
    # keeps everything, exactly as it does with `topn`.
    ored = s.propagate(
        _probe_circuit(2), policy=truncation.approx_topn(1) | truncation.weight(1)
    )
    assert len(ored.coefficients()) == 4
    assert len(
        s.propagate(
            _probe_circuit(2), policy=truncation.topn(1) | truncation.weight(1)
        ).coefficients()
    ) == 4


def test_and_combinator_requires_both_to_keep():
    # And(coeff(0.3), weight(1)): keep only terms with |c|>0.3 AND weight≤1.
    s = PauliSum.from_strings(
        {
            "II": 1.0,  # |c|=1, w=0 → kept
            "XI": 0.4,  # |c|=0.4, w=1 → kept
            "ZI": 0.1,  # |c|=0.1 → dropped (coeff fails)
            "XX": 1.0,  # w=2 → dropped (weight fails)
        },
        num_qubits=2,
    )
    policy = truncation.coeff(0.3) & truncation.weight(1)
    out = s.propagate(_probe_circuit(2), policy=policy)
    coeffs = out.coefficients()
    assert len(coeffs) == 2
    mags = sorted(abs(c) for c in coeffs)
    assert mags == pytest.approx([0.4, 1.0])


def test_or_combinator_keeps_if_either_passes():
    # Or(coeff(0.5), weight(0)): keep terms with |c|>0.5 OR weight==0.
    s = PauliSum.from_strings(
        {
            "II": 0.1,  # weight=0 passes via Or
            "XI": 1.0,  # |c|=1 passes via Or
            "ZI": 0.1,  # both fail → dropped
        },
        num_qubits=2,
    )
    policy = truncation.coeff(0.5) | truncation.weight(0)
    out = s.propagate(_probe_circuit(2), policy=policy)
    coeffs = out.coefficients()
    assert len(coeffs) == 2
    mags = sorted(abs(c) for c in coeffs)
    assert mags == pytest.approx([0.1, 1.0])


def test_nested_and_or_composition():
    # (coeff(0.2) & weight(1)) | weight(0):
    #   keep if (|c|>0.2 AND weight≤1) OR weight==0.
    s = PauliSum.from_strings(
        {
            "II": 0.05,  # weight 0 → passes via the right OR-arm
            "XI": 0.3,   # weight 1 + |c|=0.3 → passes via the AND-arm
            "ZI": 0.1,   # weight 1 but |c|=0.1 → both arms fail
            "XX": 0.5,   # weight 2 → AND fails (weight>1), OR fails (weight≠0)
        },
        num_qubits=2,
    )
    policy = (truncation.coeff(0.2) & truncation.weight(1)) | truncation.weight(0)
    out = s.propagate(_probe_circuit(2), policy=policy)
    coeffs = out.coefficients()
    assert len(coeffs) == 2
    mags = sorted(abs(c) for c in coeffs)
    assert mags == pytest.approx([0.05, 0.3])


def test_coeff_factory_is_a_truncation_object():
    # __and__ / __or__ produce a Truncation that's still composable.
    a = truncation.coeff(0.1)
    b = truncation.weight(2)
    composed = a & b
    # Should behave like a Truncation under further composition.
    deeper = composed | truncation.topn(5)
    assert deeper is not None

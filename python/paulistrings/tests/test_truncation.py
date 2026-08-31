"""Truncation factories + ``&`` / ``|`` composition.

Pin the policy semantics by running ``propagate`` through circuits whose
output coefficients straddle the cutoff, and check both the simple factories
and the ``And`` / ``Or`` combinators thread through.

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

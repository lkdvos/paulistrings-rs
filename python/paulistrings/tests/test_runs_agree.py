"""``assert_runs_agree`` accepts a different choice among boundary ties and rejects a wrong run."""

import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, truncation

from ._runs_agree import (
    _terms,
    assert_runs_agree,
    assert_runs_agree_count_only,
    discarded_norm,
    heavy_cut,
    spread,
)

# Ten distinct magnitudes from 8 down, ten tied at 0.6, ten from 0.1 down.
MAGNITUDES = [8.0 - 0.5 * k for k in range(10)] + [0.6] * 10 + [0.1 - 0.005 * k for k in range(10)]


def _sum(terms):
    """Term ``(k, m)`` is ``X`` on the bits of ``k + 1`` with coefficient ``m·e^{ik}``."""
    x = np.array([[k + 1] for k, _ in terms], dtype=np.uint64)
    z = np.zeros_like(x)
    coeffs = np.array([m * np.exp(1j * k) for k, m in terms])
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=16)


def _kept(indices):
    return _sum([(k, MAGNITUDES[k]) for k in indices])


WANT = list(range(15))
# `topn(15)` of all thirty terms drops five ties and the tail.
DISCARDED = math.sqrt(sum(m * m for m in MAGNITUDES[15:]))
TOP, APPROX = truncation.topn(15), truncation.coeff(1e-9) & truncation.approx_topn(15)


@pytest.mark.parametrize("policy", [TOP, APPROX], ids=["topn", "approx_topn"])
def test_a_different_choice_of_boundary_ties_passes(policy):
    got = _kept(list(range(10)) + list(range(15, 20)))
    assert_runs_agree(got, _kept(WANT), policy, DISCARDED, 1e-12)


def test_a_sign_flipped_on_a_large_term_fails():
    terms = [(k, MAGNITUDES[k]) for k in range(14)] + [(19, 0.6)]
    terms[0] = (0, -8.0)
    with pytest.raises(AssertionError, match="exceeds"):
        assert_runs_agree(_sum(terms), _kept(WANT), TOP, DISCARDED, 1e-12)


def test_a_large_term_dropped_fails():
    with pytest.raises(AssertionError, match="exceeds"):
        assert_runs_agree(_kept(range(1, 16)), _kept(WANT), TOP, DISCARDED, 1e-12)


def test_a_wrong_boundary_coefficient_fails():
    terms = [(k, MAGNITUDES[k]) for k in range(10)] + [(k, 0.6) for k in range(15, 20)]
    terms[14] = (19, 5.6)
    with pytest.raises(AssertionError, match="exceeds"):
        assert_runs_agree(_sum(terms), _kept(WANT), TOP, DISCARDED, 1e-12)


def test_a_different_count_fails():
    with pytest.raises(AssertionError, match="term count"):
        assert_runs_agree(_kept(range(14)), _kept(WANT), TOP, DISCARDED, 1e-12)


def test_different_keys_without_a_layer_pass_fail():
    got = _kept(list(range(10)) + list(range(15, 20)))
    with pytest.raises(AssertionError, match="different keys"):
        assert_runs_agree(got, _kept(WANT), truncation.coeff(1e-9), DISCARDED, 1e-12)


def test_a_too_heavy_cut_fails():
    with pytest.raises(AssertionError, match="too heavy"):
        assert_runs_agree(_kept(range(1, 16)), _kept(WANT), TOP, 10.0, 1e-12)


def test_discarded_norm_sums_what_each_layer_dropped():
    one = Circuit(16)
    one.h(0)
    terms = [(k, MAGNITUDES[k]) for k in range(30)]
    assert discarded_norm(_sum(terms), one, TOP) == pytest.approx(DISCARDED, rel=1e-12)


@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_different_tie_choices_over_many_layers_pass(direction):
    """A 32-layer Trotter circuit over four magnitudes spread across octaves: two bucket counts keep different survivors."""
    rng = np.random.default_rng(7)
    x = rng.integers(0, 1 << 16, size=(1_500, 1), dtype=np.uint64)
    z = rng.integers(0, 1 << 16, size=(1_500, 1), dtype=np.uint64)
    s = PauliSum.from_arrays(x, z, spread(np.tile([1.0, 0.5, 0.25, 0.125], 375), rng), num_qubits=16)
    c = Circuit(16)
    for q in range(16):
        c.pauli_rotation("ZZ", [q, (q + 1) % 16], 0.02)
    for q in range(16):
        c.rx(0.02, q)
    policy = truncation.topn(2_000)
    want = s.propagate(c, policy, direction=direction)
    got = s.propagate(c, policy, direction=direction, target_bucket_len=16, min_buckets=256)
    assert set(_terms(got)) != set(_terms(want)), "the bucket counts should choose different ties"
    assert_runs_agree(got, want, policy, discarded_norm(s, c, policy, direction), 1e-12)


def _heavy_runs(policy):
    """``heavy_cut`` at two bucket counts, and the reference's ``Δ``."""
    c, s = heavy_cut(12)
    want = s.propagate(c, policy)
    got = s.propagate(c, policy, target_bucket_len=16, min_buckets=256)
    return got, want, discarded_norm(s, c, policy)


@pytest.mark.parametrize("policy", [truncation.topn(400), truncation.approx_topn(400)], ids=["topn", "approx_topn"])
def test_a_heavy_cut_compares_counts_only(policy):
    got, want, discarded = _heavy_runs(policy)
    assert_runs_agree_count_only(got, want, policy, discarded)
    with pytest.raises(AssertionError, match="too heavy"):
        assert_runs_agree(got, want, policy, discarded, 1e-12)


def test_a_light_cut_is_refused_by_the_count_only_check():
    got = _kept(list(range(10)) + list(range(15, 20)))
    with pytest.raises(AssertionError, match="light enough"):
        assert_runs_agree_count_only(got, _kept(WANT), TOP, DISCARDED)


def test_a_count_below_n_fails():
    got, want, discarded = _heavy_runs(truncation.topn(400))
    with pytest.raises(AssertionError, match="the last layer must cut"):
        assert_runs_agree_count_only(got, want, truncation.topn(401), discarded)


def test_a_count_only_check_needs_a_bare_cut():
    got, want, discarded = _heavy_runs(truncation.topn(400))
    with pytest.raises(AssertionError, match="bare topn or approx_topn"):
        assert_runs_agree_count_only(got, want, APPROX, discarded)

"""``assert_runs_agree`` accepts a different choice among boundary ties and rejects a wrong cut."""

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

from ._runs_agree import assert_runs_agree

# Ten distinct magnitudes from 8 down, ten tied at 0.6, ten from 0.1 down.
MAGNITUDES = [8.0 - 0.5 * k for k in range(10)] + [0.6] * 10 + [0.1 - 0.005 * k for k in range(10)]


def _sum(indices, extra=()):
    """Term ``k`` is ``X`` on the bits of ``k + 1`` with ``|c| = MAGNITUDES[k]``; ``extra`` adds ``(k, magnitude)`` terms."""
    terms = [(k, MAGNITUDES[k]) for k in indices] + list(extra)
    x = np.array([[k + 1] for k, _ in terms], dtype=np.uint64)
    z = np.zeros_like(x)
    coeffs = np.array([m * np.exp(1j * k) for k, m in terms])
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=16)


WANT = list(range(10)) + list(range(10, 15))
TOP, APPROX = truncation.topn(15), truncation.coeff(1e-9) & truncation.approx_topn(15)


@pytest.mark.parametrize("policy", [TOP, APPROX], ids=["topn", "approx_topn"])
def test_a_different_choice_of_boundary_ties_passes(policy):
    got = _sum(list(range(10)) + list(range(15, 20)))
    assert_runs_agree(got, _sum(WANT), policy, 1, 1e-12)


@pytest.mark.parametrize("policy", [TOP, APPROX], ids=["topn", "approx_topn"])
def test_dropping_the_boundary_group_fails(policy):
    got = _sum(list(range(10)) + list(range(20, 25)))
    with pytest.raises(AssertionError, match="smallest kept"):
        assert_runs_agree(got, _sum(WANT), policy, 1, 1e-12)


def test_topn_swapping_a_term_above_the_boundary_fails():
    got = _sum(list(range(1, 15)), extra=[(30, 8.0)])
    with pytest.raises(AssertionError, match="above the boundary"):
        assert_runs_agree(got, _sum(WANT), TOP, 1, 1e-12)


def test_approx_topn_swapping_a_term_above_the_boundary_octave_fails():
    got = _sum(list(range(9)) + list(range(10, 16)))
    with pytest.raises(AssertionError, match="above the boundary"):
        assert_runs_agree(got, _sum(WANT), APPROX, 1, 1e-12)


def test_a_repeated_cut_compares_the_count():
    got = _sum(list(range(10)) + list(range(20, 25)))
    assert_runs_agree(got, _sum(WANT), TOP, 2, 1e-12)
    with pytest.raises(AssertionError, match="term count"):
        assert_runs_agree(_sum(range(14)), _sum(WANT), TOP, 2, 1e-12)

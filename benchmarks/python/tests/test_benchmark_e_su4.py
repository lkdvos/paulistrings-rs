"""Benchmark E -- random SU(4) brickwork, CI-safe correctness gates.

`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part A, row "E".
The full sweep (term-count explosion at n=36, error-vs-runtime, size scaling,
the PauliPropagation.jl comparison) lives in the manual driver script
`benchmarks/python/bench_e_su4.py` -- this file is the fast, CI-visible
correctness half: statevector agreement at small n (`importorskip`d so the
numpy-only CI job skips it cleanly) and the plan's explicit determinism
requirement ("same seed twice -> identical term counts and expectation to
1e-12"), which needs no optional dependency at all and always runs.

Unlike Benchmark A's Clifford points, `random_su4_staircase` has no exact
integer to check against at any size -- every gate is an independent
Haar-random SU(4) block, so the only *exact* ground truth available is dense
statevector simulation, and the only *cheap, dependency-free* check is
self-consistency (a deterministic seed must reproduce itself bit-for-bit in
term count, and to floating-point tolerance in the contracted expectation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, harness, observables, oracles  # noqa: E402

SEED = 20260831


def central_qubit(n: int) -> int:
    return n // 2


# =============================================================================
# Statevector agreement at small n, shallow depth, no truncation.
# =============================================================================


@pytest.mark.parametrize("n", (4, 6, 8, 12))
@pytest.mark.parametrize("depth", (1, 2, 4))
def test_su4_staircase_matches_statevector_untruncated(n, depth):
    """Untruncated Heisenberg propagation vs. qiskit Aer, at small n/depth.

    The two share no simulation code (dense unitary evolution vs. bucketed
    Pauli propagation), so agreement to ~1e-12 is direct evidence the SU(4)
    staircase circuit and its `unitary_2q` conjugation are both correct --
    the same "independent-path" argument `test_examples_oracles.py` makes for
    the shared oracle module itself.
    """
    pytest.importorskip("qiskit_aer")
    q = central_qubit(n)
    obs = observables.single_z(q, n)
    spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
    oracle = oracles.statevector_expectation(spec, obs, "z+").real

    circuit = spec.to_circuit()
    evolved = obs.propagate(circuit, None, direction="heisenberg")
    engine = complex(evolved.expectation("z+")).real

    assert engine == pytest.approx(oracle, abs=1e-10)


@pytest.mark.parametrize("state", ("z+", "x+", "y+"))
def test_su4_staircase_matches_statevector_for_every_uniform_state(state):
    """The three uniform product states, not just `|0...0>`."""
    pytest.importorskip("qiskit_aer")
    n, depth = 6, 3
    q = central_qubit(n)
    obs = observables.single_z(q, n)
    spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
    oracle = oracles.statevector_expectation(spec, obs, state).real

    circuit = spec.to_circuit()
    evolved = obs.propagate(circuit, None, direction="heisenberg")
    engine = complex(evolved.expectation(state)).real

    assert engine == pytest.approx(oracle, abs=1e-10)


def test_su4_staircase_matches_statevector_with_truncation():
    """A loose-but-real truncation must still land within its own error budget.

    Distinct from the untruncated tests above: this exercises
    `PropagationStats`/`min_abs_coeff` together on the SU(4) vocabulary, which
    the untruncated checks (`policy=None`) do not touch at all.
    """
    pytest.importorskip("qiskit_aer")
    from paulistrings import truncation

    n, depth, eps = 10, 4, 1e-6
    q = central_qubit(n)
    obs = observables.single_z(q, n)
    spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
    oracle = oracles.statevector_expectation(spec, obs, "z+").real

    circuit = spec.to_circuit()
    evolved, stats = obs.propagate_with_stats(
        circuit, truncation.coeff(eps), direction="heisenberg"
    )
    engine = complex(evolved.expectation("z+")).real

    assert stats.final_terms <= stats.peak_terms
    assert engine == pytest.approx(oracle, abs=1e-4)


# =============================================================================
# Determinism: the plan's explicit requirement for Benchmark E.
# =============================================================================


def test_same_seed_twice_gives_identical_term_counts_and_expectation():
    """"same seed twice -> identical term counts and expectation to 1e-12"
    (plan §6, Benchmark E). No optional dependency, so this always runs in CI.
    """
    n, depth, eps = 16, 4, 1e-4
    q = central_qubit(n)

    def run_once():
        obs = observables.single_z(q, n)
        circuit = circuits.random_su4_staircase(n, depth, SEED)
        policy = harness.make_policy(min_abs_coeff=eps)
        evolved, stats = obs.propagate_with_stats(circuit, policy, direction="heisenberg")
        return stats.final_terms, stats.peak_terms, complex(evolved.expectation("z+"))

    terms_a, peak_a, exp_a = run_once()
    terms_b, peak_b, exp_b = run_once()

    assert terms_a == terms_b
    assert peak_a == peak_b
    assert abs(exp_a - exp_b) < 1e-12


def test_different_seeds_give_different_circuits():
    """Negative control: the determinism test above is not vacuously true of
    any two runs -- two distinct seeds must (generically) disagree.
    """
    n, depth = 10, 3
    q = central_qubit(n)
    obs = observables.single_z(q, n)

    circuit_a = circuits.random_su4_staircase(n, depth, seed=1)
    circuit_b = circuits.random_su4_staircase(n, depth, seed=2)

    exp_a = complex(obs.propagate(circuit_a, None, direction="heisenberg").expectation("z+"))
    exp_b = complex(obs.propagate(circuit_b, None, direction="heisenberg").expectation("z+"))

    assert abs(exp_a - exp_b) > 1e-6


def test_su4_staircase_is_deterministic_across_repeated_construction():
    """The circuit *builder* itself is deterministic given the seed (no RNG
    state leaking from module import order, no wall-clock seeding anywhere).
    Complements the propagation-level check above at the construction level.
    """
    n, depth = 12, 5
    a = circuits.random_su4_staircase(n, depth, SEED)
    b = circuits.random_su4_staircase(n, depth, SEED)
    assert len(a) == len(b)

    obs = observables.single_z(central_qubit(n), n)
    exp_a = complex(obs.propagate(a, None, direction="heisenberg").expectation("z+"))
    exp_b = complex(obs.propagate(b, None, direction="heisenberg").expectation("z+"))
    assert exp_a == exp_b

"""Benchmark A -- Clifford-point correctness gate (adapted plan §6 Part A "A").

Heavy-hex kicked Ising, `n=127`, 5 Trotter steps, `theta_h` at the two Clifford
points (`pi/2` and `0`). At `theta_h = pi/2` (with the default
`theta_zz = -pi/2`) the published weight-10 and weight-17 observables of Kim et
al. (2023) are stabilizers of the circuit with eigenvalues exactly `+1` and
`-1` (`examples/data/kim2023_observables.json`); at `theta_h = 0` every ZZ
generator commutes with a `Z`-type observable and `rx(0)` is the identity, so
`Z_62` (Fig. 4b's weight-1 observable) is untouched. `stim_clifford_exact` is
the independent cross-check (plan §7 rule 6: engine and oracle are driven from
the same gate list via `oracles.record_gates`).

**The runtime fact that makes this test file cheap.** `cos(pi/2)` is
`6.123233995736766e-17` in `float64`, not exact zero (pinned by
`test_cos_pi_half_is_not_exactly_zero_but_sin_is` below) -- so an *untruncated*
127-qubit propagation at this angle fans out without ever collapsing back to
one term (an earlier probe of this hit 181 GB RSS). But with **any**
`min_abs_coeff` truncation above that residual (empirically, `>= 1e-12` is
already far more than enough headroom over `6.1e-17`), the numerically-dead
branch is pruned every layer and the propagation never carries more than one
term end to end: measured on this checkout, the full `n=127`, 1355-channel,
weight-17 Heisenberg propagation with `coeff(1e-12)` runs in **~1.3 ms**
single-threaded, and the `stim_clifford_exact` cross-check over the same
1355-gate circuit runs in **~35 ms**. The whole file (18 parametrized cases
plus the stim cross-checks) is well under one second, so nothing here needs
the ~60 s skip-unless-env-var escape hatch the plan reserves for a slow case --
that budget is spent by the *untruncated* configuration, which this file does
not exercise (see `research/plans/2026-08-31-examples-benchmarks-suite.md`
Benchmark A's setup note, and `test_benchmark_a_headline_case_is_fast` below,
which pins the cheap case as a tripwire against a future regression that makes
it expensive again).

Failure diagnostics (this module's `_assert_single_clifford_term`): a term
count that is not exactly 1 points at a `pi/4`-vs-`pi/2` boundary bug (a
non-Clifford angle lets the `cos(theta)` residual survive truncation and fan
out); a single term with the *wrong sign* points at a Heisenberg-adjoint-
ordering bug (`apply`/`apply_adjoint`, or `forward`/`heisenberg` reversed) --
a boundary bug would not stay at one term, and an ordering bug would not
change the term count of a bijective Clifford map.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

from paulistrings import truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, observables, oracles  # noqa: E402

N = 127
TROTTER_STEPS = 5
THETA_H_CLIFFORD = math.pi / 2

#: The truncation grid the plan asks for: three `min_abs_coeff` values spanning
#: nine orders of magnitude above the `cos(pi/2)` residual, all of which must
#: give the identical exact answer.
COEFF_GRID = (1e-12, 1e-8, 1e-4)


def _kicked_ising_heisenberg(observable, theta_h, *, policy):
    circuit = circuits.heavy_hex_kicked_ising(N, trotter_steps=TROTTER_STEPS, theta_h=theta_h)
    return observable.propagate(circuit=circuit, policy=policy, direction="heisenberg")


def _assert_single_clifford_term(evolved, expected_sign: float, *, label: str) -> None:
    """The strongest true assertion: exactly one term, bit-exact `+-1.0`.

    Raises with a diagnostic that names which of the two known failure modes
    (angle boundary vs. adjoint ordering) the observed symptom matches -- see
    the module docstring.
    """
    n_terms = len(evolved)
    if n_terms != 1:
        coeffs = sorted(
            (abs(complex(c)) for c in evolved.coefficients_array()), reverse=True
        )
        raise AssertionError(
            f"{label}: evolved operator has {n_terms} terms, not 1. This looks like a "
            "pi/4-vs-pi/2 ANGLE BOUNDARY bug: a non-Clifford angle lets the "
            "cos(theta)~6.1e-17 residual branch survive truncation and fan out layer "
            "over layer, rather than an adjoint-ordering bug (which would still "
            f"collapse to a single, merely mis-signed, term). Largest surviving "
            f"|coefficient|s: {coeffs[:5]}"
        )
    coeff = complex(evolved.coefficients_array()[0])
    if coeff.real != expected_sign or coeff.imag != 0.0:
        raise AssertionError(
            f"{label}: the single surviving term has coefficient {coeff!r}, expected "
            f"exactly {expected_sign:+.1f}. A single term with the WRONG SIGN (rather "
            "than the wrong term count) looks like a HEISENBERG-ADJOINT-ORDERING bug "
            "(apply vs. apply_adjoint, or direction='forward'/'heisenberg' reversed) -- "
            "a pi/4-vs-pi/2 boundary bug would instead fan out to many terms, not "
            "silently flip the sign of a single one."
        )


# --------------------------------------------------------------------------
# The floating-point fact this whole file leans on.
# --------------------------------------------------------------------------


def test_cos_pi_half_is_not_exactly_zero_but_sin_is():
    # cos(pi/2) in float64 is the well-known ~6.1e-17 residual (the double
    # rounding of pi/2 itself); sin(pi/2) rounds to exactly 1.0 because
    # 1 - sin(pi/2 - eps) ~ eps^2/2 ~ 1.9e-33 is far below a double's ULP at 1.0
    # (~2.2e-16), so the error is unobservable in floating point.
    assert math.cos(THETA_H_CLIFFORD) == pytest.approx(0.0, abs=1e-15)
    assert math.cos(THETA_H_CLIFFORD) != 0.0
    assert math.sin(THETA_H_CLIFFORD) == 1.0
    assert math.sin(-THETA_H_CLIFFORD) == -1.0
    assert math.cos(circuits.KICKED_ISING_CLIFFORD_THETA_ZZ) != 0.0
    assert math.sin(circuits.KICKED_ISING_CLIFFORD_THETA_ZZ) == -1.0


def test_benchmark_a_headline_case_is_fast():
    # Tripwire: the whole point of truncating at this Clifford point is that
    # the numerically-dead branch never survives to fan out, so this must stay
    # fast. A regression that makes it slow (or blows past ~1 term) is exactly
    # the catastrophic-fanout failure mode the module docstring describes.
    obs = observables.weight_17_operator(N)
    start = time.perf_counter()
    evolved = _kicked_ising_heisenberg(obs, THETA_H_CLIFFORD, policy=truncation.coeff(1e-12))
    elapsed = time.perf_counter() - start
    assert len(evolved) == 1
    assert elapsed < 5.0, f"weight-17 headline propagation took {elapsed:.3f}s, expected well under 1s"


# --------------------------------------------------------------------------
# theta_h = pi/2: the published stabilizers, invariant across the truncation grid.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("eps", COEFF_GRID)
def test_weight_10_is_exactly_plus_one_at_any_coeff_truncation(eps):
    evolved = _kicked_ising_heisenberg(
        observables.weight_10_operator(N), THETA_H_CLIFFORD, policy=truncation.coeff(eps)
    )
    _assert_single_clifford_term(evolved, 1.0, label=f"weight_10 @ eps={eps:g}")


@pytest.mark.parametrize("eps", COEFF_GRID)
def test_weight_17_is_exactly_minus_one_at_any_coeff_truncation(eps):
    evolved = _kicked_ising_heisenberg(
        observables.weight_17_operator(N), THETA_H_CLIFFORD, policy=truncation.coeff(eps)
    )
    _assert_single_clifford_term(evolved, -1.0, label=f"weight_17 @ eps={eps:g}")


@pytest.mark.parametrize("eps", COEFF_GRID)
def test_weight_10_survives_a_weight_cutoff_at_its_own_weight(eps):
    # Combining the coeff grid with a matched weight cap (plan: "with/without
    # weight cutoff >= 17" -- 10 is this observable's own weight) must not
    # change the exact answer: the surviving branch never exceeds weight 10.
    policy = truncation.weight(10) & truncation.coeff(eps)
    evolved = _kicked_ising_heisenberg(observables.weight_10_operator(N), THETA_H_CLIFFORD, policy=policy)
    _assert_single_clifford_term(evolved, 1.0, label=f"weight_10, weight<=10 @ eps={eps:g}")


@pytest.mark.parametrize("eps", COEFF_GRID)
def test_weight_17_survives_a_weight_cutoff_at_its_own_weight(eps):
    policy = truncation.weight(17) & truncation.coeff(eps)
    evolved = _kicked_ising_heisenberg(observables.weight_17_operator(N), THETA_H_CLIFFORD, policy=policy)
    _assert_single_clifford_term(evolved, -1.0, label=f"weight_17, weight<=17 @ eps={eps:g}")


def test_weight_17_negative_control_a_weight_cutoff_below_its_weight_kills_it():
    # Sanity check on the weight-cutoff tests above: a cap tighter than the
    # observable's own weight (17) must drop the stabilizer entirely, so the
    # "survives at weight<=17" results above are not vacuously true of any cap.
    policy = truncation.weight(10) & truncation.coeff(1e-12)
    evolved = _kicked_ising_heisenberg(observables.weight_17_operator(N), THETA_H_CLIFFORD, policy=policy)
    assert len(evolved) == 0


# --------------------------------------------------------------------------
# theta_h = 0: Z_62 is untouched (no truncation is even needed, but the grid
# is invariant here too).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("eps", (None,) + COEFF_GRID)
def test_z62_is_exactly_plus_one_at_theta_h_zero(eps):
    policy = None if eps is None else truncation.coeff(eps)
    evolved = _kicked_ising_heisenberg(observables.canonical_z_127(), 0.0, policy=policy)
    _assert_single_clifford_term(evolved, 1.0, label=f"Z_62 @ theta_h=0, eps={eps}")


# --------------------------------------------------------------------------
# stim cross-check: the independent oracle, on the exact same n=127 circuit.
# --------------------------------------------------------------------------


def test_stim_cross_check_weight_10_and_weight_17_at_theta_h_half_pi():
    pytest.importorskip("stim")
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N, trotter_steps=TROTTER_STEPS, theta_h=THETA_H_CLIFFORD
    )
    for obs, expected in (
        (observables.weight_10_operator(N), 1.0),
        (observables.weight_17_operator(N), -1.0),
    ):
        value = oracles.stim_clifford_exact(spec, obs)
        assert value == complex(expected), f"stim disagrees: got {value!r}, expected {expected:+.1f}"


def test_stim_cross_check_z62_at_theta_h_zero():
    pytest.importorskip("stim")
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N, trotter_steps=TROTTER_STEPS, theta_h=0.0
    )
    value = oracles.stim_clifford_exact(spec, observables.canonical_z_127())
    assert value == 1.0 + 0j


# --------------------------------------------------------------------------
# Small sublattice: engine and stim agree at BOTH Clifford points, fast.
#
# The published weight-10/17 supports only fit on the full 127-qubit device,
# so this uses the same construction the observables themselves come from
# (Kim et al.'s stabilizer relation, also re-derived at full scale by
# test_examples_circuits.py::test_published_supports_are_reproduced_by_the_
# stabilizer_relation): evolve a single-qubit Z seed FORWARD through a small
# sublattice to get that sublattice's own "stabilizer", then check that
# evolving it back through stim gives exactly the seed's <0|Z_seed|0> = 1 --
# an independent, non-vacuous round trip through the oracle, not merely a
# zero-equals-zero comparison.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("theta_h", (THETA_H_CLIFFORD, 0.0), ids=("theta_h=pi/2", "theta_h=0"))
def test_small_sublattice_stim_round_trip_at_both_clifford_points(theta_h):
    pytest.importorskip("stim")
    n = 20
    seed_qubit = 5
    circuit = circuits.heavy_hex_kicked_ising(n, trotter_steps=3, theta_h=theta_h)
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, n, trotter_steps=3, theta_h=theta_h
    )
    seed = observables.pauli_sum_from_support({seed_qubit: "Z"}, n)

    # U Z_seed U^dagger, computed by the engine (direction="forward" is the
    # Schrodinger conjugation -- see test_examples_circuits.py's docstring note).
    stabilizer = seed.propagate(circuit=circuit, policy=truncation.coeff(1e-9), direction="forward")
    assert len(stabilizer) == 1, f"theta_h={theta_h}: expected a single Clifford stabilizer term"

    # Feeding that stabilizer back into stim's independent tableau simulation
    # of the SAME circuit must undo the conjugation exactly: <0| U^dagger
    # (U Z_seed U^dagger) U |0> = <0| Z_seed |0> = +1, regardless of what sign
    # or support the intermediate stabilizer carries.
    value = oracles.stim_clifford_exact(spec, stabilizer)
    assert value == 1.0 + 0j, (
        f"theta_h={theta_h}: engine/stim round trip through the derived stabilizer "
        f"did not return to <0|Z_{seed_qubit}|0> = 1, got {value!r}"
    )

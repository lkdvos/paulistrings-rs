"""Benchmark C — CI-safe correctness gate for the deep-Trotter headline.

`benchmarks/python/bench_c_deep_trotter.py` is the real Benchmark C: `n = 127`,
up to 20 Trotter steps, dyadic cutoffs down to `2^-18`, self-converged
references computed with 16 Rayon workers, and hours of wall time. None of that
can run in CI. What *can*, and what this file pins:

1. **The physics at full depth, against an exact oracle.** On the 20-qubit
   heavy-hex sublattice (device qubits `0..19`, a real sub-piece of the Eagle
   map) a dense statevector answers the **20-step** problem exactly, at both of
   the driver's kick angles. What that oracle shows — and this is the
   measurement Benchmark C's whole report turns on — is that a deep circuit in
   the hard interior converges *slowly and from one side*: at
   `theta_h = 7pi/32`, 20 steps, the error against the exact `+0.594614476873`
   runs 4.7e-1, 2.1e-1, 9.6e-2 at `2^-8/-10/-12`, falling by only ~2.2x per
   factor of four in the cutoff while the term count grows ~18x. So the gate
   asserts the *convergence law*, not exactness, at full depth
   (`test_deep_error_decreases_toward_the_exact_answer`), and asserts exactness
   at a depth where the truncated sum genuinely saturates
   (`test_tight_truncation_matches_the_statevector`, 5 steps).
2. **That the reference procedure does not lie at full depth.** A slow one-sided
   climb is precisely the shape a naive plateau test mistakes for convergence.
   C's reference is self-converged, so
   `test_the_reference_refuses_to_declare_convergence_at_full_depth` pins that
   the imported criterion reports `converged=False` there rather than inventing a
   plateau, and `test_self_convergence_estimate_tracks_the_true_error` scores its
   estimate against the truth where it *does* converge.
   `test_a_budget_truncated_sweep_can_understate_the_true_error` pins the
   measured **sign flip** in that estimate's bias: over the full grid it
   overstates the true error (2.7x, 3.2x), but a sweep stopped by its budget
   after two points *understates* it (8.7x here, and 15.7x at the off-grid
   `theta_h = 1.0` probed during development — past the 10x slack Benchmark B's
   §3.4 validation allows). So `converged=False` has to be read as "no usable
   estimate", not "a weaker estimate".
3. **The located published anchor.**
   `test_the_published_anchor_is_recorded_and_not_transcribed` pins that the
   upstream exact benchmark is recorded as a *pointer* — URL, column-to-observable
   mapping from the paper's figure captions, and why nothing was checked in —
   with no transcribed numbers; that the depth ladder carries the 9-step rung its
   `5a` column corresponds to; and that both kick angles sit on the published
   `k*pi/32` grid, so a later byte-exact fetch lines up row for row with no
   interpolation.
4. **That C did not re-implement B's corrected plateau test.** The naive "two
   successive values agree" criterion was measured wrong in Benchmark B; C
   imports B's `_plateau_is_real` rather than re-typing it, and
   `test_c_reuses_benchmark_bs_plateau_criterion` asserts the function *object*
   is the same one.
5. **The one-ulp mitigation, from both sides.** The dyadic cutoffs are the one
   case where this engine and PauliPropagation.jl provably disagree at the
   boundary. `test_julia_one_ulp_perturbation_matches_this_engines_rule` pins
   this engine's inclusive-drop rule on a hand-built sum *and* the float
   arithmetic that makes jl's exclusive-drop rule coincide with it.
6. **A fast envelope sanity check on the real lattice.** Two loose cutoffs at
   `n = 127`, 20 steps, both angles — seconds, not hours — asserting the tracked
   set really does grow into the millions and that the peak/final collapse the
   envelope check has to cope with is real
   (`test_loose_cutoff_sanity_on_the_full_lattice`).
7. **No drift between the gate and the driver.** The angle grid, the step ladder,
   the dyadic cutoffs, the accuracy bar and the envelope are all read out of the
   driver module (`test_driver_constants`).

Runtime
-------
Dominated by the two 20-step Aer statevector references (~11 s each) and the
four 20-step sublattice grids (~8 s each). Whole file: ~1.5 min on the reference
host, with no skip-unless-env-var escape hatch.

Why the oracle calls are wrapped in `_oracle`
--------------------------------------------
CI's python job installs numpy only; qiskit and qiskit-aer come from the
optional `examples` extra. Rather than an `importorskip` at module level (which
would skip the engine-only assertions too), each oracle call goes through
`_oracle`, which turns `oracles.SkipOracle` — the module's own
"dependency-not-installed" signal — into a pytest skip. Everything that needs no
oracle still runs.
"""

from __future__ import annotations

import json
import math
import re
import sys
from functools import cache
from pathlib import Path

import pytest

from paulistrings import PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _extra_path in (_REPO_ROOT / "examples", _REPO_ROOT / "benchmarks" / "python"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

import bench_b_theta_sweep as bench_b  # noqa: E402
import bench_c_deep_trotter as driver  # noqa: E402
from common import circuits, harness, observables, oracles, report  # noqa: E402

#: The sublattice size. 20 gives a connected induced heavy-hex subgraph (19
#: edges, 3 colour classes — `heavy_hex_sublattice` *computes* connectivity and
#: raises for the sizes that leave a qubit isolated) and a 16 MiB statevector.
N = 20

#: Full depth. Not a reduced stand-in: the whole point of Benchmark C is depth,
#: so the gate runs the driver's headline step count.
DEEP_STEPS = 20

#: The shallow depth used for the *exactness* assertion. At 5 steps a single-`Z`
#: seed's reachable set on this sublattice **saturates** (measured: 34 754 terms,
#: unchanged by four further decades of cutoff), so a tight cutoff gives the
#: untruncated answer in milliseconds. At 20 steps it does not — measured
#: 3 815 001 terms at `2^-14` and still climbing, which is exactly why the
#: full-depth test asserts a convergence law instead.
SHALLOW_STEPS = 5

STATE = driver.STATE
DIRECTION = driver.DIRECTION
THETA_POINTS = driver.THETA_POINTS

#: The observable. **Synthetic**, not `Z_62`: the published support lives on
#: device qubit 62 and does not fit a 20-qubit sublattice. `Z_10` is the same
#: kind of operator — a single `Z` near the middle of the lattice — which is what
#: `Z_62` is on the full device.
OBSERVABLE_SUPPORT = {10: "Z"}

#: Dyadic grid for the **full-depth** convergence assertions, loosest first.
#: Three points, and the tight end is `2^-12`: measured on the reference host at
#: 20 steps those cost 0.06 s, 0.9 s and 7.5 s per angle, while `2^-14` costs
#: 55 s and `2^-16` (extrapolating the measured 11x per factor of four in the
#: cutoff, ~4e7 terms) is minutes. This is the CI budget's boundary, not a
#: physics one.
DEEP_COEFF_GRID = (2.0**-8, 2.0**-10, 2.0**-12)

#: The error the deep grid's tight end must already have reached, against the
#: exact statevector. Measured: 9.6e-2 at `theta_h = 7pi/32` and 9.4e-3 at
#: `5pi/16` (where the signal has decayed). The bar is loose on purpose — it is a
#: "the operator is spreading and the error is coming down" tripwire, not a
#: convergence claim, and the plan's own 0.01 target is *not* reachable at this
#: depth on any grid this file could afford.
DEEP_ERROR_BAR = 0.15

#: Dyadic grid at `SHALLOW_STEPS`, where the sum saturates and exactness holds.
SHALLOW_COEFF_GRID = (2.0**-8, 2.0**-14, 2.0**-20, 2.0**-26)

#: The tight end used for the exactness assertion, at `SHALLOW_STEPS`. Above
#: Benchmark B's `MIN_SAFE_COEFF` floor, so the `cos(pi/2)` residual branch stays
#: pruned (irrelevant at these angles, but the floor is a suite-wide invariant).
TIGHT_COEFF = 2.0**-30

#: Error bar for "tight truncation reproduces the exact answer" at 5 steps.
TIGHT_TOLERANCE = 1e-7

#: Reference-procedure budgets for the gate. Far below the driver's.
GATE_MAX_TERMS = 5_000_000
GATE_MAX_SECONDS = 60.0

#: Loose cutoffs the `n = 127` sanity check runs at, and its wall-time budget.
#: Measured on ccqlin038 at 20 steps: 0.31 s / 1.3 s at θ_h ≈ 0.69 and
#: 0.20 s / 1.5 s at θ_h ≈ 0.98. The budget is deliberately ~20x that, so a
#: loaded CI box does not turn a real regression signal into a flake.
LOOSE_COEFFS = (2.0**-8, 2.0**-10)
LOOSE_BUDGET_S = 120.0


def _theta_ids(points) -> list[str]:
    return [f"theta_h={label}" for label, _ in points]


def _oracle(call, *args, **kwargs):
    """Run an oracle, turning a missing optional dependency into a skip."""
    try:
        return call(*args, **kwargs)
    except oracles.SkipOracle as exc:  # pragma: no cover - depends on the env
        pytest.skip(str(exc))


def _observable(n: int = N):
    return observables.pauli_sum_from_support(OBSERVABLE_SUPPORT, n)


@cache
def _spec(theta: float, steps: int = DEEP_STEPS, n: int = N) -> oracles.CircuitSpec:
    """The gate list, recorded once per angle.

    `record_gates` captures the same builder the engine uses, so the oracle and
    the engine are driven from one description rather than two transcriptions of
    it (plan §7 rule 6).
    """
    return oracles.record_gates(circuits.heavy_hex_kicked_ising, n, steps, theta)


@cache
def _exact(theta: float, steps: int = DEEP_STEPS) -> float:
    """Exact `<0| U^dagger Z_10 U |0>` by dense statevector on the sublattice."""
    value = _oracle(
        oracles.statevector_expectation, _spec(theta, steps), _observable(), STATE
    )
    assert abs(complex(value).imag) < 1e-12, (
        f"the statevector oracle returned a complex expectation {value!r} for a "
        "Hermitian observable"
    )
    return complex(value).real


def _propagate(theta: float, eps: float, *, steps: int = DEEP_STEPS, n: int = N):
    """One truncated Heisenberg propagation; returns `(value, stats)`."""
    observable = _observable(n)
    evolved, stats = observable.propagate_with_stats(
        _spec(theta, steps, n).to_circuit(),
        harness.make_policy(min_abs_coeff=eps),
        direction=DIRECTION,
    )
    return complex(evolved.expectation(STATE)).real, stats


# --------------------------------------------------------------------------
# 0. The lattice, the grid, and no drift from the driver
# --------------------------------------------------------------------------


def test_sublattice_is_a_connected_piece_of_the_eagle_map():
    edges = circuits.heavy_hex_sublattice(N)
    assert len(edges) == 19
    assert all(0 <= a < b < N for a, b in edges)
    colours = circuits.heavy_hex_edge_coloring(edges)
    assert len(colours) == 3
    for group in colours:
        touched = [q for edge in group for q in edge]
        assert len(touched) == len(set(touched)), "a colour class is not a matching"


def test_driver_constants():
    """The gate and the driver must agree on the benchmark's definition."""
    assert driver.N_QUBITS == 127
    assert driver.DIRECTION == "heisenberg"
    assert driver.STATE == "z+"
    assert driver.OBSERVABLE_NAME == "z62"
    # The plan's headline: up to 20 Trotter steps, theta_h in the hard interior.
    assert max(driver.STEP_POINTS) == 20
    assert list(driver.STEP_POINTS) == sorted(driver.STEP_POINTS)
    assert all(0.6 <= theta <= 1.0 for _, theta in driver.THETA_POINTS)
    # The plan's accuracy target for this benchmark.
    assert driver.ACCURACY_EPSILON == 0.01
    # The handoff's sanity envelope, and the direction of the scoring.
    assert driver.TERM_COUNT_ENVELOPE == (1_200_000, 9_300_000)

    # The plan's truncation grid is 2^-14, 2^-16, 2^-18, verbatim...
    assert driver.PLAN_COEFF_GRID == (2.0**-14, 2.0**-16, 2.0**-18)
    # ...and the swept grid contains it, is dyadic throughout, and is ordered
    # loosest-first (`harness.time_to_accuracy`'s documented contract).
    assert set(driver.PLAN_COEFF_GRID) <= set(driver.COEFF_GRID)
    assert list(driver.COEFF_GRID) == sorted(driver.COEFF_GRID, reverse=True)
    for eps in driver.COEFF_GRID:
        exponent = round(math.log2(eps))
        assert 2.0**exponent == eps, f"{eps!r} is not an exact dyadic"
    # Every cutoff must sit above the cos(pi/2) residual-branch floor, or a
    # Clifford-angle run fans out (Benchmark B's module docstring).
    assert min(driver.COEFF_GRID) >= bench_b.MIN_SAFE_COEFF
    # A cut may only *shorten* the grid at the tight end.
    for floor in driver.COEFF_GRID_CUTS.values():
        assert floor in driver.COEFF_GRID


def test_the_reference_grid_extends_past_the_timed_grid():
    """A reference must be tighter than everything it scores.

    Otherwise the tightest timed run has an error of zero by construction, and
    the accuracy claim is circular.
    """
    grid = driver.COEFF_GRID
    reference_grid = driver.self_convergence_grid(grid)
    assert set(grid) <= set(reference_grid)
    assert min(reference_grid) < min(grid)
    assert len(reference_grid) == len(grid) + driver.SELF_CONVERGENCE_EXTRA_POWERS
    assert list(reference_grid) == sorted(reference_grid, reverse=True)
    for eps in reference_grid:
        exponent = round(math.log2(eps))
        assert 2.0**exponent == eps, f"reference grid point {eps!r} is not dyadic"


def test_dyadic_labels_round_trip():
    assert driver.dyadic_label(2.0**-14) == "2^-14"
    assert driver.dyadic_label(2.0**-18) == "2^-18"
    # A non-dyadic must not be mislabelled as one.
    assert driver.dyadic_label(1e-4) == "0.0001"


def test_c_reuses_benchmark_bs_plateau_criterion():
    """C must not re-implement the corrected self-convergence test.

    The naive criterion ("two successive values agree") was *measured* wrong in
    Benchmark B: it declared convergence with an estimated uncertainty of
    exactly zero while the value was still 5.6e-7 from the truth, because the
    expectation can sit bit-identical across four decades of cutoff while the
    sum keeps growing. Asserting the identity of the function object is what
    makes a future copy-paste-and-simplify fail here.
    """
    assert driver.plateau_is_real is bench_b._plateau_is_real
    assert driver.Reference is bench_b.Reference
    # And the shapes B measured still decide the same way through C's alias.
    tol = driver.SELF_CONVERGENCE_TOL
    flat_growing = [{"final_terms": 59}, {"final_terms": 225}, {"final_terms": 728}]
    assert driver.plateau_is_real(flat_growing, [0.0, 0.0], tol) is False
    saturated = [{"final_terms": 2_146_372} for _ in range(3)]
    assert driver.plateau_is_real(saturated, [0.0, 0.0], tol) is True
    moving = [{"final_terms": 1}, {"final_terms": 2}, {"final_terms": 3}]
    assert driver.plateau_is_real(moving, [1e-5, 2e-6], tol) is True
    emptied = [{"final_terms": 0} for _ in range(3)]
    assert driver.plateau_is_real(emptied, [0.0, 0.0], tol) is False


# --------------------------------------------------------------------------
# 1. The dyadic boundary and the one-ulp mitigation
# --------------------------------------------------------------------------


def test_this_engine_drops_a_coefficient_equal_to_the_cutoff():
    """The inclusive-drop rule, on a hand-built sum at an exact dyadic.

    `truncation/builtin.rs` keeps `|c| > eps`, so a coefficient bit-exactly
    equal to a dyadic cutoff is discarded. PauliPropagation.jl keeps it
    (`benchmarks/julia/README.md` §P3). That divergence is measure-zero for a
    power-of-ten cutoff — which is why Benchmark B could ignore it — and *not*
    measure-zero for Benchmark C's dyadics, because at a Clifford `theta_zz` the
    coefficients are exact dyadics too.
    """
    from paulistrings import Circuit

    eps = 2.0**-14
    n = 2
    policy = truncation.coeff(eps)
    # `z` on a `Z` string is the identity map with sign +1, so nothing but
    # truncation can move the coefficient.
    circuit = Circuit(n)
    circuit.z(0)

    def surviving(coefficient: float) -> int:
        sum_ = PauliSum.from_strings({"ZI": coefficient}, num_qubits=n)
        _, stats = sum_.propagate_with_stats(circuit, policy, direction=DIRECTION)
        return stats.final_terms

    assert surviving(eps) == 0, "a coefficient equal to the cutoff must be dropped"
    assert surviving(math.nextafter(eps, math.inf)) == 1
    assert surviving(math.nextafter(eps, 0.0)) == 0


def test_julia_one_ulp_perturbation_matches_this_engines_rule():
    """`nextafter(eps, inf)` turns jl's exclusive rule into this engine's.

    jl drops `|c| < eps'`. With `eps' = nextafter(eps, inf)` there is no float
    strictly between `eps` and `eps'`, so `|c| < eps'` holds exactly when
    `|c| <= eps` — which is this engine's rule. No coefficient is touched, which
    is the property that makes this a legitimate mitigation rather than a fudge.
    """
    for eps in driver.PLAN_COEFF_GRID:
        perturbed = driver.julia_min_abs_coeff(eps)
        # Strictly above, and by exactly one ulp: nothing lies between them, so
        # the only coefficient whose fate changes is `|c| == eps` itself.
        assert perturbed > eps
        assert math.nextafter(perturbed, -math.inf) == eps
        # A "one-ulp" claim, quantified: the shift is at the double's own
        # resolution, ~1e-16 relative.
        assert 0.0 < perturbed - eps < eps * 1e-15
        # jl's rule at `perturbed` and this engine's rule at `eps` agree on the
        # boundary value and on its two neighbours — which is the whole claim.
        for coefficient in (
            math.nextafter(eps, 0.0), eps, math.nextafter(eps, math.inf)
        ):
            engine_drops = abs(coefficient) <= eps
            julia_drops = abs(coefficient) < perturbed
            assert engine_drops == julia_drops, (
                f"the two rules disagree at {coefficient!r} for cutoff {eps!r}"
            )

    with pytest.raises(ValueError):
        driver.julia_min_abs_coeff(0.0)


# --------------------------------------------------------------------------
# 2. Accuracy and convergence at full depth
# --------------------------------------------------------------------------


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_tight_truncation_matches_the_statevector(theta_label, theta):
    """Exactness, at the depth where the truncated sum saturates.

    5 steps, not 20: at 5 steps a single-`Z` seed reaches every Pauli string it
    can on this sublattice and then stops growing, so a tight cutoff *is* the
    untruncated answer. A miss here is a physics error, not a truncation
    artefact — check the direction (heisenberg vs forward) and the contraction
    state before anything else.
    """
    exact = _exact(theta, SHALLOW_STEPS)
    value, stats = _propagate(theta, TIGHT_COEFF, steps=SHALLOW_STEPS)
    assert value == pytest.approx(exact, abs=TIGHT_TOLERANCE), (
        f"theta_h={theta_label}, {SHALLOW_STEPS} steps: truncated propagation {value!r} "
        f"vs exact {exact!r} ({stats.final_terms} terms kept at min_abs_coeff="
        f"{driver.dyadic_label(TIGHT_COEFF)})"
    )


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_error_improves_as_truncation_tightens(theta_label, theta):
    """Loosest-to-tightest dyadic cutoff at 5 steps: error down, term count up.

    Asserted over the whole grid rather than point to point. Benchmark B
    measured that truncation error is *not* monotone in the cutoff — the
    discarded terms carry signs, so a smaller partial sum can sit nearer the
    truth than a larger one, and a truncated Pauli sum has no variational bound
    to forbid it.
    """
    exact = _exact(theta, SHALLOW_STEPS)
    errors, terms = [], []
    for eps in SHALLOW_COEFF_GRID:
        value, stats = _propagate(theta, eps, steps=SHALLOW_STEPS)
        errors.append(abs(value - exact))
        terms.append(stats.final_terms)

    labels = [driver.dyadic_label(e) for e in SHALLOW_COEFF_GRID]
    assert terms == sorted(terms), (
        f"theta_h={theta_label}: term counts {terms} are not non-decreasing along a "
        f"loosest-to-tightest cutoff grid {labels}"
    )
    assert errors[-1] <= errors[0] + 1e-12, (
        f"theta_h={theta_label}: tightening min_abs_coeff from {labels[0]} to "
        f"{labels[-1]} made the error worse ({errors[0]:.3e} -> {errors[-1]:.3e}); the "
        "truncation is not converging"
    )
    assert errors[-1] <= TIGHT_TOLERANCE


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_deep_error_decreases_toward_the_exact_answer(theta_label, theta):
    """Full depth, against the exact statevector: the convergence *law*.

    The measurement Benchmark C's report turns on. At `theta_h = 7pi/32`, 20
    steps, `n = 20`, the exact answer is `+0.594614476873` and the truncated
    propagation climbs toward it from below — 0.1231, 0.3832, 0.4989 at
    `2^-8/-10/-12` — with the error falling by only ~2.2x per factor of four in
    the cutoff while the term count grows ~18x. Extrapolating that law (measured
    out to `2^-16` during development at the neighbouring `theta_h = 0.7`, where
    the error reaches 9.0e-3 with 3.3e7 terms) is what tells Benchmark C that the
    plan's 0.01 target needs a cutoff around `2^-16`/`2^-17` even at this size,
    and is out of reach at `n = 127`.

    So what is asserted here is the law, not the target: the error falls
    monotonically, the term count rises monotonically, and the value approaches
    the exact answer from **one side** (which is why a naive plateau test is
    dangerous here — see the next test).
    """
    exact = _exact(theta)
    errors, terms, values = [], [], []
    for eps in DEEP_COEFF_GRID:
        value, stats = _propagate(theta, eps)
        errors.append(abs(value - exact))
        terms.append(stats.final_terms)
        values.append(value)

    labels = [driver.dyadic_label(e) for e in DEEP_COEFF_GRID]
    assert terms == sorted(terms), (
        f"theta_h={theta_label}, {DEEP_STEPS} steps: term counts {terms} fell while the "
        f"cutoff tightened along {labels}"
    )
    assert errors == sorted(errors, reverse=True), (
        f"theta_h={theta_label}, {DEEP_STEPS} steps: errors {errors} against the exact "
        f"{exact!r} are not decreasing along {labels}"
    )
    assert errors[-1] < DEEP_ERROR_BAR, (
        f"theta_h={theta_label}, {DEEP_STEPS} steps: error {errors[-1]:.3e} at "
        f"{labels[-1]} is above the {DEEP_ERROR_BAR:g} tripwire; the operator is not "
        "spreading as measured (check the lattice, the direction, and theta_zz)"
    )
    # One-sided approach: every partial sum sits on the same side of the truth.
    signs = {math.copysign(1.0, value - exact) for value in values}
    assert len(signs) == 1, (
        f"theta_h={theta_label}: the truncated values {values} straddle the exact "
        f"{exact!r}. That is not a failure of the engine, but it invalidates the "
        "one-sided reading the README's convergence-law extrapolation rests on — "
        "re-measure the law before quoting it."
    )


# --------------------------------------------------------------------------
# 3. The self-converged reference, where the truth is known
# --------------------------------------------------------------------------


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_self_convergence_estimate_tracks_the_true_error(theta_label, theta):
    """C's reference procedure, scored against a statevector where it converges.

    Every Benchmark C reference is self-converged — the commutation-aware cone
    of `Z_62` after 20 kicked-Ising steps is the whole 127-qubit lattice, so no
    exact route exists there. That is only defensible if the estimate is honest,
    so the identical procedure (C's tolerance, C's dyadic grid) runs here on a
    system whose exact answer a statevector gives.
    """
    exact = _exact(theta, SHALLOW_STEPS)
    reference = bench_b.self_converged_reference(
        _spec(theta, SHALLOW_STEPS).to_circuit(),
        _observable(),
        grid=driver.self_convergence_grid(SHALLOW_COEFF_GRID),
        state=STATE,
        direction=DIRECTION,
        tol=driver.SELF_CONVERGENCE_TOL,
        max_terms=GATE_MAX_TERMS,
        max_seconds=GATE_MAX_SECONDS,
    )
    assert reference.exact is False, "a self-converged reference must never claim exactness"
    assert reference.method.startswith("self_converged")
    assert reference.evidence["converged"] is True, (
        f"theta_h={theta_label}: the procedure did not converge on C's grid; "
        f"evidence={reference.evidence['points']}"
    )

    true_error = abs(reference.value - exact)
    assert true_error <= driver.SELF_CONVERGENCE_TOL * bench_b._UNCERTAINTY_SLACK, (
        f"theta_h={theta_label}: self-converged {reference.value!r} vs exact {exact!r} "
        f"(true error {true_error:.3e}) — the plateau the procedure stopped on is not "
        "the right answer"
    )
    assert reference.uncertainty is not None
    assert true_error <= max(
        reference.uncertainty * bench_b._UNCERTAINTY_SLACK, bench_b._FP_NOISE_FLOOR
    ), (
        f"theta_h={theta_label}: estimated uncertainty {reference.uncertainty:.3e} "
        f"understates the true error {true_error:.3e} by more than "
        f"{bench_b._UNCERTAINTY_SLACK:g}x, so the estimate is not reportable"
    )
    # The bar the benchmark actually reports against.
    assert true_error < driver.ACCURACY_EPSILON


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_the_reference_refuses_to_declare_convergence_at_full_depth(theta_label, theta):
    """The honesty property C depends on, at the depth where it is hardest.

    A deep circuit in the hard interior climbs toward the truth from one side,
    with the successive differences shrinking by only ~2-3x per dyadic step. That
    is exactly the shape a plateau test can be fooled by — and the consequence of
    being fooled would be a reference that is 1e-1 wrong while claiming a 1e-3
    uncertainty, i.e. a completely fictitious "reproduces the reference within
    0.01" headline.

    So on the affordable part of the deep grid the procedure must come back
    `converged=False`, must be refused by `driver.reference_is_claimable`, and
    must report an uncertainty *above* the plan's bar — which is the reason it is
    refused. Measured on this sublattice at 20 steps, against the exact
    statevector, over the whole three-point grid:

    | theta_h | value at 2^-12 | exact | true error | estimate | ratio |
    |---|---|---|---|---|---|
    | 7pi/32 | +0.498910481 | +0.594614476873 | 9.57e-2 | 2.60e-1 | 2.7x over |
    | 5pi/16 | +0.034707098 | +0.044136003756 | 9.43e-3 | 3.02e-2 | 3.2x over |

    Conservative at both angles *here* — but see the next test, where the same
    procedure stopped early by a budget understates the true error instead.
    """
    exact = _exact(theta)
    reference = bench_b.self_converged_reference(
        _spec(theta).to_circuit(),
        _observable(),
        grid=DEEP_COEFF_GRID,
        state=STATE,
        direction=DIRECTION,
        tol=driver.SELF_CONVERGENCE_TOL,
        max_terms=GATE_MAX_TERMS,
        # No wall-clock budget: the three-point grid *is* the budget, and a
        # seconds guard would make which points ran depend on machine load.
        max_seconds=None,
    )
    true_error = abs(reference.value - exact)
    assert reference.exact is False
    # Every grid point must have run. (`stopped_early` is *not* the check: the
    # budget guard is also evaluated after the last point, where there is
    # nothing left to run, so it can be set even on a complete sweep.)
    assert len(reference.evidence["points"]) == len(DEEP_COEFF_GRID), (
        f"theta_h={theta_label}: only {len(reference.evidence['points'])} of "
        f"{len(DEEP_COEFF_GRID)} grid points ran — "
        f"{reference.evidence['stopped_early']}"
    )
    assert reference.evidence["converged"] is False, (
        f"theta_h={theta_label}, {DEEP_STEPS} steps: the procedure claimed convergence "
        f"on the loose deep grid {[driver.dyadic_label(e) for e in DEEP_COEFF_GRID]} "
        f"while the value {reference.value!r} is {true_error:.3e} from the exact "
        f"{exact!r}. That is the failure mode the imported plateau test exists to "
        "prevent."
    )
    assert driver.reference_is_claimable(reference) is False, (
        f"theta_h={theta_label}: the driver would have quoted this reference against "
        f"its {driver.ACCURACY_EPSILON:g} bar while it is {true_error:.3e} from the truth"
    )
    assert reference.uncertainty is not None
    assert reference.uncertainty > driver.ACCURACY_EPSILON, (
        f"theta_h={theta_label}: the sweep reports an uncertainty "
        f"{reference.uncertainty:.3e} inside the {driver.ACCURACY_EPSILON:g} bar while "
        "declining to declare convergence — the two verdicts must not disagree"
    )
    assert true_error <= max(
        reference.uncertainty * bench_b._UNCERTAINTY_SLACK, bench_b._FP_NOISE_FLOOR
    ), (
        f"theta_h={theta_label}: the reported uncertainty {reference.uncertainty:.3e} "
        f"understates the true error {true_error:.3e} by more than "
        f"{bench_b._UNCERTAINTY_SLACK:g}x over the full grid"
    )


#: `max_terms` that stops the `theta_h = 5pi/16` deep sweep after its **second**
#: point. Term counts are load-independent (0, 510, 67 573 at `2^-8/-10/-12` on
#: the 20-qubit sublattice at 20 steps), so gating on terms rather than seconds
#: makes the early stop reproducible instead of machine-dependent: after `2^-10`
#: the observed growth factor is `510/max(1, 0) = 510`, and `510 x 510 = 260 100`
#: is over this cap while `510` itself is not.
BUDGET_STOP_MAX_TERMS = 30_000

#: The measured understatement factor at that stop, so the assertion below has a
#: number behind it rather than just an inequality: reference `+0.004546671`,
#: exact `+0.044136003756`, true error `3.96e-2`, reported uncertainty `4.55e-3`.
MEASURED_BUDGET_STOP_UNDERSTATEMENT = 8.7


def test_the_published_anchor_is_recorded_and_not_transcribed():
    """The upstream exact benchmark is *located*, not copied in.

    Plan global rule 1 forbids a reference value that cannot be traced to a
    fetched source, and `examples/data/references/README.md` sharpens it: "the
    header is the citation, so it must be written from the fetch, not from
    memory." Only a summarizing fetch was available in the environment where
    this was checked, so `driver.PUBLISHED_ANCHOR` records the URL, the
    column-to-observable mapping taken from the paper's figure captions, and the
    reason nothing was checked in — and this test pins that no numbers leaked
    into it.

    It also pins the two structural facts the report leans on: the published
    exact file's `5a` column is `<Z_62>` at **9 steps** (which is why the depth
    ladder has a 9-step rung), and there is **no `5b` column**, i.e. no exact
    20-step value exists upstream either.
    """
    anchor = driver.PUBLISHED_ANCHOR
    assert anchor["checked_in"] is False
    assert anchor["why_not_checked_in"]
    assert "2308.05077" in anchor["paper"]
    assert anchor["data_url"].startswith("https://")
    assert "9 Trotter steps" in anchor["columns"]["5a"]
    assert "ABSENT" in anchor["columns"]["5b"]
    # No transcribed data: the anchor must carry no float-looking payload.
    flat = json.dumps(anchor)
    assert not re.search(r"\d\.\d{6,}", flat), (
        "PUBLISHED_ANCHOR contains what looks like a transcribed value; the whole "
        "point is that it records where the data is, not what it says"
    )
    # And the depth ladder actually has the rung the anchor points at.
    assert 9 in driver.STEP_POINTS
    # Both angles sit on the published k*pi/32 grid, so a later byte-exact fetch
    # lines up row for row with no interpolation.
    for label, theta in driver.THETA_POINTS:
        k = theta / (math.pi / 32)
        assert k == pytest.approx(round(k), abs=1e-12), (
            f"theta_h={label} ({theta!r}) is not a multiple of pi/32, so the "
            "published grid could not be compared against it without interpolating"
        )


def test_a_budget_truncated_sweep_can_understate_the_true_error():
    """The estimate's bias flips sign when the sweep is stopped early.

    Over the full grid the successive-difference estimate *overstates* the true
    error (2.7x and 3.2x — the previous test's table). Stop the same sweep after
    two points and it *understates* it, because the estimate is then one
    difference taken before the series has moved.

    On the 20-qubit sublattice at `theta_h = 5pi/16`, 20 steps, the value
    sequence is `0.000000000`, `+0.004546671`, `+0.034707098` at `2^-8/-10/-12`:
    the first two differ by only 4.5e-3 because almost nothing has arrived yet,
    and the *third* jumps by 3.0e-2. A sweep stopped after the second point
    reports `uncertainty = 4.5e-3` while sitting `3.96e-2` from the exact
    `+0.044136003756` — 8.7x too small. At the off-grid `theta_h = 1.0` probed
    during development the same construction gave **15.7x** (reference
    `+0.001831526`, exact `+0.030567143`, uncertainty `1.83e-3`), past the 10x
    slack Benchmark B's §3.4 validation allows — so B's "a budget-truncated sweep
    reports a larger uncertainty, not a falsely confident one" does not survive
    to this depth.

    The plateau test still does its job (`converged` stays `False`, because two
    successive small differences were never seen), so nothing downstream is
    misled — and that is exactly the point: at this depth `converged=False` must
    be read as **"no usable uncertainty estimate"**, not as "a slightly weaker
    one". `driver.reference_is_claimable` encodes that reading, and
    `benchmarks/python/deep_trotter/README.md` states it next to every
    self-converged number it quotes.
    """
    theta = dict(THETA_POINTS)["5pi/16"]
    exact = _exact(theta)
    reference = bench_b.self_converged_reference(
        _spec(theta).to_circuit(),
        _observable(),
        grid=DEEP_COEFF_GRID,
        state=STATE,
        direction=DIRECTION,
        tol=driver.SELF_CONVERGENCE_TOL,
        max_terms=BUDGET_STOP_MAX_TERMS,
        max_seconds=None,
    )
    assert len(reference.evidence["points"]) == 2, (
        "this test needs the budget guard to stop the sweep after its second point; "
        f"{len(reference.evidence['points'])} ran, so the term counts it is calibrated "
        f"against have changed: {reference.evidence['points']}"
    )
    assert reference.evidence["stopped_early"] is not None
    assert reference.evidence["converged"] is False
    assert driver.reference_is_claimable(reference) is False

    true_error = abs(reference.value - exact)
    assert reference.uncertainty is not None
    understatement = true_error / reference.uncertainty
    assert understatement > 1.0, (
        f"the budget-truncated sweep's uncertainty {reference.uncertainty:.3e} no "
        f"longer understates the true error {true_error:.3e} (ratio "
        f"{understatement:.1f}x). That would be good news, but the README quotes the "
        "sign of this bias — re-derive it there."
    )
    assert understatement == pytest.approx(
        MEASURED_BUDGET_STOP_UNDERSTATEMENT, rel=0.2
    ), (
        f"the understatement factor moved from ~{MEASURED_BUDGET_STOP_UNDERSTATEMENT}x "
        f"to {understatement:.1f}x; update the README's number in the same commit"
    )


def test_self_converged_reference_refuses_an_unsafe_grid():
    """The residual-branch floor is enforced, not documented."""
    with pytest.raises(ValueError, match="MIN_SAFE_COEFF"):
        bench_b.self_converged_reference(
            _spec(dict(THETA_POINTS)["7pi/32"], SHALLOW_STEPS).to_circuit(),
            _observable(),
            grid=(2.0**-8, 1e-20),
        )


# --------------------------------------------------------------------------
# 4. The sanity envelope
# --------------------------------------------------------------------------


def _synthetic_record(
    *, final_terms: int, peak_terms: int | None, eps: float
) -> report.RunRecord:
    return report.RunRecord(
        engine="paulistrings",
        engine_version="test",
        n_qubits=driver.N_QUBITS,
        direction=DIRECTION,
        truncation={"min_abs_coeff": eps},
        propagation_time_s=1.0,
        final_terms=final_terms,
        provenance=report.Provenance(
            commit="test", dirty=False, cpu_model="test",
            python_version="3.11", rustc_version=None,
        ),
        peak_terms=peak_terms,
        extra={"theta_h_label": "7pi/32", "trotter_steps": 20},
    )


def _synthetic_record_at(steps: int, *, peak_terms: int, eps: float) -> report.RunRecord:
    record = _synthetic_record(final_terms=peak_terms, peak_terms=peak_terms, eps=eps)
    record.extra["trotter_steps"] = steps
    return record


def test_check_envelope_only_flags_the_headline_depth():
    """A shallower rung below the floor is expected, not a semantics fault.

    Measured: at 5 steps and `theta_h ~ 0.69` the sum peaks at 389 804 terms at
    `2^-14` and *saturates* at 2.1e6 by `2^-18` — genuinely below the handoff's
    1.2e6 floor at the plan's loosest cutoff, and genuinely correct (it
    reproduces the exact light-cone reference to 6e-15 there). Flagging that as
    "investigate semantics" would cry wolf on the one depth where the answer is
    provably right.
    """
    shallow = driver.check_envelope(
        _synthetic_record_at(5, peak_terms=389_804, eps=2.0**-14), 20
    )
    assert shallow is not None
    assert shallow.inside is False
    assert shallow.needs_investigation is False
    assert "expected at 5 of 20 steps" in shallow.verdict

    deep = driver.check_envelope(
        _synthetic_record_at(20, peak_terms=389_804, eps=2.0**-14), 20
    )
    assert deep is not None
    assert deep.needs_investigation is True
    assert "BELOW the envelope floor" in deep.verdict

    # Above the ceiling is never an investigation, at any depth.
    high = driver.check_envelope(
        _synthetic_record_at(20, peak_terms=47_644_820, eps=2.0**-16), 20
    )
    assert high is not None and high.needs_investigation is False


def test_check_envelope_scores_the_peak_not_the_final_count():
    """The envelope is about what a run has to *hold*, which is the peak.

    Measured motivation: at `theta_h = 1.0`, 20 steps, `2^-14`, this benchmark
    peaks at 1.53e7 resident terms and lands on 2.0e4 — three orders apart. A
    check that scored the final count would call that run "below the envelope
    floor" and demand a semantics investigation of a run that in fact held
    fifteen million terms.
    """
    collapsed = driver.check_envelope(
        _synthetic_record(final_terms=20_140, peak_terms=15_288_166, eps=2.0**-14)
    )
    assert collapsed is not None
    assert collapsed.scored_terms == 15_288_166
    assert collapsed.inside is False
    assert "above the envelope ceiling" in collapsed.verdict

    inside = driver.check_envelope(
        _synthetic_record(final_terms=2_399_125, peak_terms=3_237_089, eps=2.0**-14)
    )
    assert inside is not None and inside.inside is True

    # The failure the check exists for: a sum that never spread.
    emptied = driver.check_envelope(
        _synthetic_record(final_terms=1, peak_terms=1, eps=2.0**-16)
    )
    assert emptied is not None and emptied.inside is False
    assert "BELOW the envelope floor" in emptied.verdict
    assert "investigate semantics" in emptied.verdict


def test_check_envelope_only_scores_the_plans_cutoffs():
    """The three looser dyadics the driver prepends are below the band by design."""
    assert driver.check_envelope(
        _synthetic_record(final_terms=372, peak_terms=2_012, eps=2.0**-8)
    ) is None
    assert driver.check_envelope(
        _synthetic_record(final_terms=1, peak_terms=1, eps=2.0**-18)
    ) is not None


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_loose_cutoff_sanity_on_the_full_lattice(theta_label, theta):
    """`n = 127`, 20 steps, two loose dyadic cutoffs — seconds, not hours.

    This is the only place CI touches the real benchmark: the real lattice, the
    real depth, the real observable, at cutoffs cheap enough to run. It cannot
    check the 1.2e6-9.3e6 envelope itself (that needs `2^-14` and minutes), but
    it does check the two things a wrong setup would break first — that the
    tracked set grows steeply with the cutoff at all, and that the peak, not the
    final count, is the large number.
    """
    import time

    observable = observables.canonical_z_127()
    circuit = circuits.heavy_hex_kicked_ising(
        driver.N_QUBITS, trotter_steps=DEEP_STEPS, theta_h=theta
    )
    assert len(circuit) == DEEP_STEPS * (144 + driver.N_QUBITS), (
        "one gate per channel is the suite's construction rule (plan D10); the "
        "channel count must be steps x (edges + qubits)"
    )

    started = time.perf_counter()
    peaks, finals, values = [], [], []
    for eps in LOOSE_COEFFS:
        evolved, stats = observable.propagate_with_stats(
            circuit, harness.make_policy(min_abs_coeff=eps), direction=DIRECTION
        )
        peaks.append(stats.peak_terms)
        finals.append(stats.final_terms)
        values.append(complex(evolved.expectation(STATE)).real)
        del evolved
    elapsed = time.perf_counter() - started

    assert elapsed < LOOSE_BUDGET_S, (
        f"theta_h={theta_label}: the loose-cutoff sanity check took {elapsed:.1f} s, "
        f"over its {LOOSE_BUDGET_S:g} s budget — this test exists to be cheap"
    )
    assert peaks == sorted(peaks), (
        f"theta_h={theta_label}: peak term counts {peaks} fell while the cutoff "
        f"tightened along {[driver.dyadic_label(e) for e in LOOSE_COEFFS]}"
    )
    # Tightening two dyadic powers must admit substantially more strings; a flat
    # curve here would mean the sum is not spreading at all.
    assert peaks[-1] > 4 * peaks[0]
    assert peaks[-1] > 10_000, (
        f"theta_h={theta_label}: only {peaks[-1]} peak terms at "
        f"{driver.dyadic_label(LOOSE_COEFFS[-1])} on the full lattice at "
        f"{DEEP_STEPS} steps — the operator is not spreading, which points at the "
        "direction or the lattice, not at truncation"
    )
    # The peak/final gap the envelope check has to cope with.
    assert peaks[-1] >= finals[-1]
    assert all(abs(v) <= 1.0 + 1e-12 for v in values), (
        f"theta_h={theta_label}: |<Z_62>| > 1 ({values}) is impossible for a "
        "single-Pauli observable"
    )

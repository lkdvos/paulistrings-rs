"""Benchmark B — CI-safe correctness gate on a 20-qubit heavy-hex sublattice.

`benchmarks/python/bench_b_theta_sweep.py` is the real Benchmark B: `n = 127`,
five Trotter steps, six kick angles, an hour of wall time and a 17 GiB
statevector reference for one of its three observables. None of that can run in
CI, and none of it needs to: the *physics* and the *procedure* are the same at
`n = 20`, where the induced heavy-hex sublattice on device qubits `0..19` (19
edges, 3-colourable) is a real sub-piece of the Eagle map, and a dense
statevector answers **every** point exactly — all six `theta_h`, all three
observables, no cone reduction, no self-converged reference.

So this file pins, cheaply and in CI:

1. **Accuracy at every `theta_h`.** Tight truncation reproduces the exact
   statevector expectation (`test_tight_truncation_matches_the_statevector`).
2. **Convergence, not just accuracy.** Tightening `min_abs_coeff` and raising
   `max_weight` monotonically improve the error, which is plan §7 rule 4's
   convergence panel expressed as an assertion rather than a figure
   (`test_error_improves_as_truncation_tightens`, `..._as_the_weight_cap_rises`).
3. **The Clifford endpoints.** At `theta_h in {0, pi/2}` (with
   `theta_zz = -pi/2`) the circuit is Clifford, so the evolved observable is a
   *single* Pauli string and the expectation is an exact integer, cross-checked
   against stim's tableau simulator — the same gate that Benchmark A applies at
   `n = 127` (`test_clifford_endpoints_are_exact_integers`).
4. **The self-convergence methodology.** Benchmark B's weight-17 interior
   references are self-converged, not exact, because the 59-qubit cone puts an
   exact answer out of reach. A self-converged value is only worth anything if
   its stated uncertainty is honest, so here — where the exact answer *is*
   known — the driver's own `self_converged_reference` is run and its estimate
   compared against the true error
   (`test_self_convergence_estimate_tracks_the_true_error`).
5. **No drift between the gate and the driver.** The `theta_h` grid, the
   truncation floor and the reference routing are read out of the driver
   module, so a change there that this file does not cover fails here
   (`test_driver_constants`).

Runtime
-------
Dominated by the 18 cached Aer statevector references (~1.1 s each on the
reference host); every propagation in the file is ~10 ms, because at this size
and these observables the truncated sum *saturates* — see
`OBSERVABLE_SUPPORTS`. Whole file: well under a minute, with no
skip-unless-env-var escape hatch.

Why the oracle calls are wrapped in `_oracle`
--------------------------------------------
CI's python job installs numpy only; qiskit, qiskit-aer and stim come from the
optional `examples` extra. Rather than an `importorskip` at module level (which
would skip the engine-only assertions too), each oracle call goes through
`_oracle`, which turns `oracles.SkipOracle` — the module's own
"dependency-not-installed" signal — into a pytest skip. Everything that needs
no oracle still runs.
"""

from __future__ import annotations

import math
import sys
from functools import cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _extra_path in (_REPO_ROOT / "examples", _REPO_ROOT / "benchmarks" / "python"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

import bench_b_theta_sweep as driver  # noqa: E402
from common import circuits, harness, observables, oracles  # noqa: E402

#: The sublattice size. 20 gives a connected induced heavy-hex subgraph (19
#: edges, 3 colour classes — `heavy_hex_sublattice` *computes* connectivity and
#: raises for the sizes that leave a qubit isolated) and a 16 MiB statevector.
N = 20
TROTTER_STEPS = driver.TROTTER_STEPS
STATE = driver.STATE
DIRECTION = driver.DIRECTION

#: The driver's own angle grid, reused verbatim so the two cannot drift.
THETA_POINTS = driver.THETA_POINTS
CLIFFORD_THETA_LABELS = driver.CLIFFORD_THETA_LABELS

#: Observables. **Synthetic**, not the published Kim et al. supports: those live
#: on device qubits up to 126 and do not fit a 20-qubit sublattice.
#:
#: All three are low weight, and that is a runtime decision with a measured
#: basis. On this sublattice a low-weight seed's reachable set *saturates* — the
#: five-step propagation reaches every Pauli string it can and then stops
#: growing (measured: `z10` 34 754 terms, `zz_9_10` 26 137, `x10` 47 144), so
#: `min_abs_coeff = 1e-9` gives the untruncated answer in ~10 ms. A weight-10
#: seed does not saturate: the same sublattice, same five steps, with the
#: Clifford-evolved image of `Z_10` (which is exactly how the published
#: weight-10 operator was built — `IIIIIIYXXXYXXZIIZZII` here) needs 5.7e7 terms
#: and 12 s at a *loose* 1e-6 and never converges inside CI's budget. Benchmark
#: B pays that cost at `n = 127`; this gate does not.
#:
#: `x10` is not redundant with `z10`: a pure-`Z` seed commutes with every `ZZ`
#: generator, so nothing spreads until the first `rx`, while an `X` seed
#: anticommutes from the first layer. Both paths matter.
OBSERVABLE_SUPPORTS: dict[str, dict[int, str]] = {
    "z10": {10: "Z"},
    "zz_9_10": {9: "Z", 10: "Z"},
    "x10": {10: "X"},
}

#: Error bar for "tight truncation reproduces the exact answer". At `TIGHT_COEFF`
#: all three observables have saturated on this sublattice, so the measured
#: worst case over all 18 `(observable, theta_h)` points is ~4e-9 — two orders
#: inside this bar.
TIGHT_TOLERANCE = 1e-7

#: `min_abs_coeff` grid for the convergence assertions, loosest first.
COEFF_GRID = (1e-2, 1e-4, 1e-6, 1e-8)

#: The tight end used for the accuracy assertion. Above
#: `driver.MIN_SAFE_COEFF`, so the `cos(pi/2)` residual branch is pruned at the
#: Clifford points (see the driver's module docstring).
TIGHT_COEFF = 1e-9

#: `max_weight` grid, loosest (smallest cap) first, paired with a coefficient
#: cutoff loose enough that the weight cap is what binds.
WEIGHT_GRID = (2, 4, 6, 8, 10)
WEIGHT_SWEEP_COEFF = 1e-9


def _theta_ids(points) -> list[str]:
    return [f"theta_h={label}" for label, _ in points]


def _oracle(call, *args, **kwargs):
    """Run an oracle, turning a missing optional dependency into a skip.

    `oracles.SkipOracle` is raised when qiskit / qiskit-aer / stim is not
    installed; that is a capability gap, not a failed cross-check (see the
    module docstring).
    """
    try:
        return call(*args, **kwargs)
    except oracles.SkipOracle as exc:  # pragma: no cover - depends on the env
        pytest.skip(str(exc))


def _observable(name: str):
    return observables.pauli_sum_from_support(OBSERVABLE_SUPPORTS[name], N)


@cache
def _spec(theta: float) -> oracles.CircuitSpec:
    """The gate list, recorded once per angle.

    `record_gates` captures the same builder the engine uses, so the oracle and
    the engine are driven from one description rather than two transcriptions
    of it (plan §7 rule 6).
    """
    return oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N, TROTTER_STEPS, theta
    )


@cache
def _exact(name: str, theta: float) -> float:
    """Exact `<0| U^dagger O U |0>` by dense statevector simulation.

    Cached: 18 parametrized cases share six circuits and would otherwise pay
    for the same Aer run several times over.
    """
    value = _oracle(
        oracles.statevector_expectation, _spec(theta), _observable(name), STATE
    )
    assert abs(complex(value).imag) < 1e-12, (
        f"the statevector oracle returned a complex expectation {value!r} for a "
        "Hermitian observable"
    )
    return complex(value).real


def _propagate(name: str, theta: float, *, min_abs_coeff=None, max_weight=None):
    """One truncated Heisenberg propagation; returns `(value, stats)`."""
    observable = _observable(name)
    policy = harness.make_policy(max_weight=max_weight, min_abs_coeff=min_abs_coeff)
    evolved, stats = observable.propagate_with_stats(
        _spec(theta).to_circuit(), policy, direction=DIRECTION
    )
    return complex(evolved.expectation(STATE)).real, stats


# --------------------------------------------------------------------------
# 0. The lattice and the grid this file is pinned to
# --------------------------------------------------------------------------


def test_sublattice_is_a_connected_piece_of_the_eagle_map():
    edges = circuits.heavy_hex_sublattice(N)
    assert len(edges) == 19
    assert all(0 <= a < b < N for a, b in edges)
    # A proper edge colouring is a partition into matchings, i.e. the hardware
    # layers; degree <= 3 makes 3 achievable and this order achieves it.
    colours = circuits.heavy_hex_edge_coloring(edges)
    assert len(colours) == 3
    for group in colours:
        touched = [q for edge in group for q in edge]
        assert len(touched) == len(set(touched)), "a colour class is not a matching"


def test_driver_constants():
    """The gate and the driver must agree on the sweep's definition."""
    assert [label for label, _ in THETA_POINTS] == [
        "0", "0.2", "pi/8", "pi/4", "3pi/8", "pi/2"
    ]
    assert dict(THETA_POINTS)["pi/2"] == pytest.approx(math.pi / 2)
    assert dict(THETA_POINTS)["3pi/8"] == pytest.approx(3 * math.pi / 8)
    assert driver.TROTTER_STEPS == 5
    assert driver.N_QUBITS == 127
    assert driver.DIRECTION == "heisenberg"
    assert driver.STATE == "z+"
    # Every coefficient cutoff this suite uses must sit above the residual
    # branch `cos(pi/2)` leaves behind, or a Clifford-point run fans out.
    assert driver.MIN_SAFE_COEFF > abs(math.cos(math.pi / 2))
    assert min(driver.COEFF_GRID) >= driver.MIN_SAFE_COEFF
    assert driver.WEIGHT_SWEEP_COEFF >= driver.MIN_SAFE_COEFF
    assert min(COEFF_GRID + (TIGHT_COEFF, WEIGHT_SWEEP_COEFF)) >= driver.MIN_SAFE_COEFF
    assert min(driver.SELF_CONVERGENCE_GRID) >= driver.MIN_SAFE_COEFF
    assert list(driver.SELF_CONVERGENCE_GRID) == sorted(
        driver.SELF_CONVERGENCE_GRID, reverse=True
    )
    # Loosest-first ordering is `time_to_accuracy`'s documented contract.
    assert list(driver.COEFF_GRID) == sorted(driver.COEFF_GRID, reverse=True)
    assert list(driver.WEIGHT_GRID) == sorted(driver.WEIGHT_GRID)


def test_cos_pi_half_is_the_residual_the_coefficient_floor_guards():
    """The one floating-point fact the whole truncation floor rests on."""
    assert math.cos(math.pi / 2) == 6.123233995736766e-17
    assert math.cos(math.pi / 2) != 0.0
    # `theta_h = 0` is the benign endpoint: there the branch really is exact.
    assert math.sin(0.0) == 0.0 and math.cos(0.0) == 1.0


# --------------------------------------------------------------------------
# 1. Accuracy at every theta_h
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OBSERVABLE_SUPPORTS))
@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_tight_truncation_matches_the_statevector(name, theta_label, theta):
    exact = _exact(name, theta)
    value, stats = _propagate(name, theta, min_abs_coeff=TIGHT_COEFF)
    assert value == pytest.approx(exact, abs=TIGHT_TOLERANCE), (
        f"{name} at theta_h={theta_label}: truncated propagation {value!r} vs exact "
        f"{exact!r} ({stats.final_terms} terms kept at min_abs_coeff={TIGHT_COEFF:g}). "
        "A miss here is a physics error, not a truncation artefact at this cutoff: "
        "check the direction (heisenberg vs forward) and the contraction state."
    )


@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_the_reference_is_confirmed_by_an_independent_oracle(theta_label, theta):
    """Statevector on the full circuit vs. the cone-reduced light-cone oracle.

    Two different reductions of the same problem: `light_cone_exact` restricts
    the circuit to the observable's backward causal cone and evaluates *that*,
    so agreement checks the cone computation as well as the simulation. This is
    the oracle Benchmark B leans on hardest (it is `Z_62`'s and weight-10's
    reference at `n = 127`), so it is checked at every angle — for one
    observable rather than all three, since a second Aer run per case would
    double this file's wall time for no new information.
    """
    name = "x10"
    exact = _exact(name, theta)
    cone_value = _oracle(
        oracles.light_cone_exact,
        _spec(theta),
        _observable(name),
        TROTTER_STEPS,
        initial_state=STATE,
        method="statevector",
    )
    assert complex(cone_value).real == pytest.approx(exact, abs=1e-10)


# --------------------------------------------------------------------------
# 2. Convergence, not just accuracy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OBSERVABLE_SUPPORTS))
@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_error_improves_as_truncation_tightens(name, theta_label, theta):
    """Loosest-to-tightest `min_abs_coeff`: error down, term count up.

    Neither is asserted point-to-point — a single grid step can flatten out
    once the error is at the level of the *next* discarded coefficient shell,
    and at a Clifford point every point is already exact. What is asserted is
    the shape over the whole grid: the tightest cutoff is at least as accurate
    as the loosest (to a tolerance covering an exactly-resolved point), and the
    term count never falls when less is thrown away.
    """
    exact = _exact(name, theta)
    errors, terms = [], []
    for eps in COEFF_GRID:
        value, stats = _propagate(name, theta, min_abs_coeff=eps)
        errors.append(abs(value - exact))
        terms.append(stats.final_terms)

    assert terms == sorted(terms), (
        f"{name} at theta_h={theta_label}: term counts {terms} are not "
        f"non-decreasing along a loosest-to-tightest cutoff grid {COEFF_GRID}"
    )
    assert errors[-1] <= errors[0] + 1e-12, (
        f"{name} at theta_h={theta_label}: tightening min_abs_coeff from "
        f"{COEFF_GRID[0]:g} to {COEFF_GRID[-1]:g} made the error worse "
        f"({errors[0]:.3e} -> {errors[-1]:.3e}); the truncation is not converging"
    )
    assert errors[-1] <= TIGHT_TOLERANCE


@pytest.mark.parametrize("name", sorted(OBSERVABLE_SUPPORTS))
@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_error_improves_as_the_weight_cap_rises(name, theta_label, theta):
    """The second knob: raising `max_weight` must also converge.

    A weight cap is a *biased* truncation — it drops whole shells of the
    operator rather than the numerically smallest terms — so this sweep is the
    one that would expose a wrong `weight <= k` boundary or an off-by-one in the
    weight computation, neither of which the coefficient sweep can see.
    """
    exact = _exact(name, theta)
    errors, terms = [], []
    for cap in WEIGHT_GRID:
        value, stats = _propagate(
            name, theta, max_weight=cap, min_abs_coeff=WEIGHT_SWEEP_COEFF
        )
        errors.append(abs(value - exact))
        terms.append(stats.final_terms)

    assert terms == sorted(terms), (
        f"{name} at theta_h={theta_label}: term counts {terms} fell while the weight "
        f"cap rose along {WEIGHT_GRID}"
    )
    assert errors[-1] <= errors[0] + 1e-12, (
        f"{name} at theta_h={theta_label}: raising max_weight from {WEIGHT_GRID[0]} "
        f"to {WEIGHT_GRID[-1]} made the error worse ({errors[0]:.3e} -> "
        f"{errors[-1]:.3e})"
    )


# --------------------------------------------------------------------------
# 3. The Clifford endpoints
# --------------------------------------------------------------------------

_CLIFFORD_POINTS = tuple(
    (label, theta) for label, theta in THETA_POINTS if label in CLIFFORD_THETA_LABELS
)


@pytest.mark.parametrize("name", sorted(OBSERVABLE_SUPPORTS))
@pytest.mark.parametrize(
    "theta_label,theta", _CLIFFORD_POINTS, ids=_theta_ids(_CLIFFORD_POINTS)
)
def test_clifford_endpoints_are_exact_integers(name, theta_label, theta):
    """At a Clifford `theta_h` the answer is an integer, and stim agrees.

    A Clifford circuit maps one Pauli string to one Pauli string, so the
    evolved observable must have **exactly one** term with coefficient
    bit-exactly `+-1`, and its `|0...0>` expectation is `+1`, `-1` or `0`. This
    is the same assertion Benchmark A makes at `n = 127`; the diagnostic below
    distinguishes its two failure modes.
    """
    observable = _observable(name)
    evolved, stats = observable.propagate_with_stats(
        _spec(theta).to_circuit(),
        harness.make_policy(min_abs_coeff=driver.MIN_SAFE_COEFF),
        direction=DIRECTION,
    )
    assert stats.final_terms == 1, (
        f"{name} at theta_h={theta_label}: a Clifford circuit must map one Pauli "
        f"string to one, but {stats.final_terms} terms survived. More than one term "
        "points at an angle-boundary bug (a non-Clifford theta lets the cos(theta) "
        "residual survive truncation and fan out), not at an adjoint-ordering bug, "
        "which would keep the count at 1 and change only the sign."
    )
    coefficient = complex(evolved.coefficients_array()[0])
    assert coefficient.imag == 0.0
    assert abs(coefficient.real) == 1.0, (
        f"{name} at theta_h={theta_label}: the surviving coefficient is "
        f"{coefficient!r}, not bit-exactly +-1"
    )

    value = complex(evolved.expectation(STATE)).real
    assert value in (-1.0, 0.0, 1.0), (
        f"{name} at theta_h={theta_label}: <0|O'|0> = {value!r} for a single Pauli "
        "string must be exactly -1, 0 or +1"
    )
    stim_value = _oracle(
        oracles.stim_clifford_exact, _spec(theta), observable, initial_state=STATE
    )
    assert complex(stim_value).real == value, (
        f"{name} at theta_h={theta_label}: engine {value!r} vs stim tableau "
        f"{complex(stim_value).real!r}. Both are exact integers here, so any "
        "difference is a convention bug (direction ordering, or the Clifford-point "
        "angle), never a tolerance issue."
    )
    assert complex(stim_value).real == pytest.approx(_exact(name, theta), abs=1e-10)


def test_a_pure_z_observable_is_untouched_at_theta_h_zero():
    """`theta_h = 0` is the one point where the answer is hand-checkable.

    `rx(0)` is the identity and every `ZZ` generator commutes with `Z_10`, so
    the observable comes back unchanged and `<0|Z_10|0> = +1` — no simulator,
    no oracle, no truncation involved.
    """
    value, stats = _propagate("z10", 0.0, min_abs_coeff=driver.MIN_SAFE_COEFF)
    assert stats.final_terms == 1
    assert value == 1.0


# --------------------------------------------------------------------------
# 4. The self-convergence methodology
# --------------------------------------------------------------------------

#: The driver's own tolerance, not a local one: the point of this test is that
#: the procedure is honest **as Benchmark B runs it**. Retuning it here would
#: validate a different procedure. (Measured consequence of getting this wrong:
#: at `tol = 1e-6` the `z10`, `theta_h = pi/8` sweep produces successive deltas
#: 8.1e-4, 1.0e-4, 1.0e-5, 1.8e-6, 0.0 — only *one* of which is under 1e-6, so
#: the criterion is never met and the driver's grid runs out.)
_SELF_CONVERGENCE_TOL = driver.SELF_CONVERGENCE_TOL

#: How far the *estimated* uncertainty may understate the true error before the
#: estimate counts as dishonest — the driver's own constant, so the bar this
#: test enforces is the bar the driver's `summary.json` scores itself against.
#: A successive-difference estimate is not a bound, so exact bracketing cannot
#: be required; the README reports the measured ratios at `n = 127` for `Z_62`
#: and weight-10 alongside these.
_UNCERTAINTY_SLACK = driver._UNCERTAINTY_SLACK
_FP_NOISE_FLOOR = driver._FP_NOISE_FLOOR


@pytest.mark.parametrize("name", sorted(OBSERVABLE_SUPPORTS))
@pytest.mark.parametrize("theta_label,theta", THETA_POINTS, ids=_theta_ids(THETA_POINTS))
def test_self_convergence_estimate_tracks_the_true_error(name, theta_label, theta):
    """The weight-17 reference procedure, validated where the truth is known.

    Benchmark B's four interior weight-17 references are self-converged: the
    59-qubit cone makes an exact answer unreachable, so truncation is tightened
    until successive values agree and the last differences are reported as the
    uncertainty. That is only defensible if the estimate is honest, so here the
    identical procedure runs on a system whose exact answer a statevector gives,
    and the estimate is compared against the true error.
    """
    exact = _exact(name, theta)
    reference = driver.self_converged_reference(
        _spec(theta).to_circuit(),
        _observable(name),
        state=STATE,
        direction=DIRECTION,
        tol=_SELF_CONVERGENCE_TOL,
        # CI budgets, far below the driver's: on this sublattice these
        # observables saturate under 5e4 terms, so a sweep that reaches either
        # guard is itself the bug.
        max_terms=5_000_000,
        max_seconds=60.0,
    )
    assert reference.exact is False, "a self-converged reference must never claim exactness"
    assert reference.method.startswith("self_converged")
    assert reference.evidence["converged"] is True, (
        f"{name} at theta_h={theta_label}: the procedure did not converge on the "
        f"driver's grid; evidence={reference.evidence['points']}"
    )

    true_error = abs(reference.value - exact)
    assert true_error <= _SELF_CONVERGENCE_TOL * _UNCERTAINTY_SLACK, (
        f"{name} at theta_h={theta_label}: self-converged {reference.value!r} vs exact "
        f"{exact!r} (true error {true_error:.3e}) — the plateau the procedure stopped "
        "on is not the right answer"
    )
    assert reference.uncertainty is not None
    assert true_error <= max(
        reference.uncertainty * _UNCERTAINTY_SLACK, _FP_NOISE_FLOOR
    ), (
        f"{name} at theta_h={theta_label}: estimated uncertainty "
        f"{reference.uncertainty:.3e} understates the true error {true_error:.3e} by "
        f"more than {_UNCERTAINTY_SLACK:g}x, so the estimate is not reportable"
    )


def test_plateau_criterion_rejects_a_flat_value_with_a_growing_sum():
    """The criterion itself, pinned directly on hand-written point lists.

    `driver._plateau_is_real` is private, but it is the single decision that
    makes a self-converged weight-17 reference reportable or not, so it gets a
    unit test rather than only the integration coverage above. The four shapes
    are the ones actually observed in the `n = 127` sweep.
    """
    tol = 1e-5

    # 1. The measured failure mode (`Z_62`, theta_h = 0.2): value bit-identical
    #    across decades while the sum keeps growing, because no pure-`Z` term
    #    has arrived yet. Rejected.
    flat_growing = [{"final_terms": 59}, {"final_terms": 225}, {"final_terms": 728}]
    assert driver._plateau_is_real(flat_growing, [0.0, 0.0], tol) is False

    # 2. Saturated (`Z_62`, theta_h = pi/4): the sum stopped growing, so the
    #    plateau is the exact answer. Accepted.
    saturated = [{"final_terms": 2146372} for _ in range(3)]
    assert driver._plateau_is_real(saturated, [0.0, 0.0], tol) is True

    # 3. Moving slowly: both differences small and strictly nonzero. Accepted.
    moving = [{"final_terms": 1}, {"final_terms": 2}, {"final_terms": 3}]
    assert driver._plateau_is_real(moving, [1e-7, 2e-8], tol) is True

    # 4. An empty sum (weight-17 at a loose cutoff) is never a converged answer,
    #    even though it looks both saturated and perfectly flat.
    emptied = [{"final_terms": 0} for _ in range(3)]
    assert driver._plateau_is_real(emptied, [0.0, 0.0], tol) is False

    # A difference above tol is not a plateau at all, and three points are the
    # minimum evidence.
    assert driver._plateau_is_real(saturated, [1e-3, 0.0], tol) is False
    assert driver._plateau_is_real(saturated[:2], [0.0], tol) is False


def test_self_converged_reference_refuses_an_unsafe_grid():
    """The residual-branch floor is enforced, not documented."""
    with pytest.raises(ValueError, match="MIN_SAFE_COEFF"):
        driver.self_converged_reference(
            _spec(0.2).to_circuit(),
            _observable("z10"),
            grid=(1e-3, 1e-20),
        )

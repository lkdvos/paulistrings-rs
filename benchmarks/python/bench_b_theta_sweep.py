"""Benchmark B — the `theta_h` sweep at 5 Trotter steps.

Heavy-hex kicked Ising, `n = 127`, 5 Trotter steps, `theta_zz = -pi/2`, six kick
angles `theta_h in {0, 0.2, pi/8, pi/4, 3pi/8, pi/2}`, three observables: `Z_62`
(Fig. 4b), the weight-10 operator (Fig. 3b) and the weight-17 operator (Fig. 3c)
of Kim et al. (2023). For each `(observable, theta_h)` the driver

1. establishes a **reference value** (§Reference strategy below),
2. sweeps `min_abs_coeff` and, separately, `max_weight` loosest-to-tightest and
   records one `report.RunRecord` per grid point (warm, single-threaded, quiet
   logging — all enforced by `harness.run_propagation`),
3. optionally runs the same task through PauliPropagation.jl and checks
   **per-layer** term-count parity at matched truncation: cross-engine
   timing is reported only after every per-layer count matches, and
4. writes the records as JSON and the figures as SVG.

This module is a driver, not a pytest module: it defines no `test_*` function,
so `pytest benchmarks/python` imports it and collects nothing. The CI-safe
correctness gate for the same physics lives in
`python/paulistrings/tests/test_benchmark_b_sweep.py`, on a 20-qubit heavy-hex
sublattice where a dense statevector covers every point.

Reference strategy
------------------

A light-cone exact reference at every point for all three observables is
**not** feasible; the reference strategy is adapted per observable, using
the deciding measurements in `benchmarks/python/theta_sweep/README.md`:

`Z_62`
    Exact at every `theta_h`. The commutation-aware cone is 19 qubits, so
    `light_cone_exact(method="both")` runs *both* an Aer statevector on the
    reduced circuit and an untruncated Pauli propagation over the same cone and
    requires them to agree — two independent simulations, ~16 s per point.

weight-10
    Exact at every `theta_h`, by `light_cone_exact(method="statevector",
    max_statevector_qubits=30)`: the cone is 30 qubits (17 GiB, ~2 min per
    point). The untruncated-Pauli path over the *same* cone was measured and
    rejected — see `_MEASURED_W10_PAULI_PATH` below.

weight-17
    The cone is 59 qubits: dense simulation is impossible and untruncated Pauli
    propagation over it is far past the weight-10 wall. So

    - the two **endpoints** (`theta_h = 0` and `pi/2`) are Clifford points and
      get exact integers from `oracles.stim_clifford_exact`, and
    - the four **interior** points get a *self-converged* reference
      (`self_converged_reference`): truncation is tightened until successive
      values agree, and the reference carries its own convergence evidence and
      an uncertainty estimate. These are labelled `self_converged`, **never
      "exact"** — `Reference.exact` is `False` for them and the JSON records say
      so.

    The self-convergence procedure is validated where the exact answer *is*
    known: `--validate-convergence` runs it for `Z_62` and weight-10 and
    reports the true error against the exact reference next to the estimated
    uncertainty. `test_benchmark_b_sweep.py` does the same on the small
    sublattice.

Both endpoints must reproduce Benchmark A's integers exactly (`+1` for
weight-10, `-1` for weight-17 at `theta_h = pi/2`), which the driver asserts.

Every oracle runs in a **spawned child process**, and not for tidiness:
qiskit-aer's statevector simulator leaves behind an OpenMP pool that persists
for the life of the process, which would trip `harness.assert_single_threaded`
on every later timed run. The child's threads die with it and only plain data
crosses back. The weight-17 self-convergence child additionally gets
`REFERENCE_THREADS` Rayon workers — a reference is an oracle, not a timing
measurement, so the single-thread rule does not bind it, and the threads buy
reach at a tighter cutoff.

The `cos(pi/2)` trap
--------------------

`cos(pi/2) == 6.123233995736766e-17`, not zero, so at a Clifford point every
rotation leaves a numerically-dead residual branch and an *untruncated*
127-qubit propagation fans out without bound (Benchmark A's docstring records a
181 GB probe). Every sweep here therefore keeps `min_abs_coeff >= 1e-12`, which
prunes the residual every layer; `MIN_SAFE_COEFF` is that floor and the driver
refuses a grid that goes below it.

Usage
-----

::

    RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --help
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py          # full run
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py \
        --observables z62 --no-julia --out-dir /tmp/pilot                      # quick look

`RAYON_NUM_THREADS=1` must be in the environment **before** the interpreter
starts: Rayon builds its global pool at the first propagate and never resizes
it (`harness` module docstring). The driver refuses to run otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
# `julia_baseline` is this file's sibling; the driver may be run from anywhere.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from common import circuits, harness, observables, oracles, report  # noqa: E402

# --------------------------------------------------------------------------
# The benchmark's fixed parameters
# --------------------------------------------------------------------------

N_QUBITS = 127
TROTTER_STEPS = 5
STATE = "z+"
DIRECTION = "heisenberg"

#: The six kick angles, in the plan's order. The label is what goes into the
#: results JSON and the figures; the value is what goes into `rx`.
THETA_POINTS: tuple[tuple[str, float], ...] = (
    ("0", 0.0),
    ("0.2", 0.2),
    ("pi/8", math.pi / 8),
    ("pi/4", math.pi / 4),
    ("3pi/8", 3 * math.pi / 8),
    ("pi/2", math.pi / 2),
)

#: The two Clifford points (`theta_zz = -pi/2` is Clifford for any of them).
CLIFFORD_THETA_LABELS = frozenset({"0", "pi/2"})

OBSERVABLE_BUILDERS: dict[str, Callable[[], Any]] = {
    "z62": observables.canonical_z_127,
    "weight_10": observables.weight_10_operator,
    "weight_17": observables.weight_17_operator,
}

#: Backward-cone size per observable at 5 steps (commutation-aware cone; these
#: are *recomputed* by `light_cone` on every run, the numbers are here only so
#: the reference routing below reads as a decision rather than a magic number).
CONE_SIZES = {"z62": 19, "weight_10": 30, "weight_17": 59}

#: `min_abs_coeff` floor. Below this the `cos(pi/2) = 6.1e-17` residual branch
#: survives truncation at a Clifford point and the propagation fans out (see
#: the module docstring).
MIN_SAFE_COEFF = 1e-12

#: Statevector cap for the weight-10 cone. 30 qubits is `2**30 * 16 = 17 GiB`
#: of state, which Aer holds one copy of; measured 16.1 GiB peak RSS.
W10_STATEVECTOR_CAP = 30

#: Why the weight-10 reference is a statevector and not untruncated Pauli
#: propagation over the same cone. Measured on ccqlin038 at `theta_h = 0.2`,
#: single-threaded, by growing the applied-gate prefix of the 30-qubit reduced
#: cone circuit (`peak terms` after the first `m` applied gates):
_MEASURED_W10_PAULI_PATH = """\
m= 40 gates: peak 6.4e1      0.00 s
m= 80 gates: peak 5.2e4      0.03 s
m=120 gates: peak 4.3e6      0.93 s
m=160 gates: exceeded a 26 GiB address-space cap
full cone (305 gates in the reduced circuit): unreachable
statevector over the same cone: 125 s, 16.1 GiB — the cheaper path at every theta_h
"""

#: Truncation grids, **loosest first** — the ordering contract
#: `harness.time_to_accuracy` documents for its `first` selection.
COEFF_GRID: tuple[float, ...] = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
WEIGHT_GRID: tuple[int, ...] = (2, 4, 6, 8, 10, 12)

#: Per-`(observable, theta_h)` **recorded cuts** to the coefficient grid.
#:
#: Cost is driven by `theta_h`, not by the observable's weight: a larger kick
#: angle mixes more, so the same cutoff keeps far more terms. Measured on
#: ccqlin038 (single-threaded, warm; `benchmarks/python/theta_sweep/README.md` has the
#: full table), the tight end of `COEFF_GRID` runs off a cliff:
#:
#: | observable | theta_h | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
#: |---|---|---|---|---|---|
#: | `z62`       | 3pi/8 | 0.8 s | 1.8 s | 2.3 s | 2.4 s |
#: | `weight_10` | 0.2   | 0.1 s | 0.3 s | 1.1 s | 3.6 s |
#: | `weight_10` | pi/8  | 1.1 s | 6.7 s | 37 s  | 140 s |
#: | `weight_10` | pi/4  | 14 s  | 212 s | —     | —     |
#: | `weight_17` | 0.2   | 2.4 s | 16 s  | 98 s  | —     |
#: | `weight_17` | pi/8  | 37 s  | —     | —     | —     |
#:
#: The time-box policy is pilot, project, then shrink the grid and record the
#: cut, and this table *is* that record: a cut entry is a deliberate,
#: reviewable shortening, not an adaptive stop whose shape depends on machine
#: load. `z62` is never cut: its whole grid is under 3 s at every angle,
#: because a `Z` seed's reachable set saturates.
COEFF_GRID_CUTS: dict[tuple[str, str], tuple[float, ...]] = {
    ("weight_10", "0.2"): (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
    ("weight_10", "pi/8"): (1e-2, 1e-3, 1e-4, 1e-5, 1e-6),
    ("weight_10", "pi/4"): (1e-2, 1e-3, 1e-4),
    ("weight_10", "3pi/8"): (1e-2, 1e-3, 1e-4),
    ("weight_17", "0.2"): (1e-2, 1e-3, 1e-4, 1e-5),
    ("weight_17", "pi/8"): (1e-2, 1e-3, 1e-4),
    ("weight_17", "pi/4"): (1e-2, 1e-3, 1e-4),
    ("weight_17", "3pi/8"): (1e-2, 1e-3, 1e-4),
}


def coeff_grid_for(observable_name: str, theta_label: str, default: Sequence[float]):
    """The coefficient grid for one point, after the recorded cuts.

    A cut only ever *shortens* the default grid (and only at the tight,
    expensive end), so `--coeff-grid` still governs the loose end everywhere.
    """
    cut = COEFF_GRID_CUTS.get((observable_name, theta_label))
    if cut is None:
        return list(default)
    return [eps for eps in default if eps >= min(cut)]


#: Fallback `min_abs_coeff` for the `max_weight` sweep, used only when no
#: coefficient grid is run at all (`--coeff-grid` with no values).
#:
#: Normally the cutoff is read off the coefficient sweep that just ran, by
#: `_weight_sweep_coeff` — see it for the rule and why it needs no cost table.
#: Pairing the weight sweep with a fixed tight cutoff instead would reintroduce
#: the cost cliff `COEFF_GRID_CUTS` exists to avoid, so the pairing varies per
#: `(observable, theta_h)` and every record carries the cutoff it ran at.
WEIGHT_SWEEP_COEFF = 1e-10

#: Accuracy bar for `harness.time_to_accuracy`'s "cheapest truncation that
#: suffices" selection. 1e-3 is the scale the utility-experiment plots are read
#: at; it is a reporting choice, not a correctness threshold.
ACCURACY_EPSILON = 1e-3

#: A run whose looser neighbour already took longer than this is timed without
#: the extra untimed warm-up pass (which would double its cost). Recorded per
#: record as `extra["warm"]`.
WARMUP_TIME_BUDGET_S = 3.0

#: `min_abs_coeff` for the cross-engine parity check. A **power of ten, not a
#: dyadic**: this repo drops `|c| <= eps` and PauliPropagation.jl keeps
#: `|c| == eps` (benchmarks/julia/README.md §P3), a divergence that is
#: measure-zero for a non-dyadic threshold and *not* measure-zero for
#: `2**-14`-style cutoffs at Clifford angles.
PARITY_COEFF = 1e-4

#: The cutoffs the parity leg is run at. More than one because "identical term
#: counts at one cutoff" is a much weaker statement than "identical term counts
#: along a truncation sweep": three cutoffs spanning two decades give a real
#: two-engine term-count-vs-truncation curve, which is what
#: `report.plot_term_count_vs_truncation` is shaped for.
PARITY_COEFFS: tuple[float, ...] = (1e-3, 1e-4, 1e-5)

#: Julia is slow to start (JIT) and the 1355-gate task is not cheap; one warm
#: repeat is enough for a parity check, which is about term counts.
PARITY_WARM_REPEATS = 1


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """One reference value, and an honest label for how it was obtained.

    `exact` is `True` only for an oracle that computes the answer with no
    truncation anywhere: a Clifford tableau, a dense statevector over a causal
    cone, or untruncated Pauli propagation over one. A self-converged value has
    `exact=False` and a non-`None` `uncertainty`, and every consumer — the JSON
    records, the README tables, the figures' captions — must carry that
    distinction through.
    """

    value: float
    method: str
    exact: bool
    uncertainty: float | None = None
    seconds: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_value": self.value,
            "reference_method": self.method,
            "reference_exact": self.exact,
            "reference_uncertainty": self.uncertainty,
            "reference_seconds": self.seconds,
            **({"reference_evidence": self.evidence} if self.evidence else {}),
        }


#: The grid `self_converged_reference` tightens along. Coefficient cutoff only:
#: a weight cap is a *biased* truncation for these observables (the exact
#: evolved operator has weight up to ~50 at 5 steps), so tightening it converges
#: from one side and its successive differences understate the remaining error.
SELF_CONVERGENCE_GRID: tuple[float, ...] = (
    1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10,
)

#: Budget guards on the self-convergence loop. Each tightening step multiplies
#: the term count by roughly 3-10x, so a point that has already crossed either
#: budget makes the *next* one unaffordable; the loop stops there and reports
#: `converged=False` rather than being killed by the OOM killer mid-sweep.
#: `None` disables a guard.
SELF_CONVERGENCE_MAX_TERMS: int | None = 200_000_000
SELF_CONVERGENCE_MAX_SECONDS: float | None = 300.0

#: How close two successive values must be, twice running, for the plateau to
#: count as reached. Named (rather than left as a default argument) because
#: `test_benchmark_b_sweep.py` validates the procedure *at the value the driver
#: uses*: a test run at a tighter tolerance would be validating a different
#: procedure. 1e-5 is one order below the 1e-4 scale the utility-experiment
#: expectation values are read at, and — measured on the 20-qubit sublattice,
#: where the exact answer is known — the resulting estimate never understated
#: the true error.
SELF_CONVERGENCE_TOL = 1e-5


#: The growth factor a *projected* next grid point is assumed to cost at least.
#: Measured decade-to-decade term-count ratios on this benchmark run 3x-70x
#: (`weight_10` at `theta_h = pi/4`: 40 129 -> 2 686 388 -> 92 751 483), so 3 is
#: a floor, not an estimate — the observed ratio is used when it is larger.
_MIN_PROJECTED_GROWTH = 3.0


def _plateau_is_real(
    points: Sequence[dict[str, Any]], deltas: Sequence[float], tol: float
) -> bool:
    """Has the self-convergence sweep reached a plateau worth believing?

    The obvious criterion — "the last two successive differences are below
    `tol`" — is **not sufficient**, and this is a measured failure, not a
    hypothetical one. Validated against the exact `Z_62` reference at
    `theta_h = 0.2`, that criterion declares convergence with an estimated
    uncertainty of *exactly zero* while the value is still 5.6e-7 away from the
    truth: the expectation is bit-identical at `min_abs_coeff` 1e-3 through
    1e-7 — a four-decade plateau — and only moves at 1e-8.

    Why: at a small kick angle the terms that contribute to `<0|O|0>` are the
    ones that have been rotated all the way to pure `Z`, and each rotation costs
    a factor `sin(theta_h)`. Loosening the cutoff by a decade admits many new
    terms, but for several decades *none of them* is pure-`Z`, so the
    expectation does not move at all while the sum keeps growing. An
    exactly-zero difference there means "no relevant term has arrived yet", not
    "the series has converged".

    The two situations are distinguishable by the **term count**:

    - the sum has *saturated* (successive term counts equal) — every Pauli
      string above the cutoff is already present, so the value cannot move
      again and the plateau is the exact answer; or
    - the value is *moving but slowly* (both differences below `tol` and
      strictly nonzero) — the ordinary picture of a converging series.

    A plateau that is neither — flat value, still-growing sum — is rejected, and
    the sweep goes on to a tighter cutoff.

    One further rejection: a sum that has been truncated to **zero terms** is
    not a converged answer, however flat it looks. Two successive empty sums
    would otherwise read as "saturated" (equal term counts) with a difference of
    exactly zero, and the weight-17 observable really does empty out at the
    loose end of the grid (`min_abs_coeff` 1e-2 and 1e-3 at `theta_h = pi/4`
    both keep 0 terms).
    """
    if len(deltas) < 2 or len(points) < 3:
        return False
    if points[-1]["final_terms"] == 0:
        return False
    if not (deltas[-1] < tol and deltas[-2] < tol):
        return False
    saturated = points[-1]["final_terms"] == points[-2]["final_terms"]
    moving_slowly = deltas[-1] > 0.0 and deltas[-2] > 0.0
    return saturated or moving_slowly


def _budget_stop(
    points: Sequence[dict[str, Any]],
    eps: float,
    total_seconds: float,
    max_terms: int | None,
    max_seconds: float | None,
) -> str | None:
    """Should the self-convergence loop stop before the next, tighter point?

    Checked **before** running the next point, not after, and by *projection*:
    a decade of extra cutoff multiplies the term count several-fold, so a
    reactive "stop once the budget is exceeded" guard reliably overshoots by one
    run — and one run is where all the cost is. The projection uses the observed
    growth ratio between the last two points, floored at
    `_MIN_PROJECTED_GROWTH`.
    """
    if not points:
        return None
    last = points[-1]
    if max_terms is not None and last["final_terms"] > max_terms:
        return (
            f"budget: {last['final_terms']} terms at min_abs_coeff={eps:g} is already "
            f"over max_terms={max_terms}"
        )
    if max_seconds is not None and total_seconds > max_seconds:
        return (
            f"budget: {total_seconds:.0f} s spent by min_abs_coeff={eps:g} is already "
            f"over max_seconds={max_seconds:.0f}"
        )
    if len(points) < 2:
        return None
    previous = points[-2]
    growth = max(
        _MIN_PROJECTED_GROWTH,
        last["final_terms"] / max(1, previous["final_terms"]),
    )
    if max_terms is not None and last["final_terms"] * growth > max_terms:
        return (
            f"budget: the next tightening past min_abs_coeff={eps:g} projects to "
            f"~{last['final_terms'] * growth:.3g} terms (growth {growth:.1f}x), over "
            f"max_terms={max_terms}"
        )
    if max_seconds is not None and total_seconds + last["seconds"] * growth > max_seconds:
        return (
            f"budget: the next tightening past min_abs_coeff={eps:g} projects to "
            f"~{last['seconds'] * growth:.0f} s (growth {growth:.1f}x), over the "
            f"remaining {max_seconds - total_seconds:.0f} s of max_seconds="
            f"{max_seconds:.0f}"
        )
    return None


def self_converged_reference(
    circuit,
    observable,
    *,
    grid: Sequence[float] = SELF_CONVERGENCE_GRID,
    state: str = STATE,
    direction: str = DIRECTION,
    tol: float = SELF_CONVERGENCE_TOL,
    max_terms: int | None = SELF_CONVERGENCE_MAX_TERMS,
    max_seconds: float | None = SELF_CONVERGENCE_MAX_SECONDS,
    log: Callable[[str], None] | None = None,
) -> Reference:
    """Tighten `min_abs_coeff` until successive expectation values agree.

    Runs the grid in order (loose to tight) and stops as soon as the **last two**
    successive differences are both below `tol`. Requiring two, not one, is what
    makes the criterion resistant to an accidental crossing: a single small
    difference can happen while the value is still moving, two in a row
    essentially cannot.

    The returned `Reference` has `exact=False`, `uncertainty = max` of those two
    differences (an estimate of the remaining truncation bias, not a bound), and
    `evidence` carrying every `(min_abs_coeff, value, terms, seconds)` point so
    a reader can see the plateau rather than take it on trust. If the criterion
    is never met the reference still comes back — with `uncertainty` set to the
    last observed difference and `evidence["converged"] = False`, so a consumer
    can refuse it.

    `max_terms` / `max_seconds` stop the loop when a point has already crossed
    the budget, since the next tightening costs several times as much. Stopping
    is recorded in `evidence["stopped_early"]` and leaves
    `evidence["converged"]` `False`: a budget-truncated sweep is a *failed*
    reference attempt, never a quietly weakened one.

    Validated against a known-exact answer by `--validate-convergence` (`Z_62`
    and weight-10 at `n = 127`) and by `test_benchmark_b_sweep.py` on the
    20-qubit sublattice.
    """
    if any(eps < MIN_SAFE_COEFF for eps in grid):
        raise ValueError(
            f"self-convergence grid {list(grid)} goes below MIN_SAFE_COEFF="
            f"{MIN_SAFE_COEFF:g}; at a Clifford angle the cos(pi/2) residual branch "
            "then survives truncation and the propagation fans out (module docstring)"
        )
    points: list[dict[str, Any]] = []
    values: list[float] = []
    deltas: list[float] = []
    converged = False
    stopped_early: str | None = None
    total_seconds = 0.0

    for eps in grid:
        policy = harness.make_policy(min_abs_coeff=eps)
        start = time.perf_counter()
        evolved, stats = observable.propagate_with_stats(
            circuit, policy, direction=direction
        )
        value = complex(evolved.expectation(state)).real
        seconds = time.perf_counter() - start
        total_seconds += seconds
        del evolved
        values.append(value)
        if len(values) > 1:
            deltas.append(abs(values[-1] - values[-2]))
        points.append(
            {
                "min_abs_coeff": eps,
                "value": value,
                "final_terms": stats.final_terms,
                "peak_terms": stats.peak_terms,
                "seconds": seconds,
                "delta_vs_previous": deltas[-1] if deltas else None,
            }
        )
        if log is not None:
            delta_text = "        —" if not deltas else f"{deltas[-1]:9.2e}"
            log(
                f"      eps={eps:.0e}  <O>={value:+.9f}  Δ={delta_text}  "
                f"terms={stats.final_terms:>9}  {seconds:7.2f}s"
            )
        if _plateau_is_real(points, deltas, tol):
            converged = True
            break
        stopped_early = _budget_stop(points, eps, total_seconds, max_terms, max_seconds)
        if stopped_early is not None:
            break

    if not values:  # pragma: no cover - an empty grid is a caller bug
        raise ValueError("self-convergence grid is empty")
    if stopped_early is not None and log is not None:
        log(f"      stopped before convergence — {stopped_early}")

    uncertainty = max(deltas[-2:]) if deltas else None
    return Reference(
        value=values[-1],
        method="self_converged(min_abs_coeff)",
        exact=False,
        uncertainty=uncertainty,
        seconds=total_seconds,
        evidence={
            "criterion": (
                f"two successive |Δ<O>| below tol={tol:g} while tightening min_abs_coeff"
            ),
            "converged": converged,
            "stopped_early": stopped_early,
            "tol": tol,
            "points": points,
        },
    )


def clifford_reference(spec, observable, *, log=None) -> Reference:
    """Exact integer at a Clifford `theta_h`, from stim's tableau simulator."""
    start = time.perf_counter()
    value = complex(
        oracles.stim_clifford_exact(spec, observable, initial_state=STATE)
    )
    seconds = time.perf_counter() - start
    if abs(value.imag) > 1e-12:  # pragma: no cover - a Hermitian observable cannot
        raise AssertionError(f"Clifford oracle returned a complex value {value!r}")
    nearest = round(value.real)
    if abs(value.real - nearest) > 1e-12:
        raise AssertionError(
            f"stim_clifford_exact returned {value.real!r} at a Clifford point; a "
            "single-term unit-coefficient observable there must be an exact integer "
            "(+1, -1 or 0). A non-integer means the circuit is not Clifford: check "
            "theta_zz and theta_h against multiples of pi/2."
        )
    if log is not None:
        log(f"      stim tableau: {float(nearest):+.1f} exactly ({seconds:.2f} s)")
    return Reference(
        value=float(nearest),
        method="stim_clifford_exact",
        exact=True,
        seconds=seconds,
        evidence={"raw_value": value.real},
    )


def light_cone_reference(spec, observable, *, method: str, cap: int, log=None) -> Reference:
    """Exact value by causal-cone reduction (`oracles.light_cone_exact`)."""
    start = time.perf_counter()
    value = complex(
        oracles.light_cone_exact(
            spec,
            observable,
            TROTTER_STEPS,
            initial_state=STATE,
            method=method,
            max_statevector_qubits=cap,
        )
    )
    seconds = time.perf_counter() - start
    if log is not None:
        log(f"      light cone ({method}): {value.real:+.12f} ({seconds:.1f} s)")
    return Reference(
        value=value.real,
        method=f"light_cone_exact:{method}",
        exact=True,
        seconds=seconds,
    )


# --------------------------------------------------------------------------
# Oracle isolation: why the dense oracles run in a child process
# --------------------------------------------------------------------------
#
# `harness.run_propagation(threads=1)` asserts that the process has gained at
# most one thread since `harness` was imported — Rayon's single pinned worker.
# qiskit-aer's statevector simulator spawns an OpenMP pool (one thread per core
# on this host) that *persists* for the life of the process, so computing a
# statevector reference in-process would make every later timed run fail that
# assert. Forcing `OMP_NUM_THREADS=1` instead would make the 30-qubit weight-10
# reference far too slow to run six times.
#
# So every oracle call happens in a spawned child whose threads die with it,
# and only plain data crosses back. The child re-imports this module (spawn,
# not fork — a forked child would inherit the parent's Rayon pool), which costs
# a couple of seconds per reference against 2-125 s of oracle work.


def _reference_worker(
    kind: str, observable_name: str, theta: float, method: str, cap: int
) -> dict[str, Any]:
    """Child-process entry point. Returns a `Reference`'s fields as plain data."""
    observable = OBSERVABLE_BUILDERS[observable_name]()
    if kind == "self_converged":
        reference = self_converged_reference(
            circuits.heavy_hex_kicked_ising(
                N_QUBITS, trotter_steps=TROTTER_STEPS, theta_h=theta
            ),
            observable,
        )
        return {
            "value": reference.value,
            "method": reference.method,
            "exact": reference.exact,
            "uncertainty": reference.uncertainty,
            "seconds": reference.seconds,
            "evidence": reference.evidence,
        }
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N_QUBITS, TROTTER_STEPS, theta
    )
    if kind == "clifford":
        reference = clifford_reference(spec, observable)
    elif kind == "light_cone":
        reference = light_cone_reference(spec, observable, method=method, cap=cap)
    else:  # pragma: no cover - a caller bug
        raise ValueError(f"unknown reference kind {kind!r}")
    return {
        "value": reference.value,
        "method": reference.method,
        "exact": reference.exact,
        "uncertainty": reference.uncertainty,
        "seconds": reference.seconds,
        "evidence": reference.evidence,
    }


#: How far the *estimated* uncertainty may understate the true error before the
#: self-convergence estimate counts as dishonest. A successive-difference
#: estimate is a heuristic, not a bound, so exact bracketing cannot be required;
#: one order of magnitude is the bar, and `test_benchmark_b_sweep.py` asserts
#: the same one on the 20-qubit sublattice.
_UNCERTAINTY_SLACK = 10.0

#: Floating-point floor for that comparison. A plateau reached by *saturation*
#: has an uncertainty of legitimately zero and a true error at the summation
#: rounding level (measured: 1.3e-14 for `Z_62` at `theta_h = pi/4`), which is
#: agreement, not a failed estimate.
_FP_NOISE_FLOOR = 1e-12

#: Rayon workers the reference child is allowed. A reference is an *oracle*, not
#: a timing measurement, so the single-thread rule (which exists to make
#: cross-engine wall times comparable) does not apply to it: nothing
#: about the reference's wall time is ever reported as a benchmark number. What
#: the threads buy is *reach* — a tighter cutoff in the weight-17
#: self-convergence, hence a better-converged reference. The parent's own pool
#: is untouched: it was built at the parent's first propagate and Rayon never
#: resizes, and a `spawn` child gets a fresh process with the overridden
#: variable in its environment.
REFERENCE_THREADS = 16


def _oracle_reference(
    kind: str,
    observable_name: str,
    theta: float,
    *,
    method: str = "",
    cap: int = 0,
    in_process: bool = False,
    threads: int = 1,
    log=None,
) -> Reference:
    """Compute one oracle reference, by default in an isolated child process."""
    start = time.perf_counter()
    if in_process:
        payload = _reference_worker(kind, observable_name, theta, method, cap)
    else:
        context = multiprocessing.get_context("spawn")
        saved = os.environ.get("RAYON_NUM_THREADS")
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=context
            ) as pool:
                payload = pool.submit(
                    _reference_worker, kind, observable_name, theta, method, cap
                ).result()
        finally:
            if saved is None:  # pragma: no cover - the driver always sets it
                os.environ.pop("RAYON_NUM_THREADS", None)
            else:
                os.environ["RAYON_NUM_THREADS"] = saved
    reference = Reference(**payload)
    if log is not None:
        log(
            f"      {reference.method}: {reference.value:+.12f} "
            f"({time.perf_counter() - start:.1f} s wall, "
            f"{'in-process' if in_process else 'isolated child'})"
        )
    return reference


def resolve_reference(
    observable_name: str,
    theta_label: str,
    observable,
    *,
    in_process: bool = False,
    log=None,
) -> Reference:
    """Route one `(observable, theta_h)` point to its reference oracle.

    The routing is the adapted strategy in the module docstring, and it is a
    routing *decision* rather than a fallback chain: nothing here silently
    degrades an exact reference into a self-converged one. A point that cannot
    be answered exactly is answered by `self_converged_reference` because that
    was decided for it up front, and the resulting `Reference.exact` is `False`.
    """
    theta = dict(THETA_POINTS)[theta_label]
    if theta_label in CLIFFORD_THETA_LABELS:
        return _oracle_reference(
            "clifford", observable_name, theta, in_process=in_process, log=log
        )
    if observable_name == "z62":
        # Two independent simulations of the reduced 19-qubit problem (Aer
        # statevector and untruncated Pauli propagation) required to agree.
        return _oracle_reference(
            "light_cone", observable_name, theta,
            method="both", cap=CONE_SIZES["z62"], in_process=in_process, log=log,
        )
    if observable_name == "weight_10":
        return _oracle_reference(
            "light_cone", observable_name, theta,
            method="statevector", cap=W10_STATEVECTOR_CAP,
            in_process=in_process, log=log,
        )
    if observable_name == "weight_17":
        # No exact interior reference exists: the cone is 59 qubits, and
        # untruncated propagation over it is past the weight-10 wall recorded
        # in `_MEASURED_W10_PAULI_PATH`. Self-converged, and labelled as such.
        # Run in the reference child so it can use `REFERENCE_THREADS` workers
        # and reach a tighter cutoff than a pinned run could afford.
        return _oracle_reference(
            "self_converged", observable_name, theta,
            in_process=in_process, threads=REFERENCE_THREADS, log=log,
        )
    raise ValueError(f"no reference route for observable {observable_name!r}")


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def _extra(observable_name: str, theta_label: str, theta: float, reference: Reference,
           sweep: str) -> dict[str, Any]:
    return {
        "benchmark": "B",
        "observable": observable_name,
        "theta_h_label": theta_label,
        "theta_h": theta,
        "trotter_steps": TROTTER_STEPS,
        "sweep": sweep,
        **reference.as_dict(),
    }


#: A weight sweep is paired with the tightest coefficient cutoff whose
#: *uncapped* run came in under this. A weight cap can only remove terms, so
#: every capped run is then cheaper than that already-measured uncapped one —
#: which bounds the whole second sweep for free, without a second cost table.
WEIGHT_SWEEP_TIME_BUDGET_S = 10.0


def _weight_sweep_coeff(accuracy: harness.AccuracyResult | None) -> float:
    """Which `min_abs_coeff` to run the `max_weight` sweep at.

    Read off the coefficient sweep that just ran: the tightest cutoff whose run
    stayed under `WEIGHT_SWEEP_TIME_BUDGET_S`, falling back to the loosest point
    of that sweep, or to `WEIGHT_SWEEP_COEFF` when no coefficient sweep ran at
    all. Measured rather than tabulated, so it needs no per-point tuning.
    """
    if accuracy is None or not accuracy.records:
        return WEIGHT_SWEEP_COEFF
    affordable = [
        spec.min_abs_coeff
        for spec, record in zip(accuracy.specs, accuracy.records)
        if spec.min_abs_coeff is not None
        and record.total_time_s <= WEIGHT_SWEEP_TIME_BUDGET_S
    ]
    if affordable:
        return min(affordable)
    loosest = [s.min_abs_coeff for s in accuracy.specs if s.min_abs_coeff is not None]
    return max(loosest) if loosest else WEIGHT_SWEEP_COEFF


def sweep_one_point(
    observable_name: str,
    theta_label: str,
    theta: float,
    reference: Reference,
    *,
    coeff_grid: Sequence[float],
    weight_grid: Sequence[int],
    library_versions: dict[str, str],
    log=None,
) -> tuple[list[report.RunRecord], harness.AccuracyResult | None]:
    """Both truncation sweeps for one `(observable, theta_h)`.

    Returns every record plus the `AccuracyResult` of the coefficient sweep (the
    "time to |error| < ACCURACY_EPSILON" selection), or `None` when the
    coefficient grid is empty.
    """
    circuit = circuits.heavy_hex_kicked_ising(
        N_QUBITS, trotter_steps=TROTTER_STEPS, theta_h=theta
    )
    observable = OBSERVABLE_BUILDERS[observable_name]()
    records: list[report.RunRecord] = []

    # `warmup` is switched off once a run gets long enough that a doubled cost
    # is not worth a warm allocator; the flag is recorded on the record.
    state = {"warm": True}

    def build_run(spec: harness.TruncationSpec, sweep: str) -> report.RunRecord:
        warm = state["warm"]
        record = harness.run_propagation(
            circuit,
            observable,
            spec,
            DIRECTION,
            state=STATE,
            warmup=warm,
            oracle_value=reference.value,
            threads=1,
            library_versions=library_versions,
            extra={
                **_extra(observable_name, theta_label, theta, reference, sweep),
                "warm": warm,
            },
        )
        if record.propagation_time_s > WARMUP_TIME_BUDGET_S:
            state["warm"] = False
        if log is not None:
            log(
                f"      {spec!s:<38} <O>={record.expectation_value:+.9f} "
                f"err={record.absolute_error:.2e} terms={record.final_terms:>9} "
                f"{record.total_time_s:7.2f}s"
            )
        return record

    grid = coeff_grid_for(observable_name, theta_label, coeff_grid)
    accuracy: harness.AccuracyResult | None = None
    if grid:
        if log is not None:
            cut = "" if len(grid) == len(coeff_grid) else (
                f" [cut from {len(coeff_grid)} points, see COEFF_GRID_CUTS]"
            )
            log(f"    min_abs_coeff sweep (loosest first){cut}:")
        accuracy = harness.time_to_accuracy(
            lambda spec: build_run(spec, "min_abs_coeff"),
            reference.value,
            ACCURACY_EPSILON,
            [harness.TruncationSpec(min_abs_coeff=eps) for eps in grid],
        )
        records.extend(accuracy.records)

    weight_coeff = _weight_sweep_coeff(accuracy)
    if weight_grid:
        if log is not None:
            log(f"    max_weight sweep (at min_abs_coeff={weight_coeff:g}):")
        state["warm"] = True
        records.extend(
            harness.convergence_sweep(
                lambda spec: build_run(spec, "max_weight"),
                [
                    harness.TruncationSpec(max_weight=w, min_abs_coeff=weight_coeff)
                    for w in weight_grid
                ],
                oracle_value=reference.value,
            )
        )

    return records, accuracy


# --------------------------------------------------------------------------
# Cross-engine parity: cross-engine timing is withheld until per-layer term
# counts match.
# --------------------------------------------------------------------------


@dataclass
class ParityOutcome:
    """One matched-truncation comparison against PauliPropagation.jl."""

    observable: str
    theta_label: str
    min_abs_coeff: float
    ok: bool
    detail: str
    rust_record: report.RunRecord | None = None
    julia_record: report.RunRecord | None = None
    layers_compared: int = 0
    first_layer_mismatch: int | None = None
    rust_layers: list[int] = field(default_factory=list)
    julia_layers: list[int] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.observable, self.theta_label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observable": self.observable,
            "theta_h_label": self.theta_label,
            "min_abs_coeff": self.min_abs_coeff,
            "ok": self.ok,
            "detail": self.detail,
            "layers_compared": self.layers_compared,
            "first_layer_mismatch": self.first_layer_mismatch,
            "rust_final_terms": None if self.rust_record is None else self.rust_record.final_terms,
            "julia_final_terms": None if self.julia_record is None else self.julia_record.final_terms,
            "rust_peak_terms": None if self.rust_record is None else self.rust_record.peak_terms,
            "julia_peak_terms": None if self.julia_record is None else self.julia_record.peak_terms,
            "rust_propagation_time_s": (
                None if self.rust_record is None else self.rust_record.propagation_time_s
            ),
            "julia_propagation_time_s": (
                None if self.julia_record is None else self.julia_record.propagation_time_s
            ),
        }


def julia_parity(
    observable_name: str,
    theta_label: str,
    theta: float,
    reference: Reference,
    *,
    min_abs_coeff: float = PARITY_COEFF,
    timeout_s: float = 3600.0,
    log=None,
) -> ParityOutcome:
    """Run one matched-truncation task on both engines and compare per layer.

    Per-layer counts, not just the final count: a divergence that cancels by the
    end is exactly the coefficient-boundary or truncation-schedule bug the
    comparison exists to catch (benchmarks/julia/README.md §P3, §P5). Both lists
    are in *application* order on both engines, so they line up index by index
    with no reversal.

    Returns a `ParityOutcome`; it never raises for a parity failure, because the
    driver's contract is to *report* the failure and withhold the cross-engine
    timing until parity holds, not to abort the whole sweep.
    """
    import julia_baseline

    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N_QUBITS, TROTTER_STEPS, theta
    )
    observable = OBSERVABLE_BUILDERS[observable_name]()
    terms = oracles.pauli_terms(observable, N_QUBITS)

    task = julia_baseline.make_task(
        n_qubits=N_QUBITS,
        gates=spec.to_circuit_json()["gates"],
        observable={label: coeff for label, coeff in terms},
        direction=DIRECTION,
        min_abs_coeff=min_abs_coeff,
        threads=1,
        state=STATE,
    )

    circuit = spec.to_circuit()
    rust = harness.run_propagation(
        circuit,
        observable,
        harness.TruncationSpec(min_abs_coeff=min_abs_coeff),
        DIRECTION,
        state=STATE,
        oracle_value=reference.value,
        threads=1,
        extra={
            **_extra(observable_name, theta_label, theta, reference, "parity"),
            "warm": True,
        },
    )
    _, rust_stats = observable.propagate_with_stats(
        circuit, harness.make_policy(min_abs_coeff=min_abs_coeff), direction=DIRECTION
    )
    rust_layers = list(rust_stats.terms_out)

    try:
        result = julia_baseline.run_task(
            task,
            threads=1,
            warm_repeats=PARITY_WARM_REPEATS,
            layer_counts=True,
            timeout=timeout_s,
        )
    except julia_baseline.JuliaBaselineError as exc:
        return ParityOutcome(
            observable_name, theta_label, min_abs_coeff, False,
            f"PauliPropagation.jl runner failed: {exc}", rust_record=rust,
            rust_layers=rust_layers,
        )

    jl_versions = result.versions
    julia_record = report.RunRecord(
        engine="PauliPropagation.jl",
        engine_version=jl_versions.get("PauliPropagation", "unknown"),
        n_qubits=N_QUBITS,
        direction=DIRECTION,
        truncation={"min_abs_coeff": min_abs_coeff},
        propagation_time_s=result.wall_warm_s or result.wall_cold_s,
        final_terms=result.final_terms,
        provenance=report.collect_provenance(
            thread_count=1,
            extra_library_versions=jl_versions,
            repo_root=_REPO_ROOT,
        ),
        peak_terms=result.peak_terms,
        expectation_value=None if result.expectation is None else result.expectation.real,
        absolute_error=(
            None
            if result.expectation is None
            else abs(result.expectation.real - reference.value)
        ),
        extra={
            **_extra(observable_name, theta_label, theta, reference, "parity"),
            "warm": result.wall_warm_s is not None,
            "wall_cold_s": result.wall_cold_s,
            "julia_notes": result.notes,
        },
    )

    jl_layers = result.per_layer_terms or []
    first_mismatch = None
    compared = min(len(rust_layers), len(jl_layers))
    for index in range(compared):
        if rust_layers[index] != jl_layers[index]:
            first_mismatch = index
            break

    record_parity = harness.check_term_parity(rust, julia_record, coeff_tol=1e-9)
    reasons = list(record_parity.reasons)
    if len(rust_layers) != len(jl_layers):
        reasons.append(
            f"per-layer count lists have different lengths: {len(rust_layers)} "
            f"(paulistrings) vs {len(jl_layers)} (PauliPropagation.jl)"
        )
    if first_mismatch is not None:
        reasons.append(
            f"per-layer term counts first differ at applied layer {first_mismatch}: "
            f"{rust_layers[first_mismatch]} vs {jl_layers[first_mismatch]}"
        )

    ok = not reasons
    detail = (
        f"{compared}/{compared} per-layer term counts identical; final "
        f"{rust.final_terms} terms on both; |Δ<O>| = "
        f"{abs((julia_record.expectation_value or 0.0) - (rust.expectation_value or 0.0)):.3e}"
        if ok
        else "; ".join(reasons)
    )
    if log is not None:
        log(f"      {'PARITY OK' if ok else 'PARITY FAILED'}: {detail}")
    return ParityOutcome(
        observable_name,
        theta_label,
        min_abs_coeff,
        ok,
        detail,
        rust_record=rust,
        julia_record=julia_record,
        layers_compared=compared,
        first_layer_mismatch=first_mismatch,
        rust_layers=rust_layers,
        julia_layers=jl_layers,
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

#: The validated categorical palette `report.py` also uses (dataviz skill,
#: `references/palette.md`). Kept as a local constant rather than reaching into
#: `report`'s private `_PALETTE`, since here the categorical dimension is
#: `theta_h`, not the engine.
_THETA_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
)
_MUTED = "#898781"
_GRID = "#e1e0d9"

#: Error floor for a log plot: an exactly-zero error (a Clifford point resolved
#: to the bit) cannot be drawn on a log axis, so it is clamped here and the
#: clamp is stated in the figure's axis label.
_ERROR_FLOOR = 1e-17


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)


def _theta_color(theta_label: str) -> str:
    labels = [label for label, _ in THETA_POINTS]
    return _THETA_COLORS[labels.index(theta_label) % len(_THETA_COLORS)]


def _legend_on_a_populated_panel(axes) -> None:
    """Put one legend on the first panel that actually drew something.

    `axes[-1]` is the natural place for it, but a panel can legitimately be
    empty — `--observables z62` leaves two of three empty, and matplotlib then
    warns "No artists with labels found" and draws nothing.
    """
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
            return


def _panel_records(records, observable_name: str, sweep: str, theta_label: str):
    return [
        r
        for r in records
        if r.extra.get("observable") == observable_name
        and r.extra.get("sweep") == sweep
        and r.extra.get("theta_h_label") == theta_label
        and r.absolute_error is not None
    ]


def plot_error_vs_truncation(
    records: Sequence[report.RunRecord],
    observable_names: Sequence[str],
    *,
    sweep: str,
    truncation_key: str,
    xscale: str,
    save_path: Path,
):
    """|error| vs a truncation knob, one panel per observable, one curve per `theta_h`.

    This panel shows error trending to 0 as truncation tightens: a curve that
    *rises* left to right on the `min_abs_coeff` panel is the expected shape,
    since x increases with how much is thrown away.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(observable_names), figsize=(4.2 * len(observable_names), 3.8), squeeze=False
    )
    for ax, name in zip(axes[0], observable_names):
        for theta_label, _ in THETA_POINTS:
            points = sorted(
                (r.truncation[truncation_key], max(r.absolute_error, _ERROR_FLOOR))
                for r in _panel_records(records, name, sweep, theta_label)
                if truncation_key in r.truncation
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs, ys, marker="o", markersize=4, linewidth=1.4,
                color=_theta_color(theta_label), label=f"θ_h = {theta_label}",
            )
        if xscale == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")
        if truncation_key == "max_weight":
            # The paired coefficient cutoff is chosen per point by
            # `_weight_sweep_coeff`, so it is not the same on every curve;
            # `results.json` records it per run.
            ax.set_xlabel("max_weight (paired min_abs_coeff varies per θ_h)")
        else:
            ax.set_xlabel(truncation_key)
        ax.set_ylabel(f"|error| vs reference (floored at {_ERROR_FLOOR:g})")
        ax.set_title(name, color=_MUTED, fontsize=10)
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_error_vs_runtime_by_theta(
    records: Sequence[report.RunRecord],
    observable_names: Sequence[str],
    *,
    sweep: str,
    save_path: Path,
):
    """|error| vs warm wall time, one panel per observable, one curve per `theta_h`.

    `report.plot_error_vs_runtime` draws one curve *per engine*, which is the
    right shape for the cross-engine figure and the wrong one here, where every
    curve is the same engine at a different kick angle.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(observable_names), figsize=(4.2 * len(observable_names), 3.8), squeeze=False
    )
    for ax, name in zip(axes[0], observable_names):
        for theta_label, _ in THETA_POINTS:
            points = sorted(
                (r.total_time_s, max(r.absolute_error, _ERROR_FLOOR))
                for r in _panel_records(records, name, sweep, theta_label)
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs, ys, marker="o", markersize=4, linewidth=1.4,
                color=_theta_color(theta_label), label=f"θ_h = {theta_label}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("warm wall time (s), single-threaded")
        ax.set_ylabel(f"|error| vs reference (floored at {_ERROR_FLOOR:g})")
        ax.set_title(name, color=_MUTED, fontsize=10)
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_parity_layers(
    outcomes: Sequence[ParityOutcome], *, min_abs_coeff: float = PARITY_COEFF,
    save_path: Path,
):
    """Per-layer term counts, both engines, for every parity case that ran.

    The two curves are drawn thick-solid (paulistrings) under thin-dashed
    (PauliPropagation.jl) precisely so that *identical* lists look like one
    line with a dashed overlay — a visible orange excursion is a real
    divergence, not a rendering artefact.
    """
    import matplotlib.pyplot as plt

    drawable = [
        o
        for o in outcomes
        if o.rust_layers and o.julia_layers and o.min_abs_coeff == min_abs_coeff
    ]
    if not drawable:
        return None
    fig, axes = plt.subplots(
        1, len(drawable), figsize=(4.2 * len(drawable), 3.6), squeeze=False
    )
    for ax, outcome in zip(axes[0], drawable):
        ax.plot(range(len(outcome.rust_layers)), outcome.rust_layers, linewidth=2.6,
                color="#2a78d6", label="paulistrings")
        ax.plot(range(len(outcome.julia_layers)), outcome.julia_layers, linewidth=1.0,
                linestyle="--", color="#eb6834", label="PauliPropagation.jl")
        ax.set_yscale("log")
        ax.set_xlabel("applied layer (one gate = one channel)")
        ax.set_ylabel("resident terms")
        ax.set_title(
            f"{outcome.observable}, θ_h = {outcome.theta_label}, "
            f"min_abs_coeff = {outcome.min_abs_coeff:g}",
            color=_MUTED, fontsize=9,
        )
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_term_counts(records: Sequence[report.RunRecord], observable_names, *, save_path: Path):
    """Term count vs `min_abs_coeff`, both engines where both ran.

    Straight `report.plot_term_count_vs_truncation` per panel: here the
    categorical dimension really is the engine, which is what that helper draws.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(observable_names), figsize=(4.2 * len(observable_names), 3.8), squeeze=False
    )
    drew = False
    for ax, name in zip(axes[0], observable_names):
        subset = [
            r
            for r in records
            if r.extra.get("observable") == name
            and r.extra.get("sweep") == "parity"
            and "min_abs_coeff" in r.truncation
        ]
        if not subset:
            continue
        drew = True
        report.plot_term_count_vs_truncation(subset, ax=ax)
        ax.set_title(
            f"{name}, θ_h = {subset[0].extra.get('theta_h_label')}",
            color=_MUTED, fontsize=10,
        )
    if not drew:
        plt.close(fig)
        return None
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


DEFAULT_OUT_DIR = _REPO_ROOT / "benchmarks" / "python" / "theta_sweep"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name, key in (("qiskit", "qiskit"), ("stim", "stim"), ("numpy", "numpy")):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[key] = getattr(module, "__version__", "unknown")
    return versions


def _validate_convergence(
    references: dict[tuple[str, str], Reference],
    observable_names,
    theta_points,
    log,
) -> list[dict[str, Any]]:
    """Run the self-convergence procedure where the exact answer is known.

    The methodology check the plan demands before a self-converged number may
    stand in for an exact one: for `Z_62` and weight-10 the interior reference
    is *exact*, so the self-converged value computed the weight-17 way can be
    scored against it — estimated uncertainty vs. true error. The exact
    references are the ones the main sweep already resolved, so this leg pays
    only for the self-convergence runs.
    """
    rows: list[dict[str, Any]] = []
    for name in observable_names:
        if name == "weight_17":
            continue  # no exact interior reference exists — that is the point
        for theta_label, theta in theta_points:
            if theta_label in CLIFFORD_THETA_LABELS:
                continue
            exact = references.get((name, theta_label))
            if exact is None or not exact.exact:
                continue
            log(f"  validate {name} at θ_h = {theta_label} (exact: {exact.method})")
            observable = OBSERVABLE_BUILDERS[name]()
            circuit = circuits.heavy_hex_kicked_ising(
                N_QUBITS, trotter_steps=TROTTER_STEPS, theta_h=theta
            )
            approx = self_converged_reference(circuit, observable, log=log)
            true_error = abs(approx.value - exact.value)
            rows.append(
                {
                    "observable": name,
                    "theta_h_label": theta_label,
                    "exact_value": exact.value,
                    "exact_method": exact.method,
                    "self_converged_value": approx.value,
                    "estimated_uncertainty": approx.uncertainty,
                    "true_error": true_error,
                    "converged": approx.evidence.get("converged"),
                    # "Did the estimate stay honest?" — the estimate is a
                    # successive-difference heuristic, not a bound, so the bar
                    # is the same one `test_benchmark_b_sweep.py` asserts: the
                    # true error must not exceed the estimate by more than
                    # `_UNCERTAINTY_SLACK`, with a floating-point floor so a
                    # plateau that converged by *saturation* (uncertainty
                    # legitimately 0, true error at the 1e-14 rounding level)
                    # is not scored as dishonest.
                    "conservative": (
                        approx.uncertainty is not None
                        and true_error
                        <= max(approx.uncertainty * _UNCERTAINTY_SLACK, _FP_NOISE_FLOOR)
                    ),
                    "uncertainty_slack": _UNCERTAINTY_SLACK,
                }
            )
            log(
                f"    exact={exact.value:+.12f}  self-converged={approx.value:+.12f}  "
                f"true error={true_error:.2e}  estimated="
                f"{'n/a' if approx.uncertainty is None else format(approx.uncertainty, '.2e')}"
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark B: theta_h sweep at 5 Trotter steps on the 127-qubit heavy hex."
    )
    parser.add_argument(
        "--observables", nargs="+", default=list(OBSERVABLE_BUILDERS),
        choices=list(OBSERVABLE_BUILDERS),
    )
    parser.add_argument(
        "--thetas", nargs="+", default=[label for label, _ in THETA_POINTS],
        choices=[label for label, _ in THETA_POINTS],
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--coeff-grid", type=float, nargs="*", default=list(COEFF_GRID),
        help="min_abs_coeff grid, loosest first",
    )
    parser.add_argument(
        "--weight-grid", type=int, nargs="*", default=list(WEIGHT_GRID),
        help="max_weight grid, loosest (smallest) first",
    )
    parser.add_argument("--no-julia", action="store_true", help="skip the parity leg")
    parser.add_argument(
        "--parity-cases", nargs="*", default=["z62", "weight_10", "weight_17"],
        help="observables to run the matched-truncation jl parity check for",
    )
    parser.add_argument("--parity-theta", default="0.2")
    parser.add_argument(
        "--parity-coeffs", type=float, nargs="*", default=list(PARITY_COEFFS),
        help="min_abs_coeff values the jl parity check is run at",
    )
    parser.add_argument("--validate-convergence", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--in-process-references", action="store_true",
        help=(
            "compute oracle references in this process instead of an isolated child. "
            "Faster to debug, but qiskit-aer's persistent OpenMP pool then trips "
            "harness.assert_single_threaded on every later timed run"
        ),
    )
    args = parser.parse_args(argv)

    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error(
            "export RAYON_NUM_THREADS=1 before starting the interpreter: Rayon builds "
            "its global pool at the first propagate and never resizes it"
        )
    harness.assert_logging_quiet()
    bad = [eps for eps in args.coeff_grid if eps < MIN_SAFE_COEFF]
    if bad or WEIGHT_SWEEP_COEFF < MIN_SAFE_COEFF:
        parser.error(
            f"min_abs_coeff values {bad or [WEIGHT_SWEEP_COEFF]} are below "
            f"MIN_SAFE_COEFF={MIN_SAFE_COEFF:g}; the cos(pi/2) residual branch then "
            "survives truncation at a Clifford angle and the propagation fans out"
        )

    def log(message: str) -> None:
        print(message, flush=True)

    theta_points = [(l, v) for l, v in THETA_POINTS if l in args.thetas]
    library_versions = _library_versions()
    started = time.perf_counter()

    records: list[report.RunRecord] = []
    references: dict[tuple[str, str], Reference] = {}
    accuracy_rows: list[dict[str, Any]] = []

    for name in args.observables:
        for theta_label, theta in theta_points:
            log(f"\n=== {name}, θ_h = {theta_label} ===")
            observable = OBSERVABLE_BUILDERS[name]()
            log("    reference:")
            reference = resolve_reference(
                name, theta_label, observable,
                in_process=args.in_process_references, log=log,
            )
            references[(name, theta_label)] = reference

            point_records, accuracy = sweep_one_point(
                name, theta_label, theta, reference,
                coeff_grid=args.coeff_grid,
                weight_grid=args.weight_grid,
                library_versions=library_versions,
                log=log,
            )
            records.extend(point_records)
            if accuracy is not None:
                accuracy_rows.append(
                    {
                        "observable": name,
                        "theta_h_label": theta_label,
                        "epsilon": accuracy.epsilon,
                        "achieved": accuracy.achieved,
                        "first_truncation": (
                            None if accuracy.first_spec is None
                            else accuracy.first_spec.as_dict()
                        ),
                        "first_time_s": (
                            None if accuracy.first is None else accuracy.first.total_time_s
                        ),
                        "cheapest_truncation": (
                            None if accuracy.cheapest_spec is None
                            else accuracy.cheapest_spec.as_dict()
                        ),
                        "cheapest_time_s": (
                            None if accuracy.cheapest is None
                            else accuracy.cheapest.total_time_s
                        ),
                        "coeff_grid_used": [s.min_abs_coeff for s in accuracy.specs],
                        "coeff_grid_cut": (
                            (name, theta_label) in COEFF_GRID_CUTS
                        ),
                    }
                )
                log("    " + accuracy.describe().replace("\n", "\n    "))

    # --- endpoint cross-check against Benchmark A's integers ---------------
    endpoint_rows = []
    for (name, theta_label), reference in sorted(references.items()):
        if theta_label not in CLIFFORD_THETA_LABELS:
            continue
        # Only the *coefficient* sweep is a fair endpoint check. The weight
        # sweep is not: at a Clifford angle the back-evolved operator passes
        # through weight ~30-40 mid-circuit even though it lands on a single
        # low-weight string, so a cap of 8 truncates the whole sum to zero
        # terms and the "deviation" is 1.0 by construction — the cap's doing,
        # not the engine's. That behaviour is reported separately below.
        def _deviation(runs):
            return max(
                (abs(r.expectation_value - reference.value) for r in runs), default=None
            )

        at_point = [
            r
            for r in records
            if r.extra.get("observable") == name
            and r.extra.get("theta_h_label") == theta_label
            and r.expectation_value is not None
        ]
        coeff_runs = [r for r in at_point if r.extra.get("sweep") == "min_abs_coeff"]
        weight_runs = [r for r in at_point if r.extra.get("sweep") == "max_weight"]
        worst = _deviation(coeff_runs)
        endpoint_rows.append(
            {
                "observable": name,
                "theta_h_label": theta_label,
                "clifford_integer": reference.value,
                "worst_engine_deviation": worst,
                "runs": len(coeff_runs),
                "weight_sweep_worst_deviation": _deviation(weight_runs),
                "weight_sweep_runs": len(weight_runs),
                "weight_caps_that_emptied_the_sum": sorted(
                    r.truncation["max_weight"] for r in weight_runs if r.final_terms == 0
                ),
            }
        )
        log(
            f"endpoint {name} θ_h={theta_label}: exact {reference.value:+.1f}, worst "
            f"deviation over {len(coeff_runs)} coefficient-sweep runs = "
            f"{'n/a' if worst is None else format(worst, '.3e')}"
        )

    # --- cross-engine parity ------------------------------------------------
    parity_outcomes: list[ParityOutcome] = []
    julia_skip_reason: str | None = None
    if not args.no_julia:
        import julia_baseline

        julia_skip_reason = julia_baseline.skip_reason()
        if julia_skip_reason is not None:
            log(f"\nPauliPropagation.jl parity skipped: {julia_skip_reason}")
        else:
            theta_label = args.parity_theta
            theta = dict(THETA_POINTS)[theta_label]
            for name in args.parity_cases:
                if (name, theta_label) not in references:
                    log(f"\nparity {name} θ_h={theta_label}: no reference, skipped")
                    continue
                for eps in args.parity_coeffs:
                    log(f"\n=== parity: {name}, θ_h = {theta_label}, "
                        f"min_abs_coeff = {eps:g} ===")
                    outcome = julia_parity(
                        name, theta_label, theta, references[(name, theta_label)],
                        min_abs_coeff=eps, log=log,
                    )
                    parity_outcomes.append(outcome)
                    # The engine record is always kept (it is a single-engine
                    # measurement); the jl record, the only thing that turns
                    # the pair into a cross-engine claim, is written only when
                    # parity holds.
                    if outcome.rust_record is not None:
                        records.append(outcome.rust_record)
                    if outcome.julia_record is not None and outcome.ok:
                        records.append(outcome.julia_record)

    # --- validation of the self-convergence procedure -----------------------
    validation_rows: list[dict[str, Any]] = []
    if args.validate_convergence:
        log("\n=== self-convergence methodology validation ===")
        validation_rows = _validate_convergence(
            references, args.observables, theta_points, log
        )

    # --- outputs ------------------------------------------------------------
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    # The committed artifact is a *snapshot*, regenerated wholesale, so the
    # file is removed before `report.write_results` (which appends by design,
    # the right discipline for the gitignored campaign directory) recreates it.
    if results_path.exists():
        results_path.unlink()
    report.write_results(records, out_dir, name="results")
    log(f"\nwrote {len(records)} records to {results_path}")

    summary = {
        "benchmark": "B",
        "n_qubits": N_QUBITS,
        "trotter_steps": TROTTER_STEPS,
        "theta_zz": circuits.KICKED_ISING_CLIFFORD_THETA_ZZ,
        "state": STATE,
        "direction": DIRECTION,
        "coeff_grid": list(args.coeff_grid),
        "weight_grid": list(args.weight_grid),
        "weight_sweep_min_abs_coeff": WEIGHT_SWEEP_COEFF,
        "accuracy_epsilon": ACCURACY_EPSILON,
        "coeff_grid_cuts": {
            f"{name}@{theta}": list(grid)
            for (name, theta), grid in COEFF_GRID_CUTS.items()
        },
        "self_convergence_grid": list(SELF_CONVERGENCE_GRID),
        "self_convergence_tol": SELF_CONVERGENCE_TOL,
        "parity_min_abs_coeffs": list(args.parity_coeffs),
        "parity_theta_h_label": args.parity_theta,
        "kim2023_provenance": observables.kim2023_provenance(),
        "references": {
            f"{name}@{theta}": ref.as_dict() for (name, theta), ref in references.items()
        },
        "endpoints": endpoint_rows,
        "time_to_accuracy": accuracy_rows,
        "julia_parity": [o.as_dict() for o in parity_outcomes],
        "julia_skip_reason": julia_skip_reason,
        "self_convergence_validation": validation_rows,
        "wall_clock_s": time.perf_counter() - started,
        "measured_weight_10_pauli_path": _MEASURED_W10_PAULI_PATH,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log(f"wrote {summary_path}")

    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            log("matplotlib not installed; figures skipped")
        else:
            names = list(args.observables)
            for path in (
                plot_error_vs_truncation(
                    records, names, sweep="min_abs_coeff",
                    truncation_key="min_abs_coeff", xscale="log",
                    save_path=out_dir / "error-vs-min-abs-coeff.svg",
                ),
                plot_error_vs_truncation(
                    records, names, sweep="max_weight",
                    truncation_key="max_weight", xscale="linear",
                    save_path=out_dir / "error-vs-max-weight.svg",
                ),
                plot_error_vs_runtime_by_theta(
                    records, names, sweep="min_abs_coeff",
                    save_path=out_dir / "error-vs-runtime.svg",
                ),
                plot_term_counts(
                    records, names, save_path=out_dir / "term-count-vs-truncation.svg"
                ),
                plot_parity_layers(
                    parity_outcomes, save_path=out_dir / "parity-per-layer-terms.svg"
                ),
            ):
                if path is not None:
                    log(f"wrote {path}")

    failures = [o for o in parity_outcomes if not o.ok]
    if failures:
        log(
            f"\n{len(failures)} parity case(s) FAILED — cross-engine timings for those "
            "cases are withheld"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

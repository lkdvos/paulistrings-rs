"""Showcase B1, phase 1 — real-time operator scrambling on a 1D chain.

Handoff item B1 (1D half); see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B and global rule 4 ("every real-time-dynamics or truncated result
ships with a convergence panel"). Analysis helpers: `scrambling.py`, next to
this file. CI-safe correctness gate:
`python/paulistrings/tests/test_showcase_b1.py`.

What this measures
------------------
A single-site Pauli `Z_c` at the centre of an open `n`-site chain is evolved in
the Heisenberg picture through `T` Trotter steps of the kicked transverse-field
Ising model (`circuits.heavy_hex_kicked_ising` driven with a chain edge list, so
1D and 2D share one builder and one truncation schedule). Three things are read
off the evolved sum at every step, all defined and derived in
`scrambling.py`'s module docstring:

1. **operator support** — `w_q(t) = sum_{P_q != I} |c_P(t)|^2`, and the number
   of sites where it clears a floor;
2. **the light cone** — the same `w_q(t)` as a `(site, t)` heat map, with a
   contour front whose slope is the **butterfly velocity**;
3. **the OTOC** — `C(r,t) = 2 sum_{P anticommuting with W_r} |c_P(t)|^2`, the
   infinite-temperature squared commutator of the evolved operator with a
   single-site probe `W_r`.

Why every curve carries a convergence panel
-------------------------------------------
Real-time dynamics is where a coefficient cutoff is least forgiving: the front
of the light cone is *made of* the smallest coefficients in the sum, which is
precisely what `min_abs_coeff` deletes. So nothing here is reported at one
truncation. Every quantity is computed on the same grid of `min_abs_coeff`
values and plotted as an overlay, and the retained Hilbert-Schmidt norm
`N(t) = sum_P |c_P|^2` — which exact unitary evolution conserves, so
`1 - N(t)` is exactly the fraction of the operator truncation threw away — is
reported next to every one of them.

Five parts, run in order by `main()`:

- `run_validation` — `n = 9`, untruncated *and* truncated, cross-checked
  against a dense `2^n x 2^n` reference built by explicit Kronecker products
  (no engine, no qiskit): coefficient by coefficient, plus the support profile
  and all three OTOC probes.
- `run_clifford_cone` — `theta_h = pi/2` is a Clifford point, so the evolved
  operator is a *single* Pauli string and its support is the exact causal cone:
  a sharp-cone reference for the front measurement that needs no reference
  data at all.
- `run_sweep` — the headline: `n = 61`, `T = 12`, the `min_abs_coeff` grid.
- `run_theta_scan` — the front velocity against the kick angle, at two cutoffs
  each. At the headline `theta_h` the front is *causally saturated*, so every
  contour returns the structural bound and the number says nothing about the
  dynamics; the scan is what shows the readout measuring something.
- `write_figures` — the committed SVGs and the results JSON.

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    python examples/b1_operator_scrambling/run_b1_1d.py

Threads: the sweep is a physics measurement, not a timing claim, so it runs
with the default Rayon pool and records the observed worker count in the
results JSON (plan §7 rule 3 keeps *timed comparisons* single-threaded; there
is no cross-engine timing here). Wall times land in the JSON as run metadata.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from paulistrings import truncation  # noqa: E402

from b1_operator_scrambling import scrambling as sc  # noqa: E402
from common import circuits, harness, observables, oracles, report  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Headline sweep parameters
# --------------------------------------------------------------------------

#: Chain length. The plan asks for `n ~ 40-100`; 61 is odd (so there is an
#: exact centre site) and lands in the `W = 1` monomorphization (<= 64 qubits).
#: Nothing in the cost depends much on `n` beyond `n >= 2T + 1`: the term count
#: is set by the causal cone, not by the chain length.
N_QUBITS = 61
CENTER = N_QUBITS // 2

#: Kick angle. Generic (non-Clifford), in the strongly-scrambling regime. The
#: ZZ angle stays at the kicked-Ising Clifford point `-pi/2` that the rest of
#: this suite uses, so one Trotter step is "maximally entangling bond layer +
#: single-qubit kick" and time is measured in steps, as is standard for a
#: Floquet model.
THETA_H = 0.9
THETA_ZZ = circuits.KICKED_ISING_CLIFFORD_THETA_ZZ

#: Trotter steps. Chosen so the *tightest* cutoff on the grid below is still
#: affordable over the whole window (see the README's cost table): at
#: `min_abs_coeff = 3e-6` step 12 holds ~2.3e8 terms.
STEPS = 12

#: The `min_abs_coeff` grid, loosest first. Four values, spanning 2.5 decades.
EPS_GRID = (1e-3, 1e-4, 1e-5, 3e-6)

#: The reference (tightest) cutoff: the truncation the "physics" panels are
#: drawn at, with the looser ones as the convergence evidence.
REFERENCE_EPS = EPS_GRID[-1]

#: Refuse to grow past this many stored terms: the plan's time-box policy
#: (§8, D15) expressed in code. A run that hits it stops early and says so,
#: rather than swapping the host.
TERM_CEILING = 3.0e8

#: Contour levels for the light-cone front. A ballistic front has a
#: threshold-independent asymptotic velocity but a threshold-dependent apparent
#: one at finite time, so the velocity is fitted at all three and the spread is
#: reported as the systematic.
FRONT_THRESHOLDS = (1e-2, 1e-4, 1e-6)
REFERENCE_FRONT_THRESHOLD = 1e-4

#: Floor for "is this site in the support at all" (`scrambling.support_size`).
SUPPORT_FLOOR = 1e-6

#: Steps included in the butterfly-velocity fit. The first two are excluded:
#: the front has not left the seed site yet, so they only measure the offset.
FIT_STEPS = tuple(range(3, STEPS + 1))

#: Probe distance for the single headline number fed to
#: `report.plot_convergence_panel`.
HEADLINE_OFFSET = 5

#: Kick angles for the front-velocity scan (Part 4). At `theta_h = 0.9` the
#: front is *causally saturated* -- the weight at the edge of the cone is still
#: far above every contour level, so every contour reports the structural
#: maximum of 1 site/step and the number says nothing about the dynamics. Weak
#: kicks are where a contour readout has something to measure, so the velocity
#: is scanned in `theta_h` and reported as a curve approaching the bound rather
#: than as one number.
THETA_SCAN = (0.2, 0.4, 0.6, 0.9)

#: Two cutoffs per scan point: the scan's own convergence evidence.
SCAN_EPS = (1e-4, 1e-5)
SCAN_TERM_CEILING = 1.0e8

# --------------------------------------------------------------------------
# Validation parameters (small n, dense reference is cheap)
# --------------------------------------------------------------------------

VALIDATION_N = 9
VALIDATION_STEPS = 4
DENSE_TOLERANCE = 1e-10

#: Cutoff for the *truncated* half of the validation. Deliberately coarse: at
#: this depth every coefficient the dynamics actually generates is O(0.1), so a
#: cutoff of 1e-3 removes only the numerically-zero dust left by cancellation
#: in the merge and loses no norm at all -- which would make the "does the
#: error track the discarded norm" calibration vacuous.
VALIDATION_EPS = 0.05

#: Cutoff used to sweep away numerically-zero terms at a Clifford point, where
#: `cos(pi/2) = 6.1e-17` leaves a dust term behind rather than an exact zero.
DUST_EPS = 1e-12

#: Clifford-point cone measurement: `theta_h = pi/2` at `theta_zz = -pi/2`.
CLIFFORD_N = 41
CLIFFORD_STEPS = 8


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def one_step_circuit(n: int, theta_h: float, theta_zz: float = THETA_ZZ):
    """One Trotter step of the chain kicked Ising, as a `Circuit`.

    The whole time series is produced by applying *this* circuit repeatedly in
    the Heisenberg direction. That is exact and not a shortcut: for
    `U(t) = S^t`, `U(t)^dagger O U(t) = (S^dagger)^t O S^t`, and because this
    engine truncates after every channel, `t` successive `propagate` calls
    apply exactly the same (apply-adjoint, truncate) sequence as one call on
    the `t`-step circuit — the same identity showcase B5 pins in
    `test_showcase_b5.py`. So a `T`-step time series costs one `T`-step
    propagation, not `T` of them.
    """
    return circuits.heavy_hex_kicked_ising(
        n, 1, theta_h, theta_zz, edges=sc.chain_edges(n)
    )


def one_step_spec(n: int, theta_h: float, steps: int = 1, theta_zz: float = THETA_ZZ):
    """The same circuit as a `CircuitSpec` (gate list), for the dense reference."""
    return oracles.record_gates(
        circuits.heavy_hex_kicked_ising,
        n,
        steps,
        theta_h,
        theta_zz,
        edges=sc.chain_edges(n),
    )


@dataclass
class StepData:
    """Everything measured at one `(min_abs_coeff, trotter step)` point."""

    step: int
    terms: int
    peak_terms: int
    seconds: float
    norm: float
    profile: np.ndarray
    otoc: np.ndarray
    correlator: np.ndarray
    support: int
    fronts: dict[float, float] = field(default_factory=dict)


SiteSumsLike = Any


def measure(pauli_sum) -> tuple[SiteSumsLike, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """One chunked pass over the evolved sum, then every diagnostic from it."""
    sums = sc.site_sums(pauli_sum)
    profile = sc.support_profile(pauli_sum, sums=sums)
    otoc = sc.otoc_from_sums(sums, "X")
    correlator = sc.single_pauli_coefficients(pauli_sum, "Z")
    return sums, profile, otoc, correlator, sums.norm, sc.support_size(
        profile, SUPPORT_FLOOR
    )


def evolve_series(
    n: int,
    center: int,
    step_circuit,
    steps: int,
    eps: float,
    *,
    distances: np.ndarray | None = None,
    term_ceiling: float = TERM_CEILING,
    verbose: bool = True,
) -> list[StepData]:
    """Step the seed `Z_center` through `steps` Trotter steps at cutoff `eps`.

    Stops early (and says so) if the stored term count passes `term_ceiling`.
    """
    policy = truncation.coeff(eps)
    evolved = observables.single_z(center, n)
    out: list[StepData] = []
    for step in range(1, steps + 1):
        start = time.perf_counter()
        evolved, stats = evolved.propagate_with_stats(
            step_circuit, policy, direction="heisenberg"
        )
        seconds = time.perf_counter() - start
        sums, profile, otoc, correlator, norm, support = measure(evolved)
        gap = sc.probe_average_gap(sums)
        if gap > 1e-9:
            raise AssertionError(
                f"the exact probe-average identity mean_W C_W(r) = (4/3) w_r is "
                f"violated by {gap:.3e} at eps={eps:.1e}, step {step}: one of the "
                "two bit-column accumulations in scrambling.site_sums is wrong"
            )
        data = StepData(
            step=step,
            terms=stats.final_terms,
            peak_terms=stats.peak_terms,
            seconds=seconds,
            norm=norm,
            profile=profile,
            otoc=otoc,
            correlator=correlator,
            support=support,
            fronts={
                thr: sc.front_position(profile, center, thr, distances=distances)
                for thr in FRONT_THRESHOLDS
            },
        )
        out.append(data)
        if verbose:
            print(
                f"    step {step:2d}  terms={data.terms:>10d}  norm={norm:.6f}  "
                f"support={support:3d}  front={data.fronts[REFERENCE_FRONT_THRESHOLD]:5.1f}  "
                f"{seconds:7.2f}s",
                flush=True,
            )
        if data.terms > term_ceiling:
            if verbose:
                print(
                    f"    term ceiling {term_ceiling:.1e} passed at step {step}; "
                    "stopping this cutoff here (plan §8 D15 time-box)",
                    flush=True,
                )
            break
    return out


# --------------------------------------------------------------------------
# Part 1 -- dense cross-check
# --------------------------------------------------------------------------


def run_validation() -> dict[str, float]:
    print("=" * 78)
    print("Part 1 -- dense cross-check (independent 2^n x 2^n reference)")
    print("=" * 78)

    n, t = VALIDATION_N, VALIDATION_STEPS
    center = n // 2
    spec = one_step_spec(n, THETA_H, steps=t)
    label = observables.pauli_string({center: "Z"}, n)

    dense = sc.dense_heisenberg(spec, label)
    dense_norm = sc.dense_hs_norm(dense)
    dense_profile = sc.dense_support_profile(dense, n)

    evolved = observables.single_z(center, n).propagate(
        spec.to_circuit(), None, direction="heisenberg"
    )
    sums, profile, _otoc_x, correlator, norm, _support = measure(evolved)

    print(f"n={n} steps={t} theta_h={THETA_H} theta_zz={THETA_ZZ:.6f}")
    print(f"  engine terms (untruncated)     = {len(evolved)}")
    print(f"  <O,O>: engine {norm:.15f}  dense {dense_norm:.15f}")

    results: dict[str, float] = {}

    # (a) every stored coefficient against the dense trace <P, O>.
    coeff_gap = max(
        abs(coefficient - sc.dense_coefficient(dense, term_label))
        for term_label, coefficient in oracles.pauli_terms(evolved)
    )
    results["coefficient_gap"] = float(coeff_gap)
    print(f"  max |c_P(engine) - <P,O>(dense)|          = {coeff_gap:.3e}")

    # (b) the norms agreeing means no term is *missing* from the sum: the
    # coefficient check above can only see the terms that are there.
    norm_gap = abs(norm - dense_norm)
    results["norm_gap"] = float(norm_gap)
    print(f"  |<O,O> engine - <O,O> dense|              = {norm_gap:.3e}")

    # (c) support profile, from the dense single-qubit Pauli twirl.
    profile_gap = float(np.max(np.abs(profile - dense_profile)))
    results["support_profile_gap"] = profile_gap
    print(f"  max |w_q(engine) - w_q(dense twirl)|      = {profile_gap:.3e}")

    # (d) all three OTOC probes, from dense commutators.
    otoc_gaps = {}
    for probe in ("X", "Y", "Z"):
        engine_curve = sc.otoc_from_sums(sums, probe)
        dense_curve = np.array(
            [sc.dense_otoc(dense, probe, r, n) for r in range(n)]
        )
        otoc_gaps[probe] = float(np.max(np.abs(engine_curve - dense_curve)))
        print(f"  max |C_{probe}(r) engine - dense|              = {otoc_gaps[probe]:.3e}")
    results.update({f"otoc_gap_{k}": v for k, v in otoc_gaps.items()})

    # (e) the two-point function G(r) = <Z_r, O(t)>.
    dense_correlator = np.array(
        [
            sc.dense_coefficient(dense, observables.pauli_string({r: "Z"}, n)).real
            for r in range(n)
        ]
    )
    correlator_gap = float(np.max(np.abs(correlator - dense_correlator)))
    results["correlator_gap"] = correlator_gap
    print(f"  max |G(r) engine - G(r) dense|            = {correlator_gap:.3e}")

    # (f) the probe-average identity, exact by construction.
    identity_gap = sc.probe_average_gap(sums)
    results["probe_average_gap"] = identity_gap
    print(f"  max |mean_W C_W(r) - (4/3) w_r|           = {identity_gap:.3e}")

    worst = max(results.values())
    assert worst <= DENSE_TOLERANCE, (
        f"the dense cross-check at n={n} disagrees by {worst:.3e} > {DENSE_TOLERANCE:.1e}: "
        f"{results}"
    )
    print(f"  worst gap {worst:.3e} <= {DENSE_TOLERANCE:.1e}  OK")

    # (g) the same comparison with truncation on: now the gap is the truncation
    # error, and it must be *bounded by the norm it deleted*, which is the whole
    # justification for using 1 - N(t) as the convergence diagnostic.
    truncated = observables.single_z(center, n).propagate(
        spec.to_circuit(), truncation.coeff(VALIDATION_EPS), direction="heisenberg"
    )
    _t_sums, t_profile, _t_otoc, _t_corr, t_norm, _ = measure(truncated)
    lost = dense_norm - t_norm
    profile_error = float(np.max(np.abs(t_profile - dense_profile)))
    results["truncated_norm_lost"] = float(lost)
    results["truncated_profile_error"] = profile_error
    ratio = profile_error / lost if lost > 1e-12 else float("nan")
    print(
        f"  truncated at min_abs_coeff={VALIDATION_EPS:.0e}: terms {len(truncated)}, "
        f"norm lost {lost:.3e}, max profile error {profile_error:.3e} "
        f"(ratio {ratio:.2f})"
    )
    assert lost > 1e-12, (
        f"the validation cutoff {VALIDATION_EPS:.1e} discarded no norm ({lost:.3e}), so "
        "the calibration below is vacuous; raise VALIDATION_EPS or VALIDATION_STEPS"
    )
    # Not a theorem, a calibration: `w_q` is a partial sum of `|c_P|^2`, so a
    # single deletion round can move it by at most the weight deleted -- but
    # truncation happens after every channel, and each round also perturbs the
    # coefficients that *survive*, so the accumulated error is only expected to
    # track the discarded norm in order of magnitude. The point of printing the
    # ratio is that it stays O(1), which is what licenses using `1 - N(t)` as
    # the convergence diagnostic for every curve in Part 3.
    assert profile_error <= 10.0 * lost + DENSE_TOLERANCE, (
        f"the per-site weight error ({profile_error:.3e}) is more than 10x the "
        f"discarded norm ({lost:.3e}); 1 - N(t) is then not a usable proxy for the "
        "error in w_q and the convergence story in Part 3 needs rethinking"
    )
    print("  per-site weight error tracks the discarded norm  OK")
    print()
    return results


# --------------------------------------------------------------------------
# Part 2 -- the Clifford sharp cone
# --------------------------------------------------------------------------


def run_clifford_cone() -> list[tuple[int, int, int]]:
    """`theta_h = pi/2`: the evolved operator is one Pauli string; its support
    is the exact causal cone.

    At `theta_zz = -pi/2` *and* `theta_h = pi/2` every gate is Clifford, so a
    single Pauli string evolves into a single Pauli string with unit
    coefficient. Its support is therefore the strict light cone of the circuit,
    measured rather than asserted, and it gives the hard bound every
    butterfly-velocity number in Part 3 must respect: no truncation, however
    loose or tight, can put weight outside it.
    """
    print("=" * 78)
    print("Part 2 -- Clifford sharp cone (theta_h = pi/2, exact single string)")
    print("=" * 78)

    n, center = CLIFFORD_N, CLIFFORD_N // 2
    step = one_step_circuit(n, np.pi / 2.0)
    # A `min_abs_coeff` of 1e-12 is not a physical truncation here: at
    # `theta_h = pi/2` the surviving branch has coefficient +-1 and the other
    # has `cos(pi/2) = 6.1e-17`, which is floating-point dust rather than an
    # exact zero. Without the cutoff the "single string" is two strings, one of
    # them at 1e-17, and the cone measurement would pick up its support.
    dust = truncation.coeff(DUST_EPS)
    evolved = observables.single_z(center, n)
    rows: list[tuple[int, int, int]] = []
    for t in range(1, CLIFFORD_STEPS + 1):
        evolved = evolved.propagate(step, dust, direction="heisenberg")
        profile = sc.support_profile(evolved)
        assert len(evolved) == 1, (
            f"a Clifford circuit must map one Pauli string to one Pauli string, got "
            f"{len(evolved)} at step {t}"
        )
        weight = int(np.count_nonzero(profile))
        reach = int(sc.front_position(profile, center, 0.5))
        rows.append((t, weight, reach))
        print(f"  step {t:2d}: single string, weight {weight:3d}, cone radius {reach:2d}")

    for t, _weight, reach in rows:
        assert reach <= t, (
            f"the exact cone radius {reach} at step {t} exceeds the structural bound "
            f"of one site per Trotter step"
        )
    slope, _ = sc.front_velocity([r[0] for r in rows], [r[2] for r in rows])
    print(
        f"  exact cone radius grows at {slope:.3f} sites/step "
        f"(structural bound: 1 site/step -- one commuting ZZ layer per step)"
    )
    print()
    return rows


# --------------------------------------------------------------------------
# Part 3 -- the headline sweep
# --------------------------------------------------------------------------


def run_sweep() -> dict[float, list[StepData]]:
    print("=" * 78)
    print("Part 3 -- min_abs_coeff sweep (the convergence evidence)")
    print("=" * 78)
    print(
        f"n={N_QUBITS} centre={CENTER} theta_h={THETA_H} theta_zz={THETA_ZZ:.6f} "
        f"steps={STEPS}"
    )
    step = one_step_circuit(N_QUBITS, THETA_H)
    print(f"channels per Trotter step: {len(step)}")
    series: dict[float, list[StepData]] = {}
    for eps in EPS_GRID:
        print(f"  min_abs_coeff = {eps:.1e}")
        series[eps] = evolve_series(N_QUBITS, CENTER, step, STEPS, eps)
    print()
    return series


def report_velocities(series: dict[float, list[StepData]]) -> dict[float, dict[float, float]]:
    """Fit the butterfly velocity at every (cutoff, contour) pair and print it."""
    print("Butterfly velocity (sites per Trotter step), by cutoff and contour level:")
    header = "  " + "min_abs_coeff".ljust(15) + "".join(
        f"w>{thr:.0e}".rjust(12) for thr in FRONT_THRESHOLDS
    )
    print(header)
    fits: dict[float, dict[float, float]] = {}
    for eps, data in series.items():
        row = {}
        for thr in FRONT_THRESHOLDS:
            points = [(d.step, d.fronts[thr]) for d in data if d.step in FIT_STEPS]
            if len(points) < 2:
                continue
            slope, _ = sc.front_velocity([p[0] for p in points], [p[1] for p in points])
            row[thr] = slope
        fits[eps] = row
        cells = "".join(
            (f"{row[thr]:12.3f}" if thr in row else " " * 12) for thr in FRONT_THRESHOLDS
        )
        print(f"  {eps:<15.1e}{cells}")
    print()
    return fits


# --------------------------------------------------------------------------
# Part 4 -- front velocity vs kick angle
# --------------------------------------------------------------------------


def run_theta_scan() -> dict[tuple[float, float], list[StepData]]:
    """Front velocity as a function of the kick angle, at two cutoffs each.

    The headline sweep answers "is the front converged in `min_abs_coeff`"; this
    answers "is the front measurement measuring anything". At small `theta_h`
    the operator spreads slowly and the contour front lags well behind the
    causal cone, so `v_B` is a real number strictly below the bound; as
    `theta_h` grows the front saturates the bound and every contour returns
    exactly 1 site/step.
    """
    print("=" * 78)
    print("Part 4 -- front velocity vs kick angle (is v_B measuring anything?)")
    print("=" * 78)
    out: dict[tuple[float, float], list[StepData]] = {}
    for theta in THETA_SCAN:
        step = one_step_circuit(N_QUBITS, theta)
        for eps in SCAN_EPS:
            print(f"  theta_h = {theta:.2f}, min_abs_coeff = {eps:.1e}")
            out[(theta, eps)] = evolve_series(
                N_QUBITS,
                CENTER,
                step,
                STEPS,
                eps,
                term_ceiling=SCAN_TERM_CEILING,
            )
    print()
    return out


def report_theta_scan(
    scan: dict[tuple[float, float], list[StepData]],
) -> dict[tuple[float, float], dict[float, float]]:
    print("Front velocity (sites/step) vs kick angle, by cutoff and contour:")
    print(
        "  "
        + "theta_h".ljust(9)
        + "min_abs_coeff".ljust(15)
        + "".join(f"w>{thr:.0e}".rjust(12) for thr in FRONT_THRESHOLDS)
    )
    fits: dict[tuple[float, float], dict[float, float]] = {}
    for (theta, eps), data in scan.items():
        row: dict[float, float] = {}
        for thr in FRONT_THRESHOLDS:
            points = [(d.step, d.fronts[thr]) for d in data if d.step in FIT_STEPS]
            if len(points) < 2:
                continue
            slope, _ = sc.front_velocity([p[0] for p in points], [p[1] for p in points])
            row[thr] = slope
        fits[(theta, eps)] = row
        cells = "".join(
            (f"{row[thr]:12.3f}" if thr in row else " " * 12) for thr in FRONT_THRESHOLDS
        )
        print(f"  {theta:<9.2f}{eps:<15.1e}{cells}")
    print()
    return fits


def plot_theta_scan(
    scan: dict[tuple[float, float], list[StepData]],
    fits: dict[tuple[float, float], dict[float, float]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_f, ax_v) = plt.subplots(1, 2, figsize=(9.5, 4))

    tightest = SCAN_EPS[-1]
    for color, theta in zip(_ramp(len(THETA_SCAN)), THETA_SCAN):
        data = scan.get((theta, tightest))
        if not data:
            continue
        ax_f.plot(
            [d.step for d in data],
            [d.fronts[REFERENCE_FRONT_THRESHOLD] for d in data],
            marker="o", markersize=4, linewidth=1.8, color=color,
            label=f"theta_h = {theta:.2f}",
        )
    ax_f.plot(
        range(1, STEPS + 1), range(1, STEPS + 1), linewidth=1.2, linestyle="--",
        color=_MUTED, label="causal bound",
    )
    ax_f.set_xlabel("Trotter step")
    ax_f.set_ylabel(f"front distance (w > {REFERENCE_FRONT_THRESHOLD:.0e})")
    ax_f.set_title(f"front vs kick angle, min_abs_coeff = {tightest:.0e}")
    _style(ax_f)
    ax_f.legend(frameon=False, fontsize=8)

    for slot, eps in enumerate(SCAN_EPS):
        xs, ys = [], []
        for theta in THETA_SCAN:
            row = fits.get((theta, eps), {})
            if REFERENCE_FRONT_THRESHOLD in row:
                xs.append(theta)
                ys.append(row[REFERENCE_FRONT_THRESHOLD])
        if not xs:
            continue
        ax_v.plot(
            xs, ys, marker="o", markersize=5, linewidth=1.8,
            color=_PALETTE[slot % len(_PALETTE)], label=_eps_label(eps),
        )
    ax_v.axhline(1.0, color=_MUTED, linewidth=1.2, linestyle="--", label="causal bound")
    ax_v.set_xlabel("kick angle theta_h")
    ax_v.set_ylabel("fitted v_B (sites / Trotter step)")
    ax_v.set_title(f"butterfly velocity, contour w > {REFERENCE_FRONT_THRESHOLD:.0e}")
    _style(ax_v)
    ax_v.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Part 5 -- records, figures
# --------------------------------------------------------------------------


def build_records(series: dict[float, list[StepData]], threads: int | None) -> list:
    # Collected once: `collect_provenance` shells out to git and rustc, and a
    # sweep produces dozens of records that all describe the same process.
    provenance = report.collect_provenance(
        thread_count=threads, repo_root=_REPO_ROOT
    )
    records = []
    for eps, data in series.items():
        for d in data:
            headline_site = CENTER + HEADLINE_OFFSET
            records.append(
                report.RunRecord(
                    engine="paulistrings",
                    engine_version=provenance.library_versions.get(
                        "paulistrings", "unknown"
                    ),
                    n_qubits=N_QUBITS,
                    direction="heisenberg",
                    truncation={"min_abs_coeff": eps},
                    propagation_time_s=d.seconds,
                    final_terms=d.terms,
                    provenance=provenance,
                    peak_terms=d.peak_terms,
                    expectation_value=float(d.otoc[headline_site]),
                    extra={
                        "quantity": "otoc_x",
                        "otoc_probe_site": headline_site,
                        "trotter_step": d.step,
                        "theta_h": THETA_H,
                        "theta_zz": THETA_ZZ,
                        "center": CENTER,
                        "hs_norm": d.norm,
                        "support_size": d.support,
                        "support_floor": SUPPORT_FLOOR,
                        "front": {f"{k:.0e}": v for k, v in d.fronts.items()},
                        "weight_profile": [float(v) for v in d.profile],
                        "otoc_x_profile": [float(v) for v in d.otoc],
                        "correlator_z_profile": [float(v) for v in d.correlator],
                    },
                )
            )
    return records


def build_scan_records(
    scan: dict[tuple[float, float], list[StepData]], threads: int | None
) -> list:
    """`RunRecord`s for the kick-angle scan.

    Tagged `quantity="theta_scan"` in `extra` so the two halves of the results
    file stay separable; the per-site profiles are left out here (the scan is
    about one scalar per step, the front distance) to keep the committed JSON
    from doubling in size for data no figure reads.
    """
    provenance = report.collect_provenance(thread_count=threads, repo_root=_REPO_ROOT)
    records = []
    for (theta, eps), data in scan.items():
        for d in data:
            records.append(
                report.RunRecord(
                    engine="paulistrings",
                    engine_version=provenance.library_versions.get(
                        "paulistrings", "unknown"
                    ),
                    n_qubits=N_QUBITS,
                    direction="heisenberg",
                    truncation={"min_abs_coeff": eps},
                    propagation_time_s=d.seconds,
                    final_terms=d.terms,
                    provenance=provenance,
                    peak_terms=d.peak_terms,
                    extra={
                        "quantity": "theta_scan",
                        "trotter_step": d.step,
                        "theta_h": theta,
                        "theta_zz": THETA_ZZ,
                        "center": CENTER,
                        "hs_norm": d.norm,
                        "support_size": d.support,
                        "support_floor": SUPPORT_FLOOR,
                        "front": {f"{k:.0e}": v for k, v in d.fronts.items()},
                    },
                )
            )
    return records


# The house plot style: `report.py`'s palette and axis treatment, reproduced
# locally because none of its helpers takes an (site, t) heat map or an
# r-profile. Categorical slots go to identity (a contour level); the ordered
# parameters -- Trotter step, and the cutoff itself -- get a single-hue
# sequential ramp, light to dark, which is what makes "tighter cutoff = darker
# curve = converged" readable at a glance.
_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
_GRID = "#e1e0d9"
_MUTED = "#898781"


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)


def _ramp(count: int, cmap: str = "Blues", lo: float = 0.35, hi: float = 1.0):
    import matplotlib.pyplot as plt

    scale = plt.get_cmap(cmap)
    if count == 1:
        return [scale(hi)]
    return [scale(lo + (hi - lo) * i / (count - 1)) for i in range(count)]


def _eps_label(eps: float) -> str:
    return f"min_abs_coeff = {eps:.0e}"


def plot_support_growth(series: dict[float, list[StepData]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(9.5, 4))
    colors = _ramp(len(series))
    for color, (eps, data) in zip(colors, series.items()):
        steps = [d.step for d in data]
        ax_s.plot(
            steps, [d.support for d in data], marker="o", markersize=4,
            linewidth=1.8, color=color, label=_eps_label(eps),
        )
        ax_n.plot(
            steps, [1.0 - d.norm for d in data], marker="o", markersize=4,
            linewidth=1.8, color=color, label=_eps_label(eps),
        )
    cone = [min(2 * t + 1, N_QUBITS) for t in range(1, STEPS + 1)]
    ax_s.plot(
        range(1, STEPS + 1), cone, linewidth=1.2, linestyle="--", color=_MUTED,
        label="causal cone (2t+1 sites)",
    )
    ax_s.set_xlabel("Trotter step")
    ax_s.set_ylabel(f"sites with w_q > {SUPPORT_FLOOR:.0e}")
    ax_s.set_title("operator support growth")
    _style(ax_s)
    ax_s.legend(frameon=False, fontsize=8)

    ax_n.set_yscale("log")
    ax_n.set_xlabel("Trotter step")
    ax_n.set_ylabel("discarded weight  1 - N(t)")
    ax_n.set_title("what truncation threw away")
    _style(ax_n)
    ax_n.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_light_cone(series: dict[float, list[StepData]], path: Path) -> None:
    """One heat map per cutoff: `w_q(t)` on a log color scale, front overlaid.

    Small multiples rather than an overlay because a heat map cannot be
    overlaid — and side by side is exactly the convergence panel: the loose
    cutoffs are visibly missing the outer edge of the cone the tight one has.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    half = min(STEPS + 2, CENTER)
    sites = np.arange(CENTER - half, CENTER + half + 1)
    vmin, vmax = 1e-8, 1.0

    fig, axes = plt.subplots(
        1, len(series), figsize=(3.3 * len(series), 3.6), sharey=True
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (eps, data) in zip(axes, series.items()):
        grid = np.full((len(data), sites.size), np.nan)
        for i, d in enumerate(data):
            grid[i] = np.clip(d.profile[sites], vmin, None)
        mesh = ax.pcolormesh(
            sites - CENTER,
            [d.step for d in data],
            grid,
            cmap="Blues",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="nearest",
        )
        ax.plot(
            [d.fronts[REFERENCE_FRONT_THRESHOLD] for d in data],
            [d.step for d in data],
            color="#e34948", linewidth=1.6, marker="o", markersize=3,
            label=f"front w>{REFERENCE_FRONT_THRESHOLD:.0e}",
        )
        ax.plot(
            [d.step for d in data], [d.step for d in data],
            color=_MUTED, linewidth=1.1, linestyle="--", label="causal bound",
        )
        ax.set_xlabel("site - centre")
        ax.set_title(_eps_label(eps), fontsize=9)
        _style(ax)
    axes[0].set_ylabel("Trotter step")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes), pad=0.02)
        bar.set_label("per-site weight  w_q(t)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_otoc(series: dict[float, list[StepData]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(9.5, 4))

    reference = series[REFERENCE_EPS]
    shown = [d for d in reference if d.step >= 2]
    for color, d in zip(_ramp(len(shown)), shown):
        offsets = np.arange(N_QUBITS) - CENTER
        keep = np.abs(offsets) <= STEPS + 1
        ax_t.plot(
            offsets[keep], np.clip(d.otoc[keep], 1e-12, None),
            linewidth=1.6, color=color, label=f"t = {d.step}",
        )
    ax_t.set_yscale("log")
    ax_t.set_ylim(1e-10, 3.0)
    ax_t.set_xlabel("probe site r - centre")
    ax_t.set_ylabel("C(r, t)")
    ax_t.set_title(f"OTOC front, {_eps_label(REFERENCE_EPS)}")
    _style(ax_t)
    ax_t.legend(frameon=False, fontsize=7, ncols=2)

    last = min(len(data) for data in series.values())
    for color, (eps, data) in zip(_ramp(len(series)), series.items()):
        d = data[last - 1]
        offsets = np.arange(N_QUBITS) - CENTER
        keep = np.abs(offsets) <= STEPS + 1
        ax_c.plot(
            offsets[keep], np.clip(d.otoc[keep], 1e-12, None),
            marker="o", markersize=3, linewidth=1.6, color=color, label=_eps_label(eps),
        )
    ax_c.set_yscale("log")
    ax_c.set_ylim(1e-10, 3.0)
    ax_c.set_xlabel("probe site r - centre")
    ax_c.set_ylabel("C(r, t)")
    ax_c.set_title(f"convergence panel, t = {last}")
    _style(ax_c)
    ax_c.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_butterfly(
    series: dict[float, list[StepData]],
    fits: dict[float, dict[float, float]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_f, ax_v) = plt.subplots(1, 2, figsize=(9.5, 4))

    for color, (eps, data) in zip(_ramp(len(series)), series.items()):
        steps = [d.step for d in data]
        fronts = [d.fronts[REFERENCE_FRONT_THRESHOLD] for d in data]
        ax_f.plot(
            steps, fronts, marker="o", markersize=4, linewidth=1.8, color=color,
            label=_eps_label(eps),
        )
        slope = fits.get(eps, {}).get(REFERENCE_FRONT_THRESHOLD)
        if slope is not None:
            fit_steps = [s for s in steps if s in FIT_STEPS]
            _, intercept = sc.front_velocity(
                fit_steps, [f for s, f in zip(steps, fronts) if s in FIT_STEPS]
            )
            xs = np.array([min(fit_steps), max(fit_steps)], dtype=float)
            ax_f.plot(xs, slope * xs + intercept, linewidth=1.0, linestyle=":", color=color)
    ax_f.plot(
        range(1, STEPS + 1), range(1, STEPS + 1), linewidth=1.2, linestyle="--",
        color=_MUTED, label="causal bound (1 site/step)",
    )
    ax_f.set_xlabel("Trotter step")
    ax_f.set_ylabel(f"front distance (w > {REFERENCE_FRONT_THRESHOLD:.0e})")
    ax_f.set_title("light-cone front and linear fits")
    _style(ax_f)
    ax_f.legend(frameon=False, fontsize=8)

    for slot, thr in enumerate(FRONT_THRESHOLDS):
        xs, ys = [], []
        for eps in EPS_GRID:
            if thr in fits.get(eps, {}):
                xs.append(eps)
                ys.append(fits[eps][thr])
        if not xs:
            continue
        ax_v.plot(
            xs, ys, marker="o", markersize=5, linewidth=1.8,
            color=_PALETTE[slot % len(_PALETTE)], label=f"contour w > {thr:.0e}",
        )
    ax_v.set_xscale("log")
    ax_v.invert_xaxis()
    ax_v.axhline(1.0, color=_MUTED, linewidth=1.2, linestyle="--", label="causal bound")
    ax_v.set_xlabel("min_abs_coeff (tighter to the right)")
    ax_v.set_ylabel("fitted v_B (sites / Trotter step)")
    ax_v.set_title("butterfly velocity vs. truncation")
    _style(ax_v)
    ax_v.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_headline_convergence(records: list, path: Path) -> None:
    """`report.plot_convergence_panel` on the headline OTOC value.

    The shared helper keys its x axis on `RunRecord.truncation["min_abs_coeff"]`
    and its y on `expectation_value`, so the records handed to it are filtered
    to a single time step; the y label is retitled afterwards because the
    helper's generic label ("expectation value") would misname an OTOC.
    """
    import matplotlib.pyplot as plt

    sweep = [r for r in records if r.extra.get("quantity") == "otoc_x"]
    last = max(r.extra["trotter_step"] for r in sweep)
    while True:
        subset = [r for r in sweep if r.extra["trotter_step"] == last]
        if len(subset) == len({r.truncation["min_abs_coeff"] for r in subset}) and len(
            subset
        ) == len(EPS_GRID):
            break
        last -= 1
        if last < 1:
            subset = []
            break
    fig, ax = plt.subplots(figsize=(5.5, 4))
    report.plot_convergence_panel(subset, truncation_key="min_abs_coeff", ax=ax)
    site = subset[0].extra["otoc_probe_site"] if subset else CENTER
    ax.set_ylabel(f"C(r = centre + {site - CENTER}, t = {last})")
    ax.set_title("headline OTOC vs. truncation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    values = [r.expectation_value for r in sorted(
        subset, key=lambda r: -r.truncation["min_abs_coeff"]
    )]
    if len(values) >= 2:
        print(
            f"headline OTOC at t={last}, r=centre+{site - CENTER}: "
            + " -> ".join(f"{v:.6f}" for v in values)
            + f"   (last two differ by {abs(values[-1] - values[-2]):.2e})"
        )


def write_figures(
    series: dict[float, list[StepData]],
    scan: dict[tuple[float, float], list[StepData]],
    records: list,
) -> None:
    plot_support_growth(series, OUT_DIR / "support_growth.svg")
    plot_light_cone(series, OUT_DIR / "light_cone_1d.svg")
    plot_otoc(series, OUT_DIR / "otoc_1d.svg")
    fits = report_velocities(series)
    plot_butterfly(series, fits, OUT_DIR / "butterfly_velocity_1d.svg")
    plot_headline_convergence(records, OUT_DIR / "convergence_panel_1d.svg")
    scan_fits = report_theta_scan(scan)
    plot_theta_scan(scan, scan_fits, OUT_DIR / "velocity_vs_kick_angle.svg")
    for name in (
        "support_growth.svg",
        "light_cone_1d.svg",
        "otoc_1d.svg",
        "butterfly_velocity_1d.svg",
        "convergence_panel_1d.svg",
        "velocity_vs_kick_angle.svg",
    ):
        print(f"wrote {OUT_DIR / name}")


def write_json(records: list) -> None:
    # `report.write_results` appends, which is right for a results directory but
    # wrong for a committed artifact that must be regenerable: drop the old file
    # first so a rerun replaces it instead of growing it.
    path = OUT_DIR / "results_1d.json"
    if path.exists():
        path.unlink()
    report.write_results(records, OUT_DIR, name="results_1d")
    print(f"wrote {path}  ({len(records)} records)")


def main() -> None:
    harness.assert_logging_quiet()
    run_validation()
    run_clifford_cone()
    series = run_sweep()
    scan = run_theta_scan()
    threads = harness.rayon_worker_estimate()
    print(f"observed Rayon workers: {threads}")
    records = build_records(series, threads) + build_scan_records(scan, threads)
    write_figures(series, scan, records)
    write_json(records)
    print()
    print("done.")


if __name__ == "__main__":
    main()

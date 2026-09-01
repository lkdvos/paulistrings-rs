"""Showcase B1, phase 2 — a 2D quench, and where the coefficient cutoff gives out.

Handoff item B1 (2D half); see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B (runtime class **manual-long**, time-boxed per §8 D15) and global rule
4. Analysis helpers: `scrambling.py`. Companion: `run_b1_1d.py`.

The setup
---------
A `L x L` open square lattice (edges built here — `circuits.py` ships heavy-hex
and chain topologies, and the plan allows a locally built square lattice), a
single-site `Z` at the centre, and the *same* kicked-Ising step builder as the
1D run, now driven with a physical Trotter step of the transverse-field Ising
Hamiltonian

    H = J sum_<ij> Z_i Z_j + h sum_i X_i,      theta_zz = 2 J dt,  theta_h = 2 h dt

so that "time" is a physical `t = steps * dt` rather than a Floquet period, and
the step count can be increased to resolve the dynamics instead of merely
advancing it. Three quantities are tracked, each with a `min_abs_coeff`
convergence panel:

- **magnetization** `<Z_c(t)>` in the quench initial state `|0...0>`
  (`state="z+"`) — the local observable a quench experiment measures;
- **the infinite-temperature two-point function**
  `G(r,t) = Tr(Z_r U^dagger Z_c U) / 2^n`, which in the Pauli basis is
  literally the coefficient of the weight-one string `Z_r` in the evolved sum
  (`scrambling.single_pauli_coefficients`) — no extra propagation needed;
- **the light cone** `w_q(t)`, as a spatial map over the lattice and as a
  radial profile against graph distance.

Why 2D is the honest hard case
------------------------------
In 1D the causal cone grows by one site per Trotter step, so the number of
Pauli strings inside it grows like `4^(2t)`. In 2D the cone is an area: after
`t` steps it holds `O(t^2)` sites and `4^(O(t^2))` strings. The coefficient
cutoff has to fight that, and it loses — not gradually, but at a specific step
this script measures. The reported diagnostic is the retained
Hilbert-Schmidt norm `N(t) = sum_P |c_P|^2`, which exact evolution conserves;
`1 - N(t)` is exactly the fraction of the operator that was deleted.

Rather than pick a truncation and present the resulting curve, this script
sweeps `min_abs_coeff` and lets each cutoff run until it hits a **term
ceiling** (`TERM_CEILING`), then reports, per cutoff, how far it got and how
much norm it had left. The converged window is therefore *measured*, and the
wall is a result, not a hidden assumption.

A weight cap is deliberately **not** used to push further. It was tried and it
does not work at this entangling strength: on a degree-4 lattice one
`exp(i pi/4 Z_i Z_j)` layer takes a weight-1 string to weight 5, so a cap below
roughly `1 + 4t` deletes the dynamics rather than its tail. Measured by
`run_weight_cap_probe` below: a cap of 4 has lost 61% of the operator by step 2,
a cap of 8 46% by step 3, a cap of 12 48% by step 4. Recording that is more
useful than shipping a curve produced by a cap that is really a hard cutoff on
time.

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    python examples/b1_operator_scrambling/run_b1_2d.py

Threads: default Rayon pool, recorded in the results JSON (this is a physics
measurement, not a timing claim; see `run_b1_1d.py`'s note).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from paulistrings import truncation  # noqa: E402

from b1_operator_scrambling import run_b1_1d as b1d  # noqa: E402
from b1_operator_scrambling import scrambling as sc  # noqa: E402
from common import circuits, harness, observables, oracles, report  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

#: Lattice size. 7x7 = 49 sites: odd, so there is an exact centre; large enough
#: that the front never reaches the boundary inside the converged window
#: (centre-to-edge distance 3 in each direction, 6 in graph distance); and
#: inside the `W = 1` monomorphization. An 8x8 pilot cost ~2x per step for
#: identical early-time physics (the term count is set by the cone, not by `n`)
#: -- the recorded cut, README §Cuts.
ROWS = COLS = 7
N_QUBITS = ROWS * COLS
CENTER = sc.square_lattice_index(ROWS, COLS, ROWS // 2, COLS // 2)

#: Physical Trotter step of `H = J sum ZZ + h sum X`. `dt = 0.15` is small
#: enough that the discarded weight per step is tiny at the start (so the
#: convergence window is set by operator growth, not by Trotter error) and
#: large enough that 10 steps reach a physical time where the front has left
#: the seed site.
J_COUPLING = 1.0
H_FIELD = 1.0
DT = 0.15
THETA_ZZ = 2.0 * J_COUPLING * DT
THETA_H = 2.0 * H_FIELD * DT

STEPS = 10

#: `min_abs_coeff` grid, loosest first.
EPS_GRID = (1e-4, 1e-5, 1e-6, 1e-7)
REFERENCE_EPS = 1e-6

#: Stored-term ceiling: the time-box, in code. At ~1.2e8 terms one step costs
#: ~40 s and the numpy export copies ~4 GiB, both comfortable; the next factor
#: of four in either is not, so a cutoff that passes the ceiling stops and the
#: script reports where.
TERM_CEILING = 1.2e8

SUPPORT_FLOOR = 1e-6

#: Times whose spatial weight map is drawn.
MAP_STEPS = (2, 4, 6, 8)

# Validation: a 3x3 lattice is 9 qubits, so the dense 2^n reference is cheap.
VALIDATION_ROWS = VALIDATION_COLS = 3
VALIDATION_STEPS = 4
DENSE_TOLERANCE = 1e-10

#: Per-term dense checks: the `COEFFICIENT_SAMPLE` largest coefficients plus
#: every `COEFFICIENT_STRIDE`-th of the remainder. See `run_validation`.
COEFFICIENT_SAMPLE = 500
COEFFICIENT_STRIDE = 97


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def lattice_step_circuit(rows: int, cols: int, steps: int = 1):
    """One (or `steps`) Trotter step(s) of the square-lattice TFIM."""
    return circuits.heavy_hex_kicked_ising(
        rows * cols,
        steps,
        THETA_H,
        THETA_ZZ,
        edges=sc.square_lattice_edges(rows, cols),
    )


def lattice_step_spec(rows: int, cols: int, steps: int = 1):
    """The same circuit as a `CircuitSpec`, for the dense reference."""
    return oracles.record_gates(
        circuits.heavy_hex_kicked_ising,
        rows * cols,
        steps,
        THETA_H,
        THETA_ZZ,
        edges=sc.square_lattice_edges(rows, cols),
    )


@dataclass
class StepData:
    """Everything measured at one `(min_abs_coeff, trotter step)` point."""

    step: int
    terms: int
    peak_terms: int
    seconds: float
    norm: float
    magnetization: float
    profile: np.ndarray
    correlator: np.ndarray
    otoc: np.ndarray
    support: int
    radial_weight: np.ndarray
    radial_correlator: np.ndarray

    @property
    def time(self) -> float:
        return self.step * DT


def radial_average(values: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Mean of `values` over the sites at each graph distance `0, 1, 2, ...`."""
    out = np.zeros(int(distances.max()) + 1, dtype=np.float64)
    for d in range(out.size):
        mask = distances == d
        out[d] = float(np.mean(values[mask])) if np.any(mask) else 0.0
    return out


def evolve_series(
    rows: int,
    cols: int,
    steps: int,
    eps: float,
    *,
    term_ceiling: float = TERM_CEILING,
    verbose: bool = True,
) -> list[StepData]:
    n = rows * cols
    center = sc.square_lattice_index(rows, cols, rows // 2, cols // 2)
    distances = sc.square_lattice_distances(rows, cols, center)
    step_circuit = lattice_step_circuit(rows, cols)
    policy = truncation.coeff(eps)
    evolved = observables.single_z(center, n)
    out: list[StepData] = []
    for step in range(1, steps + 1):
        start = time.perf_counter()
        evolved, stats = evolved.propagate_with_stats(
            step_circuit, policy, direction="heisenberg"
        )
        seconds = time.perf_counter() - start
        magnetization = complex(evolved.expectation("z+")).real
        sums = sc.site_sums(evolved)
        profile = sc.support_profile(evolved, sums=sums)
        otoc = sc.otoc_from_sums(sums, "X")
        correlator = sc.single_pauli_coefficients(evolved, "Z")
        data = StepData(
            step=step,
            terms=stats.final_terms,
            peak_terms=stats.peak_terms,
            seconds=seconds,
            norm=sums.norm,
            magnetization=magnetization,
            profile=profile,
            correlator=correlator,
            otoc=otoc,
            support=sc.support_size(profile, SUPPORT_FLOOR),
            radial_weight=radial_average(profile, distances),
            radial_correlator=radial_average(correlator, distances),
        )
        out.append(data)
        if verbose:
            print(
                f"    step {step:2d}  t={data.time:4.2f}  terms={data.terms:>10d}  "
                f"norm={data.norm:.6f}  <Z_c>={magnetization:+.8f}  "
                f"support={data.support:3d}  {seconds:7.2f}s",
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
# Part 1 -- dense cross-check on a 3x3 lattice
# --------------------------------------------------------------------------


def run_validation() -> dict[str, float]:
    print("=" * 78)
    rows, cols = VALIDATION_ROWS, VALIDATION_COLS
    print(f"Part 1 -- dense cross-check on a {rows}x{cols} lattice ({rows * cols} qubits)")
    print("=" * 78)

    n = rows * cols
    center = sc.square_lattice_index(rows, cols, rows // 2, cols // 2)
    spec = lattice_step_spec(rows, cols, VALIDATION_STEPS)
    label = observables.pauli_string({center: "Z"}, n)

    dense = sc.dense_heisenberg(spec, label)
    evolved = observables.single_z(center, n).propagate(
        spec.to_circuit(), None, direction="heisenberg"
    )
    sums = sc.site_sums(evolved)
    profile = sc.support_profile(evolved, sums=sums)
    correlator = sc.single_pauli_coefficients(evolved, "Z")

    print(
        f"{rows}x{cols} lattice, centre site {center}, {VALIDATION_STEPS} steps, "
        f"dt={DT}, theta_zz={THETA_ZZ}, theta_h={THETA_H}"
    )
    print(f"  engine terms (untruncated) = {len(evolved)}")

    results: dict[str, float] = {
        "norm_gap": abs(sums.norm - sc.dense_hs_norm(dense)),
        "support_profile_gap": float(
            np.max(np.abs(profile - sc.dense_support_profile(dense, n)))
        ),
        "correlator_gap": float(
            np.max(
                np.abs(
                    correlator
                    - np.array(
                        [
                            sc.dense_coefficient(
                                dense, observables.pauli_string({r: "Z"}, n)
                            ).real
                            for r in range(n)
                        ]
                    )
                )
            )
        ),
        "probe_average_gap": sc.probe_average_gap(sums),
    }
    for probe in ("X", "Y", "Z"):
        results[f"otoc_gap_{probe}"] = float(
            np.max(
                np.abs(
                    sc.otoc_from_sums(sums, probe)
                    - np.array([sc.dense_otoc(dense, probe, r, n) for r in range(n)])
                )
            )
        )
    # Coefficient-by-coefficient, over a bounded and deterministic sample: a
    # dense `<P, O>` costs ~1 ms and this sum holds ~6e4 terms, so checking all
    # of them would dominate the script's runtime for no extra coverage. The
    # sample is the largest-|c| terms plus a fixed stride through the rest (no
    # RNG, so it is reproducible), and the `norm_gap` above is what rules out a
    # *missing* term, which a per-term check cannot see.
    terms = sorted(oracles.pauli_terms(evolved), key=lambda kv: -abs(kv[1]))
    sample = terms[:COEFFICIENT_SAMPLE] + terms[COEFFICIENT_SAMPLE::COEFFICIENT_STRIDE]
    results["coefficient_gap"] = float(
        max(
            abs(coefficient - sc.dense_coefficient(dense, term_label))
            for term_label, coefficient in sample
        )
    )
    print(f"  coefficients checked against dense: {len(sample)} of {len(terms)}")

    # The magnetization has its own independent reference: the dense operator
    # contracted with |0...0>, which is its (0, 0) matrix element.
    dense_magnetization = float(np.real(dense[0, 0]))
    engine_magnetization = complex(evolved.expectation("z+")).real
    results["magnetization_gap"] = abs(dense_magnetization - engine_magnetization)
    print(f"  <0|O(t)|0>: engine {engine_magnetization:.15f}  dense {dense_magnetization:.15f}")

    for name, gap in sorted(results.items()):
        print(f"  {name:<24} = {gap:.3e}")
    worst = max(results.values())
    assert worst <= DENSE_TOLERANCE, (
        f"the dense cross-check on the {rows}x{cols} lattice disagrees by "
        f"{worst:.3e} > {DENSE_TOLERANCE:.1e}: {results}"
    )
    print(f"  worst gap {worst:.3e} <= {DENSE_TOLERANCE:.1e}  OK")
    print()
    return results


# --------------------------------------------------------------------------
# Part 2 -- the weight-cap probe (a recorded negative result)
# --------------------------------------------------------------------------


def run_weight_cap_probe() -> list[tuple[int, int, int, float]]:
    """Show, in numbers, why a weight cap cannot extend the 2D window.

    Run at the *kicked-Ising Clifford entangler* (`theta_zz = -pi/2`), the
    maximally entangling bond angle the rest of this suite uses: on a degree-4
    lattice one such layer takes a weight-1 string to weight 5, so a cap below
    roughly `1 + 4t` truncates the dynamics itself rather than its tail. The
    printed norms are the evidence.
    """
    print("=" * 78)
    print("Part 2 -- weight cap probe (why it cannot rescue 2D)")
    print("=" * 78)
    n = N_QUBITS
    step = circuits.heavy_hex_kicked_ising(
        n, 1, 0.9, circuits.KICKED_ISING_CLIFFORD_THETA_ZZ,
        edges=sc.square_lattice_edges(ROWS, COLS),
    )
    rows: list[tuple[int, int, int, float]] = []
    for cap in (4, 8, 12):
        evolved = observables.single_z(CENTER, n)
        policy = truncation.weight(cap) & truncation.coeff(1e-8)
        for t in range(1, 5):
            evolved = evolved.propagate(step, policy, direction="heisenberg")
            rows.append((cap, t, len(evolved), sc.hs_norm(evolved)))
        for cap_seen, t, terms, norm in rows[-4:]:
            print(f"  max_weight={cap_seen:2d}  step {t}: terms {terms:6d}, retained norm {norm:.6f}")
    print()
    return rows


# --------------------------------------------------------------------------
# Part 3 -- the sweep
# --------------------------------------------------------------------------


def run_sweep() -> dict[float, list[StepData]]:
    print("=" * 78)
    print(f"Part 3 -- min_abs_coeff sweep on the {ROWS}x{COLS} quench")
    print("=" * 78)
    print(
        f"{ROWS}x{COLS} = {N_QUBITS} sites, centre {CENTER}, dt={DT} "
        f"(theta_zz={THETA_ZZ}, theta_h={THETA_H}), up to {STEPS} steps "
        f"(t = {STEPS * DT:.2f})"
    )
    print(f"channels per Trotter step: {len(lattice_step_circuit(ROWS, COLS))}")
    series: dict[float, list[StepData]] = {}
    for eps in EPS_GRID:
        print(f"  min_abs_coeff = {eps:.1e}")
        series[eps] = evolve_series(ROWS, COLS, STEPS, eps)
    print()
    return series


def report_window(series: dict[float, list[StepData]]) -> int:
    """Print the convergence table and return the last step common to all cutoffs."""
    common = min(len(data) for data in series.values())
    print("Magnetization <Z_c(t)> by cutoff (the convergence panel, in numbers):")
    header = "  step  t    " + "".join(f"{eps:>14.0e}" for eps in EPS_GRID)
    print(header)
    reached = max(len(data) for data in series.values())
    for i in range(reached):
        cells = ""
        for eps in EPS_GRID:
            data = series[eps]
            cells += f"{data[i].magnetization:14.8f}" if i < len(data) else " " * 14
        print(f"  {i + 1:>4}  {(i + 1) * DT:4.2f} {cells}")
    print()
    print("Retained Hilbert-Schmidt norm N(t) by cutoff:")
    print(header)
    for i in range(reached):
        cells = ""
        for eps in EPS_GRID:
            data = series[eps]
            cells += f"{data[i].norm:14.6f}" if i < len(data) else " " * 14
        print(f"  {i + 1:>4}  {(i + 1) * DT:4.2f} {cells}")
    print()
    tight = EPS_GRID[-1]
    for eps in EPS_GRID:
        data = series[eps]
        print(
            f"  min_abs_coeff={eps:.0e}: reached step {len(data)} "
            f"(t={len(data) * DT:.2f}), {data[-1].terms} terms, N={data[-1].norm:.6f}"
        )
    print(f"  last step reached by every cutoff (incl. {tight:.0e}): {common}")
    gaps = []
    for i in range(common):
        a = series[EPS_GRID[-2]][i].magnetization
        b = series[EPS_GRID[-1]][i].magnetization
        gaps.append(abs(a - b))
        print(
            f"    step {i + 1}: |<Z_c> at {EPS_GRID[-2]:.0e} - <Z_c> at "
            f"{EPS_GRID[-1]:.0e}| = {gaps[-1]:.2e}"
        )
    print()
    return common


# --------------------------------------------------------------------------
# Part 4 -- 3D pilot (measured, not extrapolated)
# --------------------------------------------------------------------------

#: 3x3x3 = 27 sites, degree up to 6. Small enough that the pilot is cheap and
#: the answer is unambiguous: if the coefficient cutoff cannot follow a *27
#: site* cubic lattice, it will not follow a bigger one.
PILOT_3D = (3, 3, 3)
PILOT_3D_EPS = (1e-5, 1e-6)
PILOT_3D_STEPS = 10


def run_3d_pilot() -> dict[float, list[tuple[int, int, float, float]]]:
    """The same quench on a cubic lattice, to the term ceiling.

    The plan's 3D item is "only if trivially cheap after the 2D pilot --
    otherwise record it as deferred with the projected cost". This *is* the
    pilot: one small cubic lattice, the same `dt` and cutoffs as the 2D sweep,
    stopped by the same ceiling. What it produces is a measured cost curve for
    a degree-6 lattice, which is what the README's projection is built on
    instead of an extrapolated formula.

    Printed, not written to the results JSON: this is a cost measurement, not
    one of the showcase's physics curves, and the README §5 quotes it in full.
    """
    print("=" * 78)
    nx, ny, nz = PILOT_3D
    print(f"Part 4 -- 3D pilot on a {nx}x{ny}x{nz} cubic lattice ({nx * ny * nz} sites)")
    print("=" * 78)
    n = nx * ny * nz
    center = sc.cubic_lattice_index(nx, ny, nz, nx // 2, ny // 2, nz // 2)
    edges = sc.cubic_lattice_edges(nx, ny, nz)
    step = circuits.heavy_hex_kicked_ising(n, 1, THETA_H, THETA_ZZ, edges=edges)
    print(
        f"  {len(edges)} bonds, centre site {center}, dt={DT}, "
        f"{len(step)} channels per step"
    )
    out: dict[float, list[tuple[int, int, float, float]]] = {}
    for eps in PILOT_3D_EPS:
        evolved = observables.single_z(center, n)
        rows: list[tuple[int, int, float, float]] = []
        for t in range(1, PILOT_3D_STEPS + 1):
            start = time.perf_counter()
            evolved = evolved.propagate(step, truncation.coeff(eps), direction="heisenberg")
            seconds = time.perf_counter() - start
            norm = sc.hs_norm(evolved)
            magnetization = complex(evolved.expectation("z+")).real
            rows.append((t, len(evolved), norm, magnetization))
            print(
                f"  eps={eps:.0e} step {t:2d}  t={t * DT:4.2f}  terms={len(evolved):>10d}  "
                f"norm={norm:.6f}  <Z_c>={magnetization:+.8f}  {seconds:7.2f}s",
                flush=True,
            )
            if len(evolved) > TERM_CEILING:
                print(
                    f"  term ceiling {TERM_CEILING:.1e} passed at step {t}; stopping",
                    flush=True,
                )
                break
        out[eps] = rows
    print()
    return out


# --------------------------------------------------------------------------
# Part 5 -- records and figures
# --------------------------------------------------------------------------


def build_records(series: dict[float, list[StepData]], threads: int | None) -> list:
    provenance = report.collect_provenance(thread_count=threads, repo_root=_REPO_ROOT)
    records = []
    for eps, data in series.items():
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
                    expectation_value=d.magnetization,
                    extra={
                        "quantity": "magnetization_z",
                        "lattice": f"{ROWS}x{COLS}",
                        "trotter_step": d.step,
                        "time": d.time,
                        "dt": DT,
                        "theta_h": THETA_H,
                        "theta_zz": THETA_ZZ,
                        "center": CENTER,
                        "state": "z+",
                        "hs_norm": d.norm,
                        "support_size": d.support,
                        "support_floor": SUPPORT_FLOOR,
                        "radial_weight": [float(v) for v in d.radial_weight],
                        "radial_correlator": [float(v) for v in d.radial_correlator],
                        "weight_profile": [float(v) for v in d.profile],
                        "correlator_z_profile": [float(v) for v in d.correlator],
                        "otoc_x_profile": [float(v) for v in d.otoc],
                    },
                )
            )
    return records


def plot_quench_observables(series: dict[float, list[StepData]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_m, ax_n) = plt.subplots(1, 2, figsize=(9.5, 4))
    for color, (eps, data) in zip(b1d._ramp(len(series)), series.items()):
        times = [d.time for d in data]
        ax_m.plot(
            times, [d.magnetization for d in data], marker="o", markersize=4,
            linewidth=1.8, color=color, label=b1d._eps_label(eps),
        )
        ax_n.plot(
            times, [1.0 - d.norm for d in data], marker="o", markersize=4,
            linewidth=1.8, color=color, label=b1d._eps_label(eps),
        )
    ax_m.set_xlabel("time  t = steps x dt")
    ax_m.set_ylabel("<Z_c(t)>  in  |0...0>")
    ax_m.set_title(f"local magnetization, {ROWS}x{COLS} quench")
    b1d._style(ax_m)
    ax_m.legend(frameon=False, fontsize=8)

    ax_n.set_yscale("log")
    ax_n.set_xlabel("time  t = steps x dt")
    ax_n.set_ylabel("discarded weight  1 - N(t)")
    ax_n.set_title("where the cutoff gives out")
    b1d._style(ax_n)
    ax_n.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_light_cone(series: dict[float, list[StepData]], path: Path) -> None:
    """Spatial weight maps at four times, plus the radial convergence panel."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    reference = series[REFERENCE_EPS]
    shown = [d for d in reference if d.step in MAP_STEPS]
    # `constrained_layout` rather than `tight_layout`: this figure mixes a
    # colorbar-carrying row of square maps with a row of line plots, which
    # `tight_layout` warns it cannot handle.
    fig, axes = plt.subplots(
        2, max(len(shown), 2), figsize=(2.6 * max(len(shown), 2), 5.6),
        layout="constrained",
    )
    vmin, vmax = 1e-8, 1.0
    mesh = None
    for ax, d in zip(axes[0], shown):
        grid = np.clip(d.profile.reshape(ROWS, COLS), vmin, None)
        mesh = ax.pcolormesh(
            np.arange(COLS), np.arange(ROWS), grid, cmap="Blues",
            norm=LogNorm(vmin=vmin, vmax=vmax), shading="nearest",
        )
        ax.set_title(f"t = {d.time:.2f} (step {d.step})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.tick_params(colors=b1d._MUTED, labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[0][len(shown) :]:
        ax.axis("off")
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes[0]), pad=0.02)
        bar.set_label("per-site weight  w_q(t)", fontsize=8)

    ax_r = axes[1][0]
    for color, d in zip(b1d._ramp(len(reference)), reference):
        ax_r.plot(
            np.arange(d.radial_weight.size), np.clip(d.radial_weight, 1e-12, None),
            linewidth=1.6, color=color, label=f"t = {d.time:.2f}",
        )
    ax_r.set_yscale("log")
    ax_r.set_xlabel("graph distance from centre")
    ax_r.set_ylabel("mean w_q")
    ax_r.set_title(f"radial profile, {b1d._eps_label(REFERENCE_EPS)}", fontsize=9)
    b1d._style(ax_r)
    ax_r.legend(frameon=False, fontsize=7, ncols=2)

    ax_c = axes[1][1]
    common = min(len(data) for data in series.values())
    for color, (eps, data) in zip(b1d._ramp(len(series)), series.items()):
        d = data[common - 1]
        ax_c.plot(
            np.arange(d.radial_weight.size), np.clip(d.radial_weight, 1e-12, None),
            marker="o", markersize=3, linewidth=1.6, color=color,
            label=b1d._eps_label(eps),
        )
    ax_c.set_yscale("log")
    ax_c.set_xlabel("graph distance from centre")
    ax_c.set_ylabel("mean w_q")
    ax_c.set_title(f"convergence panel, step {common}", fontsize=9)
    b1d._style(ax_c)
    ax_c.legend(frameon=False, fontsize=7)
    for ax in axes[1][2:]:
        ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_correlator(series: dict[float, list[StepData]], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_g, ax_c) = plt.subplots(1, 2, figsize=(9.5, 4))
    reference = series[REFERENCE_EPS]
    for color, d in zip(b1d._ramp(len(reference)), reference):
        ax_g.plot(
            np.arange(d.radial_correlator.size),
            np.abs(d.radial_correlator) + 1e-16,
            marker="o", markersize=3, linewidth=1.6, color=color,
            label=f"t = {d.time:.2f}",
        )
    ax_g.set_yscale("log")
    ax_g.set_xlabel("graph distance r from centre")
    ax_g.set_ylabel("|G(r, t)|")
    ax_g.set_title(f"two-point function, {b1d._eps_label(REFERENCE_EPS)}")
    b1d._style(ax_g)
    ax_g.legend(frameon=False, fontsize=7, ncols=2)

    common = min(len(data) for data in series.values())
    for color, (eps, data) in zip(b1d._ramp(len(series)), series.items()):
        d = data[common - 1]
        ax_c.plot(
            np.arange(d.radial_correlator.size),
            np.abs(d.radial_correlator) + 1e-16,
            marker="o", markersize=3, linewidth=1.6, color=color,
            label=b1d._eps_label(eps),
        )
    ax_c.set_yscale("log")
    ax_c.set_xlabel("graph distance r from centre")
    ax_c.set_ylabel("|G(r, t)|")
    ax_c.set_title(f"convergence panel, step {common}")
    b1d._style(ax_c)
    ax_c.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_headline_convergence(records: list, common: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = [r for r in records if r.extra["trotter_step"] == common]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    report.plot_convergence_panel(subset, truncation_key="min_abs_coeff", ax=ax)
    ax.set_ylabel(f"<Z_c> at step {common} (t = {common * DT:.2f})")
    ax.set_title("headline magnetization vs. truncation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_json(records: list) -> None:
    path = OUT_DIR / "results_2d.json"
    if path.exists():
        path.unlink()
    report.write_results(records, OUT_DIR, name="results_2d")
    print(f"wrote {path}  ({len(records)} records)")


def main() -> None:
    harness.assert_logging_quiet()
    run_validation()
    run_weight_cap_probe()
    series = run_sweep()
    common = report_window(series)
    run_3d_pilot()
    threads = harness.rayon_worker_estimate()
    print(f"observed Rayon workers: {threads}")
    records = build_records(series, threads)
    plot_quench_observables(series, OUT_DIR / "quench_observables_2d.svg")
    plot_light_cone(series, OUT_DIR / "light_cone_2d.svg")
    plot_correlator(series, OUT_DIR / "correlator_2d.svg")
    plot_headline_convergence(records, common, OUT_DIR / "convergence_panel_2d.svg")
    for name in (
        "quench_observables_2d.svg",
        "light_cone_2d.svg",
        "correlator_2d.svg",
        "convergence_panel_2d.svg",
    ):
        print(f"wrote {OUT_DIR / name}")
    write_json(records)
    print()
    print("done.")


if __name__ == "__main__":
    main()

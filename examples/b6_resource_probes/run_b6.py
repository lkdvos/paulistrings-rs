"""Showcase B6 -- resource-theoretic probes of the evolved observable.

Handoff item B6; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B (and decision D12) for the adapted specification. Every diagnostic
lives in `resource_probes.py` next to this file, computed in pure Python over
`PauliSum.x_array()` / `z_array()` / `coefficients_array()` -- read the module
docstring there for the definitions, the derivations of their properties, and
the literature they do and do not correspond to.

The question this showcase answers: **how hard is this operator, and where
does the hardness come from?** Two orthogonal answers, both read straight off
the evolved Pauli sum:

1. the **Pauli-spectrum Rényi-2 entropy** `S_2 = -ln sum_P p_P^2` (and its
   linear form `L = 1 - sum_P p_P^2`) over `p_P = |c_P|^2 / sum|c|^2` -- how
   many Pauli strings actually carry the operator's weight, which is what
   truncation-based Pauli propagation pays for directly;
2. the **operator entanglement entropy** `S_op = -sum_k λ_k ln λ_k` across the
   bipartition `[0, n/2) | [n/2, n)`, from the operator Schmidt spectrum --
   how hard the same operator would be for a matrix-product-operator method,
   a different cost model on the same object.

Four parts, run in order by `main()`:

* **Part 1** (`run_exact_cross_check`) -- at `n = 6, 8, 10`, rebuild the dense
  `2^n x 2^n` matrix with `numpy.kron` and recompute both diagnostics by
  routes that share no code with the array-based probes (brute force over all
  `4^n` Paulis; reshape-and-SVD of the dense matrix). Asserts agreement to
  `1e-10`, and asserts the two Clifford points read exactly zero. Writes
  `exact_cross_check.json`.
* **Part 2** (`run_theta_sweep`) -- 1D kicked-Ising chain, `n = 16`, 5 Trotter
  steps, `theta_h` swept from `0` to `pi/2` in 17 points. **Untruncated**, so
  nothing has to converge and no convergence panel is owed -- the curves are
  exact. Both diagnostics vanish at both Clifford endpoints and are nonzero,
  non-monotone in between. Writes `theta_sweep.csv` / `.svg`.
* **Part 3** (`run_depth_sweep`) -- same chain at `n = 20`, generic
  `theta_h = 0.6`, depth 1..7, exact where affordable (through depth 6) and
  truncated at `min_abs_coeff = 1e-6` throughout. Writes `depth_sweep.csv` /
  `.svg`.
* **Part 4** (`run_convergence_panels`) -- the panel plan §7 rule 4 requires
  for Part 3's truncated curve: `min_abs_coeff` swept at depth 6 (against the
  exact value, drawn as a reference line) and at depth 7 (beyond exact reach,
  self-converged). Writes `convergence_panel.svg`.

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    python examples/b6_resource_probes/run_b6.py

No arguments; every number is printed and every artifact is written next to
this script. Nothing here is a performance claim -- the whole script is a
couple of minutes of small linear algebra -- but one measurement *is*
load-bearing for it being that fast: the operator Schmidt SVDs are pinned to
one BLAS thread below, because with LAPACK left to spawn its own pool on a
busy host the same 462x1715 SVD was observed at 56 s instead of 0.14 s. That
also keeps the script on the suite's single-thread default (plan §7 rule 3).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Before numpy: pin LAPACK/BLAS to one thread (see the module docstring).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_HERE = Path(__file__).resolve().parent
for _path in (str(_EXAMPLES_DIR), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from paulistrings import truncation  # noqa: E402

import resource_probes as probes  # noqa: E402
from common import circuits, harness, observables, oracles, report  # noqa: E402

OUT_DIR = _HERE

# --------------------------------------------------------------------------
# Part 1 -- exact dense cross-check
# --------------------------------------------------------------------------

#: `(n, trotter_steps, cut, run_the_brute_force_4^n_spectrum)`. The exhaustive
#: Pauli-spectrum oracle costs `16^n` complex multiplications -- measured at
#: 0.6 s for n=6 and 99 s for n=8 on the development host -- so it runs at n=6
#: only, and `probes.MAX_DENSE_SPECTRUM_N` refuses it past n=8 outright. The
#: dense operator-entanglement oracle is `O(8^n)` and runs at all three sizes.
#: Depth and cut both vary across the three rows on purpose: at a *fixed*
#: depth the light cone makes the evolved operator identical at n=6, 8, 10, and
#: the operator entanglement across a single cut bond of a 1D chain turns out
#: to be n-independent as well, so a fixed (depth, cut) would give three
#: oracles on one number. The `numpy.kron` build is `O(T · 4^n)`, which caps
#: depth at 3 for n=10 (132 terms there is three seconds; 1430 is a minute).
EXACT_SIZES = ((6, 4, 3, True), (8, 4, 2, False), (10, 3, 5, False))
EXACT_THETA_H = 0.6
EXACT_TOL = 1e-10

# --------------------------------------------------------------------------
# Part 2 -- theta_h sweep (exact)
# --------------------------------------------------------------------------

THETA_N = 16
THETA_STEPS = 5
#: 0 .. pi/2 inclusive in 17 points. Both endpoints are Clifford points of the
#: kicked-Ising circuit (`theta_zz = -pi/2` is fixed at its Clifford value):
#: at `theta_h = 0` the X layer is the identity, and at `theta_h = pi/2` the X
#: rotation is a Clifford quarter turn -- so a single-Pauli seed stays a single
#: Pauli string and both diagnostics must read zero. See
#: `circuits.heavy_hex_kicked_ising`.
THETA_GRID = tuple(k * math.pi / 32 for k in range(17))
#: `theta_h = 0` is exact in floating point (the X layer really is identity).
#: `theta_h = pi/2` is Clifford only in exact arithmetic: `cos(pi/4)` and
#: `sin(pi/4)` are equal to the last bit but `cos(pi/2) = 6.1e-17`, so the
#: cancelled branch survives as dust and the sum keeps thousands of terms with
#: coefficients around 1e-49. The diagnostics are quadratic in those, hence a
#: bound around 1e-25 rather than an equality.
CLIFFORD_DUST_BOUND = 1e-25

# --------------------------------------------------------------------------
# Part 3/4 -- depth sweep and its convergence panels
# --------------------------------------------------------------------------

DEPTH_N = 20
DEPTH_THETA_H = 0.6  # generic, non-Clifford: no free stabilizer shortcut
DEPTH_GRID = tuple(range(1, 8))
#: Deepest point at which the untruncated sum is still cheap to diagnose
#: (208012 terms, a 462x1715 Schmidt matrix). Depth 7 exact is 2.67 M terms
#: and a Schmidt matrix past the guard, so it is truncated-only.
DEPTH_EXACT_MAX = 6
DEPTH_EPS = 1e-6
#: `min_abs_coeff` grid for the convergence panels, loosest first.
EPS_GRID = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
PANEL_DEPTHS = (6, 7)

#: Curves plotted by `_plot_sweep`, as `(diagnostic key, axis, label)`.
DIAGNOSTIC_CURVES = (
    ("pauli_renyi2", 0, "S_2  (Pauli spectrum)"),
    ("pauli_shannon", 0, "S_1  (Pauli spectrum)"),
    ("op_entanglement", 1, "S_op  (operator entanglement)"),
    ("op_entanglement_renyi2", 1, "S_op^(2)  (operator entanglement)"),
)


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def chain_edges(n: int) -> list[tuple[int, int]]:
    """Open 1D chain `0-1-2-...-(n-1)` as an edge list for the kicked-Ising
    builder.

    A 1D chain rather than a heavy-hex sublattice, purely so that the
    bipartition `[0, n/2) | [n/2, n)` the task specifies cuts **exactly one
    lattice bond** -- the operator-entanglement curve then measures growth
    across a single, unambiguous cut. On a heavy-hex sublattice the same index
    range cuts a topology-dependent number of edges, which would confound the
    reading. Everything else (layer order, the Clifford `theta_zz = -pi/2`
    entangler, one gate per channel) is the shared suite construction.
    """
    return [(i, i + 1) for i in range(n - 1)]


def cut_bonds(edges: Sequence[tuple[int, int]], cut: int) -> int:
    """How many lattice bonds cross the bipartition boundary at `cut`."""
    return sum(1 for a, b in edges if (a < cut) != (b < cut))


def evolve(n: int, steps: int, theta_h: float, *, policy=None):
    """Heisenberg-evolve `Z_{n/2}` through `steps` kicked-Ising Trotter steps."""
    circuit = circuits.heavy_hex_kicked_ising(n, steps, theta_h, edges=chain_edges(n))
    return observables.single_z(n // 2, n).propagate(
        circuit, policy, direction="heisenberg"
    )


def all_diagnostics(pauli_sum, cut: int) -> dict[str, Any]:
    """Every diagnostic for one evolved sum, as a flat dict of floats/ints."""
    rows, cols = probes.schmidt_matrix_shape(pauli_sum, cut)
    spectrum = probes.operator_schmidt_spectrum(pauli_sum, cut)
    return {
        "terms": len(pauli_sum),
        "hs_weight": probes.hilbert_schmidt_weight(pauli_sum),
        "pauli_renyi2": probes.pauli_spectrum_renyi2(pauli_sum),
        "pauli_linear": probes.pauli_spectrum_linear(pauli_sum),
        "pauli_shannon": probes.pauli_spectrum_shannon(pauli_sum),
        "op_entanglement": probes.renyi_entropy(spectrum, 1.0),
        "op_entanglement_renyi2": probes.renyi_entropy(spectrum, 2.0),
        "schmidt_rows": rows,
        "schmidt_cols": cols,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Every key any row has, in first-seen order; missing values blank."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(
            ",".join(
                "" if row.get(col) is None else _csv_value(row[col]) for col in columns
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


# --------------------------------------------------------------------------
# Part 1 -- exact dense cross-check
# --------------------------------------------------------------------------


def run_exact_cross_check() -> dict[str, Any]:
    print("=" * 78)
    print("Part 1 -- exact dense cross-check (independent numpy oracles)")
    print("=" * 78)

    sizes: list[dict[str, Any]] = []
    for n, steps, cut, do_spectrum in EXACT_SIZES:
        evolved = evolve(n, steps, EXACT_THETA_H)
        t0 = time.perf_counter()
        dense = probes.dense_matrix(oracles.pauli_terms(evolved))
        build_s = time.perf_counter() - t0

        row: dict[str, Any] = {
            "n_qubits": n,
            "trotter_steps": steps,
            "theta_h": EXACT_THETA_H,
            "cut": cut,
            "terms": len(evolved),
            "dense_build_s": build_s,
            "exhaustive_pauli_spectrum": do_spectrum,
        }

        # (a) Hilbert-Schmidt weight against tr(O^dag O) / 2^n -- the identity
        # that makes `sum_P |c_P|^2` a property of the operator rather than of
        # the representation.
        _record(row, "hs_weight", probes.hilbert_schmidt_weight(evolved),
                float(np.trace(dense.conj().T @ dense).real) / (1 << n))

        # (b) the Pauli spectrum, rebuilt from all 4^n traces tr(P O) / 2^n.
        if do_spectrum:
            t0 = time.perf_counter()
            p_dense = probes.dense_pauli_spectrum_probabilities(dense, n)
            row["dense_spectrum_s"] = time.perf_counter() - t0
            _record(row, "pauli_renyi2", probes.pauli_spectrum_renyi2(evolved),
                    probes.renyi_entropy(p_dense, 2.0))
            _record(row, "pauli_shannon", probes.pauli_spectrum_shannon(evolved),
                    probes.renyi_entropy(p_dense, 1.0))
            _record(row, "pauli_linear", probes.pauli_spectrum_linear(evolved),
                    float(1.0 - (p_dense**2).sum()))

        # (c) the operator Schmidt spectrum, rebuilt by reshaping the dense
        # matrix -- a route that never looks at a Pauli label or a bit.
        t0 = time.perf_counter()
        lambdas_dense = probes.dense_operator_schmidt_spectrum(dense, n, cut)
        row["dense_schmidt_s"] = time.perf_counter() - t0
        lambdas_sparse = probes.operator_schmidt_spectrum(evolved, cut)
        _record(row, "op_entanglement", probes.renyi_entropy(lambdas_sparse, 1.0),
                probes.renyi_entropy(lambdas_dense, 1.0))
        _record(row, "op_entanglement_renyi2", probes.renyi_entropy(lambdas_sparse, 2.0),
                probes.renyi_entropy(lambdas_dense, 2.0))

        print(
            f"n={n} steps={steps} terms={len(evolved)} cut={cut} "
            f"(dense build {build_s:.2f} s, "
            f"{'exhaustive 4^n spectrum' if do_spectrum else 'operator entanglement only'})"
        )
        for key in sorted(k[: -len("_gap")] for k in row if k.endswith("_gap")):
            print(
                f"    {key:<24} sparse={row[key]:.15f} "
                f"dense={row[key + '_dense']:.15f} gap={row[key + '_gap']:.3e}"
            )
            assert row[key + "_gap"] <= EXACT_TOL, (
                f"n={n}: {key} disagrees between the array-based probe and the "
                f"independent dense oracle by {row[key + '_gap']:.3e} > {EXACT_TOL:.0e}"
            )
        sizes.append(row)

    # (d) the Clifford points, where the answer is known with no oracle at all.
    print()
    print("Clifford points (theta_zz = -pi/2 fixed; a single-Pauli seed stays single):")
    clifford: list[dict[str, Any]] = []
    for theta_h, name, bound in (
        (0.0, "0", 0.0),
        (math.pi / 2, "pi/2", CLIFFORD_DUST_BOUND),
    ):
        diagnostics = all_diagnostics(evolve(THETA_N, THETA_STEPS, theta_h), THETA_N // 2)
        clifford.append(
            {"theta_h": theta_h, "theta_h_name": name, "bound": bound, **diagnostics}
        )
        print(
            f"    theta_h={name:<5} terms={diagnostics['terms']:<6} "
            f"S_2={diagnostics['pauli_renyi2']:.3e} "
            f"L={diagnostics['pauli_linear']:.3e} "
            f"S_op={diagnostics['op_entanglement']:.3e}  (bound {bound:.0e})"
        )
        for key in (
            "pauli_renyi2",
            "pauli_linear",
            "pauli_shannon",
            "op_entanglement",
            "op_entanglement_renyi2",
        ):
            assert abs(diagnostics[key]) <= bound, (
                f"theta_h={name} is a Clifford point, so {key} must vanish; got "
                f"{diagnostics[key]:.3e} > {bound:.0e}"
            )

    payload = {
        "showcase": "B6",
        "what": (
            "Independent dense-matrix cross-check of the Pauli-spectrum and "
            "operator-entanglement diagnostics in resource_probes.py, plus the two Clifford "
            "points of the kicked-Ising circuit. Every gap here is computed by this "
            "script; nothing is asserted from a stored value."
        ),
        "tolerance": EXACT_TOL,
        # The process's OS thread count, not a worker count: a one-worker Rayon
        # pool still shows as 2 threads here. The pins actually in force are
        # recorded next to it.
        "process_threads": harness.observed_thread_count(),
        "thread_pins": {
            var: os.environ.get(var)
            for var in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "provenance": report.collect_provenance(repo_root=_REPO_ROOT).__dict__,
        "sizes": sizes,
        "clifford_points": clifford,
    }
    path = OUT_DIR / "exact_cross_check.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")
    return payload


def _record(row: dict[str, Any], key: str, sparse: float, dense: float) -> None:
    """Store a probe value, its dense-oracle counterpart, and the gap."""
    row[key] = sparse
    row[f"{key}_dense"] = dense
    row[f"{key}_gap"] = abs(sparse - dense)


# --------------------------------------------------------------------------
# Part 2 -- theta_h sweep (exact)
# --------------------------------------------------------------------------


def run_theta_sweep() -> list[dict[str, Any]]:
    print()
    print("=" * 78)
    print(f"Part 2 -- theta_h sweep, untruncated (n={THETA_N}, {THETA_STEPS} steps)")
    print("=" * 78)

    n, cut = THETA_N, THETA_N // 2
    print(
        f"1D chain n={n}, seed Z_{cut}, cut [0,{cut}) | [{cut},{n}) crossing "
        f"{cut_bonds(chain_edges(n), cut)} lattice bond(s); policy=None (exact)"
    )

    rows = [
        {"theta_h": theta_h, **all_diagnostics(evolve(n, THETA_STEPS, theta_h), cut)}
        for theta_h in THETA_GRID
    ]

    print(f"{'theta_h':>9} {'terms':>7} {'S_2':>9} {'L':>9} {'S_op':>9} {'S_op^(2)':>9}")
    for row in rows:
        print(
            f"{row['theta_h']:>9.5f} {row['terms']:>7} {row['pauli_renyi2']:>9.5f} "
            f"{row['pauli_linear']:>9.5f} {row['op_entanglement']:>9.5f} "
            f"{row['op_entanglement_renyi2']:>9.5f}"
        )

    for row in rows:
        # Untruncated unitary evolution is an orthogonal rotation of the
        # coefficient vector, so no spectral weight can be lost.
        assert abs(row["hs_weight"] - 1.0) < 1e-9, (
            f"untruncated evolution must preserve sum|c|^2 = 1, got "
            f"{row['hs_weight']!r} at theta_h={row['theta_h']}"
        )
        # Rényi entropies are non-increasing in alpha, on both spectra.
        assert row["pauli_renyi2"] <= row["pauli_shannon"] + 1e-9
        assert row["op_entanglement_renyi2"] <= row["op_entanglement"] + 1e-9
        # S_2 = -ln(1 - L) by construction.
        assert abs(row["pauli_renyi2"] + math.log1p(-row["pauli_linear"])) < 1e-9

    assert rows[0]["pauli_renyi2"] == 0.0, "theta_h=0 must leave the seed untouched"
    assert rows[-1]["pauli_renyi2"] <= CLIFFORD_DUST_BOUND
    interior = rows[1:-1]
    assert min(r["pauli_renyi2"] for r in interior) > 0.0, (
        "every non-Clifford kick angle must spread the Pauli spectrum"
    )
    peak = max(interior, key=lambda r: r["pauli_renyi2"])
    print(
        f"  both diagnostics vanish at the Clifford endpoints; S_2 peaks at "
        f"theta_h={peak['theta_h']:.5f} with S_2={peak['pauli_renyi2']:.5f} nats "
        f"= {math.exp(peak['pauli_renyi2']):.0f} effective Pauli strings out of "
        f"{peak['terms']} stored"
    )

    _write_csv(OUT_DIR / "theta_sweep.csv", rows)
    print(f"wrote {OUT_DIR / 'theta_sweep.csv'}")
    _plot_sweep(
        rows,
        x_key="theta_h",
        xlabel="transverse-field kick angle $\\theta_h$ (rad)",
        title=f"kicked Ising chain, n={THETA_N}, {THETA_STEPS} Trotter steps (exact)",
        save_path=OUT_DIR / "theta_sweep.svg",
        clifford_marks=(0.0, math.pi / 2),
    )
    print(f"wrote {OUT_DIR / 'theta_sweep.svg'}")
    return rows


# --------------------------------------------------------------------------
# Part 3 -- depth sweep (exact where affordable, plus truncated)
# --------------------------------------------------------------------------


def run_depth_sweep() -> list[dict[str, Any]]:
    print()
    print("=" * 78)
    print(f"Part 3 -- depth sweep (n={DEPTH_N}, theta_h={DEPTH_THETA_H})")
    print("=" * 78)

    n, cut = DEPTH_N, DEPTH_N // 2
    print(
        f"1D chain n={n}, seed Z_{cut}, cut crossing "
        f"{cut_bonds(chain_edges(n), cut)} lattice bond(s); truncated curve at "
        f"min_abs_coeff={DEPTH_EPS:.0e}, exact through depth {DEPTH_EXACT_MAX}"
    )

    rows: list[dict[str, Any]] = []
    for steps in DEPTH_GRID:
        row: dict[str, Any] = {"trotter_steps": steps}
        truncated = evolve(n, steps, DEPTH_THETA_H, policy=truncation.coeff(DEPTH_EPS))
        row.update({f"trunc_{k}": v for k, v in all_diagnostics(truncated, cut).items()})
        if steps <= DEPTH_EXACT_MAX:
            exact = evolve(n, steps, DEPTH_THETA_H)
            row.update({f"exact_{k}": v for k, v in all_diagnostics(exact, cut).items()})
        rows.append(row)

    print(
        f"{'depth':>6} {'exact T':>9} {'exact S_2':>10} {'exact S_op':>11} "
        f"{'trunc T':>9} {'trunc S_2':>10} {'trunc S_op':>11} "
        f"{'kept |c|^2':>13} {'Schmidt':>12}"
    )
    for row in rows:
        print(
            f"{row['trotter_steps']:>6} "
            f"{_cell(row, 'exact_terms', '{:d}'):>9} "
            f"{_cell(row, 'exact_pauli_renyi2', '{:.5f}'):>10} "
            f"{_cell(row, 'exact_op_entanglement', '{:.5f}'):>11} "
            f"{row['trunc_terms']:>9} {row['trunc_pauli_renyi2']:>10.5f} "
            f"{row['trunc_op_entanglement']:>11.5f} {row['trunc_hs_weight']:>13.10f} "
            f"{_shape(row, 'trunc_'):>12}"
        )

    # Where both are available, truncation at 1e-6 must not have moved either
    # diagnostic much: that is the whole claim the convergence panels support.
    for row in rows:
        if "exact_pauli_renyi2" not in row:
            continue
        for key in ("pauli_renyi2", "op_entanglement"):
            gap = abs(row[f"exact_{key}"] - row[f"trunc_{key}"])
            assert gap < 1e-3, (
                f"depth {row['trotter_steps']}: {key} moved by {gap:.3e} under "
                f"min_abs_coeff={DEPTH_EPS:.0e}, which is not a converged truncation"
            )

    _write_csv(OUT_DIR / "depth_sweep.csv", rows)
    print(f"wrote {OUT_DIR / 'depth_sweep.csv'}")
    _plot_sweep(
        rows,
        x_key="trotter_steps",
        xlabel="Trotter steps",
        title=(
            f"kicked Ising chain, n={DEPTH_N}, $\\theta_h$={DEPTH_THETA_H} "
            f"(exact vs min_abs_coeff={DEPTH_EPS:.0e})"
        ),
        save_path=OUT_DIR / "depth_sweep.svg",
        prefixes=("exact_", "trunc_"),
    )
    print(f"wrote {OUT_DIR / 'depth_sweep.svg'}")
    return rows


def _cell(row: dict[str, Any], key: str, fmt: str) -> str:
    return fmt.format(row[key]) if key in row else "-"


def _shape(row: dict[str, Any], prefix: str = "") -> str:
    """`"rows x cols"` of the operator Schmidt matrix, for the printed tables."""
    return f"{row[f'{prefix}schmidt_rows']}x{row[f'{prefix}schmidt_cols']}"


# --------------------------------------------------------------------------
# Part 4 -- truncation-convergence panels
# --------------------------------------------------------------------------


def run_convergence_panels() -> dict[int, list[report.RunRecord]]:
    print()
    print("=" * 78)
    print("Part 4 -- truncation-convergence panels (plan §7 rule 4)")
    print("=" * 78)

    n, cut = DEPTH_N, DEPTH_N // 2
    provenance = report.collect_provenance(repo_root=_REPO_ROOT)
    version = provenance.library_versions.get("paulistrings", "unknown")
    records: dict[int, list[report.RunRecord]] = {}
    references: dict[int, dict[str, float]] = {}

    for steps in PANEL_DEPTHS:
        if steps <= DEPTH_EXACT_MAX:
            references[steps] = all_diagnostics(evolve(n, steps, DEPTH_THETA_H), cut)

        def build_run(spec: harness.TruncationSpec, steps: int = steps) -> report.RunRecord:
            start = time.perf_counter()
            evolved = evolve(n, steps, DEPTH_THETA_H, policy=spec.policy())
            elapsed = time.perf_counter() - start
            diagnostics = all_diagnostics(evolved, cut)
            return report.RunRecord(
                engine="paulistrings",
                engine_version=version,
                n_qubits=n,
                direction="heisenberg",
                truncation=spec.as_dict(),
                propagation_time_s=elapsed,
                final_terms=len(evolved),
                provenance=provenance,
                # `expectation_value` is `report.plot_convergence_panel`'s y
                # axis, and this showcase converges a *diagnostic* rather than
                # an expectation value; `extra["quantity"]` says which, so the
                # record stays self-describing. Set per diagnostic below.
                expectation_value=None,
                extra={"trotter_steps": steps, **diagnostics},
            )

        grid = [harness.TruncationSpec(min_abs_coeff=eps) for eps in EPS_GRID]
        records[steps] = harness.convergence_sweep(build_run, grid)

        reference = references.get(steps)
        label = "exact reference" if reference else "no exact reference (self-converged)"
        print(f"depth {steps} ({label}):")
        print(
            f"    {'min_abs_coeff':>13} {'terms':>8} {'kept |c|^2':>13} "
            f"{'S_2':>9} {'S_op':>9} {'Schmidt':>12}"
        )
        for record in records[steps]:
            extra = record.extra
            print(
                f"    {record.truncation['min_abs_coeff']:>13.0e} {record.final_terms:>8} "
                f"{extra['hs_weight']:>13.10f} {extra['pauli_renyi2']:>9.5f} "
                f"{extra['op_entanglement']:>9.5f} {_shape(extra):>12}"
            )
        if reference:
            print(
                f"    {'exact':>13} {reference['terms']:>8} "
                f"{reference['hs_weight']:>13.10f} {reference['pauli_renyi2']:>9.5f} "
                f"{reference['op_entanglement']:>9.5f}"
            )
            for key in ("pauli_renyi2", "op_entanglement"):
                gaps = [abs(r.extra[key] - reference[key]) for r in records[steps]]
                assert gaps[-1] <= gaps[0] + 1e-12, (
                    f"depth {steps}: {key} at the tightest cutoff ({EPS_GRID[-1]:.0e}) "
                    f"is further from exact ({gaps[-1]:.3e}) than at the loosest "
                    f"({gaps[0]:.3e}) -- that is not convergence"
                )
                print(f"    |gap| vs exact, {key}: " + ", ".join(f"{g:.2e}" for g in gaps))
        else:
            for key in ("pauli_renyi2", "op_entanglement"):
                values = [r.extra[key] for r in records[steps]]
                drifts = [abs(b - a) for a, b in zip(values, values[1:])]
                assert drifts[-1] <= drifts[0], (
                    f"depth {steps}: {key} is still moving faster at the tight end of "
                    f"the grid ({drifts[-1]:.3e}) than at the loose end "
                    f"({drifts[0]:.3e}) -- no self-convergence to claim"
                )
                print(
                    f"    successive drift, {key}: " + ", ".join(f"{d:.2e}" for d in drifts)
                )

    _plot_convergence_panels(records, references, OUT_DIR / "convergence_panel.svg")
    print(f"wrote {OUT_DIR / 'convergence_panel.svg'}")
    return records


# --------------------------------------------------------------------------
# Plotting -- styled after `common/report.py`'s helpers
# --------------------------------------------------------------------------

#: Same categorical slots `report.py` draws from, so the showcase figures sit
#: next to the suite's own without a palette clash.
_COLORS = ("#2a78d6", "#d97706", "#0f9d58", "#a855c7")
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


def _plot_sweep(
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    xlabel: str,
    title: str,
    save_path: Path,
    prefixes: Sequence[str] = ("",),
    clifford_marks: Sequence[float] = (),
) -> None:
    """Two side-by-side panels -- Pauli spectrum, operator entanglement -- with
    one curve per `(prefix, diagnostic)` pair that the rows actually carry.

    Never a dual y-axis (the two families are different quantities in the same
    unit, nats, but on different objects); dashed lines mark the exact curve's
    truncated counterpart so the two are distinguishable in print.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for curve_index, (key, panel, label) in enumerate(DIAGNOSTIC_CURVES):
        for style_index, prefix in enumerate(prefixes):
            column = f"{prefix}{key}"
            points = sorted((r[x_key], r[column]) for r in rows if column in r)
            if not points:
                continue
            xs, ys = zip(*points)
            suffix = f"  [{prefix.rstrip('_')}]" if prefix else ""
            axes[panel].plot(
                xs,
                ys,
                marker="o" if style_index == 0 else "s",
                markersize=4,
                linewidth=1.5,
                linestyle="-" if style_index == 0 else "--",
                color=_COLORS[curve_index % len(_COLORS)],
                alpha=1.0 if style_index == 0 else 0.75,
                label=f"{label}{suffix}",
            )
    for mark in clifford_marks:
        for ax in axes:
            ax.axvline(mark, color=_MUTED, linewidth=1.0, linestyle=":")
    axes[0].set_ylabel("Pauli-spectrum entropy (nats)")
    axes[1].set_ylabel("operator entanglement entropy (nats)")
    for ax in axes:
        ax.set_xlabel(xlabel)
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _plot_convergence_panels(
    records: dict[int, list[report.RunRecord]],
    references: dict[int, dict[str, float]],
    save_path: Path,
) -> None:
    """A row of panels per depth, a column per diagnostic, drawn by
    `report.plot_convergence_panel`.

    That helper's y axis is `RunRecord.expectation_value`, so each diagnostic
    is copied into that field on a shallow clone just before plotting and the
    y label is overridden -- reusing the shared helper (and its reference-line
    handling) rather than reimplementing a log-x convergence plot here.
    """
    import matplotlib.pyplot as plt
    from dataclasses import replace

    depths = sorted(records)
    columns = (
        ("pauli_renyi2", "$S_2$, Pauli spectrum (nats)"),
        ("op_entanglement", "$S_{op}$, operator entanglement (nats)"),
    )
    fig, axes = plt.subplots(
        len(depths), len(columns), figsize=(9.5, 3.6 * len(depths)), squeeze=False
    )
    for row_index, depth in enumerate(depths):
        for col_index, (key, ylabel) in enumerate(columns):
            ax = axes[row_index][col_index]
            projected = [
                replace(record, expectation_value=record.extra[key])
                for record in records[depth]
            ]
            reference = references.get(depth, {}).get(key)
            report.plot_convergence_panel(
                projected, truncation_key="min_abs_coeff", reference_value=reference, ax=ax
            )
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"depth {depth}"
                + ("" if reference is not None else "  (beyond exact reach)"),
                fontsize=10,
            )
    fig.suptitle(
        f"truncation convergence, kicked Ising chain n={DEPTH_N}, "
        f"$\\theta_h$={DEPTH_THETA_H}",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_exact_cross_check()
    run_theta_sweep()
    run_depth_sweep()
    run_convergence_panels()
    print()
    print("done.")


if __name__ == "__main__":
    main()

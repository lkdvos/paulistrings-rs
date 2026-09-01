"""Showcase B7 -- stabilizer preparation in stim, estimation by Pauli propagation.

Handoff item B7; adapted spec in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part B (decision
D13: the generator-membership contraction is an *expectation* feature, not
stabilizer simulation, so it does not conflict with the `lib.rs` non-goal).
Narrative: `README.md` next to this file. Shared helpers:
`stabilizer_prep.py`. CI-safe gate:
`python/paulistrings/tests/test_showcase_b7.py`.

The capability neither tool has alone
-------------------------------------
A Clifford circuit of any depth is free for **stim** and useless for Pauli
propagation to reproduce; a non-Clifford circuit is the natural habitat of
**Pauli propagation** and outside stim's reach entirely. Chaining them across
the observable is what this showcase does:

1. prepare a stabilizer state `|psi>` with an arbitrarily deep Clifford circuit
   in stim (here a 2D cluster state on a 6x6 grid, 36 qubits);
2. read its `n` signed stabilizer generators out of the tableau
   (`interop.stabilizers_from_stim`);
3. back-propagate the observable through a **non-Clifford** tail
   (`direction="heisenberg"`, kicked-Ising `X`/`ZZ` rotations at a generic kick
   angle) with truncation;
4. contract the evolved Pauli sum against the generators
   (`PauliSum.expectation_stabilizer`), at `O(m·n²/64)` for `m` terms -- never a
   `2^n` expansion. At n=36 that expansion would be 6.9e10 amplitudes, 1.0 TiB.

Five parts, run in order by `main()`:

* **Part 1** (`run_pipeline`) -- the pipeline itself. 6x6 cluster state; two
  sites' own stabilizers `K_q = X_q prod_{n in N(q)} Z_n` as the observables
  (the centre, degree 4, and a corner, degree 2); a two-Trotter-step tail;
  `theta_h` swept from `0` to `pi/2` in 17 points. Three independent checks:
  the generators against the closed-form cluster stabilizers, the two Clifford
  endpoints against `oracles.stim_clifford_exact` on the *composed*
  preparation-plus-tail circuit, and -- the strong one -- **every** point of
  the sweep against the derived closed form `<K_q> = cos^deg(q)(theta_h)` (see
  `CLOSED_FORM_TOL`). The `|0...0>` special case is checked on one more evolved
  sum. Writes `theta_sweep.csv` / `.svg`.
* **Part 2** (`run_preparation_depth`) -- preparation depth is free. The same
  cluster state prepared by circuits from 96 to 50 236 Clifford gates (via
  provably-identity padding): byte-identical generators, identical estimate,
  stim's readout cost growing from 0.5 ms to ~2 ms, and the Pauli-propagation
  side literally unchanged (one evolved sum, contracted against each). Then
  unstructured deep Clifford preparations, which is where the honest caveat
  lives. Writes `prep_depth.csv`.
* **Part 3** (`run_contraction_scaling`) -- the `O(m·n²/64)` cost law measured:
  `n` from 64 to 1024 (every monomorphized width `W in {1,2,4,8,16}`) at fixed
  `m`, and `m` from 1e3 to 1e5 at fixed `n`. Writes `scaling.csv` / `.svg`.
* **Part 4** (`run_convergence_panels`) -- the truncation-convergence panel
  plan §7 rule 4 requires, at tail depths 4 and 5 where the evolved sum is
  genuinely large (up to 1.7e8 terms). Writes `convergence_panel.svg`.
* **Part 5** (`run_validation`) -- the dense cross-check at `n <= 12`: the
  composed circuit run on a `2^n` state vector with numpy alone
  (`stabilizer_prep.dense_expectation`), the projector route
  `Pi = prod (I + s_i G_i)/2` at n=8, and qiskit Aer where installed. Writes
  `validation_b7.json`.

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py
    RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py --quick

Full run: 116 s and 10.2 GiB peak RSS on the reference host, almost all of it
the single deepest convergence point (167 million terms at
`min_abs_coeff = 1e-6`, tail depth 5). `--quick` stops that grid one point
earlier: 40 s and 1.5 GiB, every other number identical.

Timings here are wall-clock on a shared workstation and are *not* a performance
claim -- the suite's noise floor is +-5-8% single-threaded (CLAUDE.md
§Performance discipline), and every time used quantitatively (Part 3) is a
minimum over repeats.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

# Before numpy: pin BLAS/LAPACK and Rayon to one thread. The Rayon pool is
# built once, at the first propagate, so a `setdefault` here reaches it -- but
# exporting the variable before the interpreter starts is the reliable spelling
# (see `examples/README.md`), which is why the assert in `main()` is a hard one.
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

from paulistrings import PauliSum, interop, truncation  # noqa: E402

import stabilizer_prep as sp  # noqa: E402
from common import circuits, harness, oracles, report  # noqa: E402

OUT_DIR = _HERE

# --------------------------------------------------------------------------
# Part 1 -- the pipeline
# --------------------------------------------------------------------------

#: 6x6 open square lattice: 36 qubits, inside the W=1 band (<= 64), and big
#: enough that the dense alternative is out of reach (2^36 amplitudes = 1.0 TiB)
#: while every run here is seconds. The cluster state on a 2D lattice is a
#: universal resource state and is not a product state in any local basis, so
#: `expectation(state=...)` cannot express it -- which is the point.
GRID_ROWS = 6
GRID_COLS = 6

#: Two Trotter steps of the kicked-Ising tail. Chosen because at this depth the
#: evolved sum's *physical* content is 16 terms, so Part 1's sweep is exact and
#: owes no convergence panel (Part 4 supplies the deep, truncated case).
PIPELINE_STEPS = 2

#: 17 kick angles from 0 to pi/2 inclusive. Both endpoints are Clifford points
#: of the kicked-Ising circuit at the fixed `theta_zz = -pi/2`
#: (`circuits.heavy_hex_kicked_ising`), so both are cross-checkable against
#: stim on the composed circuit; everything in between is genuinely
#: non-Clifford and reachable by no stabilizer method.
THETA_POINTS = 17

#: "Dust-only" truncation for the exact runs. At `theta_zz = -pi/2` the ZZ
#: rotation's `cos(theta_zz/2 * 2) = cos(-pi/2)` branch is 6.1e-17 rather than
#: an exact zero, so an untruncated run accumulates floating-point dust without
#: bound -- the correction recorded in plan §9(b). `min_abs_coeff = 1e-12` drops
#: exactly that dust and nothing physical: Part 1 verifies it against the fully
#: untruncated run at every sweep point, and the agreement is ~1e-15.
DUST_CUTOFF = 1e-12

#: Cross-check tolerances. The Clifford-point stim checks are exact integers, so
#: they get the tight bound; the dense and dust comparisons are floating point.
EXACT_TOL = 1e-12
DENSE_TOL = 1e-10

#: The **closed-form oracle** for Part 1, valid at exactly `PIPELINE_STEPS = 2`
#: kicked-Ising steps (order `x-then-zz`, `theta_zz = -pi/2`) and for a cluster
#: stabilizer `K_q` on a site of *even* degree:
#:
#:     <K_q> = cos(theta_h) ** deg(q)
#:
#: Derivation. Write the two-step circuit as `U = ZZ_2 X_2 ZZ_1 X_1` (rightmost
#: acts first) and push `K_q = X_q prod_{n in N(q)} Z_n` through `U^dagger . U`
#: from the left:
#:
#: 1. `ZZ_2`: at the Clifford angle each `Z_qZ_n` rotation is a Clifford, and
#:    `Z_qZ_n` anticommutes with the running operator exactly on the `deg(q)`
#:    edges incident to `q`. Composing those, the operator picks up
#:    `prod_n (Z_q Z_n) = Z_q^deg(q) prod_n Z_n`, which for even degree is
#:    `prod_n Z_n` -- cancelling the `Z`s and leaving `+-X_q`, weight one.
#: 2. `X_2`: `X_q` commutes with every `X` rotation, so nothing happens. (For
#:    *odd* degree step 1 leaves `+-Y_q` instead, which does not commute, and
#:    this layer contributes one extra factor of `cos` -- hence the even-degree
#:    restriction.)
#: 3. `ZZ_1`: the same Clifford conjugation runs backwards, `+-X_q -> +-K_q`.
#: 4. `X_1`: each of the `deg(q)` `Z_n` factors anticommutes with its own `X_n`
#:    rotation and splits, `Z_n -> cos(theta_h) Z_n - sin(theta_h)(...)Y_n`, so
#:    the sum has `2^deg(q)` terms and the all-`cos` branch is `cos^deg(q) K_q`.
#:
#: `K_q` is a group element (expectation `+1`); every other branch carries a `Y`
#: on at least one neighbour and is not, so it contracts to `0`. Hence the
#: closed form -- and hence also `<K_q> = 0` after a *single* step, where the
#: evolution stops at `+-X_q`. Verified to 1.1e-16 at every even-degree site of
#: the 6x6, 4x5 and 3x4 lattices, and pinned by the CI gate.
CLOSED_FORM_TOL = 1e-12

# --------------------------------------------------------------------------
# Part 2 -- preparation depth
# --------------------------------------------------------------------------

#: Identity-padding rounds. Each round is one of `H H`, `S S S S`,
#: `CNOT CNOT`, `CZ CZ` -- so the prepared state, and therefore every generator
#: and every estimate, is invariant by construction while the circuit grows.
PADDING_ROUNDS = (0, 200, 2000, 20000)

#: Depths for the unstructured random Clifford preparations.
RANDOM_PREP_DEPTHS = (2, 10, 50, 200)

#: Seed for every random preparation in Part 2 (recorded in the provenance).
PREP_SEED = 0xB7

#: The depth of the random preparation whose *own* stabilizer generator is used
#: as an observable, and the tail depth / cutoff for that run.
OWN_STABILIZER_DEPTH = 50
OWN_STABILIZER_STEPS = 2
OWN_STABILIZER_CUTOFF = 1e-4

# --------------------------------------------------------------------------
# Part 3 -- contraction scaling
# --------------------------------------------------------------------------

#: Qubit counts, one per monomorphized width `W in {1,2,4,8,16}` (64-1024
#: qubits; `crates/paulistrings-py` dispatches once outside any hot loop).
SCALING_QUBITS = (64, 128, 256, 512, 1024)

#: Terms held fixed while `n` varies. 20 000 random full-support Pauli strings:
#: random rather than structured because a random Pauli hits about half the
#: group's pivots, which is the generic per-term cost the bound describes.
SCALING_TERMS = 20_000

#: Term counts swept at fixed `n` for the linearity panel.
SCALING_TERM_GRID = (1_000, 5_000, 20_000, 100_000)
SCALING_LINEARITY_QUBITS = 256

#: Repeats per point; the reported number is the minimum, which is the least
#: load-contaminated estimator available on a shared host.
SCALING_REPEATS = 5

#: Seed for the random Pauli strings.
SCALING_SEED = 11

# --------------------------------------------------------------------------
# Part 4 -- convergence panels
# --------------------------------------------------------------------------

#: Generic (non-Clifford) kick angle for the deep runs.
DEEP_THETA_H = 0.6

#: `(tail_steps, coefficient cutoffs)`. Depth 4 converges inside the grid (the
#: last two points agree to 12 digits), so its reference line is that plateau;
#: depth 5 is still moving at 1e-5 and is carried one point further, which is
#: the 167-million-term, 10 GiB run `--quick` drops.
CONVERGENCE_GRIDS = (
    (4, (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)),
    (5, (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)),
)
QUICK_CONVERGENCE_GRIDS = (
    (4, (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)),
    (5, (1e-2, 1e-3, 1e-4, 1e-5)),
)

# --------------------------------------------------------------------------
# Part 5 -- dense validation
# --------------------------------------------------------------------------

#: `(rows, cols, tail_steps, theta_h)` for the dense cross-checks. 3x4 = 12
#: qubits is the largest the state-vector route is run at here (4096
#: amplitudes); 2x4 = 8 also gets the `2^n x 2^n` projector route.
VALIDATION_CASES = (
    (3, 4, 2, 0.6),
    (3, 4, 3, 0.6),
    (3, 4, 2, 1.0),
    (2, 4, 2, 0.6),
    (2, 4, 3, 0.35),
)

#: The projector route (a `2^n x 2^n` matrix) runs only at or below this size.
VALIDATION_PROJECTOR_MAX = 8


# ==========================================================================
# Small shared helpers
# ==========================================================================


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV to {path}")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def _lattice(rows: int, cols: int):
    """`(num_qubits, edges, adjacency, centre_qubit)` for a `rows x cols` grid."""
    n = rows * cols
    edges = sp.grid_edges(rows, cols)
    adjacency = sp.grid_adjacency(n, edges)
    centre = (rows // 2) * cols + cols // 2
    return n, edges, adjacency, centre


def _centre_stabilizer_label(n: int, adjacency: dict[int, list[int]], centre: int) -> str:
    """`K_c = X_c prod_{q in N(c)} Z_q`, one site's own cluster stabilizer.

    The same formula `stabilizer_prep.cluster_state_stabilizers` uses for the
    whole set; spelled separately here because the showcase wants one named
    site's generator as an observable, not the list.
    """
    support = {centre: "X"}
    for neighbour in adjacency[centre]:
        support[neighbour] = "Z"
    return sp.pauli_label(support, n)


def _stim_gate_count(circuit) -> int:
    """Single- and two-qubit gate count of a stim circuit (annotations excluded).

    `len(stim_circuit)` counts *instructions*, and one instruction carries many
    targets (`H 0 1 2 ... 35` is one), so it understates a preparation's size by
    an order of magnitude. This counts targets, halving them for two-qubit
    gates.
    """
    import stim

    total = 0
    for instruction in circuit.flattened():
        data = stim.gate_data(instruction.name)
        if not data.is_unitary:
            continue
        targets = len(instruction.targets_copy())
        total += targets // 2 if data.is_two_qubit_gate else targets
    return total


def _tail(n: int, edges, steps: int, theta_h: float):
    """The non-Clifford tail: `steps` kicked-Ising Trotter steps on `edges`.

    `circuits.heavy_hex_kicked_ising` with an explicit edge list -- the builder
    is topology-agnostic, and one gate per channel (the suite's construction
    rule, plan D10) means the layer index *is* the channel index.
    """
    return circuits.heavy_hex_kicked_ising(n, steps, theta_h, edges=edges)


def _record(
    circuit,
    observable: PauliSum,
    generators: Sequence[str],
    policy: Any,
    *,
    warmup: bool,
    oracle_value: float | None = None,
    extra: dict[str, Any] | None = None,
) -> report.RunRecord:
    """One `RunRecord` for "propagate the tail, contract against `generators`".

    `contract=` rather than `state=`: the contraction target is a stabilizer
    state, which `PauliSum.expectation(state=...)` cannot express. Everything
    else -- warm timing, separated propagation/contraction times, peak and final
    term counts, peak RSS, thread-pin assertion -- is the shared harness.
    """
    return harness.run_propagation(
        circuit,
        observable,
        policy,
        direction="heisenberg",
        contract=lambda evolved: evolved.expectation_stabilizer(list(generators)),
        warmup=warmup,
        oracle_value=oracle_value,
        threads=1,
        seeds={"preparation": PREP_SEED},
        library_versions=_library_versions(),
        extra={**(extra or {}), "warm_timing": warmup},
    )


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import stim

        versions["stim"] = stim.__version__
    except ImportError:  # pragma: no cover - stim is required by this script
        pass
    versions["numpy"] = np.__version__
    return versions


# ==========================================================================
# Part 1 -- the pipeline
# ==========================================================================


def run_pipeline() -> list[report.RunRecord]:
    """Prepare, read out generators, propagate a non-Clifford tail, contract."""
    print()
    print("=" * 78)
    print("Part 1 -- stabilizer prep (stim) -> non-Clifford tail (PP) -> estimate")
    print("=" * 78)

    n, edges, adjacency, centre = _lattice(GRID_ROWS, GRID_COLS)
    preparation = sp.cluster_prep_stim(n, edges)
    t0 = time.perf_counter()
    generators = interop.stabilizers_from_stim(preparation, num_qubits=n)
    readout_s = time.perf_counter() - t0
    prep_circuit, prep_observable = interop.circuit_from_stim(preparation)
    if prep_observable is not None:
        raise AssertionError(
            "the preparation circuit is not supposed to carry an OBSERVABLE_INCLUDE"
        )

    print(
        f"lattice        : {GRID_ROWS}x{GRID_COLS} open square grid, n={n}, "
        f"{len(edges)} edges"
    )
    print(
        f"preparation    : {_stim_gate_count(preparation)} Clifford gates in "
        f"{len(preparation)} stim instructions ({len(prep_circuit)} channels once "
        "imported)"
    )
    print(f"generators     : {len(generators)} signed strings, read out in {readout_s * 1e3:.2f} ms")
    print(f"  G_0          = {generators[0]}")
    print(
        f"dense would be : 2^{n} = {2**n:.3e} amplitudes "
        f"({2**n * 16 / 2**40:.1f} TiB) -- never built"
    )

    # --- cross-check 1: the generators, against the closed-form cluster
    # stabilizers derived from the edge list.
    hand_derived = sp.cluster_state_stabilizers(n, edges)
    worst = 0.0
    for spec in hand_derived:
        value = PauliSum.from_strings({spec[1:]: 1.0}, num_qubits=n).expectation_stabilizer(
            generators
        )
        worst = max(worst, abs(value - 1.0))
    print(
        f"generator check: all {len(hand_derived)} closed-form K_q are +1 group elements "
        f"of stim's readout (worst |dev| = {worst:.3e})"
    )
    if worst > EXACT_TOL:
        raise AssertionError(f"generator cross-check failed: {worst:.3e} > {EXACT_TOL}")

    # Two observables, both cluster stabilizers, on sites of different degree:
    # the closed form (see CLOSED_FORM_TOL) then predicts two different decay
    # exponents, which is a much sharper check than one curve.
    corner = 0
    sites = {
        "K_centre": (centre, len(adjacency[centre])),
        "K_corner": (corner, len(adjacency[corner])),
    }
    observables = {
        name: PauliSum.from_strings(
            {_centre_stabilizer_label(n, adjacency, site): 1.0}, num_qubits=n
        )
        for name, (site, _) in sites.items()
    }
    local_label = sp.pauli_label({centre: "Z"}, n)
    local_observable = PauliSum.from_strings({local_label: 1.0}, num_qubits=n)
    for name, (site, degree) in sites.items():
        print(
            f"observable     : {name} = K_{site} (degree {degree}), "
            f"<{name}> = "
            f"{observables[name].expectation_stabilizer(generators).real:+.12f} "
            "at tail depth 0"
        )

    thetas = [math.pi / 2 * k / (THETA_POINTS - 1) for k in range(THETA_POINTS)]
    rows: list[dict[str, Any]] = []
    records: list[report.RunRecord] = []
    print()
    print(
        f"{'theta_h':>9} {'kept':>6} {'untrunc':>9} {'<K_centre>':>15} "
        f"{'<K_corner>':>15} {'closed form':>12} {'dust':>9} {'stim':>9}"
    )
    for theta in thetas:
        tail = _tail(n, edges, PIPELINE_STEPS, theta)
        values: dict[str, float] = {}
        closed_form_gap = 0.0
        dust_gap = 0.0
        untruncated_terms = 0
        kept_terms = 0

        for name, (site, degree) in sites.items():
            observable = observables[name]
            analytic = math.cos(theta) ** degree
            record = _record(
                tail,
                observable,
                generators,
                {"min_abs_coeff": DUST_CUTOFF},
                warmup=(name == "K_centre"),
                oracle_value=analytic,
                extra={
                    "theta_h": theta,
                    "tail_steps": PIPELINE_STEPS,
                    "observable": name,
                    "site": site,
                    "site_degree": degree,
                    "closed_form_value": analytic,
                    "preparation": "cluster_2d",
                    "generator_readout_s": readout_s,
                },
            )
            records.append(record)
            values[name] = record.expectation_value
            closed_form_gap = max(closed_form_gap, record.absolute_error)
            kept_terms = max(kept_terms, record.final_terms)

            # The fully untruncated run: same answer, plus ~6e5 dust terms.
            exact = observable.propagate(tail, None, direction="heisenberg")
            dust_gap = max(
                dust_gap,
                abs(exact.expectation_stabilizer(generators).real - values[name]),
            )
            untruncated_terms = max(untruncated_terms, len(exact))
            del exact

        stim_cell = ""
        stim_value = None
        clifford = abs(theta) < 1e-12 or abs(theta - math.pi / 2) < 1e-9
        if clifford:
            stim_value = oracles.stim_clifford_exact(
                prep_circuit + tail, observables["K_centre"]
            ).real
            gap = abs(stim_value - values["K_centre"])
            stim_cell = f"{gap:.1e}"
            if gap > EXACT_TOL:
                raise AssertionError(
                    f"Clifford-point cross-check failed at theta_h={theta}: "
                    f"PP {values['K_centre']!r} vs stim {stim_value!r}"
                )

        if dust_gap > DENSE_TOL:
            raise AssertionError(
                f"the dust cutoff changed the answer at theta_h={theta}: "
                f"{dust_gap:.3e} > {DENSE_TOL}"
            )
        if closed_form_gap > CLOSED_FORM_TOL:
            raise AssertionError(
                f"the closed form disagrees at theta_h={theta}: "
                f"{closed_form_gap:.3e} > {CLOSED_FORM_TOL}"
            )

        rows.append(
            {
                "theta_h": theta,
                "terms_kept": kept_terms,
                "terms_untruncated": untruncated_terms,
                "centre_expectation": values["K_centre"],
                "corner_expectation": values["K_corner"],
                "centre_closed_form": math.cos(theta) ** sites["K_centre"][1],
                "corner_closed_form": math.cos(theta) ** sites["K_corner"][1],
                "closed_form_gap": closed_form_gap,
                "dust_gap": dust_gap,
                "clifford_point": clifford,
                "stim_exact": stim_value,
            }
        )
        print(
            f"{theta:9.5f} {kept_terms:6d} {untruncated_terms:9d} "
            f"{values['K_centre']:+15.12f} {values['K_corner']:+15.12f} "
            f"{closed_form_gap:12.2e} {dust_gap:9.1e} {stim_cell:>9}"
        )

    print()
    print(
        f"Every point above matches the closed form cos^deg(q)(theta_h) to "
        f"{max(r['closed_form_gap'] for r in rows):.1e} -- an analytic oracle for the\n"
        "whole sweep, not just the two Clifford endpoints (see CLOSED_FORM_TOL for the\n"
        f"derivation). The centre decays as cos^{sites['K_centre'][1]} and the corner "
        f"as cos^{sites['K_corner'][1]}, and both are exactly\n"
        "+1 at theta_h = 0 and 0 at pi/2."
    )

    # --- cross-check 2: the |0...0> special case, on one evolved sum.
    tail = _tail(n, edges, PIPELINE_STEPS, DEEP_THETA_H)
    evolved = local_observable.propagate(
        tail, truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    product_value = complex(evolved.expectation("z+"))
    generator_value = evolved.expectation_stabilizer(sp.single_z_generators(n))
    gap = abs(product_value - generator_value)
    print()
    print(
        f"|0...0> special case (Z_c, theta_h={DEEP_THETA_H}, {len(evolved)} terms): "
        f"expectation(state='z+') = {product_value.real:+.15f}, "
        f"expectation_stabilizer(+Z_q) = {generator_value.real:+.15f}, gap = {gap:.3e}"
    )
    if gap > EXACT_TOL:
        raise AssertionError(f"the |0...0> special case disagrees: {gap:.3e}")

    _write_csv(OUT_DIR / "theta_sweep.csv", rows)
    _plot_theta_sweep(
        rows,
        {name: degree for name, (_site, degree) in sites.items()},
        OUT_DIR / "theta_sweep.svg",
    )
    return records


def _plot_theta_sweep(
    rows: Sequence[dict[str, Any]],
    degrees: dict[str, int],
    save_path: Path,
) -> None:
    """Left: the two expectation curves. Right: kept vs. untruncated term count.

    Never a dual y axis: an expectation value and a term count are different
    measures, so they get their own panel (`report.py`'s plotting note).
    """
    import matplotlib.pyplot as plt

    fig, (ax_value, ax_terms) = plt.subplots(1, 2, figsize=(10, 4))
    thetas = [r["theta_h"] for r in rows]

    for index, (which, marker) in enumerate((("centre", "o"), ("corner", "s"))):
        degree = degrees[f"K_{which}"]
        ax_value.plot(
            thetas,
            [r[f"{which}_expectation"] for r in rows],
            marker=marker,
            markersize=4,
            linewidth=0,
            color=_COLORS[index],
            label=(
                r"$\langle K_q\rangle$, " + f"{which} site (degree {degree})"
            ),
        )
        ax_value.plot(
            thetas,
            [r[f"{which}_closed_form"] for r in rows],
            linewidth=1.2,
            linestyle="--",
            color=_COLORS[index],
            label=f"$\\cos^{degree}" + r"\theta_h$ (closed form)",
        )
    ax_value.set_ylabel("expectation in the 2D cluster state")

    ax_terms.plot(
        thetas,
        [r["terms_untruncated"] for r in rows],
        marker="o",
        markersize=4,
        linewidth=1.5,
        color=_COLORS[2],
        label="untruncated",
    )
    ax_terms.plot(
        thetas,
        [r["terms_kept"] for r in rows],
        marker="s",
        markersize=4,
        linewidth=1.5,
        linestyle="--",
        color=_COLORS[3],
        label=r"kept at $\epsilon = 10^{-12}$ (physical terms)",
    )
    ax_terms.set_yscale("log")
    ax_terms.set_ylabel("terms in the evolved sum")

    for ax in (ax_value, ax_terms):
        for mark in (0.0, math.pi / 2):
            ax.axvline(mark, color=_MUTED, linewidth=1.0, linestyle=":")
        ax.set_xlabel(r"kick angle $\theta_h$  (Clifford points dotted)")
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"B7: {GRID_ROWS}x{GRID_COLS} cluster state prepared in stim, "
        f"{PIPELINE_STEPS}-step non-Clifford tail propagated in the Heisenberg picture",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


# ==========================================================================
# Part 2 -- preparation depth is free
# ==========================================================================


def run_preparation_depth() -> None:
    """Grow the preparation's depth; the estimate and its cost do not move."""
    print()
    print("=" * 78)
    print("Part 2 -- preparation depth is free (and where that stops being useful)")
    print("=" * 78)

    n, edges, adjacency, centre = _lattice(GRID_ROWS, GRID_COLS)
    base = sp.cluster_prep_stim(n, edges)
    stabilizer_observable = PauliSum.from_strings(
        {_centre_stabilizer_label(n, adjacency, centre): 1.0}, num_qubits=n
    )
    tail = _tail(n, edges, PIPELINE_STEPS, DEEP_THETA_H)

    # One evolved sum, reused for every preparation below: the Heisenberg side
    # never sees the state, so there is nothing to recompute per preparation.
    evolved = stabilizer_observable.propagate(
        tail, truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    print(
        f"one evolved observable ({len(evolved)} terms) is contracted against every "
        "preparation below"
    )
    print()

    rows: list[dict[str, Any]] = []
    reference_generators: list[str] | None = None
    reference_value: float | None = None
    print(
        f"{'kind':>18} {'gates':>7} {'readout ms':>11} {'contract ms':>12} "
        f"{'same gens':>10} {'estimate':>16} {'mean |G|':>9}"
    )
    for rounds in PADDING_ROUNDS:
        padded = base + sp.identity_padding(
            n, rounds, np.random.default_rng(PREP_SEED)
        )
        generators, readout_s = _timed_readout(padded, n)
        value, contract_s = _timed_contraction(evolved, generators)
        if reference_generators is None:
            reference_generators, reference_value = generators, value
        identical = generators == reference_generators
        drift = abs(value - reference_value)
        gates = _stim_gate_count(padded)
        rows.append(
            {
                "kind": "cluster_2d+padding",
                "padding_rounds": rounds,
                "gates": gates,
                "instructions": len(padded),
                "readout_s": readout_s,
                "contraction_s": contract_s,
                "generators_identical": identical,
                "estimate": value,
                "estimate_drift": drift,
                "mean_generator_weight": _mean_weight(generators),
            }
        )
        kind = f"padding {rounds}"
        print(
            f"{kind:>18} {gates:7d} {readout_s * 1e3:11.2f} "
            f"{contract_s * 1e3:12.3f} {identical!s:>10} {value:+16.12f} "
            f"{_mean_weight(generators):9.1f}"
        )
        if not identical or drift > EXACT_TOL:
            raise AssertionError(
                f"identity padding changed the state: identical={identical}, "
                f"drift={drift:.3e} after {rounds} rounds"
            )

    print()
    for depth in RANDOM_PREP_DEPTHS:
        random_prep = sp.random_clifford_prep(
            n, depth, np.random.default_rng(PREP_SEED + depth)
        )
        generators, readout_s = _timed_readout(random_prep, n)
        value, contract_s = _timed_contraction(evolved, generators)
        gates = _stim_gate_count(random_prep)
        rows.append(
            {
                "kind": "random_clifford",
                "padding_rounds": None,
                "gates": gates,
                "instructions": len(random_prep),
                "readout_s": readout_s,
                "contraction_s": contract_s,
                "generators_identical": False,
                "estimate": value,
                "estimate_drift": None,
                "mean_generator_weight": _mean_weight(generators),
            }
        )
        kind = f"random depth {depth}"
        print(
            f"{kind:>18} {gates:7d} "
            f"{readout_s * 1e3:11.2f} {contract_s * 1e3:12.3f} {'-':>10} "
            f"{value:+16.12f} {_mean_weight(generators):9.1f}"
        )

    print()
    print(
        "The unstructured rows all read exactly 0. That is not a failure and not a\n"
        "truncation artefact: the contraction is a group-membership test, and a\n"
        "generic stabilizer state's group holds 2^n of the 4^n Pauli strings, so a\n"
        f"term lands in it with probability 2^-n = {2.0**-n:.1e} at n={n}. A deep\n"
        "*unstructured* preparation therefore annihilates a low-weight evolved\n"
        "operator term by term."
    )

    # ... so for such a state the informative observables are its own group
    # elements. One of them, propagated through the same kind of tail:
    random_prep = sp.random_clifford_prep(
        n, OWN_STABILIZER_DEPTH, np.random.default_rng(PREP_SEED + 1)
    )
    generators, readout_s = _timed_readout(random_prep, n)
    own = generators[0]
    own_label = own[1:]
    own_weight = sum(1 for ch in own_label if ch != "I")
    own_observable = PauliSum.from_strings({own_label: 1.0}, num_qubits=n)
    depth0 = own_observable.expectation_stabilizer(generators).real
    own_tail = _tail(n, edges, OWN_STABILIZER_STEPS, DEEP_THETA_H)
    own_evolved = own_observable.propagate(
        own_tail, truncation.coeff(OWN_STABILIZER_CUTOFF), direction="heisenberg"
    )
    own_value = own_evolved.expectation_stabilizer(generators).real
    print()
    print(
        f"Its own generator G_0 (weight {own_weight}, sign {own[0]}) as the observable:\n"
        f"  tail depth 0                      -> {depth0:+.12f}  (exactly the sign of G_0)\n"
        f"  tail depth {OWN_STABILIZER_STEPS}, eps={OWN_STABILIZER_CUTOFF:g}, "
        f"{len(own_evolved)} terms -> {own_value:+.12f}"
    )
    if abs(abs(depth0) - 1.0) > EXACT_TOL:
        raise AssertionError(f"a group element must read +-1, got {depth0!r}")
    rows.append(
        {
            "kind": "random_clifford_own_generator",
            "padding_rounds": None,
            "gates": _stim_gate_count(random_prep),
            "instructions": len(random_prep),
            "readout_s": readout_s,
            "contraction_s": None,
            "generators_identical": False,
            "estimate": own_value,
            "estimate_drift": None,
            "mean_generator_weight": _mean_weight(generators),
        }
    )
    _write_csv(OUT_DIR / "prep_depth.csv", rows)


def _timed_readout(preparation, n: int) -> tuple[list[str], float]:
    """`stabilizers_from_stim`, timed (minimum of three; it is sub-millisecond)."""
    best = math.inf
    generators: list[str] = []
    for _ in range(3):
        start = time.perf_counter()
        generators = interop.stabilizers_from_stim(preparation, num_qubits=n)
        best = min(best, time.perf_counter() - start)
    return generators, best


def _timed_contraction(evolved: PauliSum, generators: Sequence[str]) -> tuple[float, float]:
    """`(value, best-of-three seconds)` for one stabilizer contraction."""
    best = math.inf
    value = 0.0
    generators = list(generators)
    for _ in range(3):
        start = time.perf_counter()
        raw = evolved.expectation_stabilizer(generators)
        best = min(best, time.perf_counter() - start)
        value = raw.real
    return value, best


def _mean_weight(generators: Sequence[str]) -> float:
    weights = [sum(1 for ch in spec[1:] if ch != "I") for spec in generators]
    return sum(weights) / len(weights)


# ==========================================================================
# Part 3 -- contraction scaling
# ==========================================================================


def run_contraction_scaling() -> list[dict[str, Any]]:
    """Measure the `O(m·n²/64)` contraction against `n` and against `m`."""
    print()
    print("=" * 78)
    print("Part 3 -- contraction cost: O(m . n^2 / 64), measured")
    print("=" * 78)
    print(
        "Setup (the generators' GF(2) echelon reduction) is O(n^3/64) and is charged\n"
        "once per call, so it is measured separately as the cost of a *one-term* sum\n"
        "and subtracted to get the per-term figure. Each number is the minimum of\n"
        f"{SCALING_REPEATS} repeats on a shared host; the suite's single-threaded noise\n"
        "floor is +-5-8% (CLAUDE.md), so read the exponents, not the last digit."
    )
    print()

    rows: list[dict[str, Any]] = []
    print(
        f"{'n':>6} {'m':>8} {'setup ms':>10} {'total ms':>11} "
        f"{'per term us':>12} {'n^2/64':>8} {'ns/word-op':>11}"
    )
    for n in SCALING_QUBITS:
        generators = _chain_cluster_generators(n)
        labels = _random_labels(n, SCALING_TERMS)
        full = PauliSum.from_strings({label: 1.0 for label in labels}, num_qubits=n)
        one = PauliSum.from_strings({labels[0]: 1.0}, num_qubits=n)
        setup_s = _best_contraction(one, generators)
        total_s = _best_contraction(full, generators)
        m = len(full)
        per_term = (total_s - setup_s) / m
        word_ops = n * n / 64.0
        rows.append(
            {
                "series": "n_sweep",
                "n_qubits": n,
                "terms": m,
                "setup_s": setup_s,
                "total_s": total_s,
                "per_term_s": per_term,
                "word_ops_bound": word_ops,
            }
        )
        print(
            f"{n:6d} {m:8d} {setup_s * 1e3:10.3f} {total_s * 1e3:11.3f} "
            f"{per_term * 1e6:12.4f} {word_ops:8.0f} "
            f"{per_term / word_ops * 1e9:11.3f}"
        )

    n_sweep = [r for r in rows if r["series"] == "n_sweep"]
    print()
    print("local slope d log(per-term time) / d log(n), against the bound's 2.0:")
    for a, b in zip(n_sweep, n_sweep[1:]):
        slope = math.log(b["per_term_s"] / a["per_term_s"]) / math.log(
            b["n_qubits"] / a["n_qubits"]
        )
        print(f"  n = {a['n_qubits']:4d} -> {b['n_qubits']:4d}:  {slope:.2f}")

    print()
    print(f"{'n':>6} {'m':>8} {'total ms':>11} {'per term us':>12}")
    for m_target in SCALING_TERM_GRID:
        n = SCALING_LINEARITY_QUBITS
        generators = _chain_cluster_generators(n)
        labels = _random_labels(n, m_target)
        full = PauliSum.from_strings({label: 1.0 for label in labels}, num_qubits=n)
        total_s = _best_contraction(full, generators)
        m = len(full)
        rows.append(
            {
                "series": "m_sweep",
                "n_qubits": n,
                "terms": m,
                "setup_s": None,
                "total_s": total_s,
                "per_term_s": total_s / m,
                "word_ops_bound": n * n / 64.0,
            }
        )
        print(f"{n:6d} {m:8d} {total_s * 1e3:11.3f} {total_s / m * 1e6:12.4f}")

    _write_csv(OUT_DIR / "scaling.csv", rows)
    _plot_scaling(rows, OUT_DIR / "scaling.svg")
    return rows


def _chain_cluster_generators(n: int) -> list[str]:
    """1D open-chain cluster-state generators `Z_{q-1} X_q Z_{q+1}`.

    A valid stabilizer state at every `n`, built without stim, so the scaling
    probe measures the contraction and nothing else. Its structure is
    irrelevant to the cost: the echelon reduction sees `n` rows of `W` words
    either way.
    """
    return sp.cluster_state_stabilizers(n, [(q, q + 1) for q in range(n - 1)])


def _random_labels(n: int, count: int) -> list[str]:
    """`count` uniformly random full-length Pauli labels on `n` qubits."""
    rng = np.random.default_rng(SCALING_SEED)
    alphabet = np.array(list("IXYZ"))
    return ["".join(row) for row in alphabet[rng.integers(0, 4, size=(count, n))]]


def _best_contraction(pauli_sum: PauliSum, generators: Sequence[str]) -> float:
    generators = list(generators)
    pauli_sum.expectation_stabilizer(generators)  # warm
    best = math.inf
    for _ in range(SCALING_REPEATS):
        start = time.perf_counter()
        pauli_sum.expectation_stabilizer(generators)
        best = min(best, time.perf_counter() - start)
    return best


def _plot_scaling(rows: Sequence[dict[str, Any]], save_path: Path) -> None:
    """Left: per-term and setup cost vs. `n`. Right: total cost vs. `m`."""
    import matplotlib.pyplot as plt

    fig, (ax_n, ax_m) = plt.subplots(1, 2, figsize=(10, 4))

    n_rows = [r for r in rows if r["series"] == "n_sweep"]
    ns = [r["n_qubits"] for r in n_rows]
    per_term_us = [r["per_term_s"] * 1e6 for r in n_rows]
    setup_us = [r["setup_s"] * 1e6 for r in n_rows]
    ax_n.plot(
        ns,
        per_term_us,
        marker="o",
        markersize=5,
        linewidth=1.5,
        color=_COLORS[0],
        label=f"per term (m = {n_rows[0]['terms']:,})",
    )
    ax_n.plot(
        ns,
        setup_us,
        marker="s",
        markersize=5,
        linewidth=1.5,
        color=_COLORS[1],
        label="setup (GF(2) echelon, once per call)",
    )
    anchor = per_term_us[-1] / ns[-1] ** 2
    ax_n.plot(
        ns,
        [anchor * n * n for n in ns],
        linewidth=1.0,
        linestyle="--",
        color=_MUTED,
        label=r"$\propto n^2$ (the bound)",
    )
    ax_n.set_xscale("log", base=2)
    ax_n.set_yscale("log")
    ax_n.set_xlabel("n qubits")
    ax_n.set_ylabel(r"contraction time ($\mu$s)")

    m_rows = [r for r in rows if r["series"] == "m_sweep"]
    ms = [r["terms"] for r in m_rows]
    totals_ms = [r["total_s"] * 1e3 for r in m_rows]
    ax_m.plot(
        ms,
        totals_ms,
        marker="o",
        markersize=5,
        linewidth=1.5,
        color=_COLORS[2],
        label=f"n = {m_rows[0]['n_qubits']}",
    )
    m_anchor = totals_ms[-1] / ms[-1]
    ax_m.plot(
        ms,
        [m_anchor * m for m in ms],
        linewidth=1.0,
        linestyle="--",
        color=_MUTED,
        label=r"$\propto m$ (the bound)",
    )
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlabel("terms m in the evolved sum")
    ax_m.set_ylabel("contraction time (ms)")

    for ax in (ax_n, ax_m):
        _style(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "B7: stabilizer contraction is $O(m \\cdot n^2/64)$ -- no $2^n$ anywhere",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


# ==========================================================================
# Part 4 -- convergence panels
# ==========================================================================


def run_convergence_panels(*, quick: bool) -> list[report.RunRecord]:
    """The panel plan §7 rule 4 owes for every truncated result."""
    print()
    print("=" * 78)
    print("Part 4 -- truncation convergence of the estimate (plan rule 4)")
    print("=" * 78)

    n, edges, adjacency, centre = _lattice(GRID_ROWS, GRID_COLS)
    generators = interop.stabilizers_from_stim(
        sp.cluster_prep_stim(n, edges), num_qubits=n
    )
    observable = PauliSum.from_strings(
        {_centre_stabilizer_label(n, adjacency, centre): 1.0}, num_qubits=n
    )

    grids = QUICK_CONVERGENCE_GRIDS if quick else CONVERGENCE_GRIDS
    all_records: list[report.RunRecord] = []
    per_depth: dict[int, list[report.RunRecord]] = {}
    references: dict[int, float] = {}
    for steps, cutoffs in grids:
        tail = _tail(n, edges, steps, DEEP_THETA_H)
        print()
        print(f"tail depth {steps}, theta_h = {DEEP_THETA_H}:")
        # `peak RSS` is `VmHWM`, i.e. the *process* high-water mark, which is
        # monotone -- so the column is "how big has this run made the process so
        # far", not a per-point figure (`harness.peak_memory_kb`'s docstring).
        print(f"{'min_abs_coeff':>14} {'peak terms':>12} {'final terms':>12} "
              f"{'prop (s)':>9} {'contract (s)':>13} {'estimate':>16} "
              f"{'proc peak':>10}")

        def build_run(spec, tail=tail, steps=steps):
            # `warmup=False`: this grid is a convergence measurement, not a
            # timing claim, and the deepest point is a 47-second propagation --
            # doubling every point to warm it would buy nothing here.
            record = _record(
                tail,
                observable,
                generators,
                spec,
                warmup=False,
                extra={"tail_steps": steps, "theta_h": DEEP_THETA_H},
            )
            print(
                f"{spec.min_abs_coeff:14.0e} {record.peak_terms:12,d} "
                f"{record.final_terms:12,d} {record.propagation_time_s:9.3f} "
                f"{record.contraction_time_s:13.3f} {record.expectation_value:+16.12f} "
                f"{(record.peak_memory_kb or 0) / 2**20:9.2f}G"
            )
            return record

        records = harness.convergence_sweep(
            build_run, [{"min_abs_coeff": eps} for eps in cutoffs]
        )
        reference = records[-1].expectation_value
        references[steps] = reference
        for record in records:
            record.absolute_error = abs(record.expectation_value - reference)
        gap = abs(records[-1].expectation_value - records[-2].expectation_value)
        print(
            f"  self-converged reference = {reference:+.12f}; the last two grid points "
            f"differ by {gap:.2e}"
        )
        per_depth[steps] = records
        all_records.extend(records)

    _plot_convergence_panels(per_depth, references, OUT_DIR / "convergence_panel.svg")
    return all_records


def _plot_convergence_panels(
    per_depth: dict[int, list[report.RunRecord]],
    references: dict[int, float],
    save_path: Path,
) -> None:
    """One panel per tail depth, drawn by `report.plot_convergence_panel`.

    The shared helper's x axis is `record.truncation["min_abs_coeff"]` and its y
    axis `record.expectation_value`, which is exactly what these records carry,
    so the reference-line handling comes for free.
    """
    import matplotlib.pyplot as plt

    depths = sorted(per_depth)
    fig, axes = plt.subplots(1, len(depths), figsize=(5.2 * len(depths), 4), squeeze=False)
    for index, depth in enumerate(depths):
        ax = axes[0][index]
        report.plot_convergence_panel(
            per_depth[depth],
            truncation_key="min_abs_coeff",
            reference_value=references[depth],
            ax=ax,
        )
        ax.set_ylabel(r"$\langle K_c\rangle$ in the 2D cluster state")
        final = per_depth[depth][-1]
        ax.set_title(
            f"tail depth {depth}  ({final.final_terms:,} terms at the tightest cutoff)",
            fontsize=10,
        )
    fig.suptitle(
        f"B7: truncation convergence of the stabilizer-state estimate, "
        f"n={GRID_ROWS * GRID_COLS}, $\\theta_h$={DEEP_THETA_H}",
        fontsize=11,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


# ==========================================================================
# Part 5 -- dense validation
# ==========================================================================


def run_validation() -> dict[str, Any]:
    """Cross-check the whole pipeline against a dense `2^n` reference."""
    print()
    print("=" * 78)
    print("Part 5 -- dense cross-check at n <= 12")
    print("=" * 78)
    print(
        "The reference runs the *composed* preparation-plus-tail circuit on a 2^n\n"
        "state vector with numpy alone, reading the gate list off the very Circuit\n"
        "object the engine propagates (Circuit.gates), and contracts the observable\n"
        "against it. It knows nothing about stabilizer groups or Pauli propagation."
    )
    print()
    print(
        f"{'lattice':>9} {'n':>3} {'steps':>6} {'theta':>7} {'terms':>7} "
        f"{'PP estimate':>16} {'dense':>16} {'gap':>10} {'projector':>10}"
    )

    cases: list[dict[str, Any]] = []
    worst = 0.0
    for rows_, cols_, steps, theta in VALIDATION_CASES:
        n, edges, adjacency, centre = _lattice(rows_, cols_)
        preparation = sp.cluster_prep_stim(n, edges)
        generators = interop.stabilizers_from_stim(preparation, num_qubits=n)
        prep_circuit, _ = interop.circuit_from_stim(preparation)
        observable = PauliSum.from_strings(
            {_centre_stabilizer_label(n, adjacency, centre): 1.0}, num_qubits=n
        )
        tail = _tail(n, edges, steps, theta)

        evolved = observable.propagate(
            tail, truncation.coeff(DUST_CUTOFF), direction="heisenberg"
        )
        pp_value = evolved.expectation_stabilizer(generators)
        dense_value = sp.dense_expectation(prep_circuit + tail, observable)
        gap = abs(pp_value - dense_value)
        worst = max(worst, gap)

        projector_gap = None
        if n <= VALIDATION_PROJECTOR_MAX:
            # Second dense route, independent of the circuit: the rank-1
            # projector built from the *generator strings*.
            state = sp.dense_projector_state(generators)
            projector_value = sum(
                coefficient * sp.dense_pauli_expectation(state, label)
                for label, coefficient in oracles.pauli_terms(evolved, n)
            )
            projector_gap = abs(pp_value - projector_value)
            worst = max(worst, projector_gap)

        statevector_gap = None
        try:
            with warnings.catch_warnings():
                # qiskit's PauliEvolutionGate synthesis routes through
                # scipy.sparse and warns about matrix formats it chose itself;
                # nothing actionable here, and it would interleave with the
                # table being printed.
                warnings.simplefilter("ignore")
                statevector_value = oracles.statevector_expectation(
                    prep_circuit + tail, observable
                )
        except oracles.OracleError:
            statevector_value = None
        else:
            statevector_gap = abs(pp_value - statevector_value)
            worst = max(worst, statevector_gap)

        cases.append(
            {
                "lattice": f"{rows_}x{cols_}",
                "n_qubits": n,
                "tail_steps": steps,
                "theta_h": theta,
                "terms": len(evolved),
                "pauli_propagation": pp_value.real,
                "dense_statevector": dense_value.real,
                "dense_gap": gap,
                "projector_gap": projector_gap,
                "qiskit_aer_gap": statevector_gap,
            }
        )
        lattice = f"{rows_}x{cols_}"
        projector_cell = "-" if projector_gap is None else f"{projector_gap:.2e}"
        print(
            f"{lattice:>9} {n:3d} {steps:6d} {theta:7.3f} {len(evolved):7d} "
            f"{pp_value.real:+16.12f} {dense_value.real:+16.12f} {gap:10.2e} "
            f"{projector_cell:>10}"
        )

    aer_gaps = [c["qiskit_aer_gap"] for c in cases if c["qiskit_aer_gap"] is not None]
    print()
    if aer_gaps:
        print(f"qiskit Aer agrees too, worst gap {max(aer_gaps):.2e}")
    else:
        print("qiskit Aer not installed; the numpy routes above stand on their own")
    print(f"worst dense gap over every case and route: {worst:.3e}  (bound {DENSE_TOL:g})")
    if worst > DENSE_TOL:
        raise AssertionError(f"dense cross-check failed: {worst:.3e} > {DENSE_TOL}")

    payload = {
        "what": (
            "Showcase B7 dense cross-check: a 2^n state-vector reference for the "
            "composed stabilizer-preparation + non-Clifford-tail circuit, against "
            "Heisenberg Pauli propagation contracted on the preparation's stabilizer "
            "generators."
        ),
        "tolerance": DENSE_TOL,
        "worst_gap": worst,
        "cases": cases,
        "provenance": report.collect_provenance(
            seeds={"preparation": PREP_SEED},
            thread_count=1,
            extra_library_versions=_library_versions(),
            repo_root=_REPO_ROOT,
        ).__dict__,
    }
    path = OUT_DIR / "validation_b7.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path}")
    return payload


# ==========================================================================
# Plot styling (shared with b6's local conventions)
# ==========================================================================

_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
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


# ==========================================================================


def write_json(records: Sequence[report.RunRecord]) -> None:
    # `report.write_results` appends, which is right for a results directory but
    # wrong for a committed artifact that must be regenerable: drop the old file
    # first so a rerun replaces it instead of growing it.
    path = OUT_DIR / "results_b7.json"
    if path.exists():
        path.unlink()
    report.write_results(list(records), OUT_DIR, name="results_b7")
    print(f"wrote {path}  ({len(records)} records)")


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    unknown = [arg for arg in argv if arg != "--quick"]
    if unknown:
        raise SystemExit(f"unknown argument(s) {unknown}; the only flag is --quick")

    harness.assert_logging_quiet()
    harness.assert_single_threaded()

    records: list[report.RunRecord] = []
    records += run_pipeline()
    run_preparation_depth()
    run_contraction_scaling()
    records += run_convergence_panels(quick=quick)
    run_validation()
    write_json(records)
    print()
    print("done." + ("  (--quick: the depth-5 grid stopped at 1e-5)" if quick else ""))


if __name__ == "__main__":
    main()

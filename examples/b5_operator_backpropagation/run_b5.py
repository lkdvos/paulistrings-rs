"""Showcase B5 — hybrid depth reduction via operator backpropagation.

Handoff item B5; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B for the adapted specification and
`research/notes/2026-09-01-python-api-extensions.md` §A3/§A5 for the
`paulistrings.io` / `paulistrings.interop` surface this script exercises.

The idea: split a circuit at layer `k` from the end. Instead of running the
whole circuit on a QPU and measuring the original observable, classically
back-propagate the observable through the final `k` layers
(`direction="heisenberg"`, exact -- this is Pauli propagation, not an
approximation of the tail), then hand a QPU only the *shortened front
circuit* plus the *evolved observable*. Composing the two halves --
classical tail, then quantum front -- reproduces the full-circuit
expectation value exactly (to floating-point tolerance) when nothing is
truncated, and to a controllable, characterizable error when it is. The
"task file" this script emits is literally the artifact a QPU-side runner
would consume: a schema-v1 task JSON (`paulistrings.interop.load_task`)
carrying the residual circuit and the evolved observable.

Two independent parts, run in order by `main()`:

1. **Validation** (`run_validation`) — small `n` so an *exact* reference is
   cheap. Builds a `hardware_efficient_ansatz`, splits it at `SMALL_K`
   layers from the end, and checks three things: (a) the round trip through
   an emitted task-JSON file reproduces the full-circuit expectation to
   `<= 1e-12` at `policy=None`; (b) an independent qiskit-Aer statevector
   oracle agrees with the full-circuit (untruncated) Pauli-propagation
   answer; (c) a `min_abs_coeff` sweep shows the composed (split) expectation
   converging to the same exact reference as the cutoff tightens -- the
   truncation-error convergence panel plan §7 rule 4 requires. Emits
   `evolved_observable_exact.npz` (the `paulistrings.io` round trip) and
   `task_exact.json` / `task_truncated.json` (the `paulistrings.interop`
   round trip) as committed data artifacts, plus `convergence_panel.svg`.

2. **Depth/term-count sweep** (`run_depth_sweep`) — a bigger, ~16-qubit
   kicked-Ising sublattice with a fixed weight-cap truncation, showing the
   actual tradeoff this technique buys: as the tail depth `k` grows (more
   work done classically, less on the QPU), the residual front circuit gets
   shallower while the evolved observable's term count grows. Emits
   `depth_vs_terms.csv` and `depth_vs_terms.svg`.

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    python examples/b5_operator_backpropagation/run_b5.py

No arguments; every number is printed and every artifact is written next to
this script. All circuits here are well under 20 qubits and every propagate
call in this script finishes in well under a second on a laptop-class
machine -- there is nothing here worth timing as a performance claim.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from paulistrings import interop, truncation  # noqa: E402
from paulistrings import io as psio  # noqa: E402

from common import circuits, harness, observables, oracles, report  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Part 1 -- validation setup (small n, exact reference is cheap).
# --------------------------------------------------------------------------

SMALL_N = 8
SMALL_LAYERS = 4
SMALL_K = 1  # tail depth: back-propagate the observable through the final layer
SMALL_SEED = 0

#: `min_abs_coeff` grid for the convergence panel, loosest first.
EPS_GRID = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-6, 1e-8)

#: The eps whose task file is also committed, to show the schema's
#: `"truncation"` key populated on a real artifact (not just the exact case,
#: where it is correctly omitted). Chosen to actually drop a few terms from
#: the tail's evolved observable (32 -> 30) rather than being a no-op.
TRUNCATED_ARTIFACT_EPS = 2e-2

# --------------------------------------------------------------------------
# Part 2 -- depth/term-count sweep (a bigger, ~16-qubit demo).
# --------------------------------------------------------------------------

DEMO_N = 16
DEMO_STEPS = 6
DEMO_THETA_H = 0.6  # generic (non-Clifford) kick angle -- no free Clifford shortcut
DEMO_WEIGHT_CAP = 6


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def split_by_depth(
    spec: oracles.CircuitSpec, total_units: int, k: int
) -> tuple[oracles.CircuitSpec, oracles.CircuitSpec]:
    """Split `spec`'s gate list into `(front, tail)` at the boundary before the
    final `k` of `total_units` equal-sized structural units (Trotter steps or
    ansatz layers).

    Every builder in `circuits.py` pushes exactly the same number of channels
    per unit (the suite's one-gate-per-channel rule, plan §5 D10, applied
    uniformly across units), so `len(spec.gates)` always divides evenly by
    `total_units`; this is asserted rather than assumed.

    `front ++ tail == spec` gate-for-gate, so `front.to_circuit()` composed
    with `tail.to_circuit()` (front first) is exactly `spec.to_circuit()`.
    """
    if not 0 <= k <= total_units:
        raise ValueError(f"k must be in 0..{total_units}, got {k}")
    if total_units <= 0:
        raise ValueError(f"total_units must be positive, got {total_units}")
    if len(spec.gates) % total_units != 0:
        raise ValueError(
            f"{len(spec.gates)} gates do not divide evenly into {total_units} units"
        )
    unit = len(spec.gates) // total_units
    split = (total_units - k) * unit
    front = oracles.CircuitSpec(num_qubits=spec.num_qubits, gates=spec.gates[:split])
    tail = oracles.CircuitSpec(num_qubits=spec.num_qubits, gates=spec.gates[split:])
    return front, tail


def observable_to_task_dict(pauli_sum) -> dict[str, list[float]]:
    """A `PauliSum` as the task-JSON schema v1 `"observable"` object.

    Every coefficient is written as an `[re, im]` pair (rather than a bare
    number for the real-valued ones) so the file's own spelling never depends
    on how close to zero an imaginary residual happens to land -- both forms
    parse identically on `interop.load_task`'s side (`_parse_task_coeff`).
    """
    return {
        label: [coefficient.real, coefficient.imag]
        for label, coefficient in oracles.pauli_terms(pauli_sum)
    }


def build_task(
    front_spec: oracles.CircuitSpec,
    evolved_observable,
    *,
    n_qubits: int,
    direction: str = "heisenberg",
    state: str = "z+",
    threads: int = 1,
    truncation_knobs: dict[str, float] | None = None,
) -> dict:
    """The schema-v1 task-JSON object: run `front_spec` against
    `evolved_observable`, the "residual circuit + evolved observable" artifact
    this whole showcase is about.
    """
    task: dict = {
        "version": 1,
        "n_qubits": n_qubits,
        "circuit": front_spec.to_circuit_json(),
        "observable": observable_to_task_dict(evolved_observable),
        "run": {"direction": direction, "threads": threads, "state": state},
    }
    if truncation_knobs:
        task["truncation"] = dict(truncation_knobs)
    return task


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


# --------------------------------------------------------------------------
# Part 1 -- validation
# --------------------------------------------------------------------------


def run_validation() -> float:
    print("=" * 78)
    print("Part 1 -- validation (small n, exact reference)")
    print("=" * 78)

    n_params = circuits.hardware_efficient_ansatz_num_params(SMALL_N, SMALL_LAYERS)
    rng = np.random.default_rng(SMALL_SEED)
    params = rng.uniform(0.0, 2.0 * math.pi, n_params)

    spec = oracles.record_gates(
        circuits.hardware_efficient_ansatz, SMALL_N, SMALL_LAYERS, params, entangler="cnot"
    )
    front, tail = split_by_depth(spec, SMALL_LAYERS, SMALL_K)
    observable = observables.single_z(SMALL_N // 2, SMALL_N)

    full_circuit = spec.to_circuit()
    front_circuit = front.to_circuit()
    tail_circuit = tail.to_circuit()

    # (a) the full-circuit reference, no truncation.
    full_evolved = observable.propagate(full_circuit, None, direction="heisenberg")
    full_expectation = complex(full_evolved.expectation("z+")).real

    # (b) back-propagate the observable through only the tail (the classical
    # half of the split), exactly.
    tail_evolved = observable.propagate(tail_circuit, None, direction="heisenberg")

    npz_path = OUT_DIR / "evolved_observable_exact.npz"
    psio.save(npz_path, tail_evolved)
    # Read it back through the same file, rather than reusing the in-memory
    # sum, so the task file below is built from what a QPU-side process would
    # actually load off disk.
    reloaded_tail = psio.load(npz_path)

    task = build_task(front, reloaded_tail, n_qubits=SMALL_N)
    task_path = OUT_DIR / "task_exact.json"
    write_json(task_path, task)

    loaded = interop.load_task(task_path)
    composed_evolved = loaded.observable.propagate(
        loaded.circuit, loaded.truncation, direction=loaded.direction
    )
    composed_expectation = complex(composed_evolved.expectation(loaded.state)).real

    round_trip_gap = abs(composed_expectation - full_expectation)
    print(
        f"n={SMALL_N} layers={SMALL_LAYERS} tail_depth={SMALL_K} "
        f"(front {SMALL_LAYERS - SMALL_K} layers / {len(front)} gates, "
        f"tail {SMALL_K} layers / {len(tail)} gates)"
    )
    print(f"  full-circuit expectation      = {full_expectation:.15f}")
    print(f"  composed (task-file) value    = {composed_expectation:.15f}")
    print(f"  round-trip gap                = {round_trip_gap:.3e}  (bound: 1e-12)")
    assert round_trip_gap <= 1e-12, (
        f"composed expectation from {task_path} diverges from the full circuit by "
        f"{round_trip_gap:.3e} > 1e-12"
    )

    # (c) an independent statevector cross-check of the *full* circuit.
    sv_value = complex(oracles.statevector_expectation(spec, observable, None)).real
    sv_gap = abs(sv_value - full_expectation)
    print(f"  statevector cross-check value = {sv_value:.15f}")
    print(f"  statevector gap                = {sv_gap:.3e}  (bound: 1e-9)")
    assert sv_gap < 1e-9, f"statevector oracle diverges from Pauli propagation by {sv_gap:.3e}"

    # (d) the truncated variant: as min_abs_coeff tightens, the *composed*
    # (split) expectation should converge to the same exact reference.
    def build_run(spec_knobs: harness.TruncationSpec) -> report.RunRecord:
        policy = spec_knobs.policy()
        start = time.perf_counter()
        tail_eps = observable.propagate(tail_circuit, policy, direction="heisenberg")
        composed_eps = tail_eps.propagate(front_circuit, policy, direction="heisenberg")
        elapsed = time.perf_counter() - start
        value = complex(composed_eps.expectation("z+")).real
        provenance = report.collect_provenance(repo_root=_REPO_ROOT)
        return report.RunRecord(
            engine="paulistrings",
            engine_version=provenance.library_versions.get("paulistrings", "unknown"),
            n_qubits=SMALL_N,
            direction="heisenberg",
            truncation=spec_knobs.as_dict(),
            propagation_time_s=elapsed,
            final_terms=len(composed_eps),
            provenance=provenance,
            expectation_value=value,
        )

    grid = [harness.TruncationSpec(min_abs_coeff=eps) for eps in EPS_GRID]
    records = harness.convergence_sweep(build_run, grid, oracle_value=full_expectation)

    print("  truncation-convergence sweep (min_abs_coeff -> |gap| vs the exact value):")
    for rec in records:
        gap = rec.absolute_error
        print(
            f"    eps={rec.truncation['min_abs_coeff']:<8.1e} terms={rec.final_terms:<6} "
            f"value={rec.expectation_value:.10f}  gap={gap:.3e}"
        )
    assert records[-1].absolute_error < records[0].absolute_error, (
        "the tightest cutoff in the grid must be at least as close to the exact "
        "reference as the loosest one"
    )
    assert records[-1].absolute_error < 1e-6, (
        f"the tightest cutoff ({EPS_GRID[-1]:.1e}) should have nearly converged, got "
        f"gap={records[-1].absolute_error:.3e}"
    )

    report.plot_convergence_panel(
        records,
        truncation_key="min_abs_coeff",
        reference_value=full_expectation,
        save_path=OUT_DIR / "convergence_panel.svg",
    )
    print(f"  wrote {OUT_DIR / 'convergence_panel.svg'}")

    # A second task-file artifact with the truncation field populated, built
    # at one representative point of the sweep above.
    trunc_policy = truncation.coeff(TRUNCATED_ARTIFACT_EPS)
    tail_trunc = observable.propagate(tail_circuit, trunc_policy, direction="heisenberg")
    trunc_task = build_task(
        front,
        tail_trunc,
        n_qubits=SMALL_N,
        truncation_knobs={"min_abs_coeff": TRUNCATED_ARTIFACT_EPS},
    )
    trunc_task_path = OUT_DIR / "task_truncated.json"
    write_json(trunc_task_path, trunc_task)
    print(f"  wrote {trunc_task_path} (min_abs_coeff={TRUNCATED_ARTIFACT_EPS:.1e})")
    print(f"  wrote {npz_path}")
    print(f"  wrote {task_path}")

    return full_expectation


# --------------------------------------------------------------------------
# Part 2 -- depth/term-count sweep
# --------------------------------------------------------------------------


def run_depth_sweep() -> None:
    print()
    print("=" * 78)
    print("Part 2 -- residual depth vs. evolved-observable term count")
    print("=" * 78)

    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, DEMO_N, DEMO_STEPS, DEMO_THETA_H
    )
    observable = observables.single_z(DEMO_N // 2, DEMO_N)
    policy = truncation.weight(DEMO_WEIGHT_CAP)

    print(
        f"n={DEMO_N} trotter_steps={DEMO_STEPS} theta_h={DEMO_THETA_H} "
        f"weight_cap={DEMO_WEIGHT_CAP} total_gates={len(spec)}"
    )

    rows: list[tuple[int, int, int, int]] = []  # (k, front_layers, front_gates, tail_terms)
    for k in range(DEMO_STEPS + 1):
        front, tail = split_by_depth(spec, DEMO_STEPS, k)
        tail_circuit = tail.to_circuit()
        tail_evolved = observable.propagate(tail_circuit, policy, direction="heisenberg")
        rows.append((k, DEMO_STEPS - k, len(front), len(tail_evolved)))

    print(f"{'k (tail steps)':>15} {'front layers':>13} {'front gates':>12} {'tail terms':>11}")
    for k, front_layers, front_gates, tail_terms in rows:
        print(f"{k:>15} {front_layers:>13} {front_gates:>12} {tail_terms:>11}")

    assert rows[0][3] == 1, "k=0 (no back-propagation) must leave the observable untouched"
    assert rows[-1][3] > rows[0][3], (
        "deeper classical back-propagation must grow the evolved observable's term count"
    )

    csv_path = OUT_DIR / "depth_vs_terms.csv"
    with csv_path.open("w") as f:
        f.write("k_tail_steps,front_layers,front_gates,tail_evolved_terms\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {csv_path}")

    _plot_depth_vs_terms(rows, OUT_DIR / "depth_vs_terms.svg")
    print(f"wrote {OUT_DIR / 'depth_vs_terms.svg'}")


def _plot_depth_vs_terms(rows: list[tuple[int, int, int, int]], save_path: Path) -> None:
    """Residual (front) circuit depth (x) vs. evolved-observable term count (y).

    Styled to match `report.py`'s plot helpers (hairline grid, muted spines,
    the same categorical color as the first slot of the dataviz palette) but
    kept local to this script rather than added to `report.py`, since no
    existing helper's x/y pair (truncation-vs-terms, size-vs-time) fits a
    depth-vs-terms curve.
    """
    import matplotlib.pyplot as plt

    front_layers = [r[1] for r in rows]
    tail_terms = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        front_layers,
        tail_terms,
        marker="o",
        markersize=5,
        linewidth=1.5,
        color="#2a78d6",
    )
    ax.set_yscale("log")
    ax.set_xlabel("residual (front) circuit depth, in Trotter layers")
    ax.set_ylabel("evolved-observable term count")
    ax.set_title("operator backpropagation: depth/term-count tradeoff")
    ax.grid(True, color="#e1e0d9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#898781")
    ax.tick_params(colors="#898781")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")


def main() -> None:
    run_validation()
    run_depth_sweep()
    print()
    print("done.")


if __name__ == "__main__":
    main()

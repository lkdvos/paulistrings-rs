"""Benchmark E -- random SU(4) brickwork, the generic stress test.

Part A's row "E" benchmark in the suite. A `manual-short` benchmark: run it
while developing, not under pytest.

    RAYON_NUM_THREADS=1 python benchmarks/python/bench_e_su4.py

Unlike the Clifford-point benchmarks (A) and the kicked-Ising sweeps (B, C, D),
`examples/common/circuits.py`'s `random_su4_staircase` draws an independent
Haar-random SU(4) block for every brickwork site: there is no stabilizer
structure and no light-cone commutation shortcut anywhere in the circuit, so
this is the suite's test of pure generic fanout. **Every angle/coefficient
below is a Haar sample, never a Clifford point** -- there is no exact integer
to reproduce, only statevector agreement at small `n` / shallow depth and
self-consistency (determinism, convergence) at the `n=36` headline size.

What this script produces, each following the module's four pieces:

1. `validate()` -- small `n` (8..24), shallow depth (3), **no truncation**:
   agreement against `oracles.statevector_expectation` to floating-point
   tolerance. This is the oracle-backed half of the benchmark; everything
   past it is self-converged or cross-engine, never fabricated.
2. `term_growth()` -- the headline "term-count explosion vs depth" curve at
   `n=36`, for three `min_abs_coeff` truncations. No Clifford savings apply
   here, so peak term count grows ~exponentially with depth until the
   truncation itself starts pruning back to zero (the operator's support has
   spread past where any single-qubit-`Z`-sized amplitude survives the
   cutoff) -- both regimes are shown, not just the pre-saturation rise.
3. `error_vs_runtime()` -- fixed `(n, depth) = (16, 6)` (small enough for the
   statevector oracle, deep enough that Haar fanout is already underway),
   swept over `min_abs_coeff`: absolute error vs. oracle, and wall time,
   together.
4. `size_scaling()` -- fixed depth (6), swept `n` from 8 to the 36-qubit
   headline: wall time and peak memory vs. system size. Oracle-checked up to
   `n=24` (Aer's practical ceiling for a same-process sweep here); larger `n`
   is reported unchecked and labeled as such in `extra["oracle_checked"]`.
5. `check_determinism()` -- the same seed run twice must give byte-for-byte
   identical term counts and an expectation matching to 1e-12. This is
   also asserted in CI, in
   `python/paulistrings/tests/test_benchmark_e_su4.py`; the check here is a
   smoke re-run at the driver's own working sizes, not a substitute.
6. `julia_comparison()` -- cross-engine against PauliPropagation.jl 0.8.2.
   Schema v1's `unitary_2q` gate is exactly the SU(4) block this circuit is
   built from, and jl 0.8.2 defines no `_toschrodinger` for `TransferMapGate`
   (`benchmarks/julia/README.md` "Known gaps"), so **only
   `direction="heisenberg"` is comparable** -- the forward direction is not
   attempted here: this suite's cross-engine comparisons only ever compare a
   direction both engines implement for the gate in question.
   `min_abs_coeff=1e-4` is deliberately non-dyadic: the only known
   coefficient-boundary divergence between the engines
   (`benchmarks/julia/README.md` §P3) is triggered by *exact* dyadic
   coefficients at Clifford points, and Haar samples are irrational floats, so
   the boundary case does not arise here, but a non-dyadic cutoff removes it
   as a possible objection. **Per-layer term-count parity is checked and
   printed before any timing is reported or written**: cross-engine timing is
   a blocking gate on that parity check, and on
   this checkout it holds exactly at both a 6-qubit/depth-3 smoke case and the
   10-qubit/depth-5 case whose timing is recorded (`benchmarks/julia/README.md`
   already documents the general vocabulary parity -- this reconfirms it
   specifically for the `unitary_2q` gates this circuit is built from). If a
   future jl version regresses that parity, this function raises rather than
   reporting a number -- see its docstring.

Results land in the *committed* `benchmarks/python/su4_staircase/` directory
(`results.json` + one SVG per plot), not the gitignored
`benchmarks/results/<date>-<host>/` the rest of the suite's ad hoc campaigns
use -- Benchmark E has no `examples/<slug>/` showcase directory of its own (it
is a Part A benchmark), so its committed artifacts live next to the driver
script instead, following the same "regenerated in the same commit as the
script" rule showcases follow.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_BENCH_PY_DIR = Path(__file__).resolve().parent
for _p in (_EXAMPLES_DIR, _BENCH_PY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import circuits, harness, observables, oracles, report  # noqa: E402

import julia_baseline as jb  # noqa: E402
from test_julia_parity import compare as jl_compare  # noqa: E402

#: Fixed seed for every `random_su4_staircase` draw in this benchmark. Record
#: it wherever a number from this file is quoted.
SEED = 20260831

N_HEADLINE = 36
RESULTS_DIR = _BENCH_PY_DIR / "su4_staircase"


def central_qubit(n: int) -> int:
    return n // 2


def build(n: int, depth: int, seed: int = SEED):
    """The one circuit builder this whole file evolves: alias of
    `circuits.random_su4_staircase` fixing the seed argument order.

    `oracles.record_gates` rebinds the `Circuit`/`gates` names in its
    *builder's own* `__globals__`, so a spec must be recorded from
    `circuits.random_su4_staircase` directly -- rebinding this thin wrapper's
    globals would do nothing, since this function never calls `Circuit`/
    `gates` itself. `build(n, depth)` and
    `oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)` still
    build the identical circuit (same seed, same call).
    """
    return circuits.random_su4_staircase(n, depth, seed)


# =============================================================================
# 1. Validation: small n, shallow depth, untruncated, against the statevector
#    oracle.
# =============================================================================

VALIDATION_NS = (8, 12, 16, 20, 24)
VALIDATION_DEPTH = 3


def validate() -> list[report.RunRecord]:
    print(f"\n== validate: n in {VALIDATION_NS}, depth={VALIDATION_DEPTH}, no truncation ==")
    records = []
    for n in VALIDATION_NS:
        q = central_qubit(n)
        obs = observables.single_z(q, n)
        spec = oracles.record_gates(circuits.random_su4_staircase, n, VALIDATION_DEPTH, SEED)
        oracle_value = oracles.statevector_expectation(spec, obs, "z+").real
        circuit = spec.to_circuit()
        record = harness.run_propagation(
            circuit,
            obs,
            None,
            "heisenberg",
            state="z+",
            # threads=None, not 1: see the module docstring's "thread-pin
            # note" -- qiskit-aer's own thread pool has already run by this
            # point (statevector_expectation, just above) and pollutes
            # assert_single_threaded's process-wide thread-delta heuristic.
            # RAYON_NUM_THREADS=1 (checked once at the top of main()) still
            # pins the engine's own pool; this only skips the redundant,
            # Aer-confused re-check.
            threads=None,
            oracle_value=oracle_value,
            seeds={"circuit": SEED},
            extra={"n": n, "depth": VALIDATION_DEPTH, "role": "validation", "central_qubit": q},
        )
        records.append(record)
        print(
            f"  n={n:3d} oracle={oracle_value:+.10f} engine={record.expectation_value:+.10f} "
            f"err={record.absolute_error:.2e} terms={record.final_terms}"
        )
        assert record.absolute_error < 1e-10, (
            f"n={n}: untruncated engine result disagrees with the statevector oracle "
            f"by {record.absolute_error:.3e}"
        )
    return records


# =============================================================================
# 2. Term-count explosion vs depth, at n=36, several truncations.
# =============================================================================

#: (min_abs_coeff, depth grid). The eps=1e-4 grid is cut short at depth=7 --
#: piloted on this checkout (ccqlin038, single-threaded): depth 6 -> 7.9s,
#: depth 7 -> 24.1s, depth 8 -> 54.3s with peak_terms already past 5.4M;
#: extrapolating (~2.3x per extra depth here) depth 9 would cost ~2 minutes and
#: depth 10 ~5 minutes for a single point, which is out of proportion for a
#: `manual-short` benchmark. The looser truncations (1e-2, 1e-3) are cheap
#: (each full depth grid <2s total) precisely because they show the *other*
#: half of the curve: after the operator has spread past a coefficient scale
#: the cutoff can resolve, tightening the cutoff stops helping and peak_terms
#: plateaus, then final_terms collapses to 0 as the tracked amplitude decays
#: below threshold entirely (recorded cut; see the module docstring point 2).
TERM_GROWTH_GRID: tuple[tuple[float, tuple[int, ...]], ...] = (
    (1e-2, (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)),
    (1e-3, (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)),
    (1e-4, (1, 2, 3, 4, 5, 6, 7)),
)


def term_growth() -> list[report.RunRecord]:
    print(f"\n== term_growth: n={N_HEADLINE}, eps/depth grid {TERM_GROWTH_GRID} ==")
    q = central_qubit(N_HEADLINE)
    obs = observables.single_z(q, N_HEADLINE)
    records = []
    for eps, depths in TERM_GROWTH_GRID:
        policy = harness.make_policy(min_abs_coeff=eps)
        for depth in depths:
            circuit = build(N_HEADLINE, depth)
            t0 = time.perf_counter()
            evolved, stats = obs.propagate_with_stats(circuit, policy, direction="heisenberg")
            elapsed = time.perf_counter() - t0
            expectation = evolved.expectation("z+").real
            prov = report.collect_provenance(
                seeds={"circuit": SEED}, thread_count=1, repo_root=_REPO_ROOT
            )
            record = report.RunRecord(
                engine="paulistrings",
                engine_version=prov.library_versions.get("paulistrings", "unknown"),
                n_qubits=N_HEADLINE,
                direction="heisenberg",
                truncation={"min_abs_coeff": eps},
                propagation_time_s=elapsed,
                final_terms=stats.final_terms,
                provenance=prov,
                peak_terms=stats.peak_terms,
                expectation_value=expectation,
                extra={"depth": depth, "role": "term_growth", "central_qubit": q},
            )
            records.append(record)
            print(
                f"  eps={eps:g} depth={depth:2d} final_terms={stats.final_terms:>9d} "
                f"peak_terms={stats.peak_terms:>9d} time={elapsed:.3f}s"
            )
    return records


# =============================================================================
# 3. Error vs runtime at fixed (n, depth), swept over truncation.
# =============================================================================

ERROR_VS_RUNTIME_N = 16
ERROR_VS_RUNTIME_DEPTH = 6
ERROR_VS_RUNTIME_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8)


def error_vs_runtime() -> list[report.RunRecord]:
    n, depth = ERROR_VS_RUNTIME_N, ERROR_VS_RUNTIME_DEPTH
    print(f"\n== error_vs_runtime: n={n}, depth={depth}, eps in {ERROR_VS_RUNTIME_GRID} ==")
    q = central_qubit(n)
    obs = observables.single_z(q, n)
    spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
    oracle_value = oracles.statevector_expectation(spec, obs, "z+").real
    circuit = spec.to_circuit()
    print(f"  statevector oracle: {oracle_value:+.10f}")

    def build_run(trunc_spec: harness.TruncationSpec) -> report.RunRecord:
        return harness.run_propagation(
            circuit,
            obs,
            trunc_spec,
            "heisenberg",
            state="z+",
            threads=None,  # see the threads=None note in validate() above
            oracle_value=oracle_value,
            seeds={"circuit": SEED},
            extra={"n": n, "depth": depth, "role": "error_vs_runtime"},
        )

    records = harness.convergence_sweep(
        build_run, [{"min_abs_coeff": e} for e in ERROR_VS_RUNTIME_GRID], oracle_value=oracle_value
    )
    for eps, record in zip(ERROR_VS_RUNTIME_GRID, records):
        print(
            f"  eps={eps:g} terms={record.final_terms:>9d} time={record.propagation_time_s:.4f}s "
            f"err={record.absolute_error:.3e}"
        )
    return records


# =============================================================================
# 4. Time/memory vs n, at fixed depth.
# =============================================================================

SIZE_SCALING_DEPTH = 6
SIZE_SCALING_EPS = 1e-4
SIZE_SCALING_NS = (8, 12, 16, 20, 24, 28, 32, 36)
#: Oracle-checked up to this n (statevector cost/time still practical for a
#: same-process sweep at this depth); larger n is unchecked and labeled so.
SIZE_SCALING_ORACLE_MAX_N = 24


def size_scaling() -> list[report.RunRecord]:
    depth, eps = SIZE_SCALING_DEPTH, SIZE_SCALING_EPS
    print(f"\n== size_scaling: depth={depth}, eps={eps:g}, n in {SIZE_SCALING_NS} ==")
    records = []
    for n in SIZE_SCALING_NS:
        q = central_qubit(n)
        obs = observables.single_z(q, n)
        oracle_value = None
        if n <= SIZE_SCALING_ORACLE_MAX_N:
            spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
            oracle_value = oracles.statevector_expectation(spec, obs, "z+").real
            circuit = spec.to_circuit()
        else:
            circuit = build(n, depth)
        # Pass the knob dict, not harness.make_policy()'s resolved Truncation
        # object -- run_propagation labels RunRecord.truncation from whatever
        # it is given, and only the knob forms (TruncationSpec / dict / pair)
        # produce the {"min_abs_coeff": ...} label the plots and results.json
        # consumers expect (see harness.py's _policy_and_labels / TruncationSpec
        # docstrings). A resolved Truncation labels itself opaquely as
        # {"policy": repr(...)}.
        record = harness.run_propagation(
            circuit,
            obs,
            {"min_abs_coeff": eps},
            "heisenberg",
            state="z+",
            threads=None,  # see the threads=None note in validate() above
            oracle_value=oracle_value,
            seeds={"circuit": SEED},
            extra={
                "n": n,
                "depth": depth,
                "role": "size_scaling",
                "oracle_checked": oracle_value is not None,
            },
        )
        records.append(record)
        err = "n/a" if record.absolute_error is None else f"{record.absolute_error:.2e}"
        print(
            f"  n={n:3d} time={record.propagation_time_s:.4f}s terms={record.final_terms:>9d} "
            f"peak_mem_delta_kb={record.extra.get('peak_memory_kb_delta', 'n/a')} err={err}"
        )
    return records


# =============================================================================
# 5. Determinism smoke check (authoritative version: the CI pytest file).
# =============================================================================


def check_determinism(n: int = 16, depth: int = 6, eps: float = 1e-4) -> None:
    print(f"\n== check_determinism: n={n}, depth={depth}, eps={eps:g}, same seed twice ==")
    q = central_qubit(n)
    policy = harness.make_policy(min_abs_coeff=eps)

    def run_once():
        obs = observables.single_z(q, n)
        circuit = build(n, depth)
        evolved, stats = obs.propagate_with_stats(circuit, policy, direction="heisenberg")
        return stats.final_terms, stats.peak_terms, evolved.expectation("z+")

    terms_a, peak_a, exp_a = run_once()
    terms_b, peak_b, exp_b = run_once()
    print(f"  run A: terms={terms_a} peak={peak_a} expectation={exp_a}")
    print(f"  run B: terms={terms_b} peak={peak_b} expectation={exp_b}")
    assert terms_a == terms_b, f"final term count differs between runs: {terms_a} vs {terms_b}"
    assert peak_a == peak_b, f"peak term count differs between runs: {peak_a} vs {peak_b}"
    assert abs(exp_a - exp_b) < 1e-12, f"expectation differs between runs: {exp_a} vs {exp_b}"
    print("  determinism OK (identical term counts, expectation matches to 1e-12)")


# =============================================================================
# 6. PauliPropagation.jl comparison.
# =============================================================================

#: Non-dyadic on purpose -- see the module docstring's point 6.
JULIA_EPS = 1e-4
JULIA_SMOKE = (6, 3)  # (n, depth): parity-only, cheap
JULIA_TIMED = (10, 5)  # (n, depth): parity + recorded cross-engine timing


def _make_su4_task(n: int, depth: int, eps: float):
    q = central_qubit(n)
    spec = oracles.record_gates(circuits.random_su4_staircase, n, depth, SEED)
    gates_json = spec.to_circuit_json()["gates"]
    obs_label = observables.pauli_string({q: "Z"}, n)
    return jb.make_task(
        n_qubits=n,
        gates=gates_json,
        observable={obs_label: 1.0},
        direction="heisenberg",
        min_abs_coeff=eps,
        threads=1,
        state="z+",
    )


def julia_comparison() -> list[report.RunRecord]:
    """Parity-gated cross-engine comparison. Returns `[]` (and prints why) if
    Julia is unavailable or if parity fails -- **never** a timing record for a
    run that has not cleared the parity gate. A parity failure is reported
    here as text (what mismatched, at which n/depth), not swallowed: rerun
    this function's output into the benchmark's README by hand if that ever
    happens.
    """
    reason = jb.skip_reason()
    if reason is not None:
        print(f"\n== julia_comparison: SKIPPED ({reason}) ==")
        return []

    print(f"\n== julia_comparison: direction=heisenberg, min_abs_coeff={JULIA_EPS:g} ==")

    smoke_n, smoke_depth = JULIA_SMOKE
    smoke_task = _make_su4_task(smoke_n, smoke_depth, JULIA_EPS)
    smoke_rust, smoke_jl, smoke_problems = jl_compare(smoke_task)
    print(
        f"  smoke n={smoke_n} depth={smoke_depth}: rust_terms={smoke_rust['final_terms']} "
        f"jl_terms={smoke_jl.final_terms} problems={smoke_problems or 'none'}"
    )
    if smoke_problems:
        print(
            "  PARITY FAILED at the smoke case -- no timing will be reported. "
            "Document this in benchmarks/python/su4_staircase/README.md verbatim; "
            "do not adjust eps or fudge a passing case."
        )
        return []

    timed_n, timed_depth = JULIA_TIMED
    timed_task = _make_su4_task(timed_n, timed_depth, JULIA_EPS)
    timed_rust, timed_jl, timed_problems = jl_compare(timed_task)
    print(
        f"  timed n={timed_n} depth={timed_depth}: rust_terms={timed_rust['final_terms']} "
        f"jl_terms={timed_jl.final_terms} problems={timed_problems or 'none'}"
    )
    if timed_problems:
        print(
            "  PARITY FAILED at the timed case -- no timing will be reported. "
            "Document this in benchmarks/python/su4_staircase/README.md verbatim."
        )
        return []

    # Parity holds: record the comparable timing. Rust's own warm timing (not
    # the DEBUG-logging-instrumented path `jl_compare` used for the per-layer
    # counts) is what belongs in a comparison -- run once more, cleanly.
    q = central_qubit(timed_n)
    obs = observables.single_z(q, timed_n)
    circuit = build(timed_n, timed_depth)
    # Knob dict, not a resolved Truncation -- see the size_scaling() comment.
    rust_record = harness.run_propagation(
        circuit,
        obs,
        {"min_abs_coeff": JULIA_EPS},
        "heisenberg",
        state="z+",
        threads=None,  # see the threads=None note in validate() above
        seeds={"circuit": SEED},
        extra={"n": timed_n, "depth": timed_depth, "role": "julia_comparison"},
    )
    jl_record = report.RunRecord(
        engine="PauliPropagation.jl",
        engine_version=timed_jl.versions.get("PauliPropagation", "unknown"),
        n_qubits=timed_n,
        direction="heisenberg",
        truncation={"min_abs_coeff": JULIA_EPS},
        propagation_time_s=timed_jl.wall_warm_s or timed_jl.wall_cold_s,
        final_terms=timed_jl.final_terms,
        provenance=report.collect_provenance(
            seeds={"circuit": SEED},
            thread_count=1,
            extra_library_versions=timed_jl.versions,
            repo_root=_REPO_ROOT,
        ),
        peak_terms=timed_jl.peak_terms,
        expectation_value=(timed_jl.expectation.real if timed_jl.expectation is not None else None),
        extra={"n": timed_n, "depth": timed_depth, "role": "julia_comparison"},
    )
    print(
        f"  TIMED (parity holds, {timed_rust['final_terms']} terms both engines): "
        f"rust={rust_record.propagation_time_s:.4f}s jl={jl_record.propagation_time_s:.4f}s"
    )
    return [rust_record, jl_record]


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    harness.assert_single_threaded()
    all_records: list[report.RunRecord] = []
    all_records += validate()
    growth_records = term_growth()
    all_records += growth_records
    error_records = error_vs_runtime()
    all_records += error_records
    size_records = size_scaling()
    all_records += size_records
    check_determinism()
    all_records += julia_comparison()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = report.write_results(all_records, RESULTS_DIR, name="results")
    print(f"\nwrote {len(all_records)} records to {written}")

    _plot_term_growth(growth_records, RESULTS_DIR / "term_count_vs_depth.svg")
    report.plot_error_vs_runtime(error_records, save_path=RESULTS_DIR / "error_vs_runtime.svg")
    report.plot_time_and_memory_vs_size(
        size_records, save_path=RESULTS_DIR / "time_memory_vs_n.svg"
    )
    print(f"wrote figures to {RESULTS_DIR}")


def _plot_term_growth(records: list[report.RunRecord], save_path: Path) -> None:
    """Peak term count (y, log) vs. depth (x), one curve per `min_abs_coeff`.

    Not one of `report.py`'s generic helpers (those key on truncation or
    `n_qubits` on the x axis, grouped by engine); this benchmark's x axis is
    depth, grouped by truncation, single engine -- specific enough to this one
    plot that it lives here rather than in the shared module.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    by_eps: dict[float, list[tuple[int, int]]] = {}
    for r in records:
        eps = r.truncation.get("min_abs_coeff")
        depth = r.extra.get("depth")
        if eps is None or depth is None or r.peak_terms is None:
            continue
        by_eps.setdefault(eps, []).append((depth, r.peak_terms))

    for eps in sorted(by_eps):
        points = sorted(by_eps[eps])
        xs, ys = zip(*points)
        color = report._color_for_engine(f"eps={eps:g}")
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=f"min_abs_coeff={eps:g}")

    ax.set_yscale("log")
    ax.set_xlabel("brickwork depth")
    ax.set_ylabel("peak term count")
    ax.set_title(f"n={N_HEADLINE} random SU(4) staircase, seed={SEED}")
    report._style_axes(ax)
    ax.legend(frameon=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")


if __name__ == "__main__":
    main()

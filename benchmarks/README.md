# Benchmarks

Performance is a primary goal of this library. Three benchmark surfaces exist: Rust microbenchmarks,
Python end-to-end benchmarks (including the examples-and-benchmarks suite's Part A entries), and a
Julia cross-engine baseline.

## Rust microbenchmarks (criterion)

```
cargo bench -p paulistrings
```

Output: `target/criterion/` (HTML reports). Use these for tight inner-loop work (multiplication,
commutator, anticommutator, weight, hashing, canonicalization).

## Python end-to-end benchmarks (pytest-benchmark)

The suite is `benchmarks/python/bench_baseline.py`, manual and not run in CI — it needs the
`./scripts/setup.sh` venv (with the `bench` extras: `qiskit`, `openfermion`) and a
`maturin develop --release` build first.

```bash
./scripts/setup.sh
source .venv/bin/activate
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

It covers two groups, each parameterized over `n_terms ∈ {100, 1_000, 10_000}` at a fixed
`num_qubits = 16`:

- `construct` — building an N-term Pauli sum from `(label, coefficient)` pairs, against
  `qiskit.quantum_info.SparsePauliOp.from_list` and `openfermion.QubitOperator`.
- `conjugate_clifford` — Heisenberg-conjugating an N-term sum through a fixed `H`+`CNOT` Clifford
  circuit, against `qiskit.quantum_info.PauliList.evolve` on a `SparsePauliOp`.

`PauliStrings.jl` is excluded from `bench_baseline.py` (no PyJulia wiring; see the Julia baseline
below). `stim` was not compared here, but the Clifford path uses it in Benchmark A below.

## The examples & benchmarks suite (Part A)

Five more benchmarks follow `bench_baseline.py`'s idioms (`pytest.importorskip`, seeded fixtures
outside the timed region, `@pytest.mark.benchmark(group=...)`, assert-on-result). None run in CI;
each has a CI-safe correctness gate at smaller scale in `python/paulistrings/tests/`. Results and
figures are committed next to each driver; raw `RunRecord` JSON regenerates by rerunning the script.

| | driver | results | oracle | runtime class |
|---|---|---|---|---|
| **A** Clifford gate | [`python/bench_a_clifford.py`](python/bench_a_clifford.py) | `benchmarks/results/bench_a.json` | `stim` (Clifford-point exact ±1) | CI-gated correctness (`test_benchmark_a_clifford.py`) + manual timing |
| **B** θ_h sweep | [`python/bench_b_theta_sweep.py`](python/bench_b_theta_sweep.py) | [`python/theta_sweep/`](python/theta_sweep/) | causal-cone light-cone exact reference | manual-short |
| **C** deep Trotter (headline) | [`python/bench_c_deep_trotter.py`](python/bench_c_deep_trotter.py) | [`python/deep_trotter/`](python/deep_trotter/) | self-converged reference with documented convergence evidence | manual-long (time-boxed) |
| **D** XXZ chain | `examples/xxz_chain/run_benchmark_d.py` | `examples/xxz_chain/results/` | statevector (`n ≤ 26`) + an analytic quadratic term-growth law | manual-short — lives in `examples/xxz_chain/`, not here (its deliverable is a scaling sweep, not a `pytest-benchmark` entry) |
| **E** SU(4) brickwork | [`python/bench_e_su4.py`](python/bench_e_su4.py) | [`python/su4_staircase/`](python/su4_staircase/) | statevector at small `n` | manual-short (`test_benchmark_e_su4.py` gates correctness) |

Benchmark F (surrogate/symbolic-coefficient landscape) is not implemented: it needs a
symbolic-coefficient core the engine does not have.

## Julia baseline (PauliPropagation.jl)

[`benchmarks/julia/`](julia/) is a subprocess-driven, out-of-CI baseline against
**PauliPropagation.jl**, pinned to version 0.8.2 in `julia/Project.toml`/`Manifest.toml` (julia
1.12.6). There is no PyJulia/juliacall anywhere — the same exclusion `bench_baseline.py` records for
`PauliStrings.jl` — so the only entry points are `julia/runner.jl` (reads a task JSON in schema v1,
propagates, emits a result JSON) and [`python/julia_baseline.py`](python/julia_baseline.py) (the
`subprocess` wrapper, which skips cleanly with no `julia` binary on `PATH`).

```bash
python benchmarks/python/julia_baseline.py --self-test     # wrapper smoke test
pytest benchmarks/python/test_julia_parity.py -q           # the parity gate (blocking for timing)
```

`python/test_julia_parity.py` is the **blocking** parity gate: no cross-engine timing may be reported
for a run whose evolved Pauli sums diverge term-for-term at matched truncation, so every Part A driver
above runs this check before writing a timed number.

Known semantic divergences between this engine and PauliPropagation.jl 0.8.2, measured via
`julia/probes.jl`: the coefficient-cutoff boundary is inclusive here vs. exclusive in jl, exact-zero
coefficients are dropped here but kept there, and `direction="forward"` has no counterpart in jl for
several non-Clifford gates. All three are recorded with mitigations in
[`benchmarks/julia/README.md`](julia/README.md).

Save raw timings under `benchmarks/results/<date>-<machine>/` to plot regressions over time. Always
record: commit hash, CPU, RAM, compiler version, BLAS (if any), and number of threads.

## Ground rules

- Always benchmark `--release` / `cargo bench` (never debug).
- The governor is `powersave` and unpinnable on the reference host (no root) — rely on more samples
  and ratio metrics (IPC, % of bandwidth ceiling, speedup, cycles/string) rather than absolute ms
  across days; see `PROFILING.md`.
- Report both single-thread and multi-thread numbers when the operation is parallel.
- Keep input generation deterministic (seeded RNG) and outside the timed region.

## Profiling

For the phase-timing probe, flamegraphs, hardware counters, the memory-bandwidth roofline, and the
standard change → measure → compare loop, see [`PROFILING.md`](PROFILING.md).

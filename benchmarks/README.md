# Benchmarks

Performance is a primary goal of this library. We track three benchmark surfaces.

## 1. Rust microbenchmarks (criterion)

```
cargo bench -p paulistrings
```

Output: `target/criterion/` (HTML reports). Use these for tight inner-loop work
(multiplication, commutator, anticommutator, weight, hashing, canonicalization).

## 2. Python end-to-end benchmarks (pytest-benchmark)

```
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

These should mirror realistic user workloads: building Hamiltonians, evolving
operators, computing expectation values, BCH / Trotter expansions, etc.

## 3. Cross-library comparisons

Compare against:
- `PauliStrings.jl` (Julia) — the inspiration; same operations, same sizes.
- `qiskit.quantum_info.SparsePauliOp` / `Pauli`
- `openfermion.QubitOperator`
- `stim.PauliString` (where applicable; Clifford-focused)

Save raw timings under `benchmarks/results/<date>-<machine>/` so we can plot
regressions over time. Always record: commit hash, CPU, RAM, compiler version,
BLAS (if any), and number of threads.

## Ground rules

- Always benchmark `--release` / `cargo bench` (never debug).
- The governor is `powersave` and unpinnable on the reference host (no root) —
  rely on more samples and ratio metrics (IPC, % of bandwidth ceiling,
  speedup, cycles/string) rather than absolute ms across days; see
  `PROFILING.md`.
- Report both single-thread and multi-thread numbers when the operation is
  parallel.
- Keep input generation deterministic (seeded RNG) and outside the timed region.

## Profiling

For the phase-timing probe, flamegraphs, hardware counters, the memory-
bandwidth roofline, and the standard change → measure → compare loop, see
[`PROFILING.md`](PROFILING.md).

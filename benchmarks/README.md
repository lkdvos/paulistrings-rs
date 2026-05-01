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
- Pin the CPU governor to `performance` and disable turbo if you want stable
  numbers; otherwise run with enough samples that criterion's noise model copes.
- Report both single-thread and multi-thread numbers when the operation is
  parallel.
- Keep input generation deterministic (seeded RNG) and outside the timed region.

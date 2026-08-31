# Benchmarks

Performance is a primary goal of this library. We track two benchmark surfaces.

## 1. Rust microbenchmarks (criterion)

```
cargo bench -p paulistrings
```

Output: `target/criterion/` (HTML reports). Use these for tight inner-loop work
(multiplication, commutator, anticommutator, weight, hashing, canonicalization).

## 2. Python end-to-end benchmarks (pytest-benchmark)

The suite is `benchmarks/python/bench_baseline.py`. It is **manual, not run
in CI** — it needs the `./scripts/setup.sh` venv (with the `bench` extras:
`qiskit`, `openfermion`) and a `maturin develop --release` build first.

```
./scripts/setup.sh
source .venv/bin/activate
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

It currently covers two groups, each parameterized over `n_terms ∈ {100,
1_000, 10_000}` at a fixed `num_qubits = 16`:

- `construct` — building an N-term Pauli sum from `(label, coefficient)`
  pairs, against `qiskit.quantum_info.SparsePauliOp.from_list` and
  `openfermion.QubitOperator`.
- `conjugate_clifford` — Heisenberg-conjugating an N-term sum through a
  fixed `H`+`CNOT` Clifford circuit, against
  `qiskit.quantum_info.PauliList.evolve` on a `SparsePauliOp`.

`PauliStrings.jl` is excluded (no PyJulia wiring). `stim` is not currently
compared.

Possible future comparisons (not implemented): BCH/Trotter expansion
benchmarks, and a `stim.PauliString` comparison for the Clifford-only path.

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

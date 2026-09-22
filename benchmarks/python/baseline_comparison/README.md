# Baseline comparison — qiskit / openfermion container operations

Committed results of `benchmarks/python/bench_baseline.py`: `paulistrings` against `qiskit.quantum_info.SparsePauliOp` and `openfermion.QubitOperator` on Pauli-sum container operations (construction from string terms, one-layer Heisenberg conjugation by an H+CNOT Clifford circuit).
Tables and interpretation: [Comparisons — vs qiskit/openfermion](https://lkdvos.github.io/paulistrings-rs/examples/comparisons.html#vs-qiskitsparsepauliop-openfermionqubitoperator).

## Provenance

- Host: ccqlin038.flatironinstitute.org (2× Xeon Gold 6244, governor `powersave`), 2026-09-01,
  commit `94b3364`, single process, no explicit BLAS thread pinning (the ops are pure-Python/Rust
  container operations).
- Python 3.11.11, qiskit 2.5.2, openfermion 1.8.1, `paulistrings` built with
  `maturin develop --release`.
- Inputs: seeded `random.Random`, `n_terms ∈ {100, 1000, 10000}` at a fixed qubit count,
  generated outside the timed region.
- Raw data: [`results.json`](results.json) (pytest-benchmark JSON, committed).

## Rerun

```bash
pytest benchmarks/python/bench_baseline.py --benchmark-only \
    --benchmark-json=benchmarks/python/baseline_comparison/results.json
```

Missing backends skip rather than fail.
When rerun, update the site's comparisons page and this provenance block from the new `results.json` in the same commit.

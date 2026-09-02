# Baseline comparison — qiskit / openfermion container operations

Committed results of `benchmarks/python/bench_baseline.py`: `paulistrings` against
`qiskit.quantum_info.SparsePauliOp` and `openfermion.QubitOperator` on Pauli-sum container
operations. Scope: these are container benchmarks (construction from string terms, one-layer
Heisenberg conjugation by an H+CNOT Clifford circuit), not propagation-engine benchmarks — neither
baseline library implements truncated Pauli propagation, so no crossover concept applies.
`PauliStrings.jl` is excluded (Julia-from-pytest wiring); `stim` is a correctness oracle elsewhere
in the suite and is not timed here.

## Provenance

- Host: ccqlin038.flatironinstitute.org (2× Xeon Gold 6244, governor `powersave`), 2026-09-01,
  commit `94b3364`, single process, no explicit BLAS thread pinning (the ops are pure-Python/Rust
  container operations).
- Python 3.11.11, qiskit 2.5.2, openfermion 1.8.1, `paulistrings` built with
  `maturin develop --release`.
- Inputs: seeded `random.Random`, `n_terms ∈ {100, 1000, 10000}` at a fixed qubit count,
  generated outside the timed region.
- Raw data: [`results.json`](results.json) (pytest-benchmark JSON, committed).

## Construct (median, µs; ratio = library / paulistrings)

| n_terms | paulistrings | qiskit | openfermion | qiskit ratio | openfermion ratio |
|---:|---:|---:|---:|---:|---:|
| 100 | 99.9 | 1 053.7 | 982.4 | 10.5× | 9.8× |
| 1 000 | 683.7 | 9 693.8 | 10 570.3 | 14.2× | 15.5× |
| 10 000 | 3 070.0 | 96 566.8 | 106 078.9 | 31.5× | 34.6× |

## Conjugate by a Clifford layer (median, µs)

`openfermion` has no equivalent conjugation op and is not in this group.

| n_terms | paulistrings | qiskit | qiskit ratio |
|---:|---:|---:|---:|
| 100 | 8.9 | 2 133.8 | 240× |
| 1 000 | 71.0 | 4 978.4 | 70× |
| 10 000 | 1 057.3 | 32 642.0 | 31× |

## Rerun

```bash
pytest benchmarks/python/bench_baseline.py --benchmark-only \
    --benchmark-json=benchmarks/python/baseline_comparison/results.json
```

Missing backends skip rather than fail. When rerun, update the tables above and the site's
comparisons page from the new `results.json` in the same commit.

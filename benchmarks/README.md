# Benchmarks

Three benchmark surfaces: Rust microbenchmarks, Python end-to-end benchmarks (including the examples-and-benchmarks suite's Part A entries), and a Julia cross-engine baseline.
Full narrative, rules, and headline results: [Benchmarks](https://lkdvos.github.io/paulistrings-rs/benchmarks/index.html) and [Against other tools](https://lkdvos.github.io/paulistrings-rs/comparisons.html).

## Rust microbenchmarks (criterion)

```bash
cargo bench -p paulistrings
```

Output: `target/criterion/` (HTML reports).

## Python end-to-end benchmarks (pytest-benchmark)

```bash
./scripts/setup.sh
source .venv/bin/activate
maturin develop --release -m crates/paulistrings-py/Cargo.toml
pytest benchmarks/python --benchmark-only --benchmark-json=benchmarks/results/py.json
```

Manual, not run in CI: `bench_baseline.py` (vs qiskit/openfermion containers, see [`python/baseline_comparison/README.md`](python/baseline_comparison/README.md)) plus the five Part A benchmarks below.
None of the Part A benchmarks run in CI; each has a CI-safe correctness gate at smaller scale in `benchmarks/python/tests/`.

| | driver | results | oracle |
|---|---|---|---|
| **A** Clifford gate | [`python/bench_a_clifford.py`](python/bench_a_clifford.py) | `benchmarks/results/bench_a.json` | `stim` |
| **B** θ_h sweep | [`python/bench_b_theta_sweep.py`](python/bench_b_theta_sweep.py) | [`python/theta_sweep/`](python/theta_sweep/) | causal-cone exact |
| **C** deep Trotter | [`python/bench_c_deep_trotter.py`](python/bench_c_deep_trotter.py) | [`python/deep_trotter/`](python/deep_trotter/) | self-converged reference |
| **D** XXZ chain | `examples/xxz_chain/run_benchmark_d.py` | `examples/xxz_chain/results/` | statevector + analytic growth law |
| **E** SU(4) brickwork | [`python/bench_e_su4.py`](python/bench_e_su4.py) | [`python/su4_staircase/`](python/su4_staircase/) | statevector at small `n` |

## Julia baseline (PauliPropagation.jl)

```bash
python benchmarks/python/julia_baseline.py --self-test     # wrapper smoke test
pytest benchmarks/python/test_julia_parity.py -q           # the blocking parity gate
```

[`benchmarks/julia/`](julia/) is a subprocess-driven, out-of-CI baseline against PauliPropagation.jl; see [`julia/README.md`](julia/README.md) for the pinned version and known semantic divergences.

## Profiling

For the phase-timing probe, flamegraphs, hardware counters, the memory-bandwidth roofline, and the change → measure → compare loop, see [`PROFILING.md`](PROFILING.md).

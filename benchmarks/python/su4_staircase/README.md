# Benchmark E — Random SU(4) brickwork

The deliberate worst case: `n = 36` qubits, an independent Haar-random SU(4) block on every
brickwork site, observable `Z_18`, Heisenberg direction. No stabilizer structure, no commuting
sublattice, no light-cone shortcut past nearest-neighbour causality — the generic case Pauli
propagation faces with no help from problem structure. Driver:
[`../bench_e_su4.py`](../bench_e_su4.py). CI-safe correctness gate:
[`python/paulistrings/tests/test_benchmark_e_su4.py`](../../../python/paulistrings/tests/test_benchmark_e_su4.py).

Full writeup: https://lkdvos.github.io/paulistrings-rs/benchmarks/e-su4-brickwork.html

## Run it

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_e_su4.py
```

## Headline results

Numbers already in `results.json` (term counts, timings, errors, per-`n` oracle checks) are not
repeated here; see the writeup for those. Not in `results.json`: the small-size cross-engine smoke
comparison against `PauliPropagation.jl` 0.8.2 —

| n | depth | rust final_terms | jl final_terms | parity |
|---|---|---|---|---|
| 6 (smoke) | 3 | 4051 | 4051 | exact, all layers |

Per-layer term counts (not just the final one) matched exactly at this size, using
`benchmarks/python/test_julia_parity.py`'s `compare()`.

## Provenance

Seed **20260831**, fixed everywhere in this file and in `bench_e_su4.py`'s `SEED` constant.
`results.json` carries the full per-run provenance block (commit, CPU, rustc/julia/
PauliPropagation.jl versions, thread count) for every record. Single-threaded
(`RAYON_NUM_THREADS=1`) throughout; `RunRecord.provenance.thread_count` records `None` on the runs
that follow a qiskit-aer statevector call in the same process, since Aer's own thread pool defeats
the harness's process-wide thread-delta pinning heuristic once it has run.

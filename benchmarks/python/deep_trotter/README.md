# Benchmark C — Deep Trotter circuits

Heavy-hex kicked Ising, 127 qubits, a depth ladder of 5/9/15/20 Trotter steps in the hard interior, scored against the tightest exact or self-converged reference reachable at each depth.
Full writeup: https://lkdvos.github.io/paulistrings-rs/benchmarks/c-deep-trotter.html

## Run it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --validate-convergence
pytest benchmarks/python/tests/test_benchmark_c_deep.py    # CI gate, 25 tests, ~50 s
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts.

## Headline results

Numbers quoted on the site page that are not in `results.json` or
`summary.json`:

| metric | value |
|---|---|
| projected terms / wall (32 threads) / columns at `2⁻¹⁸`, 20 steps, 7π/32 | ~7e8 terms, ~1.1 h, ~37 GiB |
| projected terms / wall (32 threads) / columns at `2⁻²⁰`, 20 steps, 7π/32 | ~1e10 terms, ~17 h, ~560 GiB |
| Showcase B2, same point, with noise | 48× fewer peak terms, 154× smaller last difference |
| this engine's `2⁻¹⁴` sum, bucketed columns (3.1e6 terms × 48 B, `W=2`) | ~0.15 GiB |
| 42-run campaign process high-water | 1.11 GiB |

The projection is what rules out reaching the accuracy target at 20 steps
on a workstation.

## Provenance

| file | what it is |
|---|---|
| `../bench_c_deep_trotter.py` | the driver |
| `results.json` | 42 `report.RunRecord`s (39 paulistrings, 3 PauliPropagation.jl) |
| `summary.json` | references, convergence evidence, envelope checks, time-to-accuracy, parity outcomes, the published anchor |
| `error-vs-runtime.svg`, `convergence-vs-truncation.svg`, `term-count-vs-truncation.svg`, `parity-per-layer-terms.svg` | the site page's figures |

Recorded run: commit `e024d8b`, clean worktree, ccqlin038 (Xeon Gold 6244 @
3.60 GHz), rustc 1.94.0, Python 3.11.11, paulistrings 0.1.0, numpy 2.4.6,
qiskit 2.5.2, stim 1.16.0, julia 1.12.6 + PauliPropagation 0.8.2. 85.7 min
end to end.

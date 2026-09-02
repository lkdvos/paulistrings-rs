# Benchmark D — XXZ chain scaling

Trotterized open XXZ chain, `Jz = 0` (free) and `Jz = 0.5` (interacting) regimes, with an
analytic growth-law prediction to check the untruncated term count against and a cross-engine
timing comparison against `PauliPropagation.jl` whose ranking flips as the tracked set grows.
Driver: [`run_benchmark_d.py`](run_benchmark_d.py); CI gate:
[`test_benchmark_d_xxz.py`](../../python/paulistrings/tests/test_benchmark_d_xxz.py) (11 tests,
~4 s).

Full writeup: https://lkdvos.github.io/paulistrings-rs/benchmarks/d-xxz-chain.html

## Run it

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
# or one mode at a time: growth | statevector | scaling | convergence | julia | figures
pytest python/paulistrings/tests/test_benchmark_d_xxz.py
```

Results land in `results/*.json`, one file per mode, overwritten (not appended) on rerun; figures
in `figures/*.svg` regenerate from the committed JSON with the `figures` mode alone.

## Headline results

Numbers below are not in the committed `results/*.json`.

- Peak memory is **55 B/term below the `W = 1 → W = 2` boundary at 64 qubits, and 87 B/term
  above it** (64-bit vs 128-bit symplectic keys: 32 B/term of key becomes 48 B/term).
- **The cross-engine ranking changes sign somewhere between 3·10³ and 3·10⁴ tracked terms.** Below
  the crossover `PauliPropagation.jl`'s hash-map backend is 3–4× faster; above it this engine is
  ~1.5× faster and pulling away.
- The self-converged `n = 60` and `n = 100` panels (`Jz = 0.5`) produce bit-identical values and
  term counts at every cutoff, differing only in wall time (9.6 s vs 16.5 s at
  `min_abs_coeff = 1e-8`).

## Provenance

| | |
|---|---|
| driver | `run_benchmark_d.py` (`growth`, `statevector`, `scaling`, `convergence`, `julia`, `figures`, `all`) |
| CI gate | `python/paulistrings/tests/test_benchmark_d_xxz.py` (11 tests, ~4 s) |
| host | ccqlin038, Intel Xeon Gold 6244 @ 3.60 GHz, `RAYON_NUM_THREADS=1`, `RUST_LOG` unset |

Cross-engine rows use PauliPropagation.jl 0.8.2 on julia 1.12.6, `dict` backend, `-t1`.

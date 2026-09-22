# Benchmark D — XXZ chain scaling

Trotterized open XXZ chain, `Jz = 0` (free) and `Jz = 0.5` (interacting) regimes, with an analytic growth-law prediction to check the untruncated term count against and a cross-engine timing comparison against `PauliPropagation.jl` whose ranking flips as the tracked set grows.

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
# or one mode at a time: growth | statevector | scaling | convergence | julia | figures
pytest benchmarks/python/tests/test_benchmark_d_xxz.py   # 11 tests, ~4 s
```

Results land in `results/*.json`, one file per mode, overwritten (not appended) on rerun; figures in `figures/*.svg` regenerate from the committed JSON with the `figures` mode alone.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/examples/benchmarks/d-xxz-chain.html

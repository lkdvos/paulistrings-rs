# `examples/` — showcases for the examples & benchmarks suite

Every directory here is a Part B application showcase (B1, B2, B5, B6, B7), plus the shared `common/` infrastructure and provenance-tagged `data/` the showcases and the benchmarks both use.
Part A benchmarks A–E live under [`../benchmarks/`](../benchmarks/README.md); Benchmark D is the exception, living at [`xxz_chain/`](xxz_chain/) because its deliverable is a scaling sweep rather than a `pytest-benchmark` entry.
Rust examples live under `crates/paulistrings/examples/`; this tree is Python-only.

```bash
./scripts/setup.sh                                            # one-time: creates .venv, builds the extension
source .venv/bin/activate
pip install -e ".[examples]"                                  # matplotlib, stim, qiskit, qiskit-aer, numpy
maturin develop --release -m crates/paulistrings-py/Cargo.toml # rebuild after any Rust change

RAYON_NUM_THREADS=1 python examples/b1_operator_scrambling/run_b1_1d.py
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick   # full run: drop --quick
RAYON_NUM_THREADS=1 python examples/b5_operator_backpropagation/run_b5.py
RAYON_NUM_THREADS=1 python examples/b6_resource_probes/run_b6.py
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py            # --quick: 40 s instead of 116 s
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts, since Rayon's global pool is built once at the first `propagate` call and never resized.
Every CI-visible test lives under `examples/tests/` (showcases) or `benchmarks/python/tests/` (benchmarks) and `importorskip`s `stim`/`qiskit`/`matplotlib`, so the numpy-only CI job stays green without the `examples` extra installed; the scripts under this directory are not collected by CI and run manually as above.

Full writeup of every showcase, what it demonstrates, and its independent cross-check: https://lkdvos.github.io/paulistrings-rs/examples/showcases/index.html

The `PauliPropagation.jl` baseline, its task-JSON schema, and the measured semantics divergences between the two engines are documented in [`../benchmarks/julia/README.md`](../benchmarks/julia/README.md).

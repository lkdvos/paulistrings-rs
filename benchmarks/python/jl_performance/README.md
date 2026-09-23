# Head-to-head: `paulistrings` vs PauliPropagation.jl

Single-threaded, core versus core, on parity-gated configurations, under an interleaved-pair protocol.
This directory is the index for the study: driver `benchmarks/python/bench_jl_performance.py`, figures `benchmarks/python/jl_performance_figures.py`.
Method, headline numbers and interpretation: [Against other tools](https://lkdvos.github.io/paulistrings-rs/examples/comparisons.html).

Every record directory carries `results.json` (one record per configuration per engine), `summary.json` (per-pair ratios, crossovers, parity evidence), `run.log` and its figures; most also carry `tasks/`, the schema-v1 task files both engines read.

## Reproducing

```bash
# the curves, default engine (~25 min per workload pair on a quiet 32-core host)
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization

# re-render figures from committed data, no measurement
python benchmarks/python/jl_performance_figures.py benchmarks/python/jl_performance/summary.json

# the CI protocol gate (no julia, no timing, < 1 s)
pytest benchmarks/python/tests/test_jl_performance_protocol.py
```

The Julia side needs the pinned project in `benchmarks/julia/` (PauliPropagation.jl 0.8.2, Julia 1.12.6); the first run precompiles for ~30 s.
Everything degrades cleanly with no `julia` on `PATH` except the measurement itself.

## Records

| directory | sweep | engine `crates/` tree | driver commit |
|---|---|---|---|
| `post-optimization/` | kicked-Ising + XXZ curves, default engine, 5 pairs | `81c568a` | `0f00207` |
| `post-optimization/su4-curve/` | SU(4) curve, default engine, 5 pairs | `81c568a` | `0f00207` |
| `post-optimization-auto/` | loose end of all three curves with `engine="auto"`, 5 pairs | `81c568a` | `0f00207` |
| `deep-kicked-ising/` | kicked-Ising at 20 Trotter steps, 5 420 channels, 3 pairs | `4768fe4` | `e4aeccd` |
| `su4-curve/` | SU(4) curve, 5 pairs | `4768fe4` | `35ff414` |
| `.` (this directory) | kicked-Ising + XXZ curves, 5 pairs | `4768fe4` | — |

Each record's README states the build it measures and its own findings.

## Provenance

Host ccqlin038, 2 × Xeon Gold 6244 @ 3.60 GHz, 32 threads, CPU governor `powersave` and not pinnable to `performance` without root.
The box was held exclusively for every timed run: never two engines at once, never alongside a build, 205–240 GiB free throughout.
Julia side: PauliPropagation.jl 0.8.2 on Julia 1.12.6, `PP_BACKEND=dict`, `PP_FUSED=0`, `-t1`.
Rust side: rustc 1.94.0, release profile (`lto = "fat"`, `codegen-units = 1`), Python 3.11.11, built 2026-09-01 03:19:44 for `4768fe4` and 2026-09-01 13:29:28 for `81c568a`.
All sweeps ran 2026-09-01.

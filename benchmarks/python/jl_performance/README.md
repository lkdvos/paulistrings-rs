# Head-to-head: `paulistrings` vs PauliPropagation.jl

Single-threaded, core versus core, on parity-gated configurations, under an interleaved-pair protocol. This
directory is the index for the study: the method, the current headline numbers, and one record per sweep.

Driver: `benchmarks/python/bench_jl_performance.py`. Figures: `benchmarks/python/jl_performance_figures.py`.
Every record directory carries `results.json` (one record per configuration per engine), `summary.json` (per-pair
ratios, crossovers, parity evidence), `run.log` (full transcript) and its figures; most also carry `tasks/`, the
schema-v1 task files both engines read. CI gate on the protocol logic:
`python/paulistrings/tests/test_jl_performance_protocol.py`.

## What is measured

One question only: at a given tracked-set size, which engine propagates a given circuit faster, and where the
ranking changes sign. Term counts, expectations and accuracy come from the parity gate, not from timing.

`ratio = t_julia / t_paulistrings` everywhere, in `results.json` (`ratio_jl_over_rust`), in the figures and in
every table: `> 1` means `paulistrings` is faster, `< 1` means PauliPropagation.jl is, `= 1` is the crossover.

## Method

1. One task file per configuration drives both engines. Julia reads it through `benchmarks/julia/runner.jl`, this
   engine through `paulistrings.interop.load_task`, so "both engines ran the same circuit" is a property of the
   file. The workload gate lists mirror `examples/common/circuits.py` gate for gate, pinned by CI.
2. A per-layer term-count parity gate precedes every timed configuration. Both engines propagate once, untimed,
   with per-gate counts collected (Julia via `@countpaulis`, this engine via its `layer {k}/{n}` DEBUG records);
   every count must match in application order, not just the final one. A mismatch disqualifies the configuration
   and no timing for it is reported.
3. The truncation boundary is made identical by moving a threshold and nothing else. This engine drops
   `|c| <= eps`, PauliPropagation.jl drops `|c| < eps`, so the Julia task carries `nextafter(eps, +inf)`: no float
   lies between, and the two rules become bit-identical. `min_abs_coeff = 0` is banned, since Julia keeps exact
   zeros and this engine's merge drops them.
4. Warm in-process timing, one process per leg. One untimed propagation, then one timed propagation, in the same
   process, so no number contains Julia's JIT or a cold cache. Construction, contraction, oracles and logging sit
   outside the timed region. Each leg runs at `RAYON_NUM_THREADS=1` (exported ahead of the interpreter, since
   Rayon sizes its global pool once) against Julia's `-t1`, with `RUST_LOG` unset.
5. Five interleaved pairs, `abba`, accepted on direction consistency. Within-pair order alternates, so drift in
   machine state cannot masquerade as a win for whichever engine always ran second. Every pair must agree on which
   engine was faster; if they do, the median ratio is the effect size. Mixed signs give the verdict
   `indistinguishable`, which is a tie and not a small win.
6. Each engine samples its own memory from `/proc/self/status` (`VmRSS` for the floor, `VmHWM` for the peak). A
   driver-side `getrusage(RUSAGE_CHILDREN)` is never used, since it is a running maximum over every reaped child
   and leaks one engine's peak into the other's number.

Resolution bar: single-shot campaign noise on this host is ±5–8 % single-threaded, and per-pair spreads run 3–60 %
at millisecond configurations. The pair protocol resolves the direction of a difference at that floor; quoted
magnitudes still carry the within-configuration spread, which the ratio figures draw pair by pair.

## Current numbers

Engine `81c568a`, default `engine="sorted"`, five pairs per configuration. Sources: `post-optimization/` and
`post-optimization/su4-curve/`.

| workload | channels | crossover (peak terms) |
|---|---|---|
| kicked-Ising, 127 q, 5 Trotter steps | 1 355 | **2.73 × 10³** (1.88 × 10³ with `engine="auto"`) |
| XXZ chain, n = 100, 6 Trotter steps | 1 782 | **2.00 × 10⁴** |
| Haar SU(4) brickwork, n = 36, depth 6 | 105 | **none on the swept range**, faster at every sign-consistent point |

The crossover is workload-specific by an order of magnitude, which is why no single global crossover is quoted
anywhere.

| workload | peak terms | ratio | rust ns/peak-term | jl ns/peak-term |
|---|---|---|---|---|
| kicked-Ising | 6.37 × 10⁵ | 2.146, the workload's peak | 1 097 | 2 406 |
| kicked-Ising | 2.15 × 10⁶ | 1.610 | 964 | 1 550 |
| XXZ | 2.66 × 10⁶ | 2.023, still rising | 5 361 | 10 840 |
| SU(4) | 2.30 × 10⁶ | 2.921, still rising | 2 268 | 6 638 |

Below the crossover PauliPropagation.jl is faster, by up to 3.6× at 68 terms, because a hash-map insert per term
costs little at 10² terms while the bucketed per-layer pipeline costs nearly the same whatever the term count. The
opt-in `engine="auto"` path is worth 1.08–2.69× on exactly those configurations.

Memory: process floors are 37.8 MB against Julia's 0.601 GiB, a factor of 16. At the largest SU(4) configuration,
the only one whose memory was sampled on this engine, peak RSS is 0.239 GiB against 1.625 GiB, which is 95
floor-subtracted bytes per peak term against 479. Rotation-workload memory in `post-optimization/` is joined from
this directory's own sweep and is flagged as such there.

## Records

| directory | sweep | engine `crates/` tree | driver commit |
|---|---|---|---|
| `post-optimization/` | kicked-Ising + XXZ curves, default engine, 5 pairs | `81c568a` | `0f00207` |
| `post-optimization/su4-curve/` | SU(4) curve, default engine, 5 pairs | `81c568a` | `0f00207` |
| `post-optimization-auto/` | loose end of all three curves with `engine="auto"`, 5 pairs | `81c568a` | `0f00207` |
| `deep-kicked-ising/` | kicked-Ising at 20 Trotter steps, 5 420 channels, 3 pairs | `4768fe4` | `e4aeccd` |
| `su4-curve/` | SU(4) curve, 5 pairs | `4768fe4` | `35ff414` |
| `.` (this directory) | kicked-Ising + XXZ curves, 5 pairs | `4768fe4` | — |

Each record's README states the build it measures and its own findings. Deltas between engine trees are recorded
in `research/notes/2026-09-01-jl-optimization-history.md`, which is internal and is not a source for a live
number; it also carries the Julia-side drift check that bounds them, per configuration (median −0.7 %, range
−4.3 % to +4.5 % over 21 configurations).

## Provenance

Host ccqlin038, 2 × Xeon Gold 6244 @ 3.60 GHz, 32 threads, CPU governor `powersave` and not pinnable to
`performance` without root. The box was held exclusively for every timed run: never two engines at once, never
alongside a build, 205–240 GiB free throughout.

Julia side: PauliPropagation.jl 0.8.2 on Julia 1.12.6, `PP_BACKEND=dict`, `PP_FUSED=0`, `-t1`. The experimental
fused rotation kernel is excluded because it truncates during gate application and has no established term-count
parity.

Rust side: rustc 1.94.0, release profile (`lto = "fat"`, `codegen-units = 1`), Python 3.11.11, extension
`python/paulistrings/_paulistrings.abi3.so`, built 2026-09-01 03:19:44 for `4768fe4` and 2026-09-01 13:29:28 for
`81c568a`. All sweeps ran 2026-09-01.

## Caveats

* One host, one governor, one Julia version, one backend. Absolute times drift between days; ratios measured
  adjacent in time do not, which is why the protocol is built on them.
* Single-threaded on both sides, and no thread-scaling claim is made. PauliPropagation.jl 0.8.2's dict backend has
  no threaded propagation path; its `VectorPauliSum` array backend takes a `thread` keyword but has no established
  term-count parity here, so it is out of scope rather than quietly substituted.
* `Float64` coefficients throughout. A complex observable roughly doubles Julia's coefficient storage.
* Heisenberg only, and only `min_abs_coeff` / `max_weight`, the two knobs both engines have. `topn` (here) and
  `max_freq` / `max_sins` (there) are excluded from comparative runs by construction.
* Five pairs supports a direction, not a confidence interval. Contraction is excluded; only propagation is timed.
* Time to fixed accuracy and thread scaling are not measured; the driver implements both, CI-tested, behind
  `--accuracy` and `--threads`.

## Reproducing

```bash
# the curves, default engine (~25 min per workload pair on a quiet 32-core host)
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization

# the sections never run, and a 1-pair pilot for checking the plumbing
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py --accuracy --threads --pairs 5
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py --curves --workload su4 --pilot

# re-render figures from committed data, no measurement
python benchmarks/python/jl_performance_figures.py benchmarks/python/jl_performance/summary.json

# the CI protocol gate (no julia, no timing, < 1 s)
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

The Julia side needs the pinned project in `benchmarks/julia/` (PauliPropagation.jl 0.8.2, Julia 1.12.6); the
first run precompiles for ~30 s. Everything degrades cleanly with no `julia` on `PATH` except the measurement
itself.

## Published form

These numbers appear on the documentation site as
[Against other tools](https://lkdvos.github.io/paulistrings-rs/comparisons.html), source
`docs/book/src/comparisons.md`, which cites this tree for every cross-engine figure it quotes.

# Curve sweeps on the default engine

Three workloads, 21 configurations, five `abba` pairs each, `engine="sorted"` (the default bucketed engine at
every term count). The protocol, ratio convention (`ratio = t_julia / t_paulistrings`, `> 1` means `paulistrings`
is faster), acceptance rule and caveats live in [`../README.md`](../README.md) and are not restated.

Engine `crates/` tree `81c568a`, extension built 2026-09-01 13:29:28, driver at `0f00207`. The SU(4) curve is a
separate driver invocation with its data in [`su4-curve/`](su4-curve/).

**Parity.** All 21 configurations passed the per-layer term-count gate: 1355 layers identical on every
kicked-Ising configuration, 1782 on every XXZ one, 105 on every SU(4) one, expectations agreeing to
≤ 2.6 × 10⁻¹⁶ against a 1e-9 bar. No configuration was disqualified.

## kicked-Ising, 127 q, 5 Trotter steps, `theta_h = 5pi/16`, 1355 channels

| `min_abs_coeff` | peak terms | rust s | jl s | rust ns/term | jl ns/term | ratio | pairs | faster |
|---|---|---|---|---|---|---|---|---|
| 2⁻⁴ | 68 | 0.0032 | 0.0009 | 47 059 | 13 235 | 0.281 | 5/5 | Julia |
| 2⁻⁶ | 517 | 0.0060 | 0.0038 | 11 605 | 7 350 | 0.638 | 5/5 | Julia |
| 2⁻⁸ | 6 311 | 0.0296 | 0.0372 | 4 690 | 5 894 | 1.253 | 5/5 | paulistrings |
| 2⁻¹⁰ | 79 029 | 0.1232 | 0.2001 | 1 559 | 2 532 | 1.651 | 5/5 | paulistrings |
| 2⁻¹² | 637 219 | 0.6992 | 1.5331 | 1 097 | 2 406 | 2.146 | 5/5 | paulistrings |
| 2⁻¹⁴ | 1 544 083 | 1.4924 | 2.9547 | 967 | 1 914 | 1.953 | 5/5 | paulistrings |
| 2⁻¹⁶ | 2 121 774 | 1.9941 | 3.2664 | 940 | 1 539 | 1.638 | 5/5 | paulistrings |
| 2⁻¹⁸ | 2 146 424 | 2.0697 | 3.3269 | 964 | 1 550 | 1.610 | 5/5 | paulistrings |
| 2⁻¹⁸ + `max_weight=6` | 712 † | 0.0042 | 0.0023 | 5 899 | 3 230 | 0.488 | 5/5 | Julia |

† peak terms not recoverable for this configuration; its final count is used, flagged in `summary.json` as
`peak_terms_source: unavailable`. The variant is excluded from the crossover bracket, since a crossover is only
meaningful along a single-parameter family.

**Crossover 2.73 × 10³ peak terms**, bracketed by 517 @ 0.638 and 6 311 @ 1.253.

The advantage is non-monotone: it peaks at 2.146 near 6.4 × 10⁵ terms and falls back to 1.610 at 2.1 × 10⁶. That
decay is Julia's per-term cost falling faster over the same range (2 406 → 1 550 ns, −36 %) than this engine's
(1 097 → 964 ns, −12 %), not anything degrading here. It is a property of a sum near closure rather than of large
`m`, which [`../deep-kicked-ising/README.md`](../deep-kicked-ising/README.md) isolates.

## XXZ chain, n = 100, `Jz = 0.5`, `dt = 0.1`, 6 Trotter steps, 1782 channels

| `min_abs_coeff` | peak terms | rust s | jl s | rust ns/term | jl ns/term | ratio | pairs | faster |
|---|---|---|---|---|---|---|---|---|
| 1e-2 | 164 | 0.0050 | 0.0022 | 30 488 | 13 415 | 0.440 | 5/5 | Julia |
| 1e-3 | 1 625 | 0.0502 | 0.0187 | 30 892 | 11 508 | 0.372 | 5/5 | Julia |
| 1e-4 | 9 918 | 0.1358 | 0.1141 | 13 692 | 11 504 | 0.873 | 5/5 | Julia |
| 1e-5 | 48 599 | 0.4337 | 0.5199 | 8 924 | 10 698 | 1.187 | 5/5 | paulistrings |
| 1e-6 | 206 035 | 1.2961 | 2.1442 | 6 291 | 10 407 | 1.654 | 5/5 | paulistrings |
| 1e-7 | 776 432 | 4.3993 | 8.3998 | 5 666 | 10 818 | 1.909 | 5/5 | paulistrings |
| 1e-8 | 2 661 873 | 14.269 | 28.854 | 5 361 | 10 840 | 2.023 | 5/5 | paulistrings |

**Crossover 2.00 × 10⁴ peak terms**, bracketed by 9 918 @ 0.873 and 48 599 @ 1.187. XXZ is nowhere near
saturation at 2.7 × 10⁶ terms: Julia's per-term cost is flat across the sweep, this engine's keeps falling, and
the ratio rises monotonically.

A tighter `1e-9` configuration reaching 8 473 952 terms was recorded at four pairs, below this study's five-pair
bar, so the recovery tool dropped it from `summary.json` and it enters no table, crossover or verdict. As
reconnaissance only: all 1782 per-layer counts identical, `|dE| = 7.6e-17`, rust 43.5–43.7 s against Julia
97.2–97.5 s, four pairs at 2.230 / 2.227 / 2.230 / 2.233. Parity therefore holds into the 10⁷-term regime, and
XXZ's ratio has not turned over by 8.5 × 10⁶ terms.

## SU(4) brickwork, n = 36, depth 6, 105 channels

Data in [`su4-curve/`](su4-curve/). This is the only workload driving the matrix-gate path — `unitary_2q` against
Julia's dense 16×16 `TransferMapGate` — and the only one whose `W = 1`, fanout-15 layers take the engine's gated
radix sort for dense-PTM gather runs.

| `min_abs_coeff` | peak terms | rust s | jl s | rust ns/term | jl ns/term | ratio | pairs | faster |
|---|---|---|---|---|---|---|---|---|
| 1e-2 | 1 416 | 0.00820 | 0.00900 | 5 791 | 6 357 | 1.097 | 5/5 | paulistrings |
| 3e-3 | 12 924 | 0.1479 | 0.0959 | 11 441 | 7 417 | 0.658 | 4/5 | indistinguishable |
| 1e-3 | 84 836 | 0.4524 | 0.6380 | 5 333 | 7 520 | 1.416 | 5/5 | paulistrings |
| 3e-4 | 573 826 | 1.8809 | 4.3859 | 3 278 | 7 643 | 2.326 | 5/5 | paulistrings |
| 1e-4 | 2 296 294 | 5.2084 | 15.242 | 2 268 | 6 638 | 2.921 | 5/5 | paulistrings |

**No crossover on this sweep.** Every sign-consistent configuration is paulistrings-faster, so the interpolation
has no bracket and the driver reports "no direction change across the swept range". The one mixed-sign point
(3e-3) reads 0.658 / 0.658 / 0.772 / 0.593 / 1.222 across its five pairs, which is a tie and not a small loss.

Memory here was measured rather than joined: 0.239 GiB peak at 2 296 294 terms, 95 floor-subtracted bytes per peak
term, against Julia's 1.625 GiB and 479 B/term.

## The small-`m` end

Below each crossover PauliPropagation.jl wins, by up to 3.6× at 68 terms. The reason is structural: a hash-map
insert per term costs little at 10² terms, while the bucketed rebucket → permute → coset-loop → unpermute pipeline
costs nearly the same whatever the term count, which the ns/term columns show directly (47 059 ns/term at 68 terms
against 964 at 2.1 × 10⁶). That fixed cost is what the opt-in `engine="auto"` direct-apply path avoids; it is
measured on exactly these configurations in
[`../post-optimization-auto/README.md`](../post-optimization-auto/README.md), and its numbers are a different
engine setting that is never mixed into a table here.

## Data provenance

`summary.json` and `results.json` in this directory were rebuilt from `run.log` by
`benchmarks/python/jl_performance_recover.py`, which imports the driver's protocol math rather than reimplementing
it and refuses to write when a recomputed median or verdict disagrees with the logged one. Ratios, medians,
verdicts, crossovers and parity evidence come from the five-pair run itself.

Two fields are joined from [`../summary.json`](../summary.json) and tagged `"source": "joined"`. `peak_terms` is
exact rather than approximate, since it is a deterministic function of circuit and cutoff, the parity gate proves
the per-layer count sequences identical, and every final count matches. The memory block is **not** a measurement
of this engine, and no rotation-workload memory figure is quoted above; a memory rerun of the rotation workloads
is a named follow-up. The SU(4) directory is fully driver-written, with nothing recovered or joined.

Host ccqlin038 (2 × Xeon Gold 6244, `powersave`, 205–240 GiB free), Julia 1.12.6 with PauliPropagation.jl 0.8.2,
`PP_BACKEND=dict`, `PP_FUSED=0`, rustc 1.94.0, Python 3.11.11, `RAYON_NUM_THREADS=1`, `RUST_LOG` unset.

## Reproducing

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization

# re-render figures from the committed data, no measurement
python benchmarks/python/jl_performance_figures.py \
    benchmarks/python/jl_performance/post-optimization/summary.json

# the CI protocol gate (no julia, no timing, < 1 s)
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

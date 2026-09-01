# SU(4) brickwork, post-optimization — data directory

The narrative lives one level up, in [`../README.md`](../README.md) §"SU(4) brickwork". This is the
data and the reason it sits in its own directory.

**Why separate.** The Part A invocation swept `kicked_ising`, `xxz` and `su4` together, but the agent
harness killed the driver during the XXZ `1e-9` configuration, before `su4` ran. The kicked-Ising and
XXZ curves were recovered from `../run.log` with the study's own
`jl_performance_recover.py`; the SU(4) curve was simply re-run, as its own invocation, into here.
That mirrors the baseline study's own layout ([`../../su4-curve/`](../../su4-curve/)) and has one
advantage over the recovered directory: **everything here is driver-written, so the memory numbers
and `peak_terms` are measured rather than joined.**

**Result in one line.** Ratio **1.974 → 2.921** at 2 296 294 peak terms, and the crossover
**8.01 × 10⁴ → none** — every sign-consistent configuration on this sweep is now paulistrings-faster.
Our rust leg moved −6.0 / −10.6 / −26.7 / −28.5 / −33.3 % across the sweep, which is E2's gated
`W = 1` dense-PTM cell (−33.2 / −33.8 %) showing up end-to-end. Julia moved −1.7 % to +1.5 %.

| `min_abs_coeff` | peak terms | rust s → | jl s → | ratio → | pairs | verdict |
|---|---|---|---|---|---|---|
| 1e-2 | 1 416 | 0.00917 → 0.00820 | 0.00909 → 0.00900 | 0.983 → **1.097** | 5/5 | paulistrings (was a tie) |
| 3e-3 | 12 924 | 0.1574 → 0.1479 | 0.0975 → 0.0959 | 0.620 → **0.658** | 4/5 | indistinguishable (was Julia) |
| 1e-3 | 84 836 | 0.6171 → 0.4524 | 0.6286 → 0.6380 | 1.027 → **1.416** | 5/5 | paulistrings (was a tie) |
| 3e-4 | 573 826 | 2.630 → 1.881 | 4.417 → 4.386 | 1.676 → **2.326** | 5/5 | paulistrings |
| 1e-4 | 2 296 294 | 7.806 → 5.208 | 15.46 → 15.24 | 1.974 → **2.921** | 5/5 | paulistrings |

Parity held at all five: 105 layers identical each, |dE| ≤ 1.25 × 10⁻¹⁶. Memory 0.239 GiB /
95 B-per-peak-term at the largest point, against the study's 0.235 GiB / 93 — unchanged, as a sort
kernel swap should leave it.

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization/su4-curve
```

Run took **12.7 min**. Host, toolchain and extension provenance as in [`../README.md`](../README.md).

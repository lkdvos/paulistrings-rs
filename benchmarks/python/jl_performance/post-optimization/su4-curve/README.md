# SU(4) brickwork curve, default engine — data directory

The reading of these numbers is one level up, in [`../README.md`](../README.md) §"SU(4) brickwork". This
directory holds the sweep itself, run as its own driver invocation, so everything here is driver-written and both
`peak_terms` and the memory numbers are measured rather than joined.

Haar-random SU(4) brickwork, n = 36, depth 6, seed 20260831, 105 channels, `Z_18` against `z+`, Heisenberg. Five
`abba` pairs per configuration, `engine="sorted"`.

| `min_abs_coeff` | peak terms | rust s | jl s | ratio | pairs | verdict |
|---|---|---|---|---|---|---|
| 1e-2 | 1 416 | 0.00820 | 0.00900 | 1.097 | 5/5 | paulistrings |
| 3e-3 | 12 924 | 0.1479 | 0.0959 | 0.658 | 4/5 | indistinguishable |
| 1e-3 | 84 836 | 0.4524 | 0.6380 | 1.416 | 5/5 | paulistrings |
| 3e-4 | 573 826 | 1.8809 | 4.3859 | 2.326 | 5/5 | paulistrings |
| 1e-4 | 2 296 294 | 5.2084 | 15.242 | 2.921 | 5/5 | paulistrings |

Parity held at all five configurations: 105 layers identical each, `|dE| ≤ 1.25 × 10⁻¹⁶`. No crossover on the
swept range. Memory at the largest configuration is 0.239 GiB and 95 bytes per peak term against Julia's
1.625 GiB and 479.

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization/su4-curve
```

Run took 12.7 min. Host, toolchain and extension provenance as in [`../README.md`](../README.md).

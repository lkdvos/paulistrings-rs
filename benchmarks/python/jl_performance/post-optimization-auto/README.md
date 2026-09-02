# The small-`m` end with `engine="auto"`

The loose end of all three curves, measured with the opt-in direct-apply path enabled. Nine configurations, five
`abba` pairs each. The protocol, ratio convention (`ratio = t_julia / t_paulistrings`, `> 1` means `paulistrings`
is faster), acceptance rule and caveats live in [`../README.md`](../README.md).

This is a different engine setting, not a better measurement of the default one, so it gets its own directory and
its own tables. `summary.json`'s `protocol.rust_engine` and every record in `results.json` carry `"auto"`, and
`run.log` prints it per curve. The default-engine sweep of the same configurations is
[`../post-optimization/README.md`](../post-optimization/README.md), same binary and same day, differing in one
kwarg.

## What the setting does

`PauliSum.propagate` takes an `engine=` kwarg that selects the layer engine and defaults to `"sorted"`, the
bucketed sorting engine at every term count. `"auto"` additionally lets the small-sum direct path take the leading
layers while the sum is within `paulistrings.DEFAULT_SMALL_SUM_THRESHOLD` (2048) and the policy has no layer pass.
That path applies each layer term by term into a hash map, skipping the bucketed
rebucket → permute → coset-loop pipeline and, more importantly, `Channel::prepare`, which the phase breakdown puts
at 70–95 % of the per-layer fixed cost. The transition is one-way: once a layer leaves the sum above the
threshold, the rest of the circuit runs on the sorting engine.

`--max-configs 3` keeps the three loosest cutoffs of each curve and drops the kicked-Ising `max_weight` variant,
which sits at the tightest. Above the threshold the setting is inert, and the sweep carries its own control for
that: SU(4) at 84 836 peak terms reads 1.409 with the path on against 1.416 with it off, a 0.5 % difference on a
configuration whose sum leaves the threshold in its first few layers.

## Results

| workload | `min_abs_coeff` | peak terms | rust s | jl s | ratio | pairs | verdict |
|---|---|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 68 | 0.00194 | 0.00091 | 0.490 | 5/5 | Julia |
| kicked_ising | 2⁻⁶ | 517 | 0.00542 | 0.00378 | 0.696 | 4/5 | indistinguishable |
| kicked_ising | 2⁻⁸ | 6 311 | 0.02738 | 0.03599 | 1.297 | 5/5 | paulistrings |
| xxz | 1e-2 | 164 | 0.00340 | 0.00224 | 0.649 | 5/5 | Julia |
| xxz | 1e-3 | 1 625 | 0.01864 | 0.01921 | 1.040 | 3/5 | indistinguishable |
| xxz | 1e-4 | 9 918 | 0.10991 | 0.11549 | 1.051 | 4/5 | indistinguishable |
| su4 | 1e-2 | 1 416 | 0.00558 | 0.00905 | 1.660 | 5/5 | paulistrings |
| su4 | 3e-3 | 12 924 | 0.12182 | 0.09775 | 0.802 | 5/5 | Julia |
| su4 | 1e-3 | 84 836 | 0.45397 | 0.63360 | 1.409 | 5/5 | paulistrings |

Three configurations are ties whose five pairs straddle 1: `xxz 1e-3` at 0.944–1.046, `xxz 1e-4` at 0.967–1.126,
`kicked_ising 2⁻⁶` at 0.637–1.012. A mixed-sign verdict is a tie and not a small win, favourable direction
notwithstanding. Every pair of every configuration is in `results.json` under `ratio_jl_over_rust_per_pair`;
per-pair spreads run 5.1 % at the tightest SU(4) point to 58.9 % at `kicked_ising 2⁻⁶`.

## What the path is worth

Same binary, same day, same box, differing only in the kwarg. Both legs are single campaigns rather than an
interleaved A/B, so the ±5–8 % floor applies; every figure below is far outside it.

| configuration | peak terms | rust s, `sorted` | rust s, `auto` | speedup | ratio, `sorted` → `auto` |
|---|---|---|---|---|---|
| xxz 1e-3 | 1 625 | 0.0502 | 0.01864 | **2.69×** | 0.372 → 1.040 |
| kicked_ising 2⁻⁴ | 68 | 0.0032 | 0.00194 | 1.65× | 0.281 → 0.490 |
| xxz 1e-2 | 164 | 0.0050 | 0.00340 | 1.47× | 0.440 → 0.649 |
| su4 1e-2 | 1 416 | 0.00820 | 0.00558 | 1.47× | 1.097 → 1.660 |
| xxz 1e-4 | 9 918 | 0.1358 | 0.1099 | 1.24× | 0.873 → 1.051 |
| su4 3e-3 | 12 924 | 0.1479 | 0.1218 | 1.21× | 0.658 → 0.802 |
| kicked_ising 2⁻⁶ | 517 | 0.0060 | 0.00542 | 1.11× | 0.638 → 0.696 |
| kicked_ising 2⁻⁸ | 6 311 | 0.0296 | 0.02738 | 1.08× | 1.253 → 1.297 |
| su4 1e-3 | 84 836 | 0.4524 | 0.4540 | 1.00× | 1.416 → 1.409 |

The shape is the one the path's design predicts: largest where the whole sum stays under the threshold for the
whole circuit, tapering to nothing above it. The taper is visible in the three partial configurations —
`kicked_ising 2⁻⁸` (peak 6 311), `xxz 1e-4` (9 918) and `su4 3e-3` (12 924) all cross the 2048 threshold partway
through and keep only the leading layers' saving, worth 1.08–1.24×.

## Crossovers with the path enabled

| workload | crossover (peak terms) |
|---|---|
| `kicked_ising` | **1.88 × 10³**, bracketed by 68 @ 0.490 and 6 311 @ 1.297 |
| `su4` | 6.62 × 10³, bracketed by 12 924 @ 0.802 and 84 836 @ 1.409 |
| `xxz` | not localizable on this sweep |

Both of the interpolated numbers belong to a three-cutoff sweep and should be read as such. SU(4)'s 6.62 × 10³ is
not comparable with the full five-cutoff sweep's "no crossover", which comes from a range where every
sign-consistent point is a win; both statements are true of their own sweep. XXZ has no bracket here at all, since
one of its three configurations is sign-consistent (1e-2, Julia) and two are indistinguishable, so the
interpolation correctly declines. What the data says instead is stronger than a number: with the path on, XXZ is
not losing anywhere above 1.6 × 10³ terms, so its crossover lies in the 1.6 × 10³–1 × 10⁴ band and this grid
cannot localize it further.

## Parity with the path taken

The parity gate takes the same `--engine` as the timed legs, so a layer engine that changed per-layer counts
disqualifies its own configuration instead of being timed. All nine configurations passed with the direct path
enabled and actually taken:

| workload | layers compared, per configuration | configurations | max \|dE\| |
|---|---|---|---|
| `kicked_ising` | 1355 | 3 | 0.0 |
| `xxz` | 1782 | 3 | 6.6 × 10⁻¹⁷ |
| `su4` | 105 | 3 | 9.0 × 10⁻¹⁷ |

**9 618 per-layer term counts, every one identical to PauliPropagation.jl's**, against a 1e-9 expectation bar.
Final term counts match the default engine's configuration for configuration (7 / 408 / 5 038, 156 / 1 625 /
9 918, 193 / 7 089 / 84 836), so the path computes the same sum by a different route, on circuits of 105 to 1782
channels, at both `W = 1` and `W = 2`, across fully-resident and partially-resident sums.

## Caveats specific to this sweep

* `auto` is not the default, and nothing here proposes flipping it. The threshold (2048) and the one-way
  transition are unchanged.
* `auto` declines a policy with a layer pass (`topn`, `approx_topn`, or an `&` containing one). Every
  configuration here uses `min_abs_coeff` only, so the path was available at all nine; a `topn` run would measure
  `sorted` under this same flag.
* Nine configurations, one host, one day. The setting comparison is two campaigns 20 minutes apart differing in
  one kwarg, not an interleaved A/B of two binaries. The effects quoted are 1.08–2.69×, far outside the noise
  floor, which is why they are quoted at all.

## Reproducing

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 \
    --engine auto --max-configs 3 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization-auto

# re-render figures from the committed data, no measurement
python benchmarks/python/jl_performance_figures.py \
    benchmarks/python/jl_performance/post-optimization-auto/summary.json

# the CI protocol gate, which pins --engine's default and --max-configs' prefix rule
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

Host ccqlin038 (2 × Xeon Gold 6244, `powersave`), Julia 1.12.6 with PauliPropagation.jl 0.8.2, `PP_BACKEND=dict`,
`PP_FUSED=0`, rustc 1.94.0, engine `crates/` tree `81c568a` (extension mtime 2026-09-01 13:29:28), driver commit
`0f00207` with a clean tree. Run took 11.6 min.

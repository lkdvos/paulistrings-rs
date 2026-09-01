# Part B — the small-`m` end with the direct-apply path enabled

> ## ✅ The regime the study lost outright is now a tie or a win, and E3's outstanding gate (c) passes
>
> Turning on `engine="auto"` — E3's opt-in direct-apply path for sums below
> `small_sum_threshold` (2048) — takes the six small configurations the study lost and turns
> **three into ties and one into a win**, without touching a single term count.
>
> | | before (study) | after, default engine | **after, `engine="auto"`** |
> |---|---|---|---|
> | kicked-Ising crossover | 3.79 × 10³ | 2.73 × 10³ | **1.88 × 10³** |
> | XXZ at 1 625 terms | 0.453, Julia | 0.372, Julia | **1.040, indistinguishable** |
> | XXZ at 9 918 terms | 0.895, Julia | 0.873, Julia | **1.051, indistinguishable** |
> | SU(4) at 1 416 terms | 0.983, tie | 1.097 | **1.660** |
> | SU(4) crossover | 8.01 × 10⁴ | none on that sweep | **6.6 × 10³** |
>
> **And the gate: all 9 configurations passed the per-layer parity gate with the direct path
> enabled** — 1355 / 1782 / 105 layers identical against PauliPropagation.jl, expectations
> ≤ 9.0 × 10⁻¹⁷. That is E3's outstanding gate (c), run against an independent engine on real
> circuits instead of the two ours-only substitutes its note had to use.

Part A, the protocol-identical rerun on the **default** engine, is
[`../post-optimization/README.md`](../post-optimization/README.md); the parent study is
[`../README.md`](../README.md). Their protocol, ratio convention
(`ratio = t_julia / t_paulistrings`, > 1 means we are faster), acceptance rule and caveats apply
here unchanged. This document reports only what enabling the path adds.

## What `--engine auto` is, and why this is a separate directory

`PauliSum.propagate` grew an `engine=` kwarg in `81c568a`. It selects the layer engine and defaults
to `"sorted"` — the bucketed sorting engine at every term count, which is what every number in the
parent study, in Part A, and in every other committed result file measures.

`"auto"` additionally lets the **small-sum direct path** take the leading layers while the sum is
within `paulistrings.DEFAULT_SMALL_SUM_THRESHOLD` (2048) *and* the policy has no layer pass. That
path applies each layer term by term into a hash map, skipping the bucketed
rebucket → permute → coset-loop pipeline and, more importantly, `Channel::prepare` — which the
campaign's phase breakdown found to be 70–95 % of the per-layer fixed cost. The transition is
one-way: once a layer leaves the sum above the threshold, the rest of the circuit runs on the
sorting engine.

This is **a different engine setting, not a better measurement of the same one**, so it gets its own
directory and its own tables. A Part B number must never be quoted beside a Part A number as though
the two were the same configuration.

The driver records the setting: `summary.json`'s `protocol.rust_engine` and every record in
`results.json` carry `"auto"`, and `run.log` prints it per curve.

## Scope: the loose end of all three workloads

`--max-configs 3` keeps the three loosest cutoffs of each curve (and drops the kicked-Ising
`max_weight` variant, which sits at the tightest one). Those nine configurations are exactly the
regime in question — every one of them is a configuration the study **lost or tied**:

| workload | cutoffs run | study verdict at each |
|---|---|---|
| `kicked_ising` | 2⁻⁴, 2⁻⁶, 2⁻⁸ | Julia, Julia, paulistrings by 1.126 |
| `xxz` | 1e-2, 1e-3, 1e-4 | Julia, Julia, Julia |
| `su4` | 1e-2, 3e-3, 1e-3 | tie, **Julia (the study's deepest deficit)**, tie |

The SU(4) leg is an extension of the handoff's named scope (which listed kicked-Ising and XXZ). It
was added because all three of its loose cutoffs are lost-or-tied configurations, it contains the
study's worst point (0.620 at 1.29 × 10⁴), and the three of them cost ~4 min. Logged in
`research/notes/2026-09-01-large-m-campaign-log.md`.

Above the threshold `auto` is inert, and the sweep contains its own control for that: **SU(4) at
84 836 peak terms reads 1.409 on `auto` against 1.416 on `sorted`** — a 0.5 % difference on a
configuration whose sum leaves the threshold in its first few layers. That is what "inert" is
supposed to look like.

## Results

5 pairs each, `abba`, acceptance on direction consistency. Ratios are `t_julia / t_paulistrings`.

| workload | `min_abs_coeff` | peak terms | rust s: study → A → **B** | jl s (B) | ratio: study → A → **B** | pairs | verdict (B) |
|---|---|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 68 | 0.0031 → 0.0032 → **0.00194** | 0.00091 | 0.323 → 0.281 → **0.490** | 5/5 | Julia |
| kicked_ising | 2⁻⁶ | 517 | 0.0062 → 0.0060 → **0.00542** | 0.00378 | 0.629 → 0.638 → **0.696** | 4/5 | **indistinguishable** |
| kicked_ising | 2⁻⁸ | 6 311 | 0.0326 → 0.0296 → **0.02738** | 0.03599 | 1.126 → 1.253 → **1.297** | 5/5 | paulistrings |
| xxz | 1e-2 | 164 | 0.0050 → 0.0050 → **0.00340** | 0.00224 | 0.460 → 0.440 → **0.649** | 5/5 | Julia |
| xxz | 1e-3 | 1 625 | 0.0433 → 0.0502 → **0.01864** | 0.01921 | 0.453 → 0.372 → **1.040** | 3/5 | **indistinguishable** |
| xxz | 1e-4 | 9 918 | 0.1287 → 0.1358 → **0.1099** | 0.1155 | 0.895 → 0.873 → **1.051** | 4/5 | **indistinguishable** |
| su4 | 1e-2 | 1 416 | 0.00917 → 0.00820 → **0.00558** | 0.00905 | 0.983 → 1.097 → **1.660** | 5/5 | paulistrings |
| su4 | 3e-3 | 12 924 | 0.1574 → 0.1479 → **0.1218** | 0.09775 | 0.620 → 0.658 → **0.802** | 5/5 | Julia |
| su4 | 1e-3 | 84 836 | 0.6171 → 0.4524 → **0.4540** | 0.6336 | 1.027 → 1.416 → **1.409** | 5/5 | paulistrings |

Per-pair ratios, since the acceptance rule is about their agreement in sign:

| configuration | pair 0 | pair 1 | pair 2 | pair 3 | pair 4 | spread |
|---|---|---|---|---|---|---|
| kicked_ising 2⁻⁴ | 0.470 | 0.668 | 0.538 | 0.466 | 0.490 | 43.2 % |
| kicked_ising 2⁻⁶ | 0.861 | 0.637 | 0.684 | 0.696 | **1.012** | 58.9 % |
| kicked_ising 2⁻⁸ | 1.326 | 1.297 | 1.180 | 1.337 | 1.275 | 13.3 % |
| xxz 1e-2 | 0.508 | 0.776 | 0.649 | 0.720 | 0.564 | 52.9 % |
| xxz 1e-3 | **1.046** | 0.974 | **1.041** | 0.944 | **1.040** | 10.8 % |
| xxz 1e-4 | **1.126** | **1.051** | 0.967 | **1.051** | **1.067** | 16.5 % |
| su4 1e-2 | 1.572 | 1.463 | 1.897 | 1.743 | 1.660 | 29.7 % |
| su4 3e-3 | 0.793 | 0.946 | 0.829 | 0.764 | 0.802 | 23.8 % |
| su4 1e-3 | 1.381 | 1.450 | 1.385 | 1.442 | 1.409 | 5.1 % |

**Three configurations became `indistinguishable` by improving**, which is the honest description of
what happened to them: each crossed from "Julia is faster, unanimously" to "the five pairs disagree
about who is faster". `xxz 1e-3` is the clearest — its five pairs sit at 0.944–1.046, a 10.8 %
spread straddling 1, where the study's five sat unanimously at 0.453.

## How much the path is worth, on its own

Isolating E3 means comparing Part B against **Part A**, not against the study — same binary, same
day, same box, differing only in the kwarg. Both are single campaigns rather than an interleaved
A/B, so the ±5–8 % floor applies; every figure below is far outside it.

| configuration | peak terms | rust A (`sorted`) | rust B (`auto`) | **speedup from the path** |
|---|---|---|---|---|
| xxz 1e-3 | 1 625 | 0.0502 | 0.01864 | **2.69×** |
| kicked_ising 2⁻⁴ | 68 | 0.0032 | 0.00194 | **1.65×** |
| xxz 1e-2 | 164 | 0.0050 | 0.00340 | **1.47×** |
| su4 1e-2 | 1 416 | 0.00820 | 0.00558 | **1.47×** |
| xxz 1e-4 | 9 918 | 0.1358 | 0.1099 | **1.24×** |
| su4 3e-3 | 12 924 | 0.1479 | 0.1218 | **1.21×** |
| kicked_ising 2⁻⁸ | 6 311 | 0.0296 | 0.02738 | **1.08×** |
| kicked_ising 2⁻⁶ | 517 | 0.0060 | 0.00542 | **1.11×** |
| su4 1e-3 | 84 836 | 0.4524 | 0.4540 | 1.00× (inert, as designed) |

E3's own gate predicted **2.28–2.36×** on kicked-Ising 2⁻⁴ and **1.55–1.68×** on the XXZ small
points, measured as engine-only wall time. What this table measures is the *end-to-end propagation*,
which also carries construction-adjacent per-call cost the path cannot remove, so the numbers being
somewhat lower on kicked-Ising (1.65× against 2.28×) and higher on XXZ 1e-3 (2.69× against 1.68×) is
expected rather than contradictory. The shape is the one the gate found: largest where the whole sum
stays under the threshold for the whole circuit, tapering to nothing above it.

The three partial configurations are visible in that taper — `kicked_ising 2⁻⁸` (peak 6 311),
`xxz 1e-4` (9 918) and `su4 3e-3` (12 924) all cross the 2048 threshold partway through and keep only
the leading layers' saving, worth 1.08–1.24×.

## Crossovers with the path enabled

| workload | study | Part A (`sorted`) | **Part B (`auto`)** |
|---|---|---|---|
| `kicked_ising` | 3.79 × 10³ | 2.73 × 10³ | **1.88 × 10³** |
| `su4` | 8.01 × 10⁴ | none on the full sweep | **6.62 × 10³** |
| `xxz` | 1.65 × 10⁴ | 2.00 × 10⁴ | **not localizable — see below** |

Two of these need reading carefully, because a truncated sweep changes what a crossover
interpolation can say:

* **`su4`'s 6.62 × 10³ is bracketed inside this run** (12 924 @ 0.802 and 84 836 @ 1.409) and is a
  real number for the three-cutoff sweep. It is *not* comparable with Part A's "none", which comes
  from the **full** five-cutoff sweep where every sign-consistent point is a win. Both statements are
  true of their own sweep.
* **`xxz` has no bracket at all here**: of its three configurations one is sign-consistent (1e-2,
  Julia) and two are indistinguishable, so the interpolation correctly declines. What the data does
  say is stronger than a number — with the path on, XXZ is **no longer losing** anywhere above
  1.6 × 10³ terms, so its crossover has moved down into the 1.6 × 10³–1 × 10⁴ band and this sweep
  cannot localize it further. Tightening the grid there is the follow-up.

## Parity — E3's gate (c), measured against an independent engine

E3's note (`research/notes/2026-09-01-small-m-path.md` §5.3, §6(c)) could only run two ours-only
checks: cross-path Rust tests, and the harness's own parity assert. Its gate (c) against
PauliPropagation.jl was left outstanding because the driver had no way to ask for the path. It does
now, and the gate takes the same `--engine` as the timed legs by construction — a layer engine that
changed per-layer counts disqualifies its own configuration instead of being timed.

**All nine configurations passed**, with the direct path enabled and actually taken:

| workload | layers compared, per configuration | configurations | max \|dE\| |
|---|---|---|---|
| `kicked_ising` | 1355 | 3 | 0.0 |
| `xxz` | 1782 | 3 | 6.6 × 10⁻¹⁷ |
| `su4` | 105 | 3 | 9.0 × 10⁻¹⁷ |

**9 618 per-layer term counts, every one identical to PauliPropagation.jl's**, against a 1e-9
expectation bar. Final term counts are identical to Part A's and to the study's, configuration for
configuration — 7 / 408 / 5 038, 156 / 1 625 / 9 918, 193 / 7 089 / 84 836 — so the path really does
compute the same sum by a different route, on circuits of 105 to 1782 channels, at both `W = 1` and
`W = 2`, across both fully-resident and partially-resident sums.

Gate (c): **PASS**.

## Caveats specific to this section

* **`auto` is still not the default**, and nothing here proposes flipping it. The threshold (2048)
  and the one-way transition are E3's, unchanged.
* **`auto` declines a policy with a layer pass** (`topn`, `approx_topn`, or an `&` containing one).
  Every configuration here uses `min_abs_coeff` only, so the path was available at all nine; a
  `topn` run would measure `sorted` under this same flag.
* **The three `indistinguishable` verdicts are ties, not wins.** The README of the parent study is
  explicit that a mixed-sign configuration is "not a small win, not a trend, not something to average
  over", and that applies to these three exactly as written, favourable direction notwithstanding.
* **Nine configurations, one host, one day.** The Part A → Part B comparison is not an interleaved
  A/B of two binaries; it is two campaigns 20 minutes apart differing in one kwarg. The effects
  quoted are 1.08–2.69×, far outside the noise floor, which is why they are quoted at all.

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

Host ccqlin038 (2 × Xeon Gold 6244, `powersave`), Julia 1.12.6 + PauliPropagation.jl 0.8.2,
`PP_BACKEND=dict`, `PP_FUSED=0`, rustc 1.94.0, extension mtime 2026-09-01 13:29:28, driver commit
`0f00207` with a clean tree. Run took **11.6 min**.

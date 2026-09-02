# Cross-engine study: optimization history of the `paulistrings` side

**What this records.** The before/after material from the `paulistrings` vs PauliPropagation.jl head-to-head study
tree: how this engine's numbers moved across the 2026-09-01 large-`m` optimization campaign, the campaign's own
deviations, and the narrative arc of the saturation hypothesis. Internal note. Nothing here is a current-state
claim, and nothing in `docs/book/` should cite it for a live number.

**Where it came from.** Extracted 2026-09-01 from `benchmarks/python/jl_performance/README.md`,
`post-optimization/README.md`, `post-optimization-auto/README.md`, `post-optimization/su4-curve/README.md`,
`deep-kicked-ising/README.md` and `su4-curve/README.md` when those six were rewritten as current-state study
records. The measurements themselves stay in the tracked `results.json` / `summary.json` / `run.log` beside each
README; only the cross-version framing moved here.

**Why the framing had to move.** The tree is a set of records, one per engine build, and each README now says what
its own build measures. A before/after column in a study README goes stale the next time the engine moves and
invites quoting a delta as if it were a measurement. The deltas are real and worth keeping — they are just not
current state.

## The two engine builds

| build | extension mtime | `crates/` tree | measured by |
|---|---|---|---|
| pre-campaign | 2026-09-01 03:19:44 | `4768fe4` (built from `9d43886`, whose `crates/` tree is identical) | `jl_performance/` (kicked-Ising + XXZ), `su4-curve/`, `deep-kicked-ising/` |
| post-campaign | 2026-09-01 13:29:28 | `81c568a` | `post-optimization/`, `post-optimization/su4-curve/`, `post-optimization-auto/` |

Julia was identical across both: PauliPropagation.jl 0.8.2 on Julia 1.12.6, `PP_BACKEND=dict`, `PP_FUSED=0`, `-t1`.
Same host (ccqlin038), same task files, same protocol, same day.

## What the campaign merged

| | what it does | where it can act on this study |
|---|---|---|
| E1 — squared-magnitude truncation (`2026-09-01-topn-finalize.md`) | `norm()` → `norm_sqr()` on the truncation path; `\|c\| > t ⟺ \|c\|² > t²` removes a `hypot` per term. Gated at −26 % on `CoefficientThreshold` layers | every configuration; all 21 use `min_abs_coeff`, and the effect is per term, so it grows with `m` |
| E2 — gated radix sort for dense-PTM gather runs (`2026-09-01-sort-kernel.md`) | replaces the comparison sort for two-qubit dense PTMs (fanout ≥ 8). Gated at −10.5…−33.8 % on the layer, strongest at `W = 1` | `su4` only: `unitary_2q`, fanout 15, `n = 36` → `W = 1`, E2's strongest cell |
| E3 — opt-in small-sum direct-apply path (`2026-09-01-small-m-path.md`) | applies layers term by term into a hash map below `small_sum_threshold` (2048), skipping the bucketed pipeline. Off by default | the `engine="auto"` sweep only; the default `"sorted"` engine leaves E3 inert by construction |

`git diff 81c568a HEAD -- crates/` was empty at run time, so the post-campaign binary is a faithful build of the
campaign tip.

## Headline movement

| | kicked-Ising | XXZ | SU(4) |
|---|---|---|---|
| crossover, pre | 3.79 × 10³ | 1.65 × 10⁴ | 8.01 × 10⁴ |
| crossover, post | 2.73 × 10³ (−28 %) | 2.00 × 10⁴ (+21 %, unresolved) | none on the sweep |
| best ratio, pre | 1.925 | 1.798 | 1.974 |
| best ratio, post | 2.146 | 2.023 | 2.921 |

With `engine="auto"` the kicked-Ising crossover moves further, to 1.88 × 10³.

## How to read a delta on this page

The **ratio** at each configuration is protocol-grade: its two legs ran adjacent in time, `abba` across five pairs,
accepted on direction consistency.

The **pre → post delta** is not. It compares two campaigns hours apart, which is what `benchmarks/PROFILING.md`'s
±5–8 % single-threaded noise floor forbids for small effects. So a delta above ~8 % whose per-pair ranges do not
overlap is a change; a delta below that, or one inside its own run's per-pair spread, is unresolved and is
attributed to nothing. Nothing here is an `ab-compare` of two binaries.

## Per-configuration deltas, default engine

kicked-Ising, 127 q, 5 Trotter steps, 1355 channels:

| `min_abs_coeff` | peak terms | rust s pre → post | jl s pre → post | ratio pre → post | Δratio |
|---|---|---|---|---|---|
| 2⁻⁴ | 68 | 0.0031 → 0.0032 | 0.0009 → 0.0009 | 0.323 → 0.281 | −12.8 % ‡ |
| 2⁻⁶ | 517 | 0.0062 → 0.0060 | 0.0038 → 0.0038 | 0.629 → 0.638 | +1.4 % ‡ |
| 2⁻⁸ | 6 311 | 0.0326 → 0.0296 | 0.0367 → 0.0372 | 1.126 → 1.253 | +11.3 % |
| 2⁻¹⁰ | 79 029 | 0.1427 → 0.1232 | 0.1932 → 0.2001 | 1.362 → 1.651 | +21.2 % |
| 2⁻¹² | 637 219 | 0.798 → 0.699 | 1.544 → 1.533 | 1.925 → 2.146 | +11.5 % |
| 2⁻¹⁴ | 1 544 083 | 1.761 → 1.492 | 2.997 → 2.955 | 1.690 → 1.953 | +15.5 % |
| 2⁻¹⁶ | 2 121 774 | 2.367 → 1.994 | 3.346 → 3.266 | 1.431 → 1.638 | +14.5 % |
| 2⁻¹⁸ | 2 146 424 | 2.404 → 2.070 | 3.364 → 3.327 | 1.389 → 1.610 | +15.9 % |
| 2⁻¹⁸ + `max_weight=6` | 712 | 0.0050 → 0.0042 | 0.0022 → 0.0023 | 0.448 → 0.488 | +8.8 % ‡ |

‡ unresolved: per-pair spreads of 30–83 % at these millisecond configurations swamp the delta.

Crossover 3.79 × 10³ → 2.73 × 10³, −28 %, bracketed in both campaigns by the same two configurations (517 and
6 311 peak terms). The movement is carried by the 6 311-term point, whose +11.3 % is resolved; the 517-term point
moved +1.4 %, which is not.

XXZ, n = 100, 6 Trotter steps, 1782 channels:

| `min_abs_coeff` | peak terms | rust s pre → post | jl s pre → post | ratio pre → post | Δratio |
|---|---|---|---|---|---|
| 1e-2 | 164 | 0.0050 → 0.0050 | 0.0023 → 0.0022 | 0.460 → 0.440 | −4.3 % ‡ |
| 1e-3 | 1 625 | 0.0433 → 0.0502 | 0.0191 → 0.0187 | 0.453 → 0.372 | −17.9 % ‡ |
| 1e-4 | 9 918 | 0.1287 → 0.1358 | 0.1146 → 0.1141 | 0.895 → 0.873 | −2.5 % ‡ |
| 1e-5 | 48 599 | 0.4074 → 0.4337 | 0.5234 → 0.5199 | 1.264 → 1.187 | −6.1 % ◊ |
| 1e-6 | 206 035 | 1.460 → 1.296 | 2.104 → 2.144 | 1.438 → 1.654 | +15.0 % |
| 1e-7 | 776 432 | 5.018 → 4.399 | 8.399 → 8.400 | 1.682 → 1.909 | +13.5 % |
| 1e-8 | 2 661 873 | 16.34 → 14.27 | 29.36 → 28.85 | 1.798 → 2.023 | +12.5 % |

‡ unresolved, inside the per-pair spread (1e-3's own five pairs span 89.8 %).
◊ marginal: the only small-`m` point whose pre and post rust ranges do not overlap (0.4062–0.4285 against
0.4296–0.4564), so a real ~6 % regression, at the noise floor.

Crossover 1.65 × 10⁴ → 2.00 × 10⁴, +21 %, **not resolved**. Its bracket is 9 918 @ 0.873 and 48 599 @ 1.187, whose
deltas are −2.5 % (unresolved) and −6.1 % (marginal). Recorded because the interpolation is what it is, not
claimed as a real rightward shift.

SU(4) brickwork, n = 36, depth 6, 105 channels:

| `min_abs_coeff` | peak terms | rust s pre → post | jl s pre → post | ratio pre → post | Δratio |
|---|---|---|---|---|---|
| 1e-2 | 1 416 | 0.00917 → 0.00820 | 0.00909 → 0.00900 | 0.983 → 1.097 | +11.6 % |
| 3e-3 | 12 924 | 0.1574 → 0.1479 | 0.0975 → 0.0959 | 0.620 → 0.658 | +6.2 % |
| 1e-3 | 84 836 | 0.6171 → 0.4524 | 0.6286 → 0.6380 | 1.027 → 1.416 | +37.8 % |
| 3e-4 | 573 826 | 2.630 → 1.881 | 4.417 → 4.386 | 1.676 → 2.326 | +38.8 % |
| 1e-4 | 2 296 294 | 7.806 → 5.208 | 15.46 → 15.24 | 1.974 → 2.921 | +48.0 % |

Crossover 8.01 × 10⁴ → none: every sign-consistent configuration on the post sweep is paulistrings-faster, so the
interpolation has no bracket. The 3e-3 point became mixed-sign by improving (pairs
0.658 / 0.658 / 0.772 / 0.593 / 1.222). The rust leg moved −6.0, −10.6, −26.7, −28.5, −33.3 % across the sweep,
which lands on E2's gate cells (`W = 1` dense-PTM layers at −33.2 and −33.8 %). Memory was unchanged, as a sort
kernel swap should leave it: 0.239 GiB / 95 B-per-peak-term against 0.235 GiB / 93.

## The drift check: Julia did not move

The ratio embeds both engines, so a Julia that got faster would masquerade as this engine getting slower. Across
all 21 rerun configurations, jl warm time changed by:

| workload | jl Δ% per configuration, loosest → tightest |
|---|---|
| kicked-Ising | +0.0, +0.0, +1.4, +3.6, −0.7, −1.4, −2.4, −1.1, +4.5 (`max_weight`) |
| XXZ | −4.3, −2.1, −0.4, −0.7, +1.9, +0.0, −1.7 |
| SU(4) | −0.9, −1.7, +1.5, −0.7, −1.4 |

Median −0.7 %, full range −4.3 % to +4.5 %, no systematic direction. Comfortably inside the ±5–8 %
between-campaign floor and an order of magnitude below the ratio movements at the top of each sweep, so the ratio
changes are on this engine's side of the fraction.

## Per-term cost, pre → post

| workload | peak terms | rust ns pre → post | jl ns pre → post |
|---|---|---|---|
| kicked_ising | 6 311 | 5 166 → 4 690 | 5 815 → 5 894 |
| kicked_ising | 79 029 | 1 806 → 1 559 | 2 445 → 2 532 |
| kicked_ising | 637 219 | 1 252 → 1 097 | 2 424 → 2 406 |
| kicked_ising | 1 544 083 | 1 141 → 967 | 1 941 → 1 914 |
| kicked_ising | 2 121 774 | 1 116 → 940 | 1 577 → 1 539 |
| kicked_ising | 2 146 424 | 1 120 → 964 | 1 567 → 1 550 |
| xxz | 48 599 | 8 383 → 8 924 | 10 770 → 10 698 |
| xxz | 206 035 | 7 087 → 6 291 | 10 212 → 10 407 |
| xxz | 776 432 | 6 463 → 5 666 | 10 817 → 10 818 |
| xxz | 2 661 873 | 6 137 → 5 361 | 11 029 → 10 840 |
| su4 | 1 416 | 6 477 → 5 791 | 6 417 → 6 357 |
| su4 | 12 924 | 12 175 → 11 441 | 7 547 → 7 417 |
| su4 | 84 836 | 7 274 → 5 333 | 7 410 → 7 520 |
| su4 | 573 826 | 4 584 → 3 278 | 7 697 → 7 643 |
| su4 | 2 296 294 | 3 399 → 2 268 | 6 732 → 6 638 |

Large-`m` per-term cost dropped by a roughly constant 13–16 % on the rotation workloads. A constant fractional
saving across a decade of `m` is what a per-term change like E1 predicts, and it is what the truncation path's
share of a merge-dominated layer is worth.

The kicked-Ising ratio decay above 6.4 × 10⁵ terms survived the campaign unchanged and is still Julia's: post-campaign
this engine falls 1 097 → 964 ns (−12 %) across that range while Julia falls 2 406 → 1 550 (−36 %), the same
saturation discount both campaigns measured. The campaign did not target it; every point in the decaying range is
simply ~15 % better than before.

## The XXZ small-`m` band, unexplained

Between 1.6 × 10³ and 4.9 × 10⁴ terms the rust leg is 5.5–16 % slower post-campaign, while above 2 × 10⁵ it is
11–13 % faster. Three things bound what can be concluded:

1. kicked-Ising moved the other way over the same band (−3.2 % at 517, −9.2 % at 6 311 terms). Both workloads are
   `pauli_rotation` at `W = 2`, so this is not a uniform small-`m` cost of the merged changes.
2. Only the 4.9 × 10⁴ point is resolvable at all; the other two sit inside their own runs' per-pair spread.
3. LTO code layout is a live candidate and is not excluded. E1's own gate measured untouched code moving −7.5 % to
   +4.4 % on build layout alone, and three merged branches is a new layout.

What would settle it is an `ab-compare` of the pre- and post-campaign binaries at `xxz min_abs_coeff = 1e-5`,
paired adjacent in time. Logged as a follow-up, never run.

## The `engine="auto"` sweep against both baselines

| workload | peak terms | pre-campaign | post, `sorted` | post, `auto` | path speedup |
|---|---|---|---|---|---|
| kicked_ising | 68 | 0.323 | 0.281 | 0.490 | 1.65× |
| kicked_ising | 517 | 0.629 | 0.638 | 0.696 (indistinguishable) | 1.11× |
| kicked_ising | 6 311 | 1.126 | 1.253 | 1.297 | 1.08× |
| xxz | 164 | 0.460 | 0.440 | 0.649 | 1.47× |
| xxz | 1 625 | 0.453 | 0.372 | 1.040 (indistinguishable) | 2.69× |
| xxz | 9 918 | 0.895 | 0.873 | 1.051 (indistinguishable) | 1.24× |
| su4 | 1 416 | 0.983 | 1.097 | 1.660 | 1.47× |
| su4 | 12 924 | 0.620 | 0.658 | 0.802 | 1.21× |
| su4 | 84 836 | 1.027 | 1.416 | 1.409 | 1.00× (inert, as designed) |

Crossovers: kicked-Ising 3.79 × 10³ → 2.73 × 10³ → 1.88 × 10³; su4 8.01 × 10⁴ → none on the full sweep →
6.62 × 10³ on the three-cutoff sweep; xxz 1.65 × 10⁴ → 2.00 × 10⁴ → not localizable.

Three configurations moved from "Julia faster, unanimously" to a measured tie. E3's own gate had predicted
2.28–2.36× on kicked-Ising 2⁻⁴ and 1.55–1.68× on the XXZ small points as engine-only wall time; end-to-end
propagation carries construction-adjacent per-call cost the path cannot remove, so 1.65× against 2.28× on
kicked-Ising and 2.69× against 1.68× on XXZ 1e-3 is expected rather than contradictory.

E3's outstanding gate (c) — parity against an independent engine with the path enabled — passed here: 9 618
per-layer term counts, every one identical to PauliPropagation.jl's, expectations ≤ 9.0 × 10⁻¹⁷. Its note had to
substitute two ours-only checks for it.

## The saturation hypothesis, as it unfolded

The pre-campaign kicked-Ising curve showed a non-monotone advantage: 1.925 at 6.4 × 10⁵ peak terms falling to
1.389 at 2.15 × 10⁶. The study's hypothesis was that the decay is not a large-`m` weakness but a *near-closed-sum*
regime: at 5 Trotter steps a 4× tighter cutoff adds only 1.16 % more terms, so nearly every gate application lands
on a key the sum already contains, which is a hash map's cheap path (lookup and add, no insert, no rehash, no
dict growth) and gets a bucketed gather → sort → merge no discount at all. Julia's peak RSS plateauing at 1.07 GiB
across the two largest configurations was consistent with a dictionary that had stopped growing.

The falsifiable prediction: run the same circuit family deep enough that 2 × 10⁶ terms is far from the reachable
set, and the ratio should stop decaying while Julia's per-term cost stops dropping. If the ratio decayed anyway,
the effect would be a genuine large-`m` property of this engine and the hypothesis wrong.

`deep-kicked-ising/` tested it at 20 Trotter steps, where a halved cutoff still multiplies the term count by ~4.2.
The verdict was **holds**: over 8.1 × 10⁵ → 3.1 × 10⁶ peak terms the ratio moves 2.212 → 2.197 (−0.7 %) against
the 5-step curve's −27.8 % over a span of the same width, and Julia's per-term cost falls 4.5 % against 35.3 %.
One honest qualification was recorded: the prediction's stronger form, that the ratio would keep *rising*, was not
met — it plateaus at ~2.2 rather than climbing, and a plateau is weaker evidence than a rise.

The consequence for the campaign was that the 5-step decay is not a regression to fix. The actionable converse
stands: a merge path that recognizes a layer producing no new keys and skips the sort would claw back exactly the
discount a hash map gets for free in the near-closed regime. That is an opportunity in a specific regime, never
run.

## Campaign deviations, as decided

Recorded in `research/notes/2026-09-01-large-m-campaign-log.md` as they happened.

1. **`kicked_ising_deep` was not re-run** after the campaign: 117 min at 3 pairs, and its regime overlaps the
   kicked-Ising curve's top end, which the rerun does measure. Nothing merged targeted large-`m` rotations beyond
   E1. Consequence: `deep-kicked-ising/` measures the pre-campaign build and says so.
2. **The Part A driver was killed by the agent harness** during xxz `1e-9` pair 4, after the kicked-Ising curve
   and XXZ through `1e-8` had completed all five pairs. `run.log` is written and flushed line by line, so no
   measurement was lost, and the structured record was rebuilt with the study's own committed tool:

   ```bash
   python benchmarks/python/jl_performance_recover.py \
       benchmarks/python/jl_performance/post-optimization/run.log \
       --out benchmarks/python/jl_performance/post-optimization \
       --memory-from benchmarks/python/jl_performance/summary.json
   ```

   The tool imports the driver's protocol math rather than reimplementing it and refuses to write if a recomputed
   median or verdict disagrees with what the driver logged. It wrote, so every median and verdict there is the
   driver's.
3. **The SU(4) curve was re-run as its own invocation** into `post-optimization/su4-curve/` after that kill,
   mirroring the pre-campaign `su4-curve/` layout. It has a complete driver-written `summary.json` with measured
   memory; nothing about it is recovered or joined.
4. **`peak_terms` in `post-optimization/summary.json` is joined** from the pre-campaign summary, tagged
   `"source": "joined"`. Exact rather than approximate: `peak_terms` is a deterministic function of circuit and
   cutoff, the parity gate proves the per-layer count sequences identical, and every final count matches.
5. **The memory block there is likewise joined and is not a measurement of the post-campaign build.** The only
   post-campaign memory measured is su4's. A memory rerun of the rotation workloads is a named follow-up.
6. **Five pairs throughout** for the rerun; the `--pairs 3` time-box rule did not fire (projection ~61 min for 21
   configurations). `deep-kicked-ising/` used 3 pairs against a ~75 min budget (5 pairs projected ~192 min, actual
   116.9 min at 3).

## The pre-campaign study was stopped mid-run

The top-level sweep completed two of five planned sections and was cancelled on purpose, because the completed
curves exposed the large-term-count effect above and effort was redirected into the optimization campaign.

| section | status |
|---|---|
| kicked-Ising curve (9 configurations) | complete, 5 pairs each |
| XXZ curve (7 of 8 configurations) | complete for those 7, 5 pairs each |
| XXZ `min_abs_coeff = 1e-9` (8.47 M terms) | cancelled mid-pairs, reconnaissance value only |
| SU(4) brickwork curve | not run there; measured separately in `su4-curve/` |
| time to fixed accuracy (3 references) | not run |
| thread scaling (1→32 threads) | not run |

Its `summary.json` was likewise rebuilt from `run.log` by `jl_performance_recover.py`, with `peak_terms` and
memory joined from single-pair pilot runs of the identical task files. A wrinkle worth remembering: `run.log`
prints cutoffs to four significant figures, which destroys the very property `is_dyadic` tests (`2**-6`
round-trips as `0.01562`), so the recovery tool snaps each logged cutoff back to the workload's declared value and
hard-errors if one matches nothing within 0.1 %. The first recovery pass mislabelled eight of nine kicked-Ising
cutoffs as non-dyadic before that was fixed.

The pre-campaign kicked-Ising crossover was also logged as 4.74 × 10³ during the run, before the driver excluded
the `max_weight` variant from bracketing; 3.79 × 10³ is the corrected value and the driver was fixed in the same
change.

The 1e-9 reconnaissance point has two records of the same configuration on the two builds: pre-campaign, one pair
at 55.0 s rust / 99.8 s jl (ratio ≈ 1.81); post-campaign, four pairs at 43.5–43.7 s rust against 97.2–97.5 s jl
(2.230 / 2.227 / 2.230 / 2.233). Both are below the five-pair bar and enter no verdict.

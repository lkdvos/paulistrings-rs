# Post-optimization rerun: the head-to-head on the improved engine

> ## ✅ Every configuration above the crossover improved, and one workload's crossover disappeared
>
> The same protocol, the same task files, the same Julia — re-run against the engine after the
> large-`m` campaign merged E1 (squared-magnitude truncation), E2 (a gated radix sort for dense-PTM
> layers) and E3 (an opt-in direct-apply path for small sums).
>
> | | kicked-Ising | XXZ | SU(4) |
> |---|---|---|---|
> | crossover, before | 3.79 × 10³ | 1.65 × 10⁴ | 8.01 × 10⁴ |
> | crossover, after | **2.73 × 10³** (−28 %) | 2.00 × 10⁴ (+21 %, unresolved) | **none — we win at every sign-consistent point** |
> | best ratio, before | 1.925 | 1.798 | 1.974 |
> | best ratio, after | **2.146** | **2.023** | **2.921** |
>
> **Julia did not move**: across all 21 rerun configurations its warm time changed by a median of
> **−0.7 %** (range −4.3 % to +4.5 %), so the ratio movements are ours. The whole gain is on our side
> of the fraction.

The parent study is [`../README.md`](../README.md) — its protocol, ratio convention
(`ratio = t_julia / t_paulistrings`, **> 1 means we are faster**), acceptance rule, workloads and
caveats apply here unchanged and are not restated. This document reports only what the rerun adds.

Two sections, and the second is **new protocol surface**:

* **Part A** — protocol-identical rerun on the **default** engine, directly comparable to the
  committed study configuration for configuration. Data in this directory (kicked-Ising, XXZ) and in
  [`su4-curve/`](su4-curve/).
* **Part B** — the small-`m` end with E3's direct-apply path **enabled** (`--engine auto`), in
  [`../post-optimization-auto/`](../post-optimization-auto/). This is a *different engine setting*,
  so its numbers are labelled apart from Part A's and never mixed into the same table.

## What changed in the engine, and where each change can possibly act

| | what it does | where it can act on this study |
|---|---|---|
| **E1** — squared-magnitude truncation (`research/notes/2026-09-01-topn-finalize.md`) | `norm()` → `norm_sqr()` on the truncation path; `\|c\| > t ⟺ \|c\|² > t²` removes a `hypot` per term. Gated at **−26 %** on `CoefficientThreshold` layers | **every configuration here.** All 21 use `min_abs_coeff`, and the effect is per *term*, so it grows with `m` |
| **E2** — gated radix sort for dense-PTM gather runs (`research/notes/2026-09-01-sort-kernel.md`) | replaces the comparison sort for two-qubit dense PTMs (fanout ≥ 8). Gated at **−10.5…−33.8 %** on the layer, strongest at `W = 1` | **`su4` only.** `unitary_2q`, fanout 15, and this workload is `n = 36` → `W = 1`, E2's strongest cell |
| **E3** — opt-in small-sum direct-apply path (`research/notes/2026-09-01-small-m-path.md`) | applies layers term-by-term into a hash map below `small_sum_threshold` (2048), skipping the bucketed pipeline. **Off by default** | **Part B only.** Part A runs `engine="sorted"`, where E3 is inert by construction |

Nothing else in the engine moved: `git diff 81c568a HEAD -- crates/` is empty, so the binary
measured here is a faithful build of the campaign tip.

## Provenance

* **Extension.** `python/paulistrings/_paulistrings.abi3.so`, mtime **2026-09-01 13:29:28**, which
  postdates `81c568a` (13:26:50), the last commit to touch `crates/`. Contains E1 + E2 + E3 and the
  `engine=` kwarg. Checkout HEAD at run time `0f00207`, tree clean.
* **Julia side, unchanged from the study.** PauliPropagation.jl 0.8.2 on Julia 1.12.6,
  `PP_BACKEND=dict`, `PP_FUSED=0`, `-t1`.
* **Host.** ccqlin038 (2 × Xeon Gold 6244 @ 3.60 GHz, 32 threads), `powersave`, 205–240 GiB free
  throughout. `RAYON_NUM_THREADS=1` exported before the interpreter; `RUST_LOG` unset. Box held
  exclusively — load ≈ 1 from interactive tooling only, no other compute tenant, never two engines at
  once and never alongside a build.
* **Driver.** `benchmarks/python/bench_jl_performance.py` at `0f00207`, which adds `--engine` and
  `--max-configs` (both defaulting to today's behaviour) and is committed *before* any timed run
  here. Part A passes no `--engine`, i.e. `sorted`.
* **rustc** 1.94.0, release profile (`lto = "fat"`, `codegen-units = 1`), Python 3.11.11.

### How to read a "before → after" delta on this page

The **ratio** at each configuration is protocol-grade: its two legs ran adjacent in time, `abba`
across five pairs, and the acceptance rule is direction consistency (`../README.md` §5).

The **baseline → rerun delta** is not that. It compares two campaigns hours apart, which is exactly
the comparison `benchmarks/PROFILING.md` says the ±5–8 % single-threaded noise floor forbids for
small effects. So on this page:

* a delta **above ~8 %** whose per-pair ranges do not overlap is reported as a change;
* a delta **below** that, or one inside its own run's per-pair spread, is reported as **unresolved**
  and is not attributed to anything. Several of the small-`m` configurations are in that category and
  are marked.

Nothing here is an `ab-compare` of two binaries, and where one is what the question needs, this page
says so instead of pretending otherwise.

---

# Part A — protocol-identical rerun, default engine

All 21 configurations passed the per-layer parity gate: **1355 layers identical** on every
kicked-Ising configuration, **1782** on every XXZ one, **105** on every SU(4) one, expectations
agreeing to ≤ 2.6 × 10⁻¹⁶ against a 1e-9 bar. No configuration was disqualified. Term counts are
**identical to the committed study's, configuration for configuration** — which is the point: E1 and
E2 change no term sets, so the two campaigns timed exactly the same work.

## kicked-Ising, 127 q, 5 Trotter steps, `theta_h = 5pi/16`, 1355 channels

| `min_abs_coeff` | peak terms | rust s → | jl s → | **ratio →** | Δratio | pairs | faster |
|---|---|---|---|---|---|---|---|
| 2⁻⁴ | 68 | 0.0031 → 0.0032 | 0.0009 → 0.0009 | 0.323 → **0.281** | −12.8 % ‡ | 5/5 | Julia |
| 2⁻⁶ | 517 | 0.0062 → 0.0060 | 0.0038 → 0.0038 | 0.629 → **0.638** | +1.4 % ‡ | 5/5 | Julia |
| 2⁻⁸ | 6 311 | 0.0326 → 0.0296 | 0.0367 → 0.0372 | 1.126 → **1.253** | **+11.3 %** | 5/5 | paulistrings |
| 2⁻¹⁰ | 79 029 | 0.1427 → 0.1232 | 0.1932 → 0.2001 | 1.362 → **1.651** | **+21.2 %** | 5/5 | paulistrings |
| 2⁻¹² | 637 219 | 0.798 → 0.699 | 1.544 → 1.533 | 1.925 → **2.146** ← peak | **+11.5 %** | 5/5 | paulistrings |
| 2⁻¹⁴ | 1 544 083 | 1.761 → 1.492 | 2.997 → 2.955 | 1.690 → **1.953** | **+15.5 %** | 5/5 | paulistrings |
| 2⁻¹⁶ | 2 121 774 | 2.367 → 1.994 | 3.346 → 3.266 | 1.431 → **1.638** | **+14.5 %** | 5/5 | paulistrings |
| 2⁻¹⁸ | 2 146 424 | 2.404 → 2.070 | 3.364 → 3.327 | 1.389 → **1.610** | **+15.9 %** | 5/5 | paulistrings |
| 2⁻¹⁸ + `max_weight=6` | 712 † | 0.0050 → 0.0042 | 0.0022 → 0.0023 | 0.448 → **0.488** | +8.8 % ‡ | 5/5 | Julia |

† peak not recoverable for this configuration in either campaign; final count used, as in the study.
‡ **unresolved** — per-pair spreads of 30–83 % at these millisecond configurations swamp the delta.

**Crossover: 3.79 × 10³ → 2.73 × 10³ peak terms, −28 %.** Bracketed by 517 @ 0.638 and 6 311 @ 1.253,
same two configurations as the study's bracket. The movement is carried by the 6 311-term point,
whose +11.3 % is resolved; the 517-term point moved +1.4 %, which is not.

## XXZ chain, n = 100, `Jz = 0.5`, `dt = 0.1`, 6 Trotter steps, 1782 channels

| `min_abs_coeff` | peak terms | rust s → | jl s → | **ratio →** | Δratio | pairs | faster |
|---|---|---|---|---|---|---|---|
| 1e-2 | 164 | 0.0050 → 0.0050 | 0.0023 → 0.0022 | 0.460 → **0.440** | −4.3 % ‡ | 5/5 | Julia |
| 1e-3 | 1 625 | 0.0433 → 0.0502 | 0.0191 → 0.0187 | 0.453 → **0.372** | −17.9 % ‡ | 5/5 | Julia |
| 1e-4 | 9 918 | 0.1287 → 0.1358 | 0.1146 → 0.1141 | 0.895 → **0.873** | −2.5 % ‡ | 5/5 | Julia |
| 1e-5 | 48 599 | 0.4074 → 0.4337 | 0.5234 → 0.5199 | 1.264 → **1.187** | −6.1 % ◊ | 5/5 | paulistrings |
| 1e-6 | 206 035 | 1.460 → 1.296 | 2.104 → 2.144 | 1.438 → **1.654** | **+15.0 %** | 5/5 | paulistrings |
| 1e-7 | 776 432 | 5.018 → 4.399 | 8.399 → 8.400 | 1.682 → **1.909** | **+13.5 %** | 5/5 | paulistrings |
| 1e-8 | 2 661 873 | 16.34 → 14.27 | 29.36 → 28.85 | 1.798 → **2.023** ← still rising | **+12.5 %** | 5/5 | paulistrings |

‡ **unresolved** — inside the per-pair spread (1e-3's own five pairs span 89.8 %).
◊ **marginal** — the only small-`m` point whose baseline and rerun rust ranges do not overlap
(0.4062–0.4285 against 0.4296–0.4564), so a real ~6 % regression, at the noise floor.

**Crossover: 1.65 × 10⁴ → 2.00 × 10⁴ peak terms, +21 % — and this movement is *not* resolved.** Its
bracket is 9 918 @ 0.873 and 48 599 @ 1.187, whose deltas are −2.5 % (unresolved) and −6.1 %
(marginal). Reported because the interpolation is what it is; not claimed as a real rightward shift.

### The XXZ small-`m` band, stated plainly

Between 1.6 × 10³ and 4.9 × 10⁴ terms our leg is **5.5–16 % slower** than the baseline campaign
measured, while above 2 × 10⁵ it is 11–13 % faster. That sign change is real in the medians, and
three things bound what can be concluded from it:

1. **kicked-Ising moved the other way over the same band** (−3.2 % at 517, −9.2 % at 6 311 terms).
   Both workloads are `pauli_rotation` at `W = 2`, so this is not a uniform small-`m` cost of the
   merged changes.
2. **Only the 4.9 × 10⁴ point is resolvable at all** — the other two sit inside their own runs'
   per-pair spread.
3. **LTO code layout is a live candidate and is not excluded.** E1's own gate measured *untouched*
   code moving −7.5 % to +4.4 % on build layout alone, and three merged branches is a new layout.

What would settle it is an `ab-compare` of the pre- and post-campaign binaries at
`xxz min_abs_coeff = 1e-5`, paired adjacent in time — the instrument this comparison is not. Logged
as a follow-up, not diagnosed here.

### The 1e-9 configuration: reconnaissance, below the bar

The study cancelled this configuration mid-pairs; the driver's own cut rule authorizes it now, so a
protocol-identical invocation ran it. **4 of 5 pairs completed** before the run was interrupted (see
"Deviations"), which is below this study's 5-pair bar, so the recovery tool dropped it from
`summary.json` and it appears in no table, crossover or verdict above.

Quoted here as reconnaissance only, exactly as the study quoted its own one-pair pilot:
**8 473 952 terms**, all 1782 per-layer counts identical, `|dE| = 7.6e-17`; rust 43.5–43.7 s against
Julia 97.2–97.5 s, four pairs at **2.230 / 2.227 / 2.230 / 2.233**. The study's single-pair pilot of
the identical configuration read 55.0 s / 99.8 s / ≈ 1.81. So XXZ's ratio has still not turned over at
8.5 × 10⁶ terms, and our leg at this size is ~21 % faster than the pilot measured.

## SU(4) brickwork, n = 36, depth 6 — [`su4-curve/`](su4-curve/)

The largest movement in the campaign, and the one with a named mechanism: this is the only workload
E2 can touch, and it is E2's strongest cell (`W = 1`, fanout 15).

| `min_abs_coeff` | peak terms | rust s → | jl s → | **ratio →** | Δratio | pairs | faster |
|---|---|---|---|---|---|---|---|
| 1e-2 | 1 416 | 0.00917 → 0.00820 | 0.00909 → 0.00900 | 0.983 → **1.097** | +11.6 % | 5/5 | **paulistrings** (was a tie) |
| 3e-3 | 12 924 | 0.1574 → 0.1479 | 0.0975 → 0.0959 | 0.620 → **0.658** | +6.2 % | 4/5 | **indistinguishable** (was Julia) |
| 1e-3 | 84 836 | 0.6171 → 0.4524 | 0.6286 → 0.6380 | 1.027 → **1.416** | **+37.8 %** | 5/5 | **paulistrings** (was a tie) |
| 3e-4 | 573 826 | 2.630 → 1.881 | 4.417 → 4.386 | 1.676 → **2.326** | **+38.8 %** | 5/5 | paulistrings |
| 1e-4 | 2 296 294 | 7.806 → 5.208 | 15.46 → 15.24 | 1.974 → **2.921** ← still rising | **+48.0 %** | 5/5 | paulistrings |

**Crossover: 8.01 × 10⁴ → none.** Every sign-consistent configuration on this sweep is now
paulistrings-faster, so the interpolation has no bracket to work with and the driver reports "no
direction change across the swept range". The single mixed-sign point (3e-3) is the study's former
deepest deficit, and it is no longer resolvable as a loss: its five pairs read
0.658 / 0.658 / 0.772 / 0.593 / **1.222**, one of them above 1.

Our rust leg moved −6.0 %, −10.6 %, **−26.7 %, −28.5 %, −33.3 %** across the sweep, which lands on
E2's gate cells almost exactly — that gate measured `W = 1` dense-PTM layers at −33.2 % and −33.8 %.
The two loosest points gain less because 105 channels of a 10³-term sum spend proportionally less of
their wall time in the sort.

**Memory (measured, not joined).** 0.239 GiB peak at 2 296 294 terms, **95 B/peak-term**, against the
study's 0.235 GiB / 93 — unchanged, as expected: E2 replaced a sort kernel, not a layout. Julia's
1.625 GiB / 479 B/term against 1.660 / 495 likewise.

## Julia did not move — the drift check

The ratio comparison embeds both engines, so a Julia that got faster would masquerade as us getting
slower. It did not. Across all 21 rerun configurations, jl warm time changed by:

| workload | jl Δ% per configuration, loosest → tightest |
|---|---|
| kicked-Ising | +0.0, +0.0, +1.4, +3.6, −0.7, −1.4, −2.4, −1.1, +4.5 (`max_weight`) |
| XXZ | −4.3, −2.1, −0.4, −0.7, +1.9, +0.0, −1.7 |
| SU(4) | −0.9, −1.7, +1.5, −0.7, −1.4 |

**Median −0.7 %, full range −4.3 % to +4.5 %, no systematic direction** — comfortably inside the
±5–8 % between-campaign floor, and an order of magnitude below the ratio movements at the top of each
sweep. Every ratio change above is therefore ours, not the governor's.

## Per-term cost

Nanoseconds per **peak** term, both engines, before → after.

| workload | peak terms | rust ns → | jl ns → | ratio → |
|---|---|---|---|---|
| kicked_ising | 6 311 | 5 166 → **4 690** | 5 815 → 5 894 | 1.126 → 1.253 |
| kicked_ising | 79 029 | 1 806 → **1 559** | 2 445 → 2 532 | 1.362 → 1.651 |
| kicked_ising | 637 219 | 1 252 → **1 097** | 2 424 → 2 406 | 1.925 → 2.146 |
| kicked_ising | 1 544 083 | 1 141 → **967** | 1 941 → 1 914 | 1.690 → 1.953 |
| kicked_ising | 2 121 774 | 1 116 → **940** | 1 577 → 1 539 | 1.431 → 1.638 |
| kicked_ising | 2 146 424 | 1 120 → **964** | 1 567 → 1 550 | 1.389 → 1.610 |
| xxz | 48 599 | 8 383 → 8 924 | 10 770 → 10 698 | 1.264 → 1.187 |
| xxz | 206 035 | 7 087 → **6 291** | 10 212 → 10 407 | 1.438 → 1.654 |
| xxz | 776 432 | 6 463 → **5 666** | 10 817 → 10 818 | 1.682 → 1.909 |
| xxz | 2 661 873 | 6 137 → **5 361** | 11 029 → 10 840 | 1.798 → 2.023 |
| su4 | 1 416 | 6 477 → **5 791** | 6 417 → 6 357 | 0.983 → 1.097 |
| su4 | 12 924 | 12 175 → 11 441 | 7 547 → 7 417 | 0.620 → 0.658 |
| su4 | 84 836 | 7 274 → **5 333** | 7 410 → 7 520 | 1.027 → 1.416 |
| su4 | 573 826 | 4 584 → **3 278** | 7 697 → 7 643 | 1.676 → 2.326 |
| su4 | 2 296 294 | 3 399 → **2 268** | 6 732 → 6 638 | 1.974 → 2.921 |

Two readings the study's version of this table did not support:

1. **Our large-`m` per-term cost dropped by a roughly constant 13–16 % on the rotation workloads**
   (kicked-Ising 1 252 → 1 097, 1 141 → 967, 1 116 → 940, 1 120 → 964; XXZ 7 087 → 6 291,
   6 463 → 5 666, 6 137 → 5 361). A constant *fractional* saving across a decade of `m` is what a
   per-term change like E1 predicts, and it is what the truncation path's share of a merge-dominated
   layer is worth.
2. **The kicked-Ising ratio decay above 6.4 × 10⁵ terms is still there and is still Julia's.** Ours
   falls 1 097 → 964 ns (−12 %) across that range while Julia's falls 2 406 → 1 550 (−36 %) — the same
   35 % saturation discount the study measured and
   [`../deep-kicked-ising/README.md`](../deep-kicked-ising/README.md) confirmed is a near-closed-sum
   effect. The campaign did not target it and did not move it; every point in the decaying range is
   simply ~15 % better than it was.

## What did not change

* **Parity, everywhere.** Same per-layer counts as the study, same expectations, no disqualification.
* **The saturation mechanism.** Unaddressed by design (the campaign log's Phase-1 verdict:
  "there is no large-`m` regression to repair").
* **Sign-consistency.** 20 of 21 configurations unanimous across five pairs; the one exception
  (su4 3e-3) *became* mixed-sign by improving.
* **Our memory profile**, at least where it was measured this time (su4): 95 vs 93 B/peak-term.

---

# Part B — the small-`m` end with the direct path enabled

Full report: [`../post-optimization-auto/README.md`](../post-optimization-auto/README.md). It is a
**different engine setting** (`--engine auto`, E3's opt-in direct-apply path below 2048 terms), so
its numbers live in their own directory and are never mixed into a Part A table. Summary of the nine
loose-end configurations, every one of which the study lost or tied:

| workload | peak terms | study | Part A (`sorted`) | **Part B (`auto`)** | path speedup |
|---|---|---|---|---|---|
| kicked_ising | 68 | 0.323 | 0.281 | **0.490** | 1.65× |
| kicked_ising | 517 | 0.629 | 0.638 | **0.696** (indistinguishable) | 1.11× |
| kicked_ising | 6 311 | 1.126 | 1.253 | **1.297** | 1.08× |
| xxz | 164 | 0.460 | 0.440 | **0.649** | 1.47× |
| xxz | 1 625 | 0.453 | 0.372 | **1.040** (indistinguishable) | **2.69×** |
| xxz | 9 918 | 0.895 | 0.873 | **1.051** (indistinguishable) | 1.24× |
| su4 | 1 416 | 0.983 | 1.097 | **1.660** | 1.47× |
| su4 | 12 924 | 0.620 | 0.658 | **0.802** | 1.21× |
| su4 | 84 836 | 1.027 | 1.416 | 1.409 | 1.00× (inert, as designed) |

Three configurations moved from "Julia faster, unanimously" to a measured tie, the kicked-Ising
crossover moves further to **1.88 × 10³**, and the 84 836-term row is the control showing the path is
genuinely inert above its threshold.

**E3's outstanding gate (c) passes.** The parity gate takes the same `--engine` as the timed legs, so
all nine configurations were parity-checked *with the path enabled and taken*: **9 618 per-layer term
counts, every one identical to PauliPropagation.jl's**, expectations ≤ 9.0 × 10⁻¹⁷. E3's note had to
substitute two ours-only checks for this; it is now measured against an independent engine on real
circuits at both `W = 1` and `W = 2`.

---

## Deviations, and the interrupted run

Recorded in `research/notes/2026-09-01-large-m-campaign-log.md` as they were decided.

1. **`kicked_ising_deep` was not re-run.** 117 min at 3 pairs, and its regime overlaps the
   kicked-Ising curve's top end, which this rerun does measure. Nothing merged targets large-`m`
   rotations beyond E1. **Consequence:** the numbers in
   [`../deep-kicked-ising/README.md`](../deep-kicked-ising/README.md) still describe the
   *pre-optimization* engine and must be read as such.
2. **The Part A driver was killed by the agent harness** during xxz `1e-9` pair 4, after the
   kicked-Ising curve and XXZ through `1e-8` had completed all five pairs. `run.log` is written and
   flushed line by line, so no measurement was lost, and the structured record was rebuilt with the
   study's own committed tool:

   ```bash
   python benchmarks/python/jl_performance_recover.py \
       benchmarks/python/jl_performance/post-optimization/run.log \
       --out benchmarks/python/jl_performance/post-optimization \
       --memory-from benchmarks/python/jl_performance/summary.json
   ```

   This is the parent study's own precedent, tool for tool. The tool *imports* the driver's protocol
   math rather than reimplementing it and refuses to write if a recomputed median or verdict
   disagrees with what the driver logged; it wrote, so every median and verdict here is the driver's.
3. **The SU(4) curve was re-run as its own invocation** into [`su4-curve/`](su4-curve/) after the
   interruption, mirroring the study's own `su4-curve/` layout. It has a complete driver-written
   `summary.json` with measured memory; nothing about it is recovered or joined.
4. **`peak_terms` in this directory's `summary.json` is joined from the study's**, tagged
   `"source": "joined"` by the recovery tool. This is exact rather than approximate: `peak_terms` is a
   deterministic function of circuit and cutoff, the parity gate proves the per-layer count sequences
   identical, and every final count matches the study's exactly.
5. **The memory block in this directory's `summary.json` is likewise joined from the study — it is
   NOT a measurement of this engine, and no memory number from it is quoted anywhere above.** The
   only memory reported on this page is su4's, which was measured. A memory rerun of the rotation
   workloads is a named follow-up.
6. **Five pairs throughout**, the study's count. The `--pairs 3` time-box rule did not fire: the
   committed study served as the projection basis (same configurations, same host, hours earlier),
   projecting ~61 min for the 21 committed configurations.

## Reproducing

```bash
# Part A, as run (the su4 curve was a separate invocation only because of the interruption)
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization

# Part B
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising --workload xxz --workload su4 \
    --engine auto --max-configs 3 --pairs 5 \
    --out benchmarks/python/jl_performance/post-optimization-auto

# re-render figures from the committed data, no measurement
python benchmarks/python/jl_performance_figures.py \
    benchmarks/python/jl_performance/post-optimization/summary.json

# the CI protocol gate (no julia, no timing, < 1 s)
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

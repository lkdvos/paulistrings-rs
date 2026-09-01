# Orchestration log: large-m optimization campaign

Branch `large-m-optimization`, executing `research/plans/2026-09-01-large-m-optimization-campaign.md`.
Host ccqlin038 (reference host), load 0.83 at start, baseline `cargo test --workspace --release` green at 90c1a0f.

Decisions are logged here as they are made, for post-hoc review (autonomous-execution preference).

## Decisions

1. **Serialization policy**: every timed run (cross-engine pairs, phase_breakdown, perf, ab-compare) runs alone on
   the box; cargo builds never overlap a timed run. Phase-2 implementation work parallelizes in worktrees, but its
   ab-compare gates run in a serialized benchmarking pass afterwards.
2. **Saturation test configuration**: kicked-Ising 127q at **20 Trotter steps, theta_h = 7pi/32** — benchmark C's
   thread-scaling precedent (all 5420 per-layer counts proven identical at 2^-14: 2 441 936 final / 3 108 582 peak)
   rather than a new untested angle/depth. Dyadic cutoffs, same one-ulp mitigation, same interleaved-pairs protocol.
3. **jl-side per-gate instrumentation deferred**: the deep-KI curve's per-term-cost trend answers the "is jl's
   per-term time flat or falling in m" question directly; a runner.jl instrumentation pass is only commissioned if
   the saturation verdict is ambiguous.
4. Follow-up cross-engine data lands in new subdirectories under `benchmarks/python/jl_performance/` (append,
   never overwrite the committed study).
5. **`phase_breakdown` cells are auto-reps'd, never run at the default `--reps`.** The powersave governor measures
   a sub-50 ms timed region at ~1200 MHz instead of ~3600 MHz (measured 3.62× at `m = 150`), so every cell gets a
   throwaway `--reps 8` calibration pass followed by the recorded pass at
   `reps = clamp(200 ms / per-layer, 4, 40 000)`. Uncorrected batches are kept beside corrected ones rather than
   deleted, so the artefact stays documented.
6. **Probe extensions are committed before any timed run they feed, and never change a default.** Phase 1c needed
   a faithful matrix-gate layer (`--layers su4`) and a `finalize_layer`-bearing policy (`--truncation`); both went
   in as 6715918 with `--layers` defaults and `--truncation keep` preserving prior behaviour exactly, workspace
   tests green and clippy clean, so no committed baseline is invalidated.

## Timeline

- 06:12 preflight: host/load/toolchain verified, branch created, baseline release tests green.
- 06:17 `kicked_ising_deep` workload added to the jl head-to-head driver (20 steps, theta_h = 7pi/32,
  cutoffs 2^-8/-10/-12/-13/-14) + a protocol-test mirror pin; `test_jl_performance_protocol.py` green (50 tests).
- 06:25 rust-only term-count probe (decision aid, not reported): 2^-8 363/1838, 2^-10 8046/17659,
  2^-12 138220/204728, 2^-13 579312/813262, 2^-14 **2441936/3108582** — the last matches benchmark C's
  committed deep-Trotter counts exactly, so the workload really is C's configuration.
  **2^-13 -> 2^-14 term growth x4.22**, against the 5-step curve's x1.0116 from 2^-16 to 2^-18: the deep
  sweep is far from closure, which is the falsification test's premise.
- 06:33 cheap cross-engine pilot on the three loosest cutoffs (1 pair, driver's own parity_gate/run_pairs).
  Parity holds: **all 5420 layers identical** at each, |dE| <= 5.6e-17. Ratios 0.424 (1838 peak),
  1.421 (17659), **2.155 (204728)** — already above the 5-step curve's 1.925 peak.
  Deviation logged: the driver's `--pilot` sweeps all five cutoffs; its 2^-13/2^-14 tail would have
  duplicated ~40 min of the timed run's own parity gates, so the pilot was restricted to the three
  loosest. Plumbing (task files, runner.jl on a 5420-gate task, parity, pairing, memory sampling) is
  exercised identically.
- 06:35 **time-box decision: `--pairs 3`.** Projected from the pilot (pair wall ~2.2x the summed timed
  work; parity gate ~6x rust_s): 5 pairs ~192 min, 3 pairs ~116 min. The orchestrator's box is ~75 min for
  5 pairs, so the rule fires. 3 is the floor, not 2: PROFILING.md's A/B harness bar is 3 pairs and
  direction consistency needs at least that. The 2^-14 point (3.1e6 peak terms, ~81 min of the total) is
  the one the verdict rests on and is not droppable.
- 08:32 full run finished, **116.9 min** wall (projection said ~116). 5/5 configurations parity-clean
  (all 5420 layers identical each, |dE| <= 2.2e-16), 15/15 pairs sign-consistent, no cut, no disqualification.
  Ratios: 0.381 (1838 peak) / 1.462 (17 659) / **2.244 (204 728) / 2.212 (813 262) / 2.197 (3 108 582)**.
- 08:40 **VERDICT: saturation hypothesis HOLDS.** The 5-step curve's -27.8% ratio decay (1.925 -> 1.389 over
  637k -> 2.15M peak) becomes **-0.7%** here (2.212 -> 2.197 over 813k -> 3.11M, a span of the same width);
  -2.1% over the full 15.2x measured span, inside the 2-5% per-pair spread. Mechanism check passes: jl's
  ns/peak-term falls 4.5% in the deep tail against our 4.2% (5-step: jl -35.3% against our -10.6%), so jl's
  discount is absent exactly where saturation is. At matched size the deep run is 2.197 at 3.11e6 peak terms
  against the 5-step 1.431 at 2.12e6 — more terms, 1.54x more advantage. Qualification: the ratio *plateaus*
  rather than rising XXZ-like, so the prediction's stronger form is not met; the falsification criterion as
  written ("if the ratio decays anyway") is not triggered.
  **Campaign consequence: there is no large-m regression to repair.** The no-new-keys merge fast path remains
  an opportunity in the near-closed regime; any other large-m work must be justified by its own roofline
  evidence, not by this decay.
- 08:40 Memory: the study's 91 -> 125 B/term step does not recur (92 -> 99 over 3.8x terms, peak RSS
  sub-linear). Both campaigns' six large points collapse to 63-73 B/term once divided by the power-of-two
  capacity slack 2^ceil(log2 m)/m, corroborating the study's capacity-doubling diagnosis and bounding the
  exact-size-allocation win at ~1.5x peak RSS worst case, ~0 best case.

### Phase 1b: the SU(4) matrix-gate curve (the study's unmeasured leg)

- 08:39 preflight for the su4 leg: load 0.27, 240 GiB free, extension binary untouched
  (mtime 2026-09-01 03:19:44, the study's build). No Rust change, no rebuild.
- 08:41 rust-only term-count probe of the committed su4 cutoffs (decision aid, not reported),
  RAYON_NUM_THREADS=1: 1e-2 193/1416, 3e-3 7089/12924, 1e-3 **84836/84836** (reproduces the study's
  committed mirror count exactly), 3e-4 573826/573826, 1e-4 **2296294/2296294** in 7.81 s.
  **Decision: do NOT extend the cutoff tuple.** The handoff's extension trigger was "1e-4 lands well
  short of ~1e6 peak terms"; it lands at 2.30e6, 2.3x above the target and the same order as every
  rotation curve's endpoint (2.15e6 / 2.66e6 / 3.11e6). The driver is therefore left untouched, and the
  su4 numbers are the workload the committed study declared, with no "we changed the workload" caveat.
  The spare budget goes into 5 pairs rather than a sixth cutoff.
- 08:46 cross-engine pilot, all five cutoffs at 1 pair through the driver's own `--pilot`
  (262 s wall). Parity holds at every point: **all 105 layers identical**, |dE| <= 1.3e-16. Single-pair
  ratios 0.976 (1416 peak) / 0.642 (12 924) / 1.076 (84 836) / 1.672 (573 826) / **1.977 (2 296 294)**,
  crossover ~6.5e4 — the pilot's 7.6e4 estimate confirmed to within the pair spread. Two things
  already visible: the small-m ratio is **non-monotone** (0.976 dips to 0.642 before climbing), and the
  large-m ratio is still rising at the last point.
- 08:47 **time-box decision: `--pairs 5`** (the study's preferred count, not the deep-KI run's floor of
  3). Projection from the pilot: pilot = parity gates + 1 pair round = 262 s; solving the two-term model
  (julia process start ~12 s/leg) gives ~125 s of parity gates + ~137 s per pair round, so 5 pairs
  projects **~13.5 min** against the ~75 min box. Nothing is cut; the driver's own projector authorizes
  every leg (worst case a 30-min julia leg budget against a projected ~35 s).
- 09:02 full run finished, **13.4 min** wall (projection said ~13.5). 5/5 configurations parity-clean
  (all 105 layers identical each, |dE| <= 1.25e-16; the 1e-3 expectation is bit-identical to benchmark
  E's committed -0.0030947264746490136), no cut, no disqualification. Ratios: 0.983 (1416 peak) /
  **0.620 (12 924)** / 1.027 (84 836) / 1.676 (573 826) / **1.974 (2 296 294)**.
  **2 of 5 configurations are `indistinguishable`** — the first anywhere in this study (the parent's 16
  and the deep-KI run's 5 were all unanimous). Both sit at ratio ~1 (1416: pairs 0.972-1.007;
  84 836: pairs 0.995-1.040), i.e. they are ties, not noise failures.
- 09:05 **VERDICT: the matrix-gate path is a mid-`m` deficit, not a large-`m` one.**
  Crossover **8.01e4 peak terms** = **21.1x** kicked-Ising's 3.79e3 (the pilot's "20x" confirmed),
  8.6x deep-KI's, 4.9x XXZ's — the widest workload spread in the study, and the su4 grid landed a
  configuration *on* the crossover (84 836 terms, median 1.027 with pairs straddling 1). The driver's
  `inside_indistinguishable_zone: true` must be read as that, not as an unresolved band: the 12 924
  point inside the flagged min-max span is unanimous at 0.620.
  At large `m` the path is **not** weaker: 1.974 at 2.30e6 peak terms, above the parent study's
  kicked-Ising (1.431 @ 2.12e6) and XXZ (1.798 @ 2.66e6) at comparable size, and rising +17.8% per 4x
  terms. Per-term cost is the XXZ pattern: ours -72% over the span and still falling -26% in the last
  4x, jl's -11% and non-monotone. The entire deficit is created in one step (193 -> 7089 terms: our
  time x17.2 against jl's x10.7) and repaid on every later step.
  **Campaign consequence: `gu2q` optimization effort belongs in the 1e4-1e5-term band**, where the
  deficit is 1.61x; above 5e5 terms the path already wins 1.7-2.0x. The obvious next probe is a
  `phase-timing` breakdown of a `unitary_2q` layer at ~1e4 terms — the cache-residency reading of the
  dip is a hypothesis, not a measurement.
- 09:05 Memory: floors reproduce the study's (37.8 MB / 0.601 GiB, x16.3). B/term falls monotonically
  793 -> 93; no repeat of the 91 -> 125 step, and none expected — both heavy su4 points sit at the
  *same* capacity slack 1.83, so this sweep adds two consistent points (slack-normalized 56 and 51
  B/term against `W = 1`'s 32 B/term arithmetic, i.e. 1.6-1.8x) without further resolving the
  capacity-doubling model. Note the overhead factor is larger than the two `W = 2` campaigns'
  1.3-1.5x, consistent with a fixed 16-byte coefficient and `W`-independent transient buffers.
- 09:06 `pytest python/paulistrings/tests/test_jl_performance_protocol.py` green (50 tests) — the driver
  was not edited, so this is a drift check rather than a gate on a change.

### Phase 1c: the per-phase breakdown (`research/notes/2026-09-01-large-m-phase-breakdown.md`)

- 09:11 preflight: load 0.23, 229 GiB free, `cargo test --workspace` green at 44782fe, probe built
  `--release --features phase-timing`. All timed runs single-threaded unless noted, `RUST_LOG` unset, box idle.
- 09:20 **methodology stop-the-line: the powersave governor invalidates every small-`m` number taken at the
  probe's default `--reps`.** A `--reps 8` cell at `m = 150` runs for 0.25 ms and is measured at ~1200 MHz;
  `ns/term` falls 143.1 → 39.6 (**3.62×**, exactly 1200/3600 MHz) as `--reps` goes 8 → 10 000, and converges once
  the timed region exceeds ~50 ms. **Decision 5: every cell in this phase is run twice** — a throwaway
  calibration pass at `--reps 8`, then the recorded pass at `reps = clamp(200 ms / per-layer, 4, 40 000)`. The
  first (uncorrected) batch is kept beside the corrected one as `small-m.log` vs `small-m2.log` rather than
  deleted, so the artefact is documented rather than hidden.
- 09:22 **probe fidelity decision 6: `gu2q` (= `sqrt(SWAP)`) is not a matrix-gate proxy, so a `su4` layer was
  added.** Measured: `sqrt(SWAP)`'s steady-state fanout is 3.65 rows/term against a dense PTM's 14.94, and
  `sqrt(SWAP)² = SWAP` is Clifford, so repeating one block puts the term count in a **period-2 cycle**
  (10 000 ↔ 32 503) instead of a fixed point — which also makes the reported `n` depend on `--reps` parity.
  `--layers su4` uses one Haar SU(4) block from `circuits.py::haar_su4` under `default_rng(0xC0FFEE)`, i.e. the
  same distribution `su4_gates` and benchmark E draw from. Also added `--truncation keep|coeff:<t>|topn:<N>`,
  statically dispatched, since no existing layer exercised `finalize_layer`. Committed separately (6715918)
  before any timed run; `cargo test --workspace` green, clippy clean, defaults unchanged.
- 09:32 m-sweep done, `--n` 10⁴ → 10⁷ for `rotation_zz`/`gu2q` and the matching `m` grid for `su4`.
  **No cell skipped**: peak RSS 1.48 GB (`rotation_zz`, `m` = 1.5 × 10⁷), 1.65 GB (`su4`, 9.9 × 10⁶), 3.3 GB
  (`gu2q`, 2.1 × 10⁷) — two orders below the 100 GB budget, longest cell 29 s.
- 09:38 **VERDICT (large `m`): no phase is superlinear, anywhere.** Per-term cost is flat to ±10 % over 1000× in
  every channel class (`rotation_zz` 29.2–32.3 ns/term, `gu2q` 61.9–70.9, `su4` 377.9–426.1). The only phase that
  moves with `m` is `gather`, +19 % peak. This is the deep-KI cross-engine verdict reproduced from inside the
  engine.
- 09:39 **VERDICT (small `m`): the fixed cost is `prepare`, not the pipeline.** Per-layer fixed cost is
  **1.43 µs** (2q rotation), of which `prepare` is 70 %; the `rebucket → span_plan → permute → unpermute →
  recount → finalize` serial pipeline is **0.19 µs** and is independent of `m` *and* of the channel to three
  digits. Break-even at `m ≈ 49` terms; 3.3 % of the layer by `m = 1497`. The study README's attribution of the
  small-`m` regime loss to that pipeline is quantitatively wrong by ~7×. `prepare` for a dense 2q PTM is
  **4.19–5.71 µs per gate** (95 % of `su4`'s fixed cost).
- 09:40 **new finding: a bucket-count cliff in the sort at `W = 2`.** `su4` costs 841 ns/term at `m` = 980 (1
  bucket) against 378 at `m` ≥ 1.6 × 10⁴ (≥ 128 buckets) — **2.2× on the layer, 3.3× per sorted row** — and the
  trigger is `desired_bits`'s `worth_splitting` floor at `DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK = 8192`. Not
  run length (rows/run is ~14 700 at 1, 2, 4 and 8 buckets alike). perf says branch misses, not cache:
  2.50 % → 0.71 % miss rate, 535 → 329 insn/row, IPC 2.14 → 2.93, LLC load-miss under 1.2 % throughout.
- 09:48 **and it is `W`-specific, which kills the obvious projection onto the su4 curve.** A one-qubit flip
  `q = 64` (`W = 1`) → `q = 65` (`W = 2`) turns the cliff on and off: at `W = 1` the `su4` sort is a flat
  ~30.5 ns/sorted row at *every* bucket count, at `W = 2` it is 16.1 (B ≥ 128) / 52.5 (B = 1). Since the
  committed su4 curve is `n = 36` qubits, i.e. **`W = 1`**, the cliff cannot explain its 0.620 point, and the
  projection "fixing the floor turns 0.620 into a tie" was **withdrawn before it went into the note**. What
  survives instead: at `W = 1` our matrix-gate per-term cost is flat across the deficit band, both engines see
  parity-identical per-layer term counts, so Julia's per-term cost fell 17.2/10.7 = **1.61×** over that step
  while ours did not move — **the mid-`m` deficit is created on Julia's side, uniform on ours.** That is the
  answer to the phase-correlation question the handoff asked. Separately, `W = 1`'s sort is **1.9× slower per row
  than `W = 2`'s** on high-duplicate runs for half the key bytes, which is a defect in its own right.
- 09:50 **VERDICT (roofline): one phase is bandwidth-bound, and only above 8 threads.** Single-threaded, every
  cell is 2.2–11.7× below its modelled traffic because the traffic is cache-served: `rotation_zz`/`gu2q` at
  `m ≥ 10⁶` measure 2.5–3.9 GB/s = 26–41 % of the 1-core mixed ceiling with 28–36 % LLC load-miss (**latency**,
  not bandwidth); `su4` measures 0.61 GB/s = 6 % with 6.5 % LLC miss and IPC 2.73 (**compute-bound — do not
  optimize its traffic at 1 thread**). At 16 threads `su4`'s write stream hits **25.2 GB/s against the measured
  25.3 GB/s write ceiling (100 %)** and 32 threads adds no bandwidth while costing 8 % of wall time.
  Consequence for Phase 2: **the dense-PTM path must be measured at ≤ 8 threads or the memory controller is the
  independent variable.**
- 09:52 **VERDICT (`TopN`): a big constant, not superlinear.** `finalize_layer` is **61–71 % of layer wall time**
  at every `m` and thread count, 52.4 → 64.2 ns/term over a 30× `m` range (**+22 %**, sub-logarithmic). Its
  zero-finalize control `coeff:0.0` costs +40–48 % over `keep` all by itself, entirely in the merge — and both
  costs share one named cause: `Complex64::norm()` is `hypot` (num-complex `src/lib.rs:217`), one libm call per
  merged term for `CoefficientThreshold` and **two per candidate** for `TopN`. Decision: the note ranks
  `norm_sqr` ahead of histogram selection as experiment (1)'s first step, with the tie-semantics rider spelled
  out.
- 09:58 fact sheet written. Six planned Phase-2 experiments mapped to evidence: (1) `TopN` **supported, re-scoped**
  (`norm_sqr` first, histogram last); (2) bucket tuning **supported but narrowly** — `W = 2` dense PTM below 8192
  terms only, explicitly *not* the su4 curve's numbers; (3) SIMD **partially** — not `gather` (latency-bound), yes
  a radix replacement for the sort, plus the unplanned `W = 1` defect; (4) hybrid/dictionary **supported with a
  different rationale** (the sort, not the pipeline), **not evidenced** for the saturated regime; (5) memory-step
  smoothing **not supported** as a throughput experiment; (6) merge/finalize **implicated as arithmetic, not
  traffic**. Plus two unplanned items (7 `prepare` cost for dense PTMs, 8 the ≤ 8-thread measurement constraint).
- 09:58 Skips and non-measurements, for the record: `--n 10⁷` was run for every layer (nothing skipped for RSS or
  time); `su4`'s `m` grid uses `--n` scaled by its own 14.2× closure factor rather than the literal `--n` values,
  since `--n 10⁷` would have meant `m` = 1.4 × 10⁸; `scripts/host-topology.sh` is not executable in the checkout,
  so `lscpu` output stands in it in `provenance.txt`; the flamegraph HTML name is derived from layer + commit, so
  successive cells of one layer overwrite each other — `perf-report.txt` (symbol tables per cell) is the quotable
  artefact and agrees with `PhaseStats` to within 2 points.

## Phase 2 slate (orchestrator decision, 10:15, from the phase-breakdown fact sheet)

Evidence-driven re-scope of the plan's six experiments into five, run as parallel implementation agents in git
worktrees (branches `expt/<slug>`), with ALL authoritative timing deferred to a serialized ab-compare pass:

- **E1 `expt/topn-finalize`** — TopN finalize cost (61-71% of layer wall): norm_sqr instead of norm/hypot first,
  drop the per-layer candidate Vec, histogram-approximate selection last (plan's experiment 1, re-ordered per
  the fact sheet's decomposition).
- **E2 `expt/sort-kernel`** — the dense-PTM sort (58-60% of layer): radix-sort replacement evaluated against the
  current kernel, plus the W=1 sort defect (1.9x slower than W=2 on half the key bytes). Plan's experiment 3,
  narrowed; `engine/merge.rs` inline set is a known danger zone.
- **E3 `expt/small-m-path`** — plan's experiment 4 re-rationalized: a runtime-selectable direct/dictionary apply
  path for small sums that skips Channel::prepare (the measured small-m fixed cost, 70% of 1.43 us/layer;
  4.2-5.7 us/gate for dense PTMs) and the bucketed machinery below an auto threshold. Additive; sort path stays
  canonical; gate = default-path non-perturbation ab-compare + small-m cross-engine wins (effects >=1.5x, above
  single-shot noise). The saturated-regime hybrid is dropped: fact sheet found no our-side evidence.
- **E4 `expt/bucket-cliff`** — plan's experiment 2, narrowed to where evidence points: the W=2-only bucket-count
  cliff (q=64 -> 65 flips it) and dense-PTM sums below 8192 terms; diagnose, then tuned policy or fact sheet.
- **E5 `expt/mem-growth`** — plan's experiment 5, demoted to an RSS-only target (no throughput evidence): smooth
  the power-of-two capacity slack (63-73 B/term live vs 91-125 observed), A/B to confirm timing neutrality.
- Plan's experiment 6 (merge/finalize traffic) is NOT run: fact sheet says arithmetic-bound, 91% cache-served,
  and the v0.6 rejections stand.

Bench-gate order once implementations land: E1 -> E3 -> E2 -> E4 -> E5, one at a time on a quiet box; dense-PTM
probes at <=8 threads per the fact sheet's write-ceiling finding.

## Gate phase (orchestrator, 12:35)

All five implementation agents landed (E1 6 commits, E2 5, E3 5, E4 5, E5 net doc-only after a measured
self-revert). Serialized gate order on the quiet box: **E1 -> E3 -> E2 -> E4(null-check) -> merge pass**; E5 has
no timing gate (its diff is the negative-result note alone).

Decisions:
- **E4 §6.1 (channel-aware bucket-bit floor sweep) skipped this campaign.** Rationale: it targets the same
  dense-PTM comparison-sort cliff E2's radix kernel attacks from the kernel side; the forced floor costs the
  sparse paths +47-96%; E4 flagged the two changes as non-additive ("re-run the m=1e4 cell after whichever lands
  first"). Recorded as future work to re-evaluate after the radix lands. E4's gate is the null-check only (its
  diff changes no defaults; the risk is code-layout perturbation from the test_support additions).
- Cross-experiment coordination: E4's rank finding was relayed to E2 mid-flight; E2's probe independently
  confirmed the ~1.35x width residual and localized its mechanism (branchless-vs-branchy comparator codegen).
  Known merge conflict to resolve at integration: duplicate Haar-SU(4) fixtures (E4 test_support::haar_su4_matrix
  vs E2 bucketed::tests::haar_su4 -- integrate on the shared fixture), plus 3-way touches on ARCHITECTURE.md,
  phase_breakdown knobs, and truncation/mod.rs (E1 magnitude semantics x E3 finalizes_layer).
### Phase 2, E1 `expt/topn-finalize` — implemented and gated (`research/notes/2026-09-01-topn-finalize.md`)

- 10:20-11:20 implementation, TDD, five commits on `f592c43`: `d9569ee` squared-magnitude comparisons in
  `CoefficientThreshold::keep_term` + all three `TopN` sites; `31289c4` pooled, singly-written magnitude buffer
  and a suffix-only tie test; `4498ac3` `ApproxTopN` (new opt-in policy, exact `TopN` untouched and default);
  `3505dda` `--truncation atopn:<N>` probe knob; `eeb6bfb` ARCHITECTURE.md §Truncation. `cargo test --workspace`
  and `--release` green at every commit, clippy/fmt clean. Core Rust only — no Python binding touched.
- **Design decisions worth the record.** (i) Squaring changes the *tie grouping*, not the ordering; symmetry
  multiplets are exactly preserved because their members differ by a sign or a power of `i` and `re²+im²` is
  bitwise invariant under both. (ii) Squares underflow to 0 below |c| ~ 1.57e-162 — accepted and tested, not
  guarded (determinism policy: tolerance is the bar); the consequence is benign (an all-underflow sum is wiped;
  an underflowing *tail* is dropped whole, keeping the representable terms). (iii) The tie test collapses to "no
  element after the pivot equals the pivot" by substituting `select_nth_unstable`'s partition into the stated
  rule. (iv) The buffer pool is `thread_local!` + `take()`/`set()`, NOT `LayerScratch`: threading scratch into
  `finalize_layer` needs a trait-signature change that touches the bindings, and would have left the Python path
  (every `TopN` user) unimproved.
- 11:28-11:47 **gate run, box exclusively ours** (siblings done; load 0.86 at start, timed regions single-threaded
  and alone). `ab-compare.sh --a f592c43 --b expt/topn-finalize --pairs 3`, `RUST_LOG` unset:

  | leg | median Δ% wall | pairs | mechanism |
  |---|---|---|---|
  | `topn:100000`, m=1e5 | **-44.09** | 3/3 neg | `finalize` 51.0 -> 11.2 ns/term (-78%) |
  | `topn:1000000`, m=1e6 | **-45.40** | 3/3 neg | `finalize` 57.9 -> 15.2 ns/term (-73.7%) |
  | `coeff:0.0`, m=1e6 | **-25.81** | 3/3 neg | `merge` 24.72 -> 12.89 ns/term (-47.8%) |
  | `keep`, m=1.5e6 | +4.44 | 3/3 pos | gather +8.9%; `finalize_ns` ~2us on BOTH sides |

  A-side reproduces the fact sheet's published numbers (93.2 vs 91.12; merge 24.72 vs 24.76; keep 30.1 vs 29.90),
  so the effects are measured against the campaign's own baseline.
- 11:47 **VERDICT: E1 PASSES.** Legs 1-3 are decisive and land at/above the predicted range, with the whole
  effect in the phase each targeted and the other phases flat. `CoefficientThreshold`'s merge is now
  indistinguishable from no-truncation's (12.89 vs 13.12 ns/term), i.e. the +40-48% the fact sheet charged every
  `min_abs_coeff` caller is gone.
- 11:47 **Free correctness corroboration at scale:** across legs 1-3 (18 runs, ~12 000 layer applications)
  `terms_in`/`terms_out`/`rows_gathered`/`rows_sorted`/`cosets` are **identical between the two sides in every
  run**. The squared-magnitude selection keeps exactly the terms the `hypot` version kept on realistic data, and
  `CoefficientThreshold(0.0)` drops nothing extra — the tie/underflow riders never fire.
- 11:47 **METHODOLOGICAL FINDING, applies to every remaining phase-2 gate: a non-perturbation control cannot be
  tested by pairing two fixed binaries.** The `keep` leg came back +4.44% with 3/3 direction consistency, on a
  configuration where `AlwaysKeep` means *none of the change executes* (`finalize_ns` ~2us on both sides, work
  counts identical, gather source byte-identical). Two extra legs with the same probe:
  `f592c43 -> d9569ee` (norm_sqr only) is **-7.48%** on the keep path (merge -17.9%, 3/3);
  `f592c43 -> 4498ac3` (library only, no probe change) is **+2.42%**; the tip is +4.44%. A ~12-point spread on
  untouched source, i.e. LTO code layout — the effect CLAUDE.md already documents for `engine/merge.rs`.
  Direction consistency resolves *run-to-run noise*, which it does well; a build-to-build layout difference is
  not noise, so it presents as a perfectly consistent direction while saying nothing about semantics. E3/E2/E4/E5
  should expect the same artefact in their own default-path controls and should read a consistent few-% move
  there as layout until shown otherwise. Testing layout neutrality properly needs per-side layout perturbation
  (several builds with a trivial unrelated code-size change, or randomized function ordering) — not attempted.
- 11:47 Open risk carried to the merge decision, not resolved by argument: on the *untruncated* path this branch
  sits +4.4% (= +1.3 ns/term) up inside that layout band, against -39.9 and -42.7 ns/term wherever `TopN` runs
  and -11.8 wherever a coefficient threshold does. Options if the orchestrator wants it addressed: gate
  `ApproxTopN` behind a feature to keep it out of the binary, re-order the new code, or measure layout properly.
- 11:47 **`ApproxTopN`, measured within the candidate binary** (not an A/B — no committed baseline accepts
  `atopn:`), 3 interleaved rounds: `finalize` **-16.5%** at N=1e5 and **-18.6%** at 1e6 (3/3 rounds negative at
  both), wall -4.9%/-6.0%, peak RSS **-11.7 MB** at m~1e6, and the approximate-N shortfall is **0.23%** at
  N=1e5 / **1.23%** at 1e6. So the fact sheet's ranking was right: the histogram is the *smallest* of the three
  wins — once `hypot` and the double-write are gone both policies are dominated by the two coefficient passes
  they share, and the histogram's remaining edge is the buffer more than the arithmetic. It earns opt-in status
  for memory-bounded runs, not the default.
- 11:47 Consequence for the slate: E1's `norm_sqr` result means **any later experiment quoting a `finalize` or a
  `min_abs_coeff` merge cost from the phase-1 fact sheet must re-baseline** — those two numbers moved by 74-78%
  and 48%. Nothing else in the fact sheet is affected.
### Phase 2 gate: E3 `expt/small-m-path` (`research/notes/2026-09-01-small-m-path.md`)

- 11:44 gate slot opened (E1 finished, E2/E4 queued behind). Preflight: load 0.34, RUST_LOG unset, worktree clean at
  448d75b, branch point f592c43.
- 11:45-11:52 **gate (a), non-perturbation of the default path: PASS, and the residual is LTO layout, not
  perturbation.** Four ab-compare legs (rotation_zz and su4 at m ~ 1e4 and ~1e6, 1 thread, 3 abba pairs). Wall medians
  -2.47% (3/3), -2.27% (MIXED), +3.94% (3/3), +3.54% (3/3) -- all inside E1's freshly calibrated +/-4-7% layout band.
  Applying E1's discriminator: **every work counter is bit-identical between the two sides and constant across all
  three runs on each side** (layers, terms_in, terms_out, rows_gathered, rows_sorted, rows_id, cosets, n, reps, seed),
  so no phase does more work; what moved is the distribution -- rotation_zz merge -17% against gather +9 / sort +14,
  su4 sort +3.2 (58-60% of that layer) plus merge +22 (7% of it), which reconstructs its +3.4% wall arithmetically.
  Two legs favour the branch, two the baseline. Noted for the record: the remedy E3's note had reserved (give
  `propagate_with_scratch` a verbatim `for k in 0..n` loop) would NOT reliably remove this, since the mechanism is new
  code in a fat-LTO single-codegen-unit build rather than the loop bound.
- 11:53-12:05 **gate (b), ours-only ON vs OFF at the study's small configurations: effect confirmed.** 5 pairs at the
  shipped threshold plus a 5-point threshold sweep {128, 512, 1024, 2048, 4096}, 3 pairs each, reps 30, load ~1.
  `--check` passed at every point: the Rust-ported kicked-Ising and XXZ circuits reproduce the study's committed
  final/peak term counts exactly (7/68, 408/517, 5038/6311, 156/164, 1625/1625, 9918/9918), and the two selections'
  full per-layer count vectors were equal in all 30 leg pairs. The configurations the study loses worst gain
  **2.28-2.36x** (kicked-Ising 2^-4, study ratio 0.323) and **1.48-1.58x** (XXZ 1e-2, 0.460), at every threshold --
  both peak below 512, so they run fully direct regardless.
- 12:05 **threshold decision: 512 -> 2048.** The sweep exposed a second channel the crossover argument missed: a
  threshold above a workload's *peak* keeps the whole run on one path, and being undivided is worth a lot. XXZ 1e-3
  (peak 1625) is a 1.02-1.04 null at 128/512/1024 and **1.682 (3/3)** at 2048. 2048 is the largest value at which
  nothing regresses; **4096 is the cliff** -- kicked-Ising 2^-8 falls to **0.676 (3/3)** and XXZ 1e-4 to 0.926 (3/3).
  Cost of 2048 over 512: kicked-Ising 2^-6 goes 1.079 (5/5) -> 1.043 (MIXED), i.e. a small win becomes a null. Clock
  check on the one sub-ramp cell (kicked-Ising 2^-4's leg is 22 ms at reps 30, under the governor's ~50 ms): re-run at
  reps 300 (region 0.227 s) gives 2.278 (3/3) against 2.350, a 3% difference, so the short region was not
  manufacturing the result.
- 12:06 **gate (c), parity: PASS.** `cargo test -p paulistrings --release --test small_sum_path` 14/14 green (green in
  debug too), every cross-path case asserting per-layer terms_in/terms_out equality; harness parity assert held on all
  six configurations; `cargo test --workspace` green in both profiles, clippy clean, no tripwire moved. The literal
  cross-engine gate with the path enabled still needs the PyO3 kwarg E3's note specifies (plus a `finalizes_layer`
  override on `SpecPolicy`), and is a follow-up.
- 12:06 **VERDICT: E3 passes all three gates.** `EngineSelection::Auto` stays **off by default** as planned -- flipping
  it is a separate decision, and the one piece of evidence still missing for it is a multi-thread cell (the direct
  path has no parallelism, so its crossover moves down with thread count). Not merged; the orchestrator merges.
### Phase 2 gate: E2 `expt/sort-kernel` -- **PASS** (`research/notes/2026-09-01-sort-kernel.md`)

- 12:08 box handed over exclusively (E1 and E3 gates done and PASSED, E4's null-check queued behind). Waited
  load down to 0.85 before the first build; `RUST_LOG` unset; `cargo test --workspace` and `--release` green at
  968598c.
- 12:12-12:34 seven `ab-compare.sh` invocations, `--a f592c43 --b expt/sort-kernel --pairs 3 --order abba`
  (one per `(n, qubits, reps)` triple, since those are per-invocation while `--layers`/`--threads` are CSV).
  **`--order abba` rather than the default `abab`**: with the box exclusively held there was no reason not to
  take the stronger protocol, and it removes the one alternative explanation a uniformly-negative result invites.
- 12:35 **VERDICT: PASS, 8/8 dense cells, 3/3 pairs each.** Layer medians: `W = 2` `q = 128` **-24.0%**
  (m=9884) / **-15.9%** (1e5) / **-15.2%** (1e6) at 1 thread, **-30.3% / -14.0% / -10.5%** at 8 threads;
  `W = 1` `q = 64` **-33.8%** (m=63364, `r = 3`) / **-33.2%** (m=991060, `r = 4`). Sort phase -17% to -50%.
  Acceptance was median <= -10% (`W = 2`) / <= -25% (`W = 1`); tightest margin 0.5 points (8t, m=1e6).
- 12:35 The baseline arm reproduces Phase-1 independently -- 58.9% sort share and 16.18 ns/sorted row at
  `W = 2` against the fact sheet's 58-60% and 16.1; 22.08 ns/row at `W = 1`, `r = 4` against E4's 22.12-22.28.
  That agreement is what licenses reading the deltas as a measurement of the change rather than of the box.
- 12:35 **The `W = 1` width residual is removed, not merely reduced**: 11.11 ns/row against `W = 2`'s 12.07 at
  matched `m` and rank (0.92x), where the comparison kernel is 1.36x -- E4's ~1.35x reproduced and then closed.
  The kernel is order-oblivious, so the near-identical Delta in the `r = 3` and `r = 4` cells is expected: it
  does not repair a deficient rank draw, it removes the sort's sensitivity to one.
- 12:35 **Controls: layout, not perturbation -- no module move.** Applying the discriminator calibrated on the
  E1/E3 gates: work counts (terms_in/out, rows_gathered/sorted/id, cosets, runs) **bit-identical** on every
  control pair; the sort phase, the only phase whose source gained a neighbour, moves +1.0 / -2.2 / -0.6%; the
  delta is concentrated in the **byte-identical** `merge2_into` (-15.0 / -14.0 / +7.2%); and the three layer
  deltas **disagree in sign** (-7.3% rotation_zz q128, -4.0% cnot q128, **+2.2%** rotation_zz q64), all inside
  or at the edge of the calibrated +-4-7% band. The largest magnitude is a *speedup* on an untouched path.
  Risk #1's contingency ("a consistent sparse regression means move the kernel to its own module") is therefore
  **not triggered**; `engine/merge.rs` keeps its shape and its `#[inline]` set.
- 12:35 Free correctness result: `terms_out` is bit-identical between the two sides on every cell including the
  dense-PTM ones, over 40 layers at m=9.9e5 and 2000 layers at m=9884. The kernels are only required to agree
  to FP tolerance, so an identical surviving term count at that scale is a differential check far wider than
  the unit tests reach.
- 12:35 **Campaign consequence for E4**: this kernel already takes the mid-`m` cell -24.0% (layer) / -37.9%
  (sort) with no policy change, so E4's bucket-count constant must be chosen against the **post-merge** sort,
  not against the comparison kernel's numbers -- otherwise it is tuned to close a gap that no longer exists,
  at the +46.6% rotation-layer cost it carries. The two are not additive; gate in sequence and re-run the
  m=1e4 cell after whichever lands first.
- 12:35 Unmeasured and flagged: 16/32 threads for this kernel (the gate stopped at 8 per the write-ceiling
  finding; the 8-thread trend -30.3% -> -10.5% as `m` grows is consistent with a compute win being absorbed as
  the layer approaches the ceiling), 2-7 rest streams (`sqrt(SWAP)`'s regime, left on the comparison kernel),
  and an `#[inline]` hint on the new function.
### Phase 2 gate: E4 `expt/bucket-cliff` (`research/notes/2026-09-01-bucket-cliff.md`)

- 12:41 E4 gate handed the box exclusively. Waited for load to fall below 1.0 (0.95 at 12:45) before starting;
  `RUST_LOG` unset throughout.
- 12:45 **scope decision (orchestrator): E4's §6.1 channel-aware-floor sweep is SKIPPED this campaign.** E2's
  radix kernel passed its gate first and already takes the mid-`m` dense cell -24.0% layer / -37.9% sort with no
  policy change; the two are **non-additive** (a radix sort is indifferent to how presorted its input is, hence
  indifferent to the coset dimension `r`, which is E4's whole mechanism); and E4's forced floor costs the sparse
  path +47-96%, so it is strictly a post-radix re-evaluation. Recorded as future work in the note's §6.1, not
  cancelled.
- 12:46-12:56 **§6.2 null-check ran: 6 cells, `ab-compare.sh --a f592c43 --b expt/bucket-cliff --order abba
  --pairs 3`,** one invocation per cell. Artefacts `benchmarks/results/2026-09-01-ccqlin038/null-*`.
  Wall medians: su4 q=128 m=980 **+2.21%** (3/3), m=7868 **+0.22%** (3/3), m=63518 **+3.16%** (3/3), su4 q=65
  m=63518 **+3.87%** (3/3), su4 8t m=1e6 **-0.10% null**, rotation_zz 1t m=1.5e6 **+0.13% null**.
- 12:58 **VERDICT: layout-band motion, not perturbation.** Four discriminators, all pointing the same way.
  (1) **Work counts bit-identical** in every cell and every run -- `terms_in`/`terms_out`/`rows_gathered`/
  `rows_sorted`/`rows_id`/`layers`/`cosets`/`runs`; layout cannot change these and a behavioural change could
  not avoid changing them. (2) Deltas concentrated in the motion-sensitive kernels and **sign-inconsistent
  across cells**: merge +27..+34% on the 1t su4 cells but **-13%** on rotation_zz; gather +5% / -2..-4% / +9.5%.
  (3) **Every cell's wall delta is exactly its share-weighted phase sum** at the fact sheet's own phase shares
  (predicted +2.22 / +0.21 / +3.02 / +3.77 / -0.03 against observed +2.21 / +0.22 / +3.16 / +3.87 / +0.13).
  (4) Worst median +3.87%, worst single pair +5.01% -- inside the +-4-7% band the three finished gates
  calibrated. Not investigated further, per the gate's own rule.
- 12:58 **The layout shift is structurally probe-only.** Verified mechanically: every line E4 changes under
  `crates/paulistrings/src/` is a doc comment (zero non-comment added lines in `bucket/sum.rs`,
  `engine/bucketed.rs`, `engine/merge.rs`), code inside `#[cfg(test)] mod tests`, or one of the 93 new lines in
  `test_support.rs`, which is `#[cfg(any(test, feature = "test-utils"))]`. `phase_breakdown` links `test-utils`,
  so with `lto = "fat"` + `codegen-units = 1` the extra fixture code reshuffles the probe binary -- but **a
  shipped build compiles identical code on both sides**, so no user-visible delta exists. Caveat recorded in the
  note: the four cells that moved are one code path at four sizes, not four independent paths, and the two
  genuinely different paths measured are both null.
- E4 gate **PASSED**. No default constant changed on the branch; branch tip 204a529 plus the results commit.

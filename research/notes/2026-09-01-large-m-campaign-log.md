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

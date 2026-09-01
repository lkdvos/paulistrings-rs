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

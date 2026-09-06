# Decisions log

Numbered record of every choice made while the user was away (2026-09-06). Each entry: the question, what
was chosen, the alternative(s), why, and where to look to revisit. Entries are appended, never rewritten.
Items that genuinely need the user are collected under **Needs you** at the end.

## Entries

1. **Naive baseline = the shipped direct path.** `EngineSelection::SmallSumDirect` with
   `small_sum_threshold = usize::MAX` runs the whole propagation on one hashbrown/FxHash map. Alternative: a
   hand-written map loop in the bench crate. Chosen because it is production code with an existing differential
   test, so the "naive" bar is not a strawman. Revisit: `presentation/bench/src/naive.rs`.
2. **Old multithreading attempts are reconstructions.** Per-thread maps + merge, and flat-array parallel
   mergesort, re-implemented in Rust on the same circuit. The slides label them as such; the remembered
   "≤2× on 32 cores" is quoted as the historical memory, the plotted numbers are the fresh ones.
   Revisit: `bench/src/{threadmaps,mergesort}.rs`, `data/thread_scaling.jsonl`.
3. **Bucket-size lever lands in the engine** as `PropagateOptions { target_bucket_len, min_buckets }` with
   defaults preserved and no Python exposure. Alternative: probe-only hack. Chosen per the user's mid-turn
   note that engine levers are allowed. Revisit: `crates/paulistrings/src/engine/mod.rs`.
4. **Spine figure uses total propagate wall time**, not per-layer time: the naive and threadmap baselines have
   no cheap per-layer instrumentation. Per-layer curves appear only for the bucketed engine.
5. **Bench crate is a separate cargo workspace** (`presentation/bench`, excluded from the root workspace) so
   `phase-timing`/`test-utils` never unify into the shipped build and the `target-cpu=native` build has its own
   target dir.

## Needs you

(none yet)

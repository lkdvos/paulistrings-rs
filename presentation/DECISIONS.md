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
6. **F0's observable is `Z_62` built via `observables.single_z(62, 127)`**, not `observables.canonical_z_127()` /
   `kim2023_operator("weight_1_z62")`. Same operator (the module docstring notes `single_z` builds it without the
   provenance detour); `single_z` avoids a dependency on `examples/data/kim2023_observables.json` for a plot that
   isn't citing the paper's numbers. Revisit: `presentation/plots/collect_term_growth.py`.
7. **Untruncated growth curve is built by re-propagating `circuit[:m]` from scratch for increasing `m`**
   (`Circuit.__getitem__` slicing), not a fallback to `trotter_steps=1`/`2` as the task spec's contingency
   suggested. `propagate_with_stats` can't stop mid-circuit, but slicing gives a real "stop early" without
   needing a reduced circuit: cheap while the sum is small (measured 2026-09-06 on ccqlin038: whole sweep to
   m=1146/1355 channels, peak 100482 terms, in ~24s), and it naturally halts one channel before the next
   explosive jump (peak_terms next channel: 82.5M) rather than needing a guessed step count. Revisit:
   `presentation/plots/collect_term_growth.py::_untruncated_growth`.
8. **F0's 9 truncation curves are colour-coded by a viridis sampling**, not the shared categorical 8-color
   palette (only 8 slots, and the eps sweep is an ordered quantity, which a sequential colormap communicates
   better than categorical colors). Revisit: `presentation/plots/fig0_term_growth.py`.

## Needs you

(none yet)

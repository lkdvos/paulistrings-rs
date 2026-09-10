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

9. **Working point moved from 5 to 10 Trotter steps at ε = 2^-12** (2710 layers, 1.07 M peak terms, half the
   layers above 10⁵ terms). At 5 steps only the last ~60 processed layers are heavy (backward light cone), so a
   run measures mostly per-layer fixed cost and the parallel speedup is Amdahl-capped by ~1300 tiny serial
   layers (32 threads: 8.5×). The 5-step results are kept in `data/alt-5steps/` for comparison. Revisit:
   `bench/scripts/collect_all.sh` (`STEPS`, `EPS`), `data/calibration.jsonl`.
10. **`-C target-cpu=native` is not a null result.** Paired abba A/B at 5 steps: bucketed −9.2 % (10/10 pairs
    same sign), naive −2.7 % (10/10). Slide 9 was reframed from "does not matter" to "a real, consistent few
    percent — two orders of magnitude short of what follows". Revisit: `data/alt-5steps/targetcpu_ab.md`,
    `data/targetcpu_ab.md` (10-step rerun).
11. **Single-thread bucket size does not move rotation layers** (5 steps: 65 536 and 262 144 terms per bucket
    run within 1 % of the 1024 default). The L2 argument therefore rests on the multi-thread sweep and the
    hardware counters, and slide 23 is worded to match whatever the 10-step sweep shows. Revisit:
    `data/bucket_sweep.jsonl`, `data/bucket_sweep_perf.jsonl`.
12. **Reconstructed baselines run with 2 repetitions and no warm-up** in the scaling stage (a single-thread
    per-thread-map propagation takes ~140 s, mergesort ~55 s); bucketed cells keep 5 repetitions + warm-up.
    The engine-ladder stage no longer re-measures them; F1 takes their best-of-threads from
    `thread_scaling.jsonl`.
13. **Contamination note.** The `threadmaps` 1-thread scaling cell (05:02–05:31) overlapped with a stray
    single-thread A/B chain left over from the aborted 5-step campaign (killed at 05:31). Two single-thread
    processes on a 32-core box: expected effect a few percent on that one cell. Not rerun unless time permits.
14. **Engine knob LTO check** (`scripts/ab-compare.sh`, 3 pairs abba): 1-thread rotation −0.29 % (3/3 but far
    inside the ±4–7 % layout band), 32-thread rotation and su4 cells null. Load was 5.8 at the start (sibling
    agents building). Logs: `benchmarks/results/2026-09-06-ccqlin038/knob-*-ab.log` (gitignored).
15. **Two delegated agents stalled without output** (bench crate, figure scripts) and were replaced: the bench
    crate was written by the orchestrator; the figure scripts were adopted from the agent's uncommitted files
    and patched (F1 reads the old-attempt rows from `thread_scaling.jsonl`).
16. **Deck theme**: Carlito 18 pt body, Source Code Pro, New Computer Modern Math; navy section dividers;
    palette from `examples/common/report.py`. Alternative (Libertinus Serif) not produced.

17. **Slide 22 was rewritten from "more than linear" to "the bucket size decides whether cores help at all".**
    The campaign found no superlinear speedup: 10.5× on 16 cores at the default bucket size, 9.5× at 3.9 M
    terms, and every bucket size runs at the same single-thread speed (F5). What the data do show is that the
    same 16 threads deliver 11× or 2× depending only on the bucket size (F4b), with the parallel-efficiency
    panel separating task starvation from the memory system. The original superlinear observation may have
    come from a different host, width (`W = 1`), or the Julia-era code; see **Needs you**.
18. **Coarse arm of the thread-scaling stage** uses `--target-bucket-len 16384 --min-buckets 64` (64 buckets
    at 1.07 M terms, 256 at 3.9 M). It tracks the default arm within 20 % — the collapse needs ≤ 32 buckets,
    which the sweep stage provides. Both are shown (main text: sweep; appendix: thread curves).
19. **F0 on the exponential-wall slide is the 10-step curve** (`term_growth_10steps.jsonl`); the 5-step
    figure is kept (`fig0_term_growth.svg`) but not used. The untruncated curve is partial (stops at 2 M).
20. **fig7 (per-layer profile) was not produced** — no run used `--layer-times`; the binary supports it
    (`presentation-bench bucketed --layer-times`) if a per-layer slide is wanted.
21. **Slide 28's two superseded bullets became one shipped-result bullet** after the partitioned engine
    merged to `main` (2026-09-10). "NUMA-aware placement — the smart version has to steal within a socket
    first" and "Distributed prototype — the exchange plan exists on paper" are both answered, and not the way
    the slide guessed: the answer is cut partition rows along the circuit graph plus one pinned pool per
    domain, not smarter stealing. The slide carries one number (4 of 271 layers remote, 21 % per step at
    `P = 2`); the rest — −20.7 % / −14.5 % at 16 / 32 threads, 125 / 133 / 136 ms at 2 / 4 / 8 MPI ranks,
    cut rows 1.9–2.4× over random — is in the speaker notes, per the deck's usual split. Static coset
    placement stays in the appendix as the negative result it is. Slide count unchanged (32 pages).
22. **No figure or number was regenerated for the merge.** The library-dependent ladder cell was re-measured
    on the merged engine, same host and working point: identical peak terms (1 071 093) and layer count
    (2710), 15.63 s at 1 thread and 1.569 s at 32 against the campaign's 15.3 s / 1.46 s — inside the
    ±5–8 % / ±10–26 % single-shot bands in `benchmarks/PROFILING.md`. The talk's measurements stand as taken.
    `presentation/bench/Cargo.lock` did need updating: the core links `libc` for placement now.

## Needs you

- **Superlinear speedup.** Not reproduced here (DECISIONS #17). If you remember the setting where you saw it
  (host, qubit count / `W`, term count, bucket count, Julia or Rust), the sweep is one command:
  `presentation/bench/scripts/collect_all.sh` with `STEPS`/`EPS`, or `phase_breakdown --target-bucket-len`.
  Until then the talk says "cores bring cache; buckets let the working set live there" without the word
  superlinear.
- **Working point.** 10 Trotter steps at 2⁻¹² instead of the 5-step utility-experiment depth (DECISIONS #9).
  The 5-step numbers are in `data/alt-5steps/` if you prefer the shallower circuit for the story.
- **target-cpu=native slide.** Reframed to "real but small" (DECISIONS #10). If you would rather drop the
  kernel-flags slide, slides 9 and F1 stage 2 are self-contained.
- **Speaker identity line** on the title slide ("Lukas Devos · CCQ, Flatiron Institute · 2026") and the venue
  are placeholders.
- **The abstract still frames distributed memory as a direction** — "extends naturally towards distributed
  memory", "a statically known communication plan" (`ABSTRACT.md` lines 17 and 36). It is now built and
  measured, so the abstract could claim the result instead. Left untouched on the assumption the announcement
  text has already gone out; say the word if it has not.


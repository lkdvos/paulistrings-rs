# Handoff: large-m performance investigation & optimization campaign

**For**: a fresh orchestration session on this repo (`paulistrings-rs`). You orchestrate: delegate subtasks to subagents (pick model/effort per task complexity), parallelize what's independent, start by creating a branch, finish with a PR. The user prefers autonomous end-to-end execution with decisions logged for post-hoc review (see memory `autonomous-execution-preference`); escalate only genuine blockers.

## Mission

The 2026-09-01 cross-engine study (`benchmarks/python/jl_performance/`, PARTIAL — kicked-Ising 9 configs and XXZ 7 configs complete with 80/80 sign-consistent pairs; SU(4)/accuracy/thread-scaling cancelled in favor of this campaign) measured our single-threaded advantage over PauliPropagation.jl 0.8.2 as **non-monotone in term count** on kicked-Ising: Julia wins below the ~3.79×10³-peak-term crossover, our lead peaks at **1.93× near 6.4×10⁵ terms, then decays to ~1.39× at 2.15×10⁶**. XXZ crosses at ~1.65×10⁴ and is still *climbing* (1.80× at 2.66×10⁶). Ratio convention: t_jl/t_ours, >1 = we're faster.

**The study's leading hypothesis (test it FIRST, before optimizing anything): the kicked-Ising decay is a saturation discount for Julia's dictionary, not a large-m weakness of ours.** Measured per-term costs across 6.4×10⁵ → 2.15×10⁶ terms: ours 1252 → 1120 ns/term (−11%), Julia 2424 → 1567 (−35%). At 5 Trotter steps the reachable Pauli set is nearly exhausted (2⁻¹⁶→2⁻¹⁸ cutoff adds only 1.16% terms; Julia's RSS pins at 1.07 GiB), so nearly every gate application is a lookup+add with no insert/rehash/growth — while our gather→sort→merge costs the same whether keys are new. XXZ is far from saturation (10× tighter cutoff → 3.43× terms) and shows no decay. **Falsifiable test: kicked-Ising at more Trotter steps so 2×10⁶ terms is far from closure — if the ratio keeps rising and jl's per-term cost stays flat, the hypothesis holds; if the decay persists, it's a genuine large-m property of our engine.** Run this before Phase 2 — it decides where the optimization effort goes.

Whatever the verdict, three measured targets stand regardless:
- **The small-m regime (below ~4×10³–1.6×10⁴ terms) is lost outright** to the dictionary (0.31–0.63× on kicked-Ising) — our fixed per-layer costs dominate there.
- **The matrix-gate path is unmeasured and looks much worse**: a 1-pair SU(4)/gu2q pilot put its crossover near 7.6×10⁴ — 20× the rotation workloads'. Measure it properly; fanout-16 gather/merge is the suspect.
- **An our-side memory step**: floor-subtracted 91 → 125 B/term between 1.54M and 2.15M terms (peak RSS ×1.69 for ×1.37 terms) — a capacity-doubling signature. Doesn't explain any timing effect, but it's a concrete allocation target. (Baselines: our floor 37.7 MB vs jl 0.600 GiB; steady-state ours ~92–123 B/term vs jl 237–398.)

## Hard constraint

**The sorting-based engine stays.** It is the GPU-readiness pillar (`ARCHITECTURE.md §GPU-Readiness`, §Determinism, Pod layout). A dictionary/hash path may be **added** as a runtime-selectable alternative or hybrid (e.g. small-m regime), never as a replacement, and every accepted change must keep `cargo test --workspace` green and the sort path canonical.

## Phase 1 — re-measure where the time goes (before any optimization)

- **First: the saturation falsification test above** (deep kicked-Ising, same protocol as the study — reuse `benchmarks/python/bench_jl_performance.py`; its README documents the interleaved-pairs discipline, and `jl_performance_recover.py` shows how to rebuild results from run.log if a run is interrupted). Its verdict gates everything below.
- **Second: measure the matrix-gate (gu2q) curve properly** — the study's biggest blind spot and possibly the biggest win.
- `cargo run --release --features phase-timing --example phase_breakdown` across m ∈ {1e5, 3e5, 1e6, 3e6, 1e7} on the study's workloads (kicked-Ising ZZ rotations; XXZ; gu2q), both near and far from saturation. Which phase (gather/sort/merge/rescale/finalize) dominates per-term cost, and does any grow superlinearly? `PhaseStats` field semantics: `crates/paulistrings/src/engine/stats.rs` (wall vs worker-busy clock domains).
- Roofline: `scripts/bandwidth.sh` ceiling is in `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`; `benchmarks/PROFILING.md §Roofline` has the bytes-moved model. If a phase is ≥70% of ceiling, it's bandwidth-bound — stop optimizing arithmetic there.
- `scripts/perf-stat.sh` for cache-miss/IPC per m; `scripts/profile.sh` flamegraphs at small vs large m.
- jl side: instrument `benchmarks/julia/runner.jl` runs at matching m (per-gate wall, GC time via `@timed`/GC stats, RSS already emitted). Establish whether jl's per-term-per-gate time is flat or falling in m.
- Also measure TopN's `finalize_layer` cost separately (`truncation/builtin.rs`) at large m — it's a per-layer partial selection over the whole sum and a prime superlinear suspect; note most study configs used coeff-only truncation, so check whether the decay exists with and without TopN in the policy.
- Deliverable: a fact-sheet note `research/notes/2026-09-XX-large-m-phase-breakdown.md` naming the culprit phase(s) with numbers, before any optimization work starts.

## Phase 2 — optimization experiments (parallel subagents, one per idea)

Every experiment: its own subagent, TDD where behavior changes, and **`scripts/ab-compare.sh` direction-consistency as the merge gate** (median Δ% effect size; sign-disagreeing pairs = no change = not merged). Write a `research/notes/` entry per experiment, **including negative results**. Read the existing negative-result notes first and do not re-attempt them: `2026-08-26-why-s5-concatenation-fails.md`, `2026-08-30-static-coset-placement.md`, `2026-08-31-v0.6-results.md`.

1. **TopN → histogram-approximate selection** (user-suggested): replace exact partial selection with a coefficient-magnitude histogram (e.g. log₂-bucketed |c|) that picks a threshold keeping ≈N terms. Semantics change: approximate N and a different tie story — allowed (determinism policy: bitwise preservation is NOT required, tolerance is the bar; the byte-identity/fingerprint tests are tripwires to regenerate or demote to `assert_terms_close` in the same commit). Document the new contract; keep exact TopN available; A/B both the truncation cost and end-to-end.
2. **Bucket count/size tuning**: today's policy is in `ARCHITECTURE.md §Bucketing/§Bucket-Policy` (`bucket/hash.rs`, GF(2) hash h(v)=H·v; channel output buckets statically predictable). Sweep bucket counts at fixed m across workloads; relate the optimum to (working set per bucket vs L2/L3 sizes) × (channel type — fanout-1 rescale vs fanout-2 rotations vs 16-fanout gu2q). Deliverable: either an adaptive bucket policy (re-bucketing already exists: `refine`/`coarsen`/`rebucket`) with a measured model, or a tuned constant + the fact sheet explaining it. Host cache topology: `scripts/host-topology.sh` conventions.
3. **SIMD kernels**: the sort/merge/gather kernels (`engine/merge.rs`, `engine/coset.rs`) — evaluate explicit SIMD (nightly `std::simd` is off-limits if it breaks the pinned toolchain 1.94.0; prefer stable intrinsics or autovectorization-friendly restructuring). **Danger zone**: `engine/merge.rs`'s `#[inline]` set is A/B-verified load-bearing in both directions (one hint ±20-34%) — read its comments; every SIMD change needs its own ab-compare, and LTO layout effects can masquerade as SIMD wins.
4. **Dictionary path** (additive): a hash-map merge engine variant (FxHashMap like `BuildAccumulator`) selectable at runtime (or auto below a size threshold), to (a) win the sub-4×10³-term regime jl currently owns, (b) exploit the saturation regime if the hypothesis holds — a saturated sum is exactly where lookup+add beats re-sorting unchanged keys, so a hybrid that detects saturation (terms_out ≈ terms_in across layers, via the always-on TermTrace) and switches merge strategy is a measured-motivation design, (c) serve as an in-repo baseline for what dictionary scaling actually looks like at large m. Must not perturb the sort path when unselected (ab-compare the default path against pre-change).
5. **Our memory step**: find the ×1.69-RSS capacity jump between 1.5M and 2.15M terms (bucket column growth policy? scratch doubling?) and smooth it (e.g. reserve from TermTrace-predicted sizes, or a gentler growth factor) — allocation-only, but A/B it anyway (allocator behavior moves timings).
6. (If Phase 1 implicates it) **merge/finalize memory traffic**: only with Phase-1 evidence, and mind the v0.6 note's three already-rejected gather/merge variants.

## Phase 3 — re-validate

Re-run the cross-engine study protocol (`benchmarks/python/bench_jl_performance.py`, its README documents the interleaved-pairs protocol) on the improved engine: crossover moved? decay gone? Update the study results dir (append, don't overwrite) and the docs site's comparisons page numbers if they change (site: `docs/book/`, numbers-must-cite-committed-files policy).

## Discipline (binding, from CLAUDE.md — read it in full)

- Benchmark `--release` only; `RUST_LOG` unset for campaigns; single-shot noise ±5-8% ST / ±10-26% MT — anything smaller needs ab-compare; `maturin develop --release` after any Rust change; workspace tests green at every commit; commit style `<type>: <lowercase imperative>` + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- jl comparisons: one gate per channel (parity rule), non-dyadic cutoffs or one-ulp perturbation (`benchmarks/julia/README.md` §P3), per-layer term-count parity gates every timed config, warm in-process timings both sides, `RAYON_NUM_THREADS=1`/`JULIA_NUM_THREADS=1` before the interpreter.
- Never execute Slurm commands; heavy runs are sequential on the workstation, time-boxed with recorded projections; other sessions may share the box — timed runs need it quiet (check load).

## Key data pointers

- Study: `benchmarks/python/jl_performance/` (protocol README, per-pair results.json, figures, `jl_performance_recover.py`). Kicked-Ising: crossover 3.79e3, ratios 0.32 → 1.13 (6.3e3) → 1.93 (6.4e5) → 1.39 (2.15e6, saturated). XXZ: crossover 1.65e4, still climbing at 1.80 (2.66e6). All 80 pairs sign-consistent; parity held everywhere (≤2.6e-16).
- Memory (each engine sampling its own /proc — never getrusage(CHILDREN)): floors 37.7 MB vs 0.600 GiB; steady-state ours 92–123 B/term vs jl 237–398 B/term.
- Engine phase hot-spot precedent: `benchmarks/results/2026-08-31-ccqlin038/a2-*.log` (the A2 ab-compare runs and their per-phase tables).
- Reference workload drivers: `benchmarks/python/bench_{c_deep_trotter,e_su4}.py`, `examples/xxz_chain/run_benchmark_d.py`.

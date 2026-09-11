# Research

Negative-result records, hardware fact sheets, and forward design notes.
Nothing here is load-bearing for the build — design documentation lives in
`ARCHITECTURE.md`, operational procedure in `benchmarks/PROFILING.md`.

Naming convention: `YYYY-MM-DD-short-slug.md`. Keep the date so the timeline
of measurement is preserved.

**Check the negative-result notes before re-attempting an optimization idea**
— each records what was tried, the measured reason it lost, and what a future
attempt would need to do differently:

- `notes/2026-08-26-why-s5-concatenation-fails.md` — support-bit bucket
  concatenation cannot replace a sort (four-term counterexample).
- `notes/2026-08-30-static-coset-placement.md` — static coset→worker
  assignment loses 1.25–1.9× to work-stealing.
- `notes/2026-08-31-v0.6-results.md` — three rejected gather/merge variants
  (recompute-in-merge borrow, segment-copy merge, interleaved key layout).

## The 2026-09-10 front-end campaign

Seven notes from one day, and they are a chain rather than seven topics — read in this
order. It began as a SIMD evaluation, which failed, and the anomaly that killed it turned
out to be worth ~25% on the sparse layers.

1. `notes/2026-09-10-simd-evaluation.md` — **SIMD rejected.** The build was compiling to
   SSE2 (zero `popcnt`, zero AVX). Enabling the ISA is *net negative* on two of three
   priority layers, and a targeted `#[target_feature]` gather made an **untouched sort
   34.66% slower**. §3c diagnosed that as hot-path code size; it is struck in place —
   wrong cause, right symptom, and the inference the next reader will also reach for.
2. `notes/2026-09-10-hot-path-code-size.md` — the real cause: the **JCC erratum
   (SKX102)**. One build flag takes DSB residency 45.8% → 98% and is worth −9..−13% on the
   reference host. This is what the "adding any code moves untouched hot paths ±4–7%"
   folklore always was.
3. `notes/2026-09-10-inline-set-repost.md` — **the `merge.rs` `#[inline]` folklore is
   dead.** Re-measured on the padded build, three of the four hints are *provably
   codegen-inert* at `lto = "fat"` + `codegen-units = 1`: adding or removing one leaves
   `.text` byte-identical. The surviving constraint is the sort *algorithm*, and it is
   narrower than recorded — `sort_unstable_by` costs +44% on cnot but is neutral on
   rotation_zz, which gathers one stream.
4. `notes/2026-09-10-branch-misprediction.md` — bad speculation measured at 7–16% of
   cycles and ~40% of it removed. Also **a branchless `merge2_into` rejected**: it removed
   no mispredicts (they relocated to the drain test at the same ~35% rate) and lengthened
   the loop-carried chain. Third confirmation of the rule that branchless only pays when
   the branch genuinely misses *and* the predicate is not itself the dependency.
5. `notes/2026-09-10-constant-recalibration.md` — both pre-padding constants **survive**,
   and the cheapest finding in the set: every built-in plan has 1, 3 or 15 rest streams, so
   the gate has four settings, not fourteen.
6. `notes/2026-09-10-presortedness-predictor.md` — **presortedness rejected** as the
   missing predictor; it is a *constant* across every built-in layer. The real separator is
   branch predictability, which the PTM's amplitude support already knows.
7. `notes/2026-09-10-jcc-portability.md` — the flag is a **~1% tax off Skylake** (13 of 13
   direction-consistent phase results across rome/genoa/icelake). Hence it is *not* in
   `.cargo/config.toml`: the default is portable and measurement hosts opt in via
   `scripts/jcc-rustflags.sh`.

Two methodological rules earned here, both cheap to forget:

- **A direction-consistent phase delta is not an effect if the total is flat.** Seen twice:
  `gather_ns` +1.04% at 7/7 against `merge_ns` −2.33% with instructions flat. Let
  instruction count settle it.
- **Suspect any constant tuned by wall-clock A/B before 2026-09-10.** Four recorded
  conclusions dissolved on re-measurement, all the same artifact.

Fact sheets and open items:

- `notes/2026-08-30-bandwidth-ceiling-ccqlin038.md` — the reference host's
  measured DRAM ceiling (the denominator for every roofline claim). Rerun
  only after hardware changes.
- `notes/2026-08-31-local-ptm-generalization.md` — design sketch for
  supporting custom channels on more than two qubits.
- `notes/2026-08-31-python-test-triage.md` — the first execution of the
  Python test suite, and the Y-phase convention conflict it exposed
  (resolved: parsers now use the core's Hermitian convention).

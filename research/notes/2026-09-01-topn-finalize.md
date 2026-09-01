# E1 `expt/topn-finalize`: the truncation path's arithmetic and its candidate array

Branch `expt/topn-finalize`, forked from the campaign tip `f592c43`. Executes experiment **E1** of the phase-2
slate in `research/notes/2026-09-01-large-m-campaign-log.md`, in the priority order the phase-1 fact sheet
(`research/notes/2026-09-01-large-m-phase-breakdown.md` §7 item 1) set: `norm_sqr` first, the candidate `Vec`
second, histogram selection last and opt-in.

**Gated 2026-09-01 11:28–11:47 on an exclusively-ours ccqlin038** (§3): `TopN` layers are **−44 to −45 %** on
wall, `CoefficientThreshold` **−26 %**, both 3/3 pairs, the whole effect in the phase it was aimed at. The
untruncated control moved +4.4 % — and two extra legs show the same untouched code moving anywhere from −7.5 % to
+4.4 % on build layout alone, which is what that control actually measured (§3.2). Verdict and its qualification
in §4. Development was done without authoritative timing (four siblings were live on the box then); the smoke
numbers that stood here have been replaced by the gate's, which agree with them.

## 0. What the evidence asked for, and what it got

| fact sheet item | measured cost it named | what shipped |
|---|---|---|
| (1a) `norm()` → `norm_sqr()` | 11.8–14.4 ns/term in `CoefficientThreshold` (merge nearly doubles); 2 calls/candidate in `TopN`, ~24 of ~56 ns/term | `d9569ee` |
| (1b) stop materializing the magnitude `Vec` | 12 MB/layer at `m` = 1.5 × 10⁶ | `31289c4` — pooled, filled once, plus one fewer full pass |
| (1c) exponent-histogram selection | "the *smallest* of the three" | `4498ac3` — a **new, opt-in** policy; exact `TopN` unchanged and still the default |

Also fixed as the fact sheet asked ("worth fixing while there"): `CoefficientThreshold` cost **+40–48 % of total
layer time** versus `keep`, paid by every Python caller passing `min_abs_coeff`. Same one-line change.

## 1. Design

### 1.1 Squared magnitudes (`d9569ee`)

`x ↦ x²` is strictly increasing on `[0, ∞)`, so `|c| > t ⟺ |c|² > t²` and every *ordering* the truncation path
needs is preserved. `norm_sqr()` is `re·re + im·im`; `norm()` is `hypot`, a libm call worth 42–52 cycles.

The design question the fact sheet flagged was the **tie semantics**, since `TopN`'s rule turns on exact `f64`
equality at the threshold. Resolved as follows, and documented on the types:

- **Symmetry multiplets — the reason the tie rule exists — are exactly preserved.** Their members differ by a
  sign or a power of `i`; `re² + im²` is bitwise invariant under negation and under swapping the two squares
  (addition commutes), so bitwise-equal magnitudes stay bitwise-equal squares.
- Two magnitudes differing by ~1 ulp may merge into one squared tie group, or (with different `re`/`im` splits)
  tie under `hypot` and differ by an ulp when squared. Both are inside floating-point tolerance, and neither can
  break the `≤ n` bound: `t²` is the `n`-th largest square, so `count(> t²) ≤ n − 1` by construction.
- **Underflow, decided and tested rather than guarded.** `|c|²` is subnormal below `|c| ≈ 1.49e-154` and rounds
  to `0.0` below `|c| ≈ 1.57e-162`, so that band collapses into one tie group. Two consequences, both pinned by
  unit tests: a sum whose magnitudes *all* underflow is wiped (it is zero to any tolerance, and this is the
  documented all-tied wipe, not a new failure mode); and a cut landing inside an underflowing *tail* drops that
  tail whole and keeps the representable terms — strictly better than padding the result with denormal noise.
  `CoefficientThreshold(0.0)` likewise drops that band. Per the determinism policy, tolerance is the bar.
- **Negative `ε` needed a guard.** `|c| > ε` is vacuously true for `ε < 0`, which squaring inverts. The predicate
  is `eps < 0.0 || c.norm_sqr() > eps * eps`; both the load and `eps * eps` hoist out of the merge loop (`&self`
  is `noalias readonly`, no interior mutability). NaN `ε` drops everything, as before.

The `layer_fingerprints_are_stable` net runs under `AlwaysKeep`, so no byte-identity tripwire is in this change's
path and none needed regenerating.

### 1.2 The candidate array (`31289c4`)

Two costs here, and the allocation was the smaller one:

1. `collect()` on an **unindexed** parallel iterator (`flat_map_iter`) cannot write the array once. Rayon takes
   the `ListVecConsumer` path: one `Vec` per split, grown by doubling, then concatenated into the result. So
   every magnitude was written twice and the intermediate buffers churned. It is now filled through one
   `&mut [f64]` handle per bucket, carved off with `split_at_mut`, into a buffer taken from a thread-local pool.
   The handle vector is 16 B per bucket against 8 B per *term* — ~1/500 of the array at the default 1024-term
   bucket target.
2. The tie-group test read all `len` magnitudes. Substituting `select_nth_unstable`'s own partition guarantee
   into the stated rule collapses it: with `e_pre`/`e_suf` the counts of ties before/after index `n − 1`,
   everything before is `≥ t²` and everything after is `≤ t²`, so `count_gt = (n − 1) − e_pre` and
   `count_eq = e_pre + 1 + e_suf`, whose sum is `n + e_suf`. Hence **the group fits iff no element after the
   pivot equals the pivot** — a scan of the `len − n` suffix that stops at the first tie. The equivalence is
   checked by the pre-existing oracle test across straddling and fitting cuts at four partitions each.

The pool is `thread_local!`, not `LayerScratch`: reaching `finalize_layer` from the engine's scratch would mean a
trait-signature change, which the PyO3 bindings implement — out of scope for this experiment, and it would have
left the Python path (every `TopN` user) unimproved anyway. The buffer is **borrowed out with `take()` and
returned at the end**, never held as a live `RefCell` borrow across a parallel section: rayon work-steals on a
blocked thread, so a nested `propagate` (several observables under one `par_iter`) can re-enter `finalize_layer`
on the very thread already inside one. Re-entry then finds an empty slot and allocates — correct, merely
unoptimized — where a held borrow would panic. Two tests guard the mechanism: a smaller layer after a larger one
must read only its own prefix, and `finalize_layer` must survive running inside a rayon job.

**Deliberately not done: exact radix-select.** Histogram the octaves, collect only the straddling octave's
members, select within that — exact, and it would shrink the array from `m` to one octave's population. It costs
a *third* full pass over the 16 B/term coefficient columns, which at `m ≥ 10⁶` are the cache-cold stream (§4 of
the fact sheet: the sparse-PTM path is latency-bound with a 28–36 % LLC load-miss rate), against saving an 8 B/term
array that fits better. The direction is not predictable without measurement, and this experiment cannot measure.
Left as a documented option.

### 1.3 `ApproxTopN` (`4498ac3`)

A new policy, not a mode: `TopN` is untouched and remains the default.

Bin every term by the binade (octave) of `|c|²` — the 11-bit `f64` exponent, i.e. a factor 2 in `|c|²` and `√2`
in `|c|`. 2048 bins of `u32` = **8 KB, L1-resident**, which is the whole reason the bin count is what it is:
adding 3 mantissa bits for a 8× finer threshold makes the counter array 64 KB and turns the one increment per
term into an L2 access. Then walk the bins down from the top to the lowest edge whose cumulative count still fits
in `n`, and `PauliSum::retain` against that edge (`norm_sqr() >= f64::from_bits(k << 52)`, one compare, which
also drops NaN where the bin index would keep it).

Semantics, all documented on the type and pinned by tests:

- `kept ≤ n` **exactly** — the memory bound `TopN` exists to provide is preserved, which is why the rounding goes
  down rather than up.
- `kept > n − p`, where `p` is the population of the coarsest octave that did not fit: including it would have
  overshot. So the shortfall is bounded by the number of terms inside one `√2`-wide magnitude band at the cut.
- Every kept magnitude ≥ every dropped one; the retained set is a union of *whole* octaves.
- **The tie story is simpler, not harder.** Equal magnitudes have equal squares and so share an octave: a
  multiplet is always kept or dropped whole, with no fits-or-not rule and no tie counting. The price is a *wider*
  degenerate case — a sum confined to a single octave of `|c|²` with `len > n` is wiped, exactly as `TopN` wipes
  an all-tied sum, for exactly the same reason (keeping it would let the policy retain unboundedly more than `n`).
- The retained set is a pure function of the magnitude multiset, so it is independent of the bucket partition,
  the hash seed and the thread count — checked in the engine's `tie_tests` beside `TopN`'s equivalents.

Tests: hand-tabulated cumulative octave populations on a powers-of-two fixture; exact agreement with `TopN` at
the `n` where the histogram resolves exactly; monotonicity in `n` (larger `n` keeps a superset — the property
that makes the policy safe to tune); the shortfall bound against `p` recomputed from the input; tie-group
integrity on `tie_heavy_sum`; the single-octave wipe; `n = 0`; `n ≥ len`; `W = 2`; complex coefficients; and a
proptest for the whole contract over tie-dense multisets.

No Python binding — out of scope for this experiment, and it should follow only if the policy survives its gate.

### 1.4 Probe knob (`3505dda`)

`--truncation atopn:<N>` joins the probe's statically dispatched menu; defaults unchanged, existing specs parse
identically. It **cannot be an ab-compare leg** — probe args go verbatim to both sides and no committed baseline
accepts `atopn:` — so the comparison it enables is `topn:<N>` versus `atopn:<N>` *within one binary*.

## 2. Commit map

| commit | what |
|---|---|
| `d9569ee` | `perf:` squared magnitudes in `CoefficientThreshold::keep_term` and all three `TopN` sites; docs + 4 tests (3 red first) |
| `31289c4` | `perf:` pooled, singly-written magnitude buffer; suffix-only tie test; 2 mechanism guards |
| `4498ac3` | `feat:` `ApproxTopN` + 10 unit tests, 1 proptest, 1 engine partition test (all red first against a no-op stub) |
| `3505dda` | `bench:` `--truncation atopn:<N>` |
| `eeb6bfb` | `docs:` `ARCHITECTURE.md` §Truncation |

Files touched: `crates/paulistrings/src/truncation/builtin.rs`, `crates/paulistrings/src/truncation/mod.rs`,
`crates/paulistrings/src/engine/bucketed.rs` (one test), `crates/paulistrings/examples/phase_breakdown.rs`,
`ARCHITECTURE.md`. **No Python binding, no engine hot path, no `engine/merge.rs`.**

`cargo test --workspace` and `cargo test --workspace --release` green at every commit (370 lib tests at the tip);
`cargo fmt --check` clean; `cargo clippy --workspace --all-targets` clean, and again with `--features
phase-timing`.

## 3. Gate results — AUTHORITATIVE

Run 2026-09-01 11:28–11:47 on ccqlin038, box exclusively ours (sibling experiments finished; load at leg start
0.86, decaying build-phase average thereafter — the timed regions are single-threaded with nothing else running).
`RUST_LOG` unset. Protocol: `scripts/ab-compare.sh`, `--a f592c43` (campaign tip) `--b expt/topn-finalize`
(`c30648d`), `--pairs 3`, order `abab`, both binaries built up front and alternated adjacent in time. Acceptance
is **direction consistency across every pair**, median Δ% the effect size (`benchmarks/PROFILING.md`).

Artefacts (gitignored): `benchmarks/results/2026-09-01-ccqlin038/{topn-1e5,topn-1e6,coeff-1e6,keep-1e6,
keep-1e6-libonly,keep-1e6-normsqr}-ab.log` plus per-side `.probe.jsonl` sidecars and the exact binaries.

### 3.1 The four planned legs

| leg | policy | `m` | median Δ% wall | pairs | where it came from |
|---|---|---|---|---|---|
| 1 | `topn:100000` | 1.0 × 10⁵ | **−44.09** (−45.61 … −42.73) | **3/3 negative** | `finalize` 51.0 → 11.2 ns/term (**−78 %**); gather +14.5 % (3/3), merge +3.7 % (3/3) |
| 2 | `topn:1000000` | 1.0 × 10⁶ | **−45.40** (−46.05 … −44.69) | **3/3 negative** | `finalize` 57.9 → 15.2 ns/term (**−73.7 %**); gather +4.4 % (3/3), sort −0.7 % (3/3), merge no consistent change |
| 3 | `coeff:0.0` | 1.0 × 10⁶ | **−25.81** (−31.17 … −24.92) | **3/3 negative** | `merge` 24.72 → 12.89 ns/term (**−47.8 %**, 3/3); gather no consistent change |
| 4 | `keep` | 1.5 × 10⁶ | **+4.44** (+4.40 … +5.98) | 3/3 positive | gather 14.50 → 15.78 ns/term (+8.9 %, 3/3); `finalize_ns` ≈ 2 µs on **both** sides |

Legs 1–3 land at or above the top of the predicted range (−30…−45 %, −20…−30 %). Leg 4 is the interesting one
and gets its own section.

**Baseline sanity.** The A side reproduces the phase-1 fact sheet's own §6/§1 numbers on the quiet box:
`topn:1000000` 93.2 ns/term against 91.12 published; `coeff:0.0` merge 24.72 against 24.76; `keep` 30.1 against
29.90. So the −44 % is measured against a baseline that is the fact sheet's, not a drifted one.

**Correctness corroboration at scale, for free.** Across legs 1–3 (18 runs, 8–100 layers each) `terms_in`,
`terms_out`, `rows_gathered`, `rows_sorted` and `cosets` are **identical between the two sides in every single
run**. So on this data the squared-magnitude selection keeps exactly the terms the `hypot` version kept — 12 000
layer applications' worth of agreement — and `CoefficientThreshold(0.0)` drops nothing extra. The tie/underflow
riders in §1.1 are real but do not fire on realistic coefficients.

### 3.2 Leg 4: the stated PASS condition was the wrong test, and two extra legs show why

Leg 4 **fails as written** ("no consistent direction"): it is 3/3 positive, +4.44 %. But under `--truncation keep`
the policy is `AlwaysKeep` — no `keep_term` override, no `finalize_layer` — so **none of this branch's code
executes**, and `finalize_ns` is ~2 µs on both sides, confirming it. The work counts are identical. The gather
loop's source is byte-identical. Two extra legs, same probe, same protocol:

| A → B | what changed in the library | `keep` wall | phases |
|---|---|---|---|
| `f592c43` → `d9569ee` | `norm_sqr` only (commit 1 of 3) | **−7.48 %** (3/3 negative) | merge **−17.9 %** (3/3), sort +13.3 % (3/3), gather −1.4 % (3/3) |
| `f592c43` → `4498ac3` | all three library commits, no probe change | **+2.42 %** (3/3 positive) | gather +7.5 % (3/3), sort −13.7 % (3/3) |
| `f592c43` → `c30648d` | + the probe's `atopn` variant | **+4.44 %** (3/3 positive) | gather +8.9 % (3/3) |

**A commit that cannot run in this configuration makes it 7.5 % *faster*; the branch tip makes it 4.4 % slower; a
12-point spread, all of it on untouched source.** That is LTO code layout, exactly the effect CLAUDE.md documents
for `engine/merge.rs` (an `#[inline]` hint worth ~6 % in one direction and +20–34 % in the other), and it is
*not* something the change did to the engine.

The methodological lesson, which matters for the rest of the campaign: **non-perturbation cannot be tested by
pairing two fixed binaries.** The interleaved-pairs protocol resolves *run-to-run* noise, and it does that well —
but a build-to-build layout difference is not noise, so it shows up as a perfectly consistent direction across
every pair while saying nothing about the change's semantics. Direction consistency is the right acceptance
criterion for a change that *executes* in the measured configuration and the wrong one for a control where it
does not. Testing layout neutrality properly needs layout *perturbation* — several builds per side with a
trivial unrelated code-size change, or randomized function ordering — which is out of E1's scope. Until then the
honest statement is the one the three legs above license: **on the untruncated path this branch sits somewhere in
a ±8 % layout band whose position no one controls, and its tip happens to land +4.4 % on the unfavourable side.**

For scale: +4.4 % on the untruncated path is +1.3 ns/term, against −39.9 ns/term (leg 1) and −42.7 ns/term
(leg 2) wherever `TopN` is active, and −11.8 ns/term wherever a coefficient threshold is.

### 3.3 `ApproxTopN` versus `TopN`, one binary

Not an A/B: both policies live in the candidate binary (`topn-1e6-b-c30648d`, the exact archived gate build), run
interleaved, 3 rounds, medians below. `sidecar: atopn-vs-topn.jsonl`.

| policy | rested `m` | `finalize` ns/term | wall ns/term | VmHWM |
|---|---|---|---|---|
| `topn:100000` | 100 000 | 11.08 | 46.06 | 24.9 MB |
| `atopn:100000` | 99 772 | **9.25** | **43.79** | 23.8 MB |
| `topn:1000000` | 1 000 000 | 15.28 | 51.46 | 205.9 MB |
| `atopn:1000000` | 987 743 | **12.44** | **48.39** | **194.2 MB** |

- `finalize` **−16.5 %** at `N` = 10⁵ and **−18.6 %** at 10⁶, 3/3 rounds negative at both sizes (per-round
  deltas −17.2/−16.7/−16.2 and −16.8/−17.2/−21.5). Wall −4.9 % and −6.0 %.
- **The `≈N` shortfall is 0.23 % at `N` = 10⁵ and 1.23 % at 10⁶** — a propagated sum's magnitudes spread over
  many octaves, so the straddling octave's population `p` is a small fraction of `N`. This is the loosest part of
  the contract and it is comfortably tight on the workload the campaign cares about.
- **Peak RSS −11.7 MB at `m` ≈ 10⁶**, of which ~2.4 MB is the 1.2 % smaller sum and ~9.3 MB is the pooled
  magnitude buffer `ApproxTopN` does not have at all.

Honest reading: **the histogram is the smallest of the three wins, as the fact sheet predicted** (§7 item 1c,
"the *smallest* of the three"). Once `hypot` and the double-write are gone, both policies are dominated by the
two coefficient passes they share, and the histogram's remaining edge is the buffer more than the arithmetic.
It earns its place as an opt-in for memory-bounded runs, not as a default.

## 4. Verdict

**PASS**, with one qualification recorded rather than waved away.

- **Legs 1 and 2 (the gate): PASS decisively.** −44.09 % and −45.40 % on wall, 3/3 pairs each, the entire effect
  in `finalize` (−73 to −78 %), other phases flat or explained. Above the predicted −30…−45 %.
- **Leg 3: PASS.** −25.81 %, 3/3, entirely `merge` (−47.8 %). Inside the predicted −20…−30 %. This one reaches
  every Python caller passing `min_abs_coeff`.
- **Leg 4: FAILS its stated condition, but the condition does not test what it claimed.** +4.44 % consistent on
  a path that executes none of the change; two extra legs put the same untouched code anywhere from −7.5 % to
  +4.4 % depending only on build layout. Recorded as a layout observation and a protocol correction, not as an
  engine regression. §5 keeps it as an open risk for the orchestrator to weigh rather than resolving it by
  argument.
- **`ApproxTopN`: works as specified and is the modest win the evidence predicted** — finalize −16 to −19 %,
  shortfall 0.2–1.2 %, ~10 MB less resident. Opt-in, exact `TopN` still the default.

Reproduction — exactly what was run, unchanged from what this note proposed before the gate:

```bash
scripts/ab-compare.sh topn-1e5   --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 100000 --qubits 128 --threads 1 --layers rotation_zz --reps 100 --truncation topn:100000'
scripts/ab-compare.sh topn-1e6   --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation topn:1000000'
scripts/ab-compare.sh coeff-1e6  --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 668000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation coeff:0.0'
scripts/ab-compare.sh keep-1e6   --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation keep'
```

plus, for §3.2, the same `keep` probe against `--b 4498ac3` and `--b d9569ee`; and for §3.3
`/tmp/.../atopn.sh`, reproduced by running the candidate binary at `--truncation topn:<N>` and `atopn:<N>`
alternately. `--json-out` is injected by the script and `--features` defaults to `phase-timing`, so neither
belongs in `--probe`.

Thread scaling was **not** gated: `finalize_layer` was the phase that parallelized worst (5.0× on 32 threads vs
11–14× for the coset loop), so the multi-thread win should be larger, not smaller, and the ±10–26 % multi-thread
noise makes single-thread the cheaper place to resolve a −44 % effect. A 32-thread `topn:1000000` leg remains
available if the orchestrator wants the number.

## 5. Risks and caveats

- **The untruncated path sits in a ±8 % LTO layout band and this tip lands +4.4 % up** (§3.2). Not caused by
  executed code, but real for a user who runs no truncation policy at all. Mitigations, none attempted here:
  re-order the new code, gate `ApproxTopN` behind a feature to keep it out of the binary, or measure layout
  neutrality properly with per-side layout perturbation. Worth a decision, not worth blocking a −44 % win.
- **The `≈N` bound is data-dependent** and the 0.2–1.2 % measured shortfall is one workload's. A magnitude
  distribution clustered inside a couple of octaves gives a much larger shortfall; one confined to a single
  octave is wiped. `ApproxTopN` is opt-in and documents this; `TopN` stays the default because of it.
- **The thread-local buffer retains 8 B per term of the largest layer that thread ever finalized** until the
  thread exits — measured as ~10 MB at `m` ≈ 10⁶. Small against the sum's ~100 B/term, but it is a steady-state
  RSS change for a process that finalizes one huge layer then many small ones.
- **`CoefficientThreshold(0.0)` now drops magnitudes below `≈1.57e-162`** where it used to keep them. Zero
  occurrences across 12 000 measured layer applications (§3.1), and such a coefficient is numerically zero.
- The suffix-only tie test relies on `select_nth_unstable`'s partition guarantee, which NaN magnitudes void. So
  did the previous form's `count_gt < n` reasoning; neither is NaN-safe and coefficients are never NaN.
- `ApproxTopN`'s bin counters are `u32`: 2³² terms would overflow them and would also need >100 GB of columns.
  `debug_assert`ed.
- The probe only builds with `--features phase-timing`; `ab-compare.sh` defaults to exactly that, so the gate
  commands do not pass it, but a hand-run `cargo build` of the probe must.

## 6. Not done, deliberately

- **No Python binding for `ApproxTopN`** — the brief scoped this experiment to core Rust. A `truncation_spec`
  variant is a small follow-up now that the policy has a gate number.
- **No exact radix-select** for `TopN` (§1.2) — it trades an 8 B/term array for a third pass over the 16 B/term
  coefficient columns. §3.3 now argues *against* it: the two shared coefficient passes are what remains, so
  adding a third to remove an array is the wrong direction.
- **No finer histogram** for `ApproxTopN`. The measured shortfall (0.2–1.2 %) does not motivate leaving L1.
- **No `LayerScratch` threading** into `finalize_layer` — needs a trait-signature change that touches the
  bindings (§1.2). Note the pooled buffer already reaches the Python path, which a scratch-threaded version
  would not have.
- **No layout remediation** for §3.2 — diagnosed, quantified, and left as a decision rather than a silent tweak.
- **No merge.** The orchestrator merges.

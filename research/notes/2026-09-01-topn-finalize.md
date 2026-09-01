# E1 `expt/topn-finalize`: the truncation path's arithmetic and its candidate array

Branch `expt/topn-finalize`, forked from the campaign tip `f592c43`. Executes experiment **E1** of the phase-2
slate in `research/notes/2026-09-01-large-m-campaign-log.md`, in the priority order the phase-1 fact sheet
(`research/notes/2026-09-01-large-m-phase-breakdown.md` §7 item 1) set: `norm_sqr` first, the candidate `Vec`
second, histogram selection last and opt-in.

**No authoritative timing is in this note.** Four sibling experiments were building and running on ccqlin038
throughout (load 4.6 at the time of the smoke runs), so every number in §3 is contaminated and is recorded only
as a direction check. The merge gate is §4, to be run serialized by the orchestrator.

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

## 3. Smoke observations — NOT AUTHORITATIVE, NO MERGE CLAIM

ccqlin038, load 4.62 (four sibling experiments building/running), single thread, `--qubits 128`, `W = 2`,
`rotation_zz`, `--reps` chosen so the timed region clears the 50 ms governor threshold (§0.1 of the fact sheet).
The "A" column is the fact sheet's own §6 table, measured at `6715918` on a **quiet** box; the "B" column is this
branch at `3505dda` on a **contaminated** one. Only the finalize/merge columns are being read; `gather` is +8 to
+13 % in every cell including the controls, which is the contamination and the reason nothing here is a claim.

| cell | `m` | ns/term A | ns/term B | finalize A | finalize B | finalize % of wall A → B |
|---|---|---|---|---|---|---|
| `keep` | 1.5 × 10⁵ | 29.33 | 30.60 | 0 | 0 | 0 → 0 |
| `coeff:0.0` | 1.0–1.5 × 10⁵ | 40.78 | 30.50 | 0 | 0 | 0 → 0 |
| `topn:100000` | 1.0 × 10⁵ | 85.73 | 46.73 | 52.44 | 11.21 | 61.2 % → 24.0 % |
| `topn:1000000` | 1.0 × 10⁶ | 91.12 | 51.81 | 56.12 | 15.16 | 61.6 % → 29.3 % |

- **`CoefficientThreshold` is now free.** Its `merge` is 12.90 ns/term against `keep`'s 13.12 in the same session
  (`-1.7 %`, i.e. noise), where the fact sheet measured `24.67` against `13.09` (`+88 %` on the merge, `+41 %` on
  the layer). This is the cleanest of the four cells because the control ran adjacent in time at the same `m`.
- **`TopN`'s `finalize` fell ~73–79 %** and is no longer the dominant phase — 24–29 % of the layer against
  61–62 %. `merge` and `sort` are unchanged in the same runs (17.17 vs 17.01, 1.84 vs 1.60), which is what says
  the change landed where it was aimed.
- `ApproxTopN`, same session, same `--n`: `finalize` 9.37 ns/term at `m` = 9.98 × 10⁴ and 12.30 at 9.88 × 10⁵,
  i.e. **−16 to −19 % against the improved `TopN`** and −5 to −7 % on the layer. Notably *small*: once the
  `hypot` and the double-write are gone, both policies are dominated by the two coefficient passes they share.
  The histogram's remaining edge is the buffer, not the arithmetic.
- **Shortfall on real data is tiny**: `ApproxTopN(100000)` rested at 99 772 terms (−0.23 %) and
  `ApproxTopN(1000000)` at 987 743 (−1.23 %). A propagated Pauli sum's magnitudes spread over many octaves, so
  `p` is a small fraction of `n` — the regime the bound is loosest in (clustered magnitudes) is not this one.
- **Peak RSS at `m ≈ 10⁶`: 210 996 kB (`topn`) vs 198 764 kB (`atopn`)**, a 12.2 MB gap of which ~2.4 MB is
  `atopn`'s 1.2 % smaller sum and ~10 MB is the pooled magnitude buffer it does not have. That is the fact
  sheet's 12 MB/layer array, now allocated once per thread instead of once per layer — and zero for `ApproxTopN`.

## 4. Proposed gate

Four `ab-compare` invocations, `--a f592c43` (campaign tip) against `--b expt/topn-finalize`. Both sides accept
every flag below, since `6715918` is in both histories. `--json-out` is injected by the script and `--features`
already defaults to `phase-timing`, so neither is in `--probe`. Run serialized on a quiet box, `RUST_LOG` unset.

```bash
# 1. TopN at m = 1e5 — the primary target. reps 100 keeps the timed region ~470 ms.
scripts/ab-compare.sh topn-1e5 --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 100000 --qubits 128 --threads 1 --layers rotation_zz --reps 100 --truncation topn:100000'

# 2. TopN at m = 1e6 — same target, 10x the working set.
scripts/ab-compare.sh topn-1e6 --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation topn:1000000'

# 3. The keep_term half of the change: CoefficientThreshold, the fact sheet's zero-finalize control.
#    --n 668000 lands m ~ 1e6 (coeff:0.0 drops nothing, so m is the closed key set of --n).
scripts/ab-compare.sh coeff-1e6 --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 668000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation coeff:0.0'

# 4. Non-perturbation of the default path: no truncation policy at all, m ~ 1.5e6.
scripts/ab-compare.sh keep-1e6 --a f592c43 --b expt/topn-finalize --pairs 3 \
  --probe '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8 --truncation keep'
```

Acceptance, per `benchmarks/PROFILING.md`:

- **(1) and (2) are the gate.** Direction consistency across all 3 pairs, with the median Δ% as the effect size.
  Expected: −30 to −45 % on wall, concentrated entirely in `finalize`; `gather`/`sort`/`merge` unchanged.
- **(3)** expected −20 to −30 % on wall, concentrated in `merge`. This leg matters beyond the experiment: every
  Python caller passing `min_abs_coeff` is on it.
- **(4)** expected **no consistent direction** — pairs disagreeing in sign is the *pass* condition here. A
  consistent regression on this leg would mean the change perturbed a path it does not touch (LTO code layout;
  `truncation/builtin.rs` has no `#[inline]` in the merge's inlining set, but the fact sheet's own warning about
  layout effects applies).
- Thread scaling is *not* gated: `finalize_layer` was the phase that parallelized worst (5.0× on 32 threads vs
  11–14× for the coset loop), so the multi-thread win should be larger, not smaller, and the fact sheet's
  ±10–26 % multi-thread noise makes single-thread the cheaper place to resolve it. A 32-thread `topn:1000000`
  leg is worth adding only if the orchestrator has budget.

`ApproxTopN` cannot be gated by ab-compare (§1.4). Its within-binary measurement, on the candidate side only:

```bash
for spec in topn:1000000 atopn:1000000; do
  ./target/release/examples/phase_breakdown --n 1000000 --qubits 128 --threads 1 \
      --layers rotation_zz --reps 8 --truncation "$spec" --format json
done
```

Read `finalize_ns / terms_in` and `vmhwm_kb`, and read `n` — the two policies rest at different term counts by
construction, so per-term is the only comparable figure.

## 5. Risks and caveats

- **The `≈N` bound is data-dependent and the note's 0.2–1.2 % shortfall is one workload.** A magnitude
  distribution clustered inside a couple of octaves gives a much larger shortfall, and one confined to a single
  octave is wiped. `ApproxTopN` is opt-in and documents this; `TopN` stays the default precisely because of it.
- **The thread-local buffer retains 8 B per term of the largest layer that thread ever finalized**, until the
  thread exits. Small against the sum's own ~100 B/term, and it is the point of the cache, but it is a real
  change in steady-state RSS for a process that finalizes one huge layer and then many small ones.
- **`CoefficientThreshold(0.0)` now drops magnitudes below `≈1.57e-162`** where it used to keep them. This
  changes term *counts* in principle; no test in the suite generates such a coefficient, and the workloads cannot.
- The suffix-only tie test relies on `select_nth_unstable`'s partition guarantee, which NaN magnitudes void. So
  did the previous form's `count_gt < n` reasoning; neither is NaN-safe and the coefficients are never NaN.
- `ApproxTopN`'s bin counters are `u32`: a sum of 2³² terms would overflow them, and would also need >100 GB of
  columns. `debug_assert`ed.
- The probe only builds with `--features phase-timing`; `ab-compare.sh` defaults to exactly that, so the gate
  commands do not pass it, but a hand-run `cargo build` of the probe must.

## 6. Not done, deliberately

- **No Python binding for `ApproxTopN`** — the brief scoped this experiment to core Rust. A `truncation_spec`
  variant is a small follow-up if the policy survives.
- **No exact radix-select** for `TopN` (§1.2) — it trades an 8 B/term array for a third pass over the 16 B/term
  coefficient columns, and the direction needs measurement this experiment could not do.
- **No finer histogram** for `ApproxTopN` (mantissa bits, or a second-level refinement inside the straddling
  octave). Both tighten the `≈N` bound at the cost of leaving L1 or adding a pass. The 8 KB / L1 choice is
  documented at the constant so the trade is visible.
- **No `LayerScratch` threading** into `finalize_layer` — it needs a trait-signature change that touches the
  bindings (§1.2).
- **No authoritative benchmark, no merge claim.** §3 is a direction check on a contaminated box.

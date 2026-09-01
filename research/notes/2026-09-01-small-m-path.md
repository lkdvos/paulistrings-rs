# E3: a direct-apply path for small sums

Branch `expt/small-m-path`, branched from the campaign tip `f592c43`. Experiment E3 of the phase-2 slate in
`research/notes/2026-09-01-large-m-campaign-log.md`, re-rationalized per the Phase-1 fact sheet
(`research/notes/2026-09-01-large-m-phase-breakdown.md` §7(4)).

**Status: implemented, tested, and gated.** All three gates were run serialized on the reference host at load ≈ 1 on
2026-09-01 (§5, §6). `EngineSelection::Auto` is **off by default** — the default engine is the sorting engine for
every layer, unchanged.

Gate verdicts in one line each:

- **(a) non-perturbation — PASS, layout not perturbation.** All four legs do **bit-identical work** (`terms_in`,
  `terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`, `cosets` all equal across both sides and every run); wall
  moves −2.5 %, −2.3 % (null), +3.9 %, +3.5 %, inside E1's measured ±4–7 % LTO layout band, with phase times
  *redistributed* rather than any phase doing more work.
- **(b) small-`m` effect — CONFIRMED.** The three configurations the study loses worst gain **2.28–2.36×**
  (kicked-Ising 2⁻⁴), **1.55–1.58×** (XXZ 1e-2) and **1.68×** (XXZ 1e-3), 3/3 to 5/5 sign-consistent; nothing
  regresses at the shipped threshold.
- **(c) parity — PASS.** 14/14 cross-path tests green in release, per-layer term-count vectors equal everywhere, and
  the harness's own parity assert held on all six study configurations.

The gate moved the default threshold **512 → 2048** (§1.5, §5).

---

## 0. What the evidence asked for, and what it did not

The target is the small-`m` regime the head-to-head study loses outright: PauliPropagation.jl at 0.32–0.63× our time
below ~4 × 10³ terms (`benchmarks/python/jl_performance/README.md`), 45.6 µs/term against Julia's 13.2 at 68 terms.

The fact sheet corrected the *mechanism* the study README guessed at. The fixed cost is **not** the bucketed serial
pipeline (rebucket → span_plan → permute → unpermute → recount → finalize = **0.19 µs/layer**, flat across five
channel types and six decades of `m`); it is **`Channel::prepare`** — 70 % of a 1.43 µs/layer fixed cost for a
two-qubit rotation at `W = 2`, 78–95 % of `su4`'s ~5.4 µs, and 4.19–5.71 µs *per gate* for a dense two-qubit PTM,
which a circuit of distinct SU(4) blocks pays once per block and cannot cache.

The saturated-regime half of the plan's experiment (4) is **dropped**: §1 of the fact sheet found no phase superlinear
in `m` anywhere out to 2.1 × 10⁷ terms, so there is no our-side evidence for a near-closed-sum fast path. Nothing here
touches it.

One thing the evidence did **not** say, which the measurements below did: the win is not only the fixed cost. The
sorting engine's per-term cost is flat in `m` (±10 % over three decades); the direct path's *rises* as its hash map
leaves cache. So there is a crossover, it is workload-dependent, and picking the threshold is the whole design
question (§1.5, §5).

---

## 1. The five design decisions

### 1.1 Where the dispatch lives, and what the knob looks like

`propagate_with_scratch` is the single layer loop and the only place that knows the term count between layers, so the
selection lives there — but the existing signatures are load-bearing (the PyO3 bindings, the examples, the benches and
six integration-test files call them), so the knob arrives on **new entry points** rather than by changing old ones:

```rust
pub enum EngineSelection { SortedOnly /* default */, Auto, SmallSumDirect }
pub struct PropagateOptions { pub engine: EngineSelection, pub small_sum_threshold: usize }

pub fn propagate_with_options(circuit, sum, policy, direction, options) -> PauliSum<W>;
pub fn propagate_with_scratch_and_options(circuit, sum, policy, direction, scratch, options) -> PauliSum<W>;
```

`propagate` and `propagate_with_scratch` delegate with `PropagateOptions::default()`, whose `engine` is `SortedOnly`.
So the default is today's behaviour, and `tests/small_sum_path.rs::default_options_match_propagate_bitwise` pins that
it is today's behaviour *bitwise*, not just to tolerance.

Three variants rather than two because the third is the only way to measure the direct path against a `TopN`-style
policy:

| variant | takes the direct path when | why |
|---|---|---|
| `SortedOnly` | never | the default; today's engine, today's bits |
| `Auto` | `len <= threshold` **and** `!policy.finalizes_layer()` | where the direct path is expected to be faster |
| `SmallSumDirect` | `len <= threshold` | A/B knob; correct with any policy, not always fast |

The predicate is `PropagateOptions::starts_direct`, evaluated **once**, before the loop, from the starting term count.
Under the default it is one not-taken branch before the loop and nothing inside it. Two further precautions against
perturbing the sorting path, whose merge kernels move 6–34 % under a few bytes of code motion:

- the direct prefix is a separate `#[inline(never)]` function in `engine::direct`, so the sorting loop's body can
  reach its call site but never its code;
- **`LayerScratch` gains no field.** The direct path allocates its own map per `propagate` call (one allocation of at
  most `threshold` entries against a run of hundreds of layers), rather than adding fields that would change the
  layout of a struct the hot loop dereferences every layer. Cross-call capacity reuse is a possible follow-up, not
  something to buy with a layout change before the non-perturbation gate has been run.

The one residual perturbation risk is that the sorting loop is now `for k in start..n` instead of `for k in 0..n`,
with `start` a runtime value that is always 0 under the default. That is deliberate — duplicating a ~90-line hot loop
to avoid it would leave two copies to keep in sync — and gate (a) is exactly the measurement that decides whether it
was affordable. If gate (a) shows a consistent regression on the default path, the fallback is to give
`propagate_with_scratch` back its own verbatim `for k in 0..n` copy.

### 1.2 Truncation: reuse the policy machinery, and a hint for the round trip

`keep_term` is applied by the direct path **per layer, on summed coefficients**, in the same position the merge
applies it, through the same `TruncationPolicy` object — one `retain` pass over the output map after the whole layer
has accumulated. Same zero-drop rule too: sum every emitted row (including exact zeros, which the merge deliberately
keeps in the sum so the sign of a zero sum is not flipped), then drop exact-zero sums, then filter.

`finalize_layer` needs a real `PauliSum`. Honouring it means materializing → finalizing → re-ingesting **per layer**,
and a materialize is a sort of the whole sum — several times the fixed cost the path exists to save. So it is done, but
only when the policy says there is something to do:

```rust
trait TruncationPolicy<W> { … fn finalizes_layer(&self) -> bool { true } }
```

The default is `true`, the **conservative** answer: an existing external policy that overrides `finalize_layer` and
never hears about this method still gets its layer pass, on every path. The built-ins answer for themselves —
`CoefficientThreshold`/`WeightCutoff` false, `TopN` true, `And` the disjunction of its sides, `Or` false (its
`finalize_layer` is the trait's no-op default, not either child's). `TopN` therefore behaves identically on either
selection, which `topn_matches_on_the_direct_path` and `topn_tie_groups_match_on_the_direct_path` check against the
sorting engine including the whole-tie-group rule.

`Auto` additionally *declines* a finalizing policy — a performance decision, not a correctness one, since the round
trip would eat the win. `SmallSumDirect` takes it anyway and stays correct.

Rejected alternative: defaulting the hint to `false` and relying on documentation. A policy that silently loses its
layer pass is a wrong answer, and this repo's bar is correctness first; the cost of the conservative default is that
an external policy has to opt in to get the fast route, which is documented on the method.

### 1.3 The transition, both ways

**Small → large: rebuild the buckets, once.** The direct path holds the terms in its map *between* layers — that is
the whole win, because a per-layer round trip through `PauliSum` costs more than the fixed cost being removed. When a
layer leaves the sum above the threshold, `DirectSum::to_sum` sorts by key once and scatters through the existing
`from_key_sorted`, under the *entering* hash's rows and seed at `max(desired_bits(len), entering bits)` — the same
clamp `PauliSum::rebucket` applies, so the grow-only bucket-count invariant survives the detour and the sorting suffix
partitions the sum exactly as it would have.

**Large → small: never, mid-run.** Three reasons, in order of weight:

1. Each crossing costs an `O(n)` ingest plus an `O(n log n)` materialize. A sum oscillating around the threshold —
   which is the normal state of a truncated propagation, per `rebucket`'s own doc comment on why it is grow-only —
   would pay them *per layer*.
2. The upside is bounded and small: the sorting engine at sub-threshold `m` is worse by the fixed cost (1.43 µs/layer
   for a rotation) plus whatever per-term difference remains, and the measurements below put that at 1.0–2.4× on a
   *whole run* of small layers, not on the tail of a run that has already been large.
3. One crossing per call is trivially reasoned about and trivially tested: the records are the same records in the
   same order wherever it happened, and `crossing_the_threshold_mid_circuit_agrees` sweeps the threshold so the
   crossing lands on every layer in turn.

Entry is re-decided on every `propagate` call, so a Trotter driver stepping a small observable through many short
circuits gets the direct path each call. A run that *starts* above the threshold never enters — including under
`SmallSumDirect`.

### 1.4 Logging and `TermTrace` parity

The direct prefix emits, per layer, in this order: the `TermTrace` push (via the same `#[cold]`
`record_layer_terms`), then the `DEBUG` line with the same target, the same format string and the same fields
(`layer {k}/{n} [{name}]: {before} -> {after} terms, {ms} ms`). The `INFO` entry/exit pair is emitted by the shared
caller, so it is unconditionally identical.

This is a hard requirement, not a nicety: the cross-engine head-to-head driver parses per-layer term counts out of
exactly those `DEBUG` records to gate term-count parity against PauliPropagation.jl. Every cross-path test asserts the
**whole** `terms_in`/`terms_out` vectors are equal — not close — and so does the A/B harness, before it reports any
timing (§6b).

Residual risk, stated plainly: term counts can only diverge if a coefficient sums to exactly zero in one summation
order and not the other. Equal-key summation order differs between the two paths (map iteration order versus the
coset gather order), so this is possible in principle, exactly as it already is between two bucket counts
(ARCHITECTURE.md §Determinism). It did not happen in any test, proptest, or in the six real study configurations
(which include kicked-Ising at the Clifford `theta_zz = -π/2`, where coefficients are exact dyadics and cancellations
are therefore exact and order-independent).

### 1.5 The threshold

`DEFAULT_SMALL_SUM_THRESHOLD = 2048`, and `PropagateOptions::small_sum_threshold` is a public field so any measurement
can move it.

Three successive answers, and the third is the shipped one:

1. **Fixed-cost arithmetic → 4096.** Against the sorting engine's ~29 ns/term at `W = 2`, the 1.43 µs fixed cost is
   9 % of a 500-term layer, 3.3 % of a 1500-term one and 1.2 % at 4096 — so 4096 is where there stops being anything
   to win *if the two paths cost the same per term*.
2. **Smoke → 512.** They do not cost the same per term. The sorting engine's per-term cost is flat in `m`; the direct
   path's rises as its map leaves cache. So there is a crossover, measured at **≈ 1.5 × 10² resident terms on
   kicked-Ising** and **≈ 2 × 10³ on XXZ** — a 14× spread mirroring the 4.4–21× spread of the study's own
   cross-engine crossovers. 512 is their geometric mean.
3. **The gated sweep → 2048.** A five-point sweep (§5) shows the threshold acts through a *second* channel the
   crossover argument misses: setting it above a workload's **peak** keeps the whole run on one path, and being
   undivided is worth a lot. XXZ's 1 625-peak configuration is a 1.02–1.04 × null at 128/512/1024 and **1.68×** at
   2048, purely because 2048 is the first value that exceeds its peak. 2048 is the largest value at which no
   configuration regresses; 4096 is the cliff (kicked-Ising 2⁻⁸ → 0.68×, XXZ 1e-4 → 0.93×, both sign-consistent).

What the constant does *not* affect: the two configurations where the study loses worst — kicked-Ising 2⁻⁴ (ratio
0.323) and XXZ 1e-2 (0.460) — peak at 68 and 164 terms, so they are fully direct at every threshold in the sweep and
their 2.28–2.36× and 1.48–1.58× are threshold-insensitive. The constant only prices the middle of the range, where the
trade is asymmetric: 2048 gives up ~6 points on kicked-Ising 2⁻⁶ (1.08× at 512 becomes a 1.02× null) to gain 0.64 on
XXZ 1e-3.

2048 also sits below `desired_bits`'s `worth_splitting` floor (128 × 64 = 8192), so a sum on this path is one the
sorting engine would have run in few buckets anyway.

---

## 2. What the direct path is

One `Channel::apply` (or `apply_adjoint`) per resident term into a `max_fanout`-sized `OutputBuffer`, every emitted
row accumulated into an `FxHashMap<PauliString<W>, Complex64>`, then one `retain` pass applying the zero-drop and
`keep_term`. That is `test_support::naive_apply_layer`'s algorithm — the differential oracle *is* the fast path now,
which is why `engine::direct`'s unit tests are only a plumbing check and the real algebra check is
`tests/small_sum_path.rs` against the bucketed engine.

No `prepare` (no PTM derivation, no delta plan, no `4^|support|` probe), no `rebucket`, no `Gf2Span`, no
permute/unpermute, no per-run sort, no merge, and no Rayon: a sub-threshold layer has nothing to spread.

### An unplanned capability

The direct path calls only `Channel::apply`, so it applies channels of **any** support width — including the
`> MAX_LOCAL_SUPPORT` channels for which `Channel::prepare` returns `None` and `propagate` panics today. It is the
whole-sum fallback `research/notes/2026-08-31-local-ptm-generalization.md` says does not exist, restricted to small
sums.

That asymmetry is a capability, not a divergence, and it is tested in both directions:
`the_direct_path_runs_a_channel_the_sorting_engine_refuses` (it works below the threshold) and
`a_channel_the_sorting_engine_refuses_panics_after_the_transition` (the same message, from the same place, on the
layer after the sum crosses). It is also what makes `Auto`'s policy exclusion *observable* in a test at all
(`auto_declines_a_finalizing_policy`).

---

## 3. Commits

| commit | what |
|---|---|
| `6c92e50` | `feat: add a truncation layer-finalization hint` — `TruncationPolicy::finalizes_layer`, defaulted `true`; overrides on `CoefficientThreshold`, `WeightCutoff`, `And`, `Or`; tests for the composition rules and the conservative default. |
| `e56f021` | `feat: add a runtime-selectable direct-apply small-sum path` — `engine::direct` (`DirectSum`, `apply_layer`, `to_sum`, `reload`, `run_direct_prefix`), `PropagateOptions`/`EngineSelection`/`DEFAULT_SMALL_SUM_THRESHOLD`, the two new entry points, the wiring, and `tests/small_sum_path.rs`. |
| `a133762` | `bench: the small-m A/B harness, and a threshold from its crossovers` — `examples/small_m_ab.rs`, and the threshold moved 4096 → 512 on smoke evidence. |
| `003f79c` | `test: gate the release-only invariant check in the small-sum tests` — `assert_invariants` is debug-only. |
| `448d75b` | `research: note the small-m direct path -- design, smoke, gates` — this note, plus front-door doc pointers. |
| `bc154d3` | `bench: gate results for the small-m path, and a threshold from the sweep` — authoritative gate results here, campaign-log entry, and the threshold moved 512 → 2048 on the gated sweep. |

`cargo test --workspace` and `cargo test --workspace --release` are green at each of them; `cargo clippy --workspace
--all-targets` is clean; `cargo fmt` applied. No fingerprint or byte-identity tripwire moved — the default path is not
reached differently, and none of them selects a non-default engine.

### Tests added

`engine::direct` unit tests: hand-computed `H: Z → X` and `exp(-iθZ/2): X → cos θ·X + sin θ·Y`; `keep_term` on summed
coefficients; the differential oracle across the channel zoo at `W ∈ {1, 2}` in both directions; a `> 2`-support
channel; ingest/materialize round-trip identity and non-coarsening; `reload` replacement.

`tests/small_sum_path.rs` (14 tests) against the **sorting engine**: bitwise identity of the default options; the zoo
(Cliffords h/s/cnot/cz/swap, rotations at generator weight 1/2/4, dense 2q PTM, depolarizing 1q/2q, amplitude damping)
at `W ∈ {1, 2}` × both directions × three policies; an above-threshold start; a threshold sweep
`{0, 1, 2, 3, 8, 33, 200, 2²⁰}` that puts the crossing on every layer in turn; degenerate empty sum and zero-layer
circuit; `TopN` and tie-heavy `TopN` through the forced selection; the two wide-channel behaviours; and two proptests
over random sums, channel orders, directions and thresholds. Every one of them asserts per-layer term-count equality
as well as the terms.

---

## 4. What the Python binding would need (deliberately not done)

Core Rust only in this experiment. For the cross-engine parity gate to run with the path enabled, `crates/paulistrings-py`
needs two additive changes:

1. **An engine kwarg on `PauliSum.propagate`** — e.g. `engine="sorted"|"auto"|"direct"` plus
   `small_sum_threshold=None`, mapped to `PropagateOptions` and passed to `propagate_with_scratch_and_options`. One
   `match` at the boundary, outside every loop; the width dispatch is unchanged.
2. **`SpecPolicy` must override `finalizes_layer`** (`src/truncation_spec.rs`). It overrides `finalize_layer`
   unconditionally today, so it inherits the conservative `true` and `Auto` would never choose the direct path from
   Python — even for `PolicySpec::NoOp`. The override is `matches!(spec, TopN(_)) || And(a, b) if either`, i.e. the
   same recursion `finalize_spec` already does.

Until then the cross-engine gate can only be run on the sorting path, and gate (c) below is its Rust-side equivalent.

---

## 5. Gate results — authoritative

Reference host ccqlin038, 2026-09-01, box exclusive to this experiment (the orchestrator serialized E1 before and
E2/E4 after), `powersave` governor, `RUST_LOG` unset, one thread, load 0.99–1.3 at the start of each gate. Raw
ab-compare logs, sidecars and archived binaries in the gitignored
`benchmarks/results/2026-09-01-ccqlin038/e3-nonperturb-*`.

### 5.1 Gate (a): the default path is not perturbed — and the residual is layout

`phase_breakdown` drives `propagate_with_scratch`, i.e. `PropagateOptions::default()`, i.e. `SortedOnly`. Three abba
pairs per leg, A = `f592c43`, B = `expt/small-m-path` @ `448d75b`.

| leg | `m` | wall median Δ% | pairs | work counts | phase deltas (median, all 3/3 unless noted) |
|---|---|---|---|---|---|
| rotation_zz | 1.5 × 10⁴ | **−2.47** | 3/3 B lower | **identical** | gather **+8.9**, sort **+13.7**, merge **−17.3** |
| rotation_zz | 1.5 × 10⁶ | −2.27 | **MIXED** (2/3) | **identical** | gather +8.2, sort +15.7, merge −16.9 |
| su4 | 9.9 × 10³ | **+3.94** | 3/3 B higher | **identical** | gather +0.3 (MIXED), sort **+3.8**, merge **+22.3** |
| su4 | 9.9 × 10⁵ | **+3.54** | 3/3 B higher | **identical** | gather +0.3 (MIXED), sort **+3.2**, merge **+21.7** |

**"Identical" is literal.** Every work counter is equal between the two sides and constant across all three runs on
each side — `layers`, `terms_in`, `terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`, `cosets`, `n`, `reps`,
`seed`. Example, the su4 1e6 leg: `terms_in = terms_out = 3 957 072`, `rows_gathered = 59 113 152`,
`rows_sorted = 55 156 080`, `rows_id = 3 957 072`, `cosets = 256` on both sides. The layer loop does exactly the same
work, exactly as many times.

**So this is LTO code layout, not a perturbation**, by the discriminator E1's gate established (±4–7 % on untouched
source between builds): a genuine perturbation from the `for k in start..n` residual would show *changed work counts*
in some phase, and there are none. What the numbers show instead is phase *redistribution* at fixed work, and it is
concentrated exactly where CLAUDE.md says code layout bites — the merge kernels, A/B-verified to move 6–34 % under a
few bytes of motion:

- On `rotation_zz` the merge gets **17 % faster** and gather/sort **9–16 % slower**, netting −2.3 to −2.5 % of wall,
  i.e. B is *faster* on the default path.
- On `su4` the arithmetic closes: sort is 58–60 % of that layer and moved +3.2 %, merge is ~7 % and moved +21.7 %,
  giving 0.58 × 3.2 + 0.07 × 21.7 ≈ **+3.4 %** against a measured +3.5 %. Both moved-phase costs are on unchanged
  source over identical row counts.

Two of four legs go our way, two against, all four inside the layout band, and the mechanism is identified in each
case. **Verdict: PASS (layout).** Note that the named remedy in §1.1 — giving `propagate_with_scratch` a verbatim
`for k in 0..n` loop — would *not* reliably remove this, because the mechanism is not the loop bound but the presence
of new code in a `lto = "fat"`, `codegen-units = 1` unit. Chasing it would be re-rolling the layout dice, which
CLAUDE.md's determinism and performance sections both say not to do.

### 5.2 Gate (b): the small-`m` effect, and the threshold sweep

`examples/small_m_ab.rs`, `--pairs 5 --reps 30 --check` at the shipped threshold, then `--pairs 3 --reps 30` per sweep
point. `--check` passed at every point: all six configurations reproduce the study's committed final/peak term counts
(7/68, 408/517, 5038/6311, 156/164, 1625/1625, 9918/9918), and the two selections' full per-layer count vectors were
equal in every leg or the harness would have aborted. `speedup = sorted / direct`, so `> 1` means the direct path is
faster. `×` marks a sign-inconsistent (MIXED) cell, i.e. no consistent change.

| workload | cutoff | peak | mean `m` | study ratio | t=128 | t=512 | t=1024 | **t=2048** | t=4096 |
|---|---|---|---|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 68 | 21.6 | 0.323 | 2.344 | 2.350 | 2.340 | **2.359** | 2.328 |
| kicked_ising | 2⁻⁶ | 517 | 114.3 | 0.629 | **1.199** | 1.079 | 1.056 | 1.043 × | 1.048 |
| kicked_ising | 2⁻⁸ | 6 311 | 746.4 | 1.126 | 1.091 | **1.111** | 1.105 | 1.081 | **0.676** |
| xxz | 1e-2 | 164 | 63.6 | 0.460 | 1.479 | **1.577** | 1.566 | 1.546 | 1.567 |
| xxz | 1e-3 | 1 625 | 434.3 | 0.453 | 1.020 × | 1.039 × | 1.017 × | **1.682** | 1.702 |
| xxz | 1e-4 | 9 918 | 2 197.5 | 0.895 | 1.006 × | 0.986 × | 0.996 × | 1.016 × | **0.926** |

The five-pair run at the shipped default, with per-layer costs:

| workload | cutoff | mean `m` | sorted µs/layer | direct µs/layer | speedup | pairs |
|---|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 21.6 | 1.281 | 0.540 | **2.350** | 5/5 |
| kicked_ising | 2⁻⁶ | 114.3 | 3.208 | 2.970 | **1.079** | 5/5 |
| kicked_ising | 2⁻⁸ | 746.4 | 22.473 | 20.281 | **1.111** | 5/5 |
| xxz | 1e-2 | 63.6 | 2.429 | 1.547 | **1.577** | 5/5 |
| xxz | 1e-3 | 434.3 | 18.517 | 17.689 | 1.039 × | 3/5 |
| xxz | 1e-4 | 2 197.5 | 67.814 | 68.219 | 0.986 × | 2/5 |

(That run was taken at threshold 512, before the sweep moved the default; at 2048 the 1e-3 row becomes 1.682 (3/3)
and 2⁻⁶ becomes a 1.02–1.04 null, per the sweep table.)

**Clock check.** kicked-Ising 2⁻⁴'s timed leg is only 22 ms at `--reps 30`, under the governor's ~50 ms ramp. Re-run
at `--reps 300` (region 0.227 s, comfortably clear): **2.278× (3/3)** against 2.350 at `--reps 30` — a 3 % difference,
so the short region was not manufacturing that result. The same re-run gives 2⁻⁶ 1.025 × (MIXED) and 2⁻⁸ 1.078 (3/3)
at t=2048, matching the `--reps 30` sweep to within a point.

Readings:

- **The regime the study loses worst is where the win is largest, and it is threshold-insensitive.** kicked-Ising 2⁻⁴
  (study ratio 0.323, i.e. we were 3.1× slower) gains **2.28–2.36×** at every threshold tested; XXZ 1e-2 (0.460)
  gains **1.48–1.58×**. Both peak below every threshold in the sweep, so they run fully direct regardless. Applying
  those factors to the study's own ratios would move 0.323 → ~0.75 and 0.460 → ~0.72: still behind
  PauliPropagation.jl, by 1.3–1.4× instead of 2.2–3.1×.
- **Two channels, not one.** The per-term crossover (~1.5 × 10² kicked-Ising, ~2 × 10³ XXZ) explains the *shape*, but
  the sweep's biggest single effect is a peak effect: XXZ 1e-3 is a null at 128/512/1024 and 1.68× at 2048, and 2048
  is simply the first threshold above its 1 625-term peak. Being undivided is worth 0.64 there.
- **The win is not purely the fixed cost.** Taking the fact sheet's 1.43 µs as the sorting engine's fixed cost:
  kicked-Ising 2⁻⁶ is 15.5 ns/term sorted against 26 ns/term direct — the direct path is *worse per term* there and
  only the fixed cost carries it. XXZ 1e-3 is 40.5 ns/term sorted against 25 direct. The difference is effective
  fanout: kicked-Ising's entangling layers sit at the Clifford angle −π/2, whose `cos` row is 6.1e-17 and is truncated
  away, so the sorting engine moves about half the rows XXZ's generic-angle rotations do.
- **The direct path's per-term cost rises with `m`** — ~25 ns/term at `m` ≈ 100–400, ~35 at `m` ≈ 2 200, ~60 on
  kicked-Ising at `m` ≈ 750 — consistent with the map (48 B/entry, two of them) leaving L1 around 700 entries. The
  sorting engine's is flat in `m`. Hence a cliff, measured between 2048 and 4096 on kicked-Ising 2⁻⁸
  (1.081 → 0.676).
- **Nothing regresses at 2048.** Every cell is a sign-consistent win or a MIXED null. At 4096 two cells are
  sign-consistent regressions.

### 5.3 Gate (c): parity

- `cargo test -p paulistrings --release --test small_sum_path` — **14/14 green** (and green in debug), including the
  per-layer `terms_in`/`terms_out` equality assertion in every cross-path comparison, the threshold sweep that puts
  the transition on every layer, TopN and tie-heavy TopN, and both proptests.
- `small_m_ab --check` — parity assert held on all six study configurations at all five thresholds (30 leg pairs),
  and the ported circuits still reproduce the study's committed term counts exactly.
- `cargo test --workspace` and `cargo test --workspace --release` green; clippy clean. No fingerprint or
  byte-identity tripwire moved.

The literal cross-engine gate (Julia in the loop, path enabled) still needs the binding kwarg of §4 and is a
follow-up; what is verified here is the same quantity that gate compares — per-layer term counts — between our two
paths.

---

## 6. Gate commands — exactly what was run, and how to re-run it

Run serialized, `RUST_LOG` unset, box otherwise idle. `f592c43` is the campaign tip this branch left. Results in §5.

### (a) Non-perturbation of the default path — the pass/fail gate

`phase_breakdown` drives `propagate_with_scratch`, i.e. `PropagateOptions::default()`, i.e. `SortedOnly`. Any effect
here is code layout, not the new path.

```bash
# rotation_zz and su4, m ~ 1e4 and ~1e6, one thread. --reps per the fact sheet's
# 0.1 rule (timed region >= 200 ms); su4's --n is scaled by its 14.2x closure factor.
scripts/ab-compare.sh e3-nonperturb-rotzz-1e4 --a f592c43 --b expt/small-m-path \
    --pairs 3 --order abba --probe '--n 10000 --qubits 128 --threads 1 --layers rotation_zz --reps 186'
scripts/ab-compare.sh e3-nonperturb-rotzz-1e6 --a f592c43 --b expt/small-m-path \
    --pairs 3 --order abba --probe '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8'
scripts/ab-compare.sh e3-nonperturb-su4-1e4 --a f592c43 --b expt/small-m-path \
    --pairs 3 --order abba --probe '--n 700 --qubits 128 --threads 1 --layers su4 --reps 400'
scripts/ab-compare.sh e3-nonperturb-su4-1e6 --a f592c43 --b expt/small-m-path \
    --pairs 3 --order abba --probe '--n 70000 --qubits 128 --threads 1 --layers su4 --reps 4'
```

**Pass criterion, as applied:** a sign-inconsistent null, a negligible median Δ%, *or* a small sign-consistent Δ%
whose phase composition shows **identical work counts** — the layout signature E1's gate calibrated (±4–7 % on
untouched source between builds). A sign-consistent Δ% accompanied by *changed* work counts (`terms_in`,
`terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`, `cosets`) is a genuine perturbation and a fail. Discriminate
by reading the two `*-{a,b}.probe.jsonl` sidecars: they carry every one of those counters per run.

### (b) Ours-only small-`m` effect, path ON vs OFF

```bash
cargo build --release --example small_m_ab -p paulistrings
# from the repository root (the harness reads examples/data/heavy_hex_127.edges)
RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 5 --reps 30 --check
# and the threshold sweep the shipped default rests on
for t in 128 512 1024 2048 4096; do
  RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 3 --reps 30 --threshold $t
done
# clock check for the one cell whose timed leg is under the governor's ramp
RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --workload kicked_ising --pairs 3 --reps 300
```

`--check` hard-fails if the ported circuits stop reproducing the study's committed final/peak term counts (7/68,
408/517, 5038/6311, 156/164, 1625/1625, 9918/9918). Raise `--reps` until every `region_s` clears 0.05 s (0.2 s
preferably); the harness prints it. Read the `pairs` column first: `MIXED` means no consistent change, whatever the
median says.

### (c) Cross-engine parity with the path enabled

The literal gate needs the binding kwarg of §4, which this experiment deliberately does not add. Two substitutes,
both already in place:

```bash
# per-layer term-count parity between the two paths, on the study's own circuits
RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 1 --check   # aborts on any per-layer mismatch
# per-layer term-count parity as a CI test, across the channel zoo and the transition
cargo test -p paulistrings --test small_sum_path
# and the sorting path is untouched, so the committed cross-engine gate must still pass as-is
pytest python/paulistrings/tests/test_jl_performance_protocol.py
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py --curves --workload kicked_ising --pilot
```

The cross-engine driver compares per-layer counts; the two substitutes compare the same quantity between our two
paths, which is the property the driver would exercise. Running the real thing with the path enabled is a follow-up
gated on §4.

---

## 7. Risks, and what was deliberately not done

**Risks**

1. **Code layout on the default path — measured, and it is layout.** Gate (a) settled the mechanism (identical work
   counts, redistributed phase times) but the *cost* is real on one side: `su4` is +3.5–3.9 % wall, sign-consistent,
   while `rotation_zz` is −2.3 to −2.5 %. Any commit that adds code to this crate re-rolls that die; the remedy named
   in §1.1 (a verbatim `for k in 0..n` loop) would not reliably remove it, since the mechanism is the presence of new
   code in a `lto = "fat"` / `codegen-units = 1` unit rather than the loop bound.
2. **One global threshold for a workload-dependent crossover.** 2048 is non-regressive across the six measured
   configurations, but the cliff on kicked-Ising 2⁻⁸ sits just one octave above it (1.081 at 2048, 0.676 at 4096), so
   a workload whose cliff is lower would regress. A per-workload or adaptive threshold — time the first few layers
   both ways, then decide — is the obvious follow-up and is not attempted here.
3. **Exact-zero cancellation could in principle change a per-layer term count** between the two paths (§1.4). Not
   observed anywhere, but it is the one way parity could break on a workload not tested here.
4. **The map's memory profile is not measured.** A `DirectSum` holds `2 × threshold` map slots plus the fanout
   buffers; at 2048 entries that is a couple of hundred KB and cannot matter next to the sums this engine targets, but
   the peak-RSS claim is unmeasured.
5. **`Auto`'s policy exclusion is invisible from Python** until `SpecPolicy` overrides the hint (§4), so a Python
   caller who eventually gets the kwarg and passes `engine="auto"` with any policy at all would silently stay on the
   sorting engine. Named here so it is not diagnosed twice.

**Deliberately not done**

- **No Python binding changes** (mission constraint). §4 says exactly what they would be.
- **`Auto` is not the default.** Gates (a) and (b) both pass, so the evidence for flipping it now exists at one
  thread; what is still missing is a multi-thread cell. The direct path has *no* parallelism, so at 8–32 threads its
  crossover moves down by roughly the sorting engine's speedup at that `m` — which at `m` ≈ 10² is small, but it has
  not been measured and the default serves threaded callers too. Recommend: flip only after a `--threads 8` cell of
  gate (b), and consider gating `Auto` on `rayon::current_num_threads() == 1` if that cell disappoints.
- **No ARCHITECTURE.md section.** The design lives here while the path is experimental and off by default; a
  `§Small-Sum-Path` section is the right home if and when the default flips, and writing one now would collide with
  four sibling experiments editing the same file.
- **No cross-call scratch reuse** for the map, to keep `LayerScratch`'s layout untouched (§1.1).
- **No zero-row skip in the accumulation loop.** It was implemented and reverted: it is provably outcome-neutral (a
  key receiving only zero rows would be inserted and then dropped; adding ±0.0 to a nonzero sum changes nothing), but
  the channels in these workloads emit almost no *exact* zeros — a Clifford-angle rotation's suppressed row is
  6.1e-17, not 0.0 — so it was an unmeasured branch in the hot loop. Worth revisiting only with a workload whose
  `Channel::apply` really does emit exact zeros.
- **No saturated-regime work** — the fact sheet found no evidence for it (§0).
- **No dense-PTM-specific tuning.** `prepare` for a dense two-qubit PTM is 4.19–5.71 µs *per gate*, so a small-`m`
  SU(4) brickwork is where this path should win biggest; the harness does not cover it because the study's SU(4) curve
  is `n = 36` (`W = 1`) and its own small-`m` points are already ties (0.983, 0.620 at 1.3 × 10⁴). Adding an `su4`
  workload to `small_m_ab` is a cheap follow-up.

# E3: a direct-apply path for small sums

Branch `expt/small-m-path`, branched from the campaign tip `f592c43`. Experiment E3 of the phase-2 slate in
`research/notes/2026-09-01-large-m-campaign-log.md`, re-rationalized per the Phase-1 fact sheet
(`research/notes/2026-09-01-large-m-phase-breakdown.md` §7(4)).

**Status: implemented, tested, and measured only by non-authoritative smoke.** The authoritative gates (§6) are the
orchestrator's to run serialized on a quiet box. `EngineSelection::Auto` is **off by default** — the default engine is
the sorting engine for every layer, unchanged.

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

`DEFAULT_SMALL_SUM_THRESHOLD = 512`, and `PropagateOptions::small_sum_threshold` is a public field so any measurement
can move it.

The arithmetic the plan asked for: against the sorting engine's ~29 ns/term at `W = 2`, the 1.43 µs fixed cost is 9 %
of a 500-term layer, 3.3 % of a 1500-term one, 1.2 % at 4096 — so 4096 is where there stops being anything to win
*even if the two paths cost the same per term*. That was the first choice.

The measurements (§5) then showed the per-term costs are **not** the same, and differ by workload: the crossover is
**≈ 1.5 × 10² resident terms on kicked-Ising** and **≈ 2 × 10³ on XXZ**, a 14× spread that mirrors the 4.4–21×
spread of the study's own cross-engine crossovers. 512 is the geometric mean of the two, rounded to a power of two.
At 512 nothing in the harness regresses; at 4096 the kicked-Ising configuration just above its crossover loses 1.4×.
512 also sits below `desired_bits`'s `worth_splitting` floor (128 × 64 = 8192), so a sum on this path is one the
sorting engine would have run in few buckets anyway.

---

## 2. What the direct path is

One `Channel::apply` (or `apply_adjoint`) per resident term into a `max_fanout`-sized `OutputBuffer`, every emitted
row accumulated into an `FxHashMap<PauliString<W>, Complex64>`, then one `retain` pass applying the zero-drop and
`keep_term`. That is `test_support::naive_apply_layer`'s algorithm — the differential oracle *is* the fast path now,
which is why `engine::direct`'s unit tests are only a plumbing check and the real algebra check is
`tests/small_sum_path.rs` against the bucketed engine.

No `prepare` (no PTM derivation, no delta plan, no `4^|support|` probe), no `rebucket`, no `Gf2Span`, no
permute/unpermute, no per-run sort, no merge, and no Rayon: a sub-512-term layer has nothing to spread.

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
| `a133762` | `bench: the small-m A/B harness, and a threshold from its crossovers` — `examples/small_m_ab.rs`, and the threshold moved 4096 → 512 on its evidence. |
| `003f79c` | `test: gate the release-only invariant check in the small-sum tests` — `assert_invariants` is debug-only. |

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

## 5. Smoke observations — **NOT AUTHORITATIVE**

Sibling experiments were building on the same box throughout. Load 0.71–6.2 across these runs, `powersave` governor,
one thread, `--reps 30` so each timed leg is 20 ms–3.7 s (the sub-50 ms legs are measured at the idle clock per the
fact sheet §0.1, which inflates the absolute `µs/layer` columns but not a ratio between two equally short legs). Three
abba pairs per cell; `speedup = sorted / direct`, so `> 1` means the direct path is faster. Numbers below are the
cleanest run (load 0.71); every cell was sign-consistent 3/3 unless marked.

At the shipped threshold **512**:

| workload | cutoff | final | peak | mean `m` | sorted µs/layer | direct µs/layer | speedup |
|---|---|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 7 | 68 | 21.6 | 1.295 | 0.549 | **2.36** |
| kicked_ising | 2⁻⁶ | 408 | 517 | 114.3 | 3.236 | 2.949 | **1.10** |
| kicked_ising | 2⁻⁸ | 5 038 | 6 311 | 746.4 | 22.91 | 20.45 | **1.12** |
| xxz | 1e-2 | 156 | 164 | 63.6 | 2.490 | 1.542 | **1.62** |
| xxz | 1e-3 | 1 625 | 1 625 | 434.3 | 18.65 | 18.15 | **1.02** |
| xxz | 1e-4 | 9 918 | 9 918 | 2 197.5 | 69.33 | 68.74 | 0.97 (MIXED) |

At **4096**, the same cells:

| workload | cutoff | mean `m` | sorted µs/layer | direct µs/layer | speedup |
|---|---|---|---|---|---|
| kicked_ising | 2⁻⁴ | 21.6 | 1.286 | 0.551 | 2.34 |
| kicked_ising | 2⁻⁶ | 114.3 | 3.171 | 2.991 | 1.06 |
| kicked_ising | 2⁻⁸ | 746.4 | 22.31 | 31.26 | **0.71** ← regression |
| xxz | 1e-2 | 63.6 | 2.402 | 1.536 | 1.57 |
| xxz | 1e-3 | 434.3 | 14.68 | 10.75 | **1.38** |
| xxz | 1e-4 | 2 197.5 | 67.87 | 69.02 | 0.985 |

Readings, all provisional:

- **The three configurations the study lost all improve, and 2⁻⁴ improves most** — 2.36× at mean `m` = 21.6, where
  the study measured ratio 0.323 (we were 3.1× slower). A 2.36× on our side would put that configuration at ~0.76,
  i.e. still behind Julia but by 1.3× instead of 3.1×.
- **The win is not purely the fixed cost.** Per-term costs, taking the sorting engine's fixed cost as the fact
  sheet's 1.43 µs: kicked-Ising 2⁻⁶ is 15.5 ns/term sorted against 26 ns/term direct — the direct path is *worse per
  term* there and only the fixed cost saves it (hence 1.06–1.10×). XXZ 1e-3 is 40.5 ns/term sorted against 25 direct
  — cheaper per term, hence 1.38× at threshold 4096. The difference between the two workloads is the effective
  fanout: kicked-Ising's entangling layers are at the Clifford angle `-π/2`, whose `cos` row is 6.1e-17 and is
  truncated away, so the sorting engine's gather/sort/merge move about half the rows XXZ's generic-angle rotations do.
- **The direct path's per-term cost rises with `m`** — ~25 ns/term at `m` ≈ 100–400, ~35 at `m` ≈ 2200, ~60 on
  kicked-Ising at `m` ≈ 750 — consistent with the map (48 B/entry) leaving L1 at ~700 entries and L2 at ~20 000.
  The sorting engine's is flat. That is the crossover, and why the threshold cannot be "as high as possible".
- Crossovers by log-interpolation of the ratios: **≈ 1.5 × 10² (kicked-Ising)**, **≈ 2.0 × 10³ (XXZ)**.
- A `--threshold 1024` run was attempted and came out noise-dominated (three cells MIXED, absolute times 20–30 %
  above the adjacent runs); it is not reported as evidence either way.

**Nothing above is a gate result.** Both the ±5–8 % single-thread noise floor and the sibling builds are in these
numbers; the paired-and-interleaved protocol is what makes the *directions* worth quoting at all.

---

## 6. Gate commands

Run serialized, `RUST_LOG` unset, box otherwise idle. `f592c43` is the campaign tip this branch left.

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

**Pass:** a sign-inconsistent null (pairs disagreeing in direction) or a negligible median Δ%. A sign-consistent
regression on any cell is a fail, and the remedy is named in §1.1 (give `propagate_with_scratch` its own verbatim
loop).

### (b) Ours-only small-`m` effect, path ON vs OFF

```bash
cargo build --release --example small_m_ab -p paulistrings
# from the repository root (the harness reads examples/data/heavy_hex_127.edges)
RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 5 --reps 30 --check
# and the threshold sweep the shipped default rests on
for t in 128 512 2048 4096; do
  RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 3 --reps 30 --threshold $t
done
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

1. **Code layout on the default path.** `for k in start..n` is the only change inside the sorting entry point. Gate
   (a) is the measurement; §1.1 has the remedy.
2. **One global threshold for a 14×-spread crossover.** At 512, an XXZ-shaped workload leaves ~1.4× on the table and
   a kicked-Ising-shaped one is safe; the opposite choice regresses kicked-Ising 1.4×. A per-workload or adaptive
   threshold (measure the first few layers, then decide) is the obvious follow-up and is not attempted here.
3. **Exact-zero cancellation could in principle change a per-layer term count** between the two paths (§1.4). Not
   observed anywhere, but it is the one way parity could break on a workload not tested here.
4. **The map's memory profile is not measured.** A `DirectSum` holds `2 × threshold` map slots plus the fanout
   buffers; at 512 entries that is tens of KB and cannot matter, but the peak-RSS claim is unmeasured.
5. **`Auto`'s policy exclusion is invisible from Python** until `SpecPolicy` overrides the hint (§4), so a Python
   caller who eventually gets the kwarg and passes `engine="auto"` with any policy at all would silently stay on the
   sorting engine. Named here so it is not diagnosed twice.

**Deliberately not done**

- **No Python binding changes** (mission constraint). §4 says exactly what they would be.
- **`Auto` is not the default.** Flipping it is a separate decision, gated on (a) and (b), and on at least one
  multi-thread cell — everything measured here is single-threaded, which is the regime the study compares in, but the
  default serves threaded callers too. Note the direct path has *no* parallelism, so at 32 threads its crossover moves
  down by roughly the sorting engine's speedup at that `m`.
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

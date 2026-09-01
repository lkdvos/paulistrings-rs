# The dense-PTM bucket cliff is a delta-span rank effect

Experiment **E4** of the large-`m` campaign (`expt/bucket-cliff`), narrowed per
`research/notes/2026-09-01-large-m-campaign-log.md` §"Phase 2 slate" to two questions:

- **(a)** why does a one-qubit flip (`q = 64` → `q = 65`) turn the `W = 2` bucket-count cliff on and off?
- **(b)** what bucket-count policy do dense-PTM sums below ~8192 terms want?

**Answer to (a): the flip is not about `W` and not about the near-empty second word.** It is the GF(2) rank of
the partitioning hash restricted to the layer's key-delta space. `Gf2Hash::new` draws `2W` words per row, so
`W = 1` and `W = 2` get *unrelated* row patterns in word 0, and whether the four delta generators come out
linearly independent is an independent coin flip at each width. At the default seed the `su4` probe's support
`(0, 1)` lands rank 3 at `W = 1` and rank 4 at `W = 2`.

**And it dissolves a Phase-1 finding.** The fact sheet's §3.2 — *"the `W = 1` sort is 1.9× slower per row than
`W = 2` on high-duplicate runs for half the key bytes, which is a defect in its own right"*, handed to E2 as its
cheapest item — is **mostly not a width effect**. Flipping only the hash seed at fixed `W = 1` recovers
**−29.6%** of the sort and **−22.5%** of the layer, 3/3 paired. The genuine residual width penalty is **≈1.35×**
(1.34–1.37 across pairs), not 1.9×, and it is a pure per-comparison cost: the comparison counts are byte-identical (4.93/row) across the
two widths at matched rank.

**Answer to (b): the mechanism says "more buckets", the measurements say "not unconditionally", and the constant
is not resolvable on this box today.** Forcing the floor is worth **−38.6%** on a dense-PTM layer at 980 terms
and costs **+46.6%** on a rotation layer at 1497 terms (both 3/3 paired). A fix therefore has to key off the
channel's fanout, and choosing its constant needs the campaign's serialized `ab-compare` pass — the bit-count
sweep is *non-monotone* on single-shot data and entangled with the parallel task count. §6 specifies that gate.
**No default constant was changed in this experiment.**

---

## 0. What the sort actually costs

`merge::sort_rows_with_scratch` sorts a `Vec<u32>` permutation through an indirect key comparator, using Rust's
**stable** `sort_by` — driftsort — chosen there explicitly *for its run adaptivity* (its own comment records
+77% for `sort_unstable_by`). So its cost is comparison count, and comparison count is set by how presorted
its input is.

Comparison counts are **exact functions of the configuration**. Wall times on this host are not: single-shot
noise is ±5–8% single-threaded (`benchmarks/PROFILING.md`), the box had three sibling agents building
throughout (load 1–14), and the small-`m` cells are microseconds. So every structural claim below rests on
counts from `crates/paulistrings/examples/delta_span_diagnostics.rs`, and timings appear only as corroboration,
always labelled.

The two agree well enough to make the counts sufficient. Against the Phase-1 fact sheet's measured
ns/sorted-row for the same configurations (`su4`, `q = 128`, one thread):

| `bits` | `r` = coset dim | gather order | **comparisons / row** | fact-sheet ns / sorted row | ns per comparison |
|---|---|---|---|---|---|
| 0 | 0 | input-major | 13.53 | 53.5 | 3.96 |
| 1 | 1 | input-major | 16.23 | 50.3 | 3.10 |
| 2 | 2 | input-major | 13.62 | 39.7 | 2.91 |
| 3 | 3 | output-major | 9.18 | 27.4 | 2.98 |
| 7 | **4** | output-major | **4.91** | **16.3** | 3.32 |

**ns per comparison is flat to ±15% across a 3.3× spread in ns/row.** The entire configuration sensitivity the
fact sheet found is comparison count. And `4.91` is `log2(15) + 1`: the information-theoretic floor for merging
the 15 non-identity delta streams. **At full rank this sort has no headroom left.**

---

## 1. Mechanism

### 1.1 The chain

```
rank(h(D))  and  bits   ──►   r = min(rank, bits)   ──►   are the gather blocks ascending?   ──►   comparisons
   (H's rows)   (term count)     (coset dimension)              (yes iff r is full)                 (4.9 vs 9–16)
```

**Step 1 — `r = min(rank(h(D)), bits)`.** A channel supported on `{i, j}` can only change those qubits' `x` and
`z` bits, so its key-delta set lies in the 4-dimensional `span{X_i, Z_i, X_j, Z_j}`; a dense 16×16 PTM
populates all 16 of them. The engine's unit of work is a coset of `span(h(D))`, of dimension
`r = Gf2Span::r()`. Pinned by `engine::coset::tests::coset_dimension_is_the_delta_span_rank_capped_by_the_bucket_bits`.

**Step 2 — the sum's steady state.** A repeated dense-PTM layer closes to a fixed point that is exactly
*(every off-support pattern) × (all 16 local patterns on the support)*. Confirmed by the probe: `terms_out ==
terms_in` and the closed count is ~14.2× the seed. This is the key set that matters, and it is what a random-key
fixture misses entirely — the first version of the test below passed vacuously for that reason.

**Step 3 — where those 16 local variants land.** `h(v) = h(rest) ⊕ h(local)`, so the 16 local variants of one
off-support pattern land in exactly `2^r` distinct buckets, `16 / 2^r` per bucket.

- `r = 4`: **one per bucket.** No two keys in any bucket differ only inside the support.
- `r = 3`: **two per bucket** — and since the support bits are the *least significant* bits of the primary sort
  key `x[0]`, those two are **adjacent** in the bucket's ascending key column.

**Step 4 — order under translation.** The gather emits, per delta `d`, the block `{v ⊕ d : v ∈ bucket}`. For
`a < b`, `a ⊕ d < b ⊕ d` iff `d`'s bit at `a`'s and `b`'s most-significant differing position is 0. At `r = 4`
consecutive keys differ high (in the off-support bits), where every support delta is 0 → **every block is fully
ascending, and driftsort merges 15 sorted runs**. At `r = 3` consecutive keys often differ *only* at the kernel
delta `g`'s bits → half the deltas (those with a 1 at `g`'s top bit) **invert every adjacent pair**, shattering
those blocks into runs of ~2.

Pinned by `bucket::hash::tests::support_delta_preserves_bucket_order_iff_the_delta_span_is_full_rank`, which
partitions the closed key set and checks all 15 non-identity deltas against every bucket, at `W ∈ {1, 2}`.

**The arithmetic checks out.** At `r = 3`, 7 of the 15 blocks stay sorted and 8 shatter into `L/2` runs of 2:
predicted `7 + 8·(125/2) = 507` runs, measured **456** for a 1876-row stream. At `r = 4`: predicted 15,
measured **15**.

### 1.2 The gather order is the *second-order* term, not the cliff

The engine picks input- or output-major on `m = 2^r ≥ 8` (`GATHER_OUTPUT_MAJOR_MIN_R = 3`). Comparisons per
row, both orders, `su4` at `q = 128`:

| `r` | deltas per coset coordinate | input-major | output-major | Δ |
|---|---|---|---|---|
| 0 | 16 | 13.53 | 14.51 | **+7% (worse)** |
| 1 | 8 | 16.23 | 13.94 | −14% |
| 2 | 4 | 13.62 | 12.75 | −6% |
| 3 | 2 | 13.88 | **9.18** | −34% |
| 4 | 1 | 4.91 | 4.91 | **0%** |

Two things fall out.

- **At `r = 4` the two orders are identical**, because each delta owns a coordinate and both emit one contiguous
  ascending block per delta. So the `r = 4` choice is a pure gather-cost question, and it is already settled in
  the source (input-major at `r = 4` measured +48% at 32 threads). The comment claiming that branch is
  "unmeasured, as a guard for a custom full-rank channel" was wrong and is corrected — it is the hot path for
  every matrix-gate layer at `B ≥ 128`.
- **Retuning `GATHER_OUTPUT_MAJOR_MIN_R` downward is not the fix.** Below `r = 3` output-major is worth −14%,
  −6%, and *+7%* — inconsistent and small. **Rejected on deterministic evidence, before any timing.**

### 1.3 Rank deficiency is a draw, and not a rare one

`rank(h(D)) < 4` is a property of `H`'s rows, not of the channel. Measured over all support pairs at
`DEFAULT_HASH_SEED` (`bucket::hash::tests::support_delta_rank_is_usually_full_but_not_always`):

| qubits | `W` | support pairs | rank-deficient at `bits = 7` |
|---|---|---|---|
| 36 | 1 | 630 | 82 (13.0%) |
| 64 | 1 | 2016 | 205 (10.2%) |
| 65 | 2 | 2080 | 205 (9.9%) |
| 128 | 2 | 8128 | 818 (10.1%) |

And over hash seeds at `nq = 128`, support `(0, 1)`: **70.0%** deficient at `bits = 4`, 40.0% at 5, 22.0% at 6,
**11.8%** at 7, 5.8% at 8, 2.9% at 9. (Four random `b`-bit vectors are dependent with probability
`1 − ∏(1 − 2^{i−b})`, which is what these are.)

So: **about one two-qubit gate placement in ten pays ~1.8× on a dense-PTM layer's sort, at any term count, for
no reason visible from the workload.** At the default seed the `su4` probe's `(0, 1)` is one of them at `W = 1`,
and rank only reaches 4 at `bits = 10` there — which is why `su4-w1.jsonl` shows a *flat* ~30.5 ns/sorted row
across `m = 980 → 283 074`: it never leaves the `r = 3` regime.

### 1.4 Why the cliff sits at exactly 8192 terms

`desired_bits(len, 1024, 128)` gates the `min_buckets` floor on `len ≥ 128 × 64 = 8192`. Below that
`bits = ceil(log2(len/1024)) ∈ {0, 1, 2, 3}`, so `r ≤ 3` **by construction, regardless of rank** — the fast
regime is unreachable. The committed Phase-1 data already contains the proof: `runs / cosets` in
`su4-fine.jsonl` is exactly `2^min(rank, bits)` at every cell (1, 2, 4, 8, 16 as `bits` goes 0, 1, 2, 3, 7).

### 1.5 One second-order effect worth knowing

Driftsort only exploits runs above a minimum length. The `m = 9884` cell has `r = 4` and 15 runs but only ~62
rows per run, and its comparison count is **9.72/row**, not 4.9 — matching the fact sheet's 19.5 ns/row there
against 16.3 at `m = 15 806`. So very short gather runs lose the adaptivity even at full rank, which is the
upper bound on how far "more buckets" can be pushed.

---

## 2. Evidence index

Deterministic (no timing, reproducible anywhere):

- `cargo run --release -p paulistrings --example delta_span_diagnostics` — the table in §0, the order comparison
  in §1.2, the run counts in §1.1.
- `cargo test -p paulistrings --lib bucket::hash` — rank flip across the word boundary, rank monotonicity in
  `bits`, the ~10% deficiency rate, and the order-preservation lemma.
- `cargo test -p paulistrings --lib engine::coset` — `r = min(rank, bits)` for the real channel, and the 16-vs-8
  member coset at the two widths.
- Committed Phase-1 raw data: `runs`/`cosets` in `benchmarks/results/2026-09-01-ccqlin038/{su4-fine,su4-w1,width}.jsonl`.

## 3. Smoke timings — **NON-AUTHORITATIVE**

Host ccqlin038, one thread, `--reps` auto-scaled to ~200 ms per §0.1 of the fact sheet, `RUST_LOG` unset.
**The box was not quiet**: three sibling E-agents were building and testing throughout, load 1.2–14.2. Runs are
interleaved A/B/A/B adjacent in time, 3 pairs, reported per `PROFILING.md`'s criterion (direction consistency;
median Δ% as effect size). These numbers must not be quoted as measurements — they exist to say which way each
knob points, at effect sizes far above the noise.

| # | cell | A | B | Δ per pair | pairs consistent | median Δ |
|---|---|---|---|---|---|---|
| a | `su4` `q=128` `m=980` | policy (`bits=0`) | `--bucket-bits 7` | −38.6 / −38.0 / −39.3 % | **3/3** | **−38.6 %** |
| b | `su4` `q=64` `m=63 364` | default seed (`r=3`) | `--hash-seed 0x8C03…` (`r=4`) | −21.6 / −22.5 / −25.2 % | **3/3** | **−22.5 %** |
| c | `rotation_zz` `q=128` `m=1497` | policy (`bits=1`) | `--bucket-bits 7` | +43.9 / +46.6 / +83.0 % | **3/3** | **+46.6 %** |
| d | `su4` `q=128` `m=63 518` | policy (`bits=7`) | `--bucket-bits 7` | +2.0 / −6.7 / +2.1 % | 2/3 → **null** | — |

(d) is the designed null: both arms are the same configuration, and the signs disagree — which is also the
clearest single indicator of how perturbed the box was (its pair-1 baseline is 388.7 ns/term, its pair-3
baseline 628.1).

Supporting single-shot cells, same caveat, from the wider sweep:

| cell | policy | `--bucket-bits 7` | Δ |
|---|---|---|---|
| `su4` `m = 980` | 853.3 ns/term | 523.8 | −39 % |
| `su4` `m = 2002` | 815.1 | 548.5 | −33 % |
| `su4` `m = 3976` | 657.3 | 573.8 | −13 % |
| `su4` `m = 7868` | 538.9 | 568.4 | **+5 %** |
| `cnot` `m = 1000` | 29.1 | 57.0 | **+96 %** |
| `cnot` `m = 4000` | 38.1 | 50.5 | **+33 %** |
| `rotation_zz` `m = 5999` | 28.8 | 32.8 | +14 % |

Sort cost per row for (b), the fact-sheet correction: 31.08/31.41/32.61 (`r = 3`) against 22.14/22.12/22.28
(`r = 4`), 3/3, median **−29.6%** — with byte-identical comparison counts on both sides of the pair at
4.93/row, so the whole difference is the presortedness, and the remaining gap to `W = 2`'s 16.15 ns/row at the
same rank and the same rows-per-run is the **≈1.35×** residual width penalty.

---

## 4. Decision: what changed, and what deliberately did not

**Landed** (this branch, `expt/bucket-cliff`):

1. Tests pinning the mechanism (`bucket/hash.rs`, `engine/coset.rs`), parameterized over `W ∈ {1, 2}`, plus
   shared fixtures `test_support::{haar_su4_matrix, gf2_rank, support_delta_rank}` — `haar_su4_matrix` also
   de-duplicates `phase_breakdown`'s copy.
2. `phase_breakdown --hash-seed` and `--bucket-bits`. The fact sheet named both as prerequisites for this
   experiment. Defaults are preserved exactly, so every committed baseline still reproduces; the JSON schema
   gains two fields additively.
3. `examples/delta_span_diagnostics.rs`, the deterministic instrument.
4. Corrections to `MIN_TERMS_PER_TASK`, `DEFAULT_TARGET_BUCKET_LEN` and `GATHER_OUTPUT_MAJOR_MIN_R`'s doc
   comments and to ARCHITECTURE.md §Bucket-Policy.

**Not landed, deliberately: no default constant was changed.** Three reasons, in order of weight.

1. **A blanket fix is wrong and measured wrong.** Dropping or lowering the `worth_splitting` gate costs the
   sparse-PTM path +14% to +96% (§3), because its sort is 7% of the layer and extra buckets are pure per-run
   overhead. Any real fix must key off the channel's fanout, which means a channel-aware floor —
   `Channel::max_fanout()` is available before `prepare` and separates the cases exactly (16 for
   `GeneralUnitary2Q`, 2 for `PauliRotation`, 1 for `Clifford2Q`).
2. **The constant is not resolvable on single-shot data.** The bit sweep at `m = 980` is *non-monotone*: 846
   (`bits=0`), **503** (3), 650 (4), 615 (5), 588 (6), 518 (7), 478 (9), 464 (10) ns/term. The comparison counts
   are non-monotone too (7.69 at `bits=3`, 9.69 at 4, 8.62 at 5, 6.13 at 6) but *not in the same pattern* —
   below `bits = 7` each extra bit also shortens the gather run, so run-length effects (§1.5) and rank effects
   are superimposed and neither the counts nor the single-shot times pick a winner. And `bits = 3`, the best
   single-thread cell, gives the whole sum **one coset**, i.e. one parallel task, so it is inadmissible above one
   thread. Picking between "3", "the existing floor of 7", and "refine until the delta span separates" needs
   thread-count data on a quiet box.
3. **The campaign says so.** Decision 1 of the orchestration log defers all authoritative timing to a serialized
   `ab-compare` pass; §7(2) of the fact sheet asked for the knob first. Committing a constant tuned at load 14
   is precisely the failure `PROFILING.md` is written against.

Also **rejected on deterministic evidence** (§1.2): retuning `GATHER_OUTPUT_MAJOR_MIN_R`. Worth −14%/−6%/+7% in
comparisons at `r = 1/2/0` — inconsistent and small, and it does not touch the rank problem.

And **not attempted**: choosing `H` to guarantee full delta rank for every support pair. It is a *partial
spread* of 2-dimensional subspaces — `V_i = ⟨h(X_i), h(Z_i)⟩` must satisfy `V_i ∩ V_j = 0` for all `i ≠ j` — and
the maximum partial spread in `GF(2)^b` has ~`2^b/3` members, so 128 qubits need `bits ≥ 10` for a full
solution. A greedy construction could cut the 10% deficiency to ~1.6% at `bits = 7`, but it would replace the
universal-hash argument in §Hash with a structured one, invalidate every committed `H`-dependent trajectory, and
buy `0.084 × 1.8×` on dense-PTM layers only. Recorded here so it is not re-derived; not recommended.

---

## 5. Consequences for the other Phase-2 experiments

- **E2 (`expt/sort-kernel`)**: its "cheapest, unplanned" item — the `W = 1` sort defect at 1.9× — is **≈1.35×**
  once rank is controlled, and the mechanism it hypothesised (derived `Ord` for `[u64; 1]` vs `[u64; 2]`) is now
  the *only* candidate left, since comparison counts are identical across widths at matched rank. Better
  targeted, smaller prize. More important for E2: **at full rank this sort is already at its comparison floor
  (`log2(15) + 1`)**, so a radix replacement's value is not "fewer comparisons at the asymptote" but "immunity
  to the 3.3× configuration spread" — a robustness argument, and it should be sold and gated as one.
- **E5 (`expt/mem-growth`)**: unaffected, but note that any policy that changes `B` changes the per-bucket
  capacity slack, so E4's proposed fix and E5's RSS accounting interact. Gate them in the log's order (E5 last).
- **E1, E3**: unaffected.
- **Fact sheet §3.2 and §7(3)** should be read with §0/§1 of this note; §3.1's numbers all stand, only their
  attribution changes (bucket count → coset dimension, of which the bucket count is one of two inputs).

---

## 6. Proposed gate (for the serialized benchmarking pass)

Two independent questions, in this order. Both dense-PTM cells at **`--threads 1` and `--threads 8`** only —
the fact sheet's §4 shows `su4` saturates the write path from 16 threads, so 16/32 would measure the memory
controller.

### 6.1 Does a channel-aware bucket floor pay? (no code change needed to answer)

`--bucket-bits` forces the partition without touching the policy, so the question is answerable with the
committed probe, in one arm each:

```bash
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown
P=./target/release/examples/phase_breakdown

# reps per cell: a throwaway --reps 8 calibration pass, then reps = clamp(200ms / per-layer, 4, 40000).
# dense PTM, below the 8192-term gate — the cells the fix targets
for n in 70 140 280 560; do for b in 0 3 4 7 8; do
  $P --n $n --qubits 128 --threads 1 --layers su4 --reps <auto> --bucket-bits $b --format json --json-out gate.jsonl
  $P --n $n --qubits 128 --threads 8 --layers su4 --reps <auto> --bucket-bits $b --format json --json-out gate.jsonl
done; done

# the sparse-PTM cost of the same floor — the veto arm
for n in 1000 4000 8000; do for b in 0 7; do
  $P --n $n --qubits 128 --threads 1,8 --layers rotation_zz,cnot --reps <auto> --bucket-bits $b \
     --format json --json-out gate.jsonl
done; done

# large-m non-regression: the floor must be inert here (both arms are already bits >= 7)
for n in 4480 70500; do for b in 0 7; do
  $P --n $n --qubits 128 --threads 1,8 --layers su4 --reps <auto> --bucket-bits $b --format json --json-out gate.jsonl
done; done
```

**Decision rule.** Implement a channel-aware floor only if, on a quiet box: the dense-PTM cells below 8192 terms
improve by ≥ 20% at both thread counts; the `rotation_zz`/`cnot` arms at `bits = 0` (i.e. the *unchanged* path)
are untouched; and the large-`m` arms show no consistent change. If the best `bits` differs between 1 and 8
threads, the floor must be the multi-thread choice — a one-coset partition is not admissible.

Sketch of the change, if the gate passes: `engine::propagate` already has `ch.max_fanout()` before `prepare`, so
`rebucket`'s floor can be raised for a fanout-`≥ 8` plan only. That needs `MIN_TERMS_PER_TASK` to become a
parameter of `desired_bits`/`rebucket` (both `pub`), so it is an API change, not a constant edit.

### 6.2 Non-perturbation of everything else

Nothing on this branch changes a default, so an A/B against the campaign tip should be a **null**. Run it as
such — it is a check on the probe edit and the `test_support` move, not on a performance hypothesis. One
invocation per cell (`ab-compare.sh` takes a single `--probe` string):

```bash
TIP=<campaign-tip-sha>
for probe in \
  '--n 70    --qubits 128 --threads 1 --layers su4 --reps 300' \
  '--n 560   --qubits 128 --threads 1 --layers su4 --reps 55' \
  '--n 4480  --qubits 128 --threads 1 --layers su4 --reps 6' \
  '--n 4480  --qubits 65  --threads 1 --layers su4 --reps 6' \
  '--n 70500 --qubits 128 --threads 8 --layers su4 --reps 40' \
  '--n 1000000 --qubits 128 --threads 1 --layers rotation_zz --reps 8' ; do
  scripts/ab-compare.sh null-check --a "$TIP" --b . --features phase-timing \
    --probe "$probe" --pairs 3
done
```

Expected: no consistent change in any cell (signs disagree). A consistent change anywhere means the
`haar_su4_matrix` move or the `run_cell` knob plumbing perturbed code layout, which would be worth knowing —
`merge.rs`'s `#[inline]` history says such effects are real here.

### 6.3 Optional, cheap, and independent of both

If the seed-dependence of §1.3 is worth acting on at all, the one-line version is to check whether a *different*
`DEFAULT_HASH_SEED` gives full delta rank at `bits = 7` for more support pairs than the current one's 90%
(seeds 1–6 of `DEFAULT_HASH_SEED ^ k·0x123456789ABCDEF1` all give rank 4 for `(0, 1)` at both widths). This is a
one-constant change with a large blast radius (every committed `H`-dependent trajectory) and an expected value
of a few percent averaged over gate placements. Listed for completeness; not recommended ahead of E2's radix
sort, which would make the whole rank question moot.

---

## 7. Reproduction

```bash
# deterministic — no quiet box needed, no timing
cargo run --release -p paulistrings --example delta_span_diagnostics
cargo test  -p paulistrings --lib bucket::hash
cargo test  -p paulistrings --lib engine::coset

# the two smoke knobs (label anything from these non-authoritative)
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown
P=./target/release/examples/phase_breakdown
$P --n 70   --qubits 128 --threads 1 --layers su4 --reps 300 --format json                  # policy, bits=0
$P --n 70   --qubits 128 --threads 1 --layers su4 --reps 300 --bucket-bits 7 --format json  # forced floor
$P --n 4480 --qubits 64  --threads 1 --layers su4 --reps 6   --format json                  # W=1, rank 3
$P --n 4480 --qubits 64  --threads 1 --layers su4 --reps 6 --hash-seed 0x8C032FC1E5F6A2E4 \
   --format json                                                                            # W=1, rank 4
```

`--n` is the *seed* term count; the probe reports the closed count as `n` (closure factor ~14.2× for `su4`).
`runs / cosets` in the JSON is the coset width `2^r` — the single most useful diagnostic field for this
mechanism, and it needs no new instrumentation.

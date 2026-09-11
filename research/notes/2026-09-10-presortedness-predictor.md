# The radix gate's missing predictor is not presortedness — it is branch predictability, and the PTM already knows it

`2026-09-10-constant-recalibration.md` §3 left one open result: `cnot` and `gu2q` have the same
rest-stream count (3) and want opposite sort kernels, 19 points of wall apart, so
`RADIX_MIN_REST_STREAMS` cannot express `cnot`'s −5.5%. Its §6.1 conjectured the missing quantity
was **presortedness** — how many maximal ascending runs a gather run arrives as — and pointed at
`examples/delta_span_diagnostics.rs`, which measures exactly that offline.

**The conjecture is wrong, and measurably so.** Every built-in `Local` layer's gather run arrives
as *exactly* `k` ascending runs for `k` rest streams — zero inversions, on `cnot`, `gu2q`, `su4`
and `rotation_zz` alike — and comparisons per row are equal on the two 3-stream layers (2.65 both).
Presortedness carries no information the stream count does not already carry. It is not a weak
predictor; it is a constant.

What separates them is **nanoseconds per comparison**: 4.23 on `cnot`, 2.74 on `gu2q`, from branch
misprediction in the `k`-way merge. `cnot` is a key *permutation*, so its rest streams are pairwise
disjoint key sets and "which stream is next" is a coin flip; `gu2q` fans out, so all three streams
carry every output key, step in lock-step, and predict. The radix kernel is branchless and flat in
both. The quantity is readable straight out of the prepared PTM's amplitude support at plan time —
`rest_rows_per_key`, 1.00 for `cnot`, 3.00 for `gu2q`, 14.00 for `su4`, matching the engine's own
instrumented `rows_sorted / distinct keys` to the digit.

**Shipped**: a second arm on the radix gate — `rest_streams >= 3` **and** `rest_rows_per_key < 2`.
`cnot` moves to the radix kernel for **−5.26% wall / −21.67% sort, 14/14 pairs**; `cz` and `swap`
move with it; `gu2q`, `su4`, `rotation_zz`, `tfim_step` and `heavyhex_step` keep the kernel they
had.

---

## 0. Protocol and provenance

Host ccqlin038 (reference host, 2× Xeon Gold 6244 Cascade Lake-SP, governor `powersave`),
2026-09-10, rustc 1.94.0, branch `simd` at `4e1eea2`.

- Cell: `--n 1000000 --qubits 128 --threads 1 --reps 20`, `taskset -c 6`, `RUST_LOG` unset.
  Exceptions as in the recalibration note: `su4 --reps 2`, `tfim_step --reps 6 --truncation
  coeff:1.220703125e-4`.
- Four binaries built up front into separate `CARGO_TARGET_DIR`s and copied out; no build ever ran
  during a measurement. `env -u RUSTFLAGS` throughout, so `.cargo/config.toml`'s JCC padding
  applied intact.
  | name | source |
  |---|---|
  | `a_r8` | `4e1eea2` unmodified — the incumbent |
  | `b_r2` | `RADIX_MIN_REST_STREAMS = 2`, the forced-radix arm of the recalibration sweep |
  | `c_new` | the two-arm gate, with the overlap derived inside `DeltaPlan::new` |
  | `d_final` | the shipped source: `c_new` plus the `LayerKnobs::rows_per_key` override the partitioned path needs (see §3) |

  Every control cell in §5 was taken on `c_new`; `cnot` and `gu2q` were re-taken on `d_final`. The
  two differ only in plan-time code — one `Option<f64>` on a struct built once per layer — and the
  re-take agrees, which is the check that says so.
- Alternated `abba`, 7 pairs (14 comparisons), `scripts/ab-report.py --all-phases`. Acceptance is
  direction consistency across every pair.
- Work counters `terms_in`, `terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`, `cosets`,
  `runs`, `layers` bit-identical between every pair of sides, `rows_sorted` included: the gate
  changes *how* a run is sorted, never what is gathered or sorted. Nothing here is a work-count
  difference in disguise.
- Load 1.7–5.0 (a neighbouring agent session). Not a quiet box; `abba` pairing is the mitigation,
  and every cell reported below is 14/14 or a declared null.
- The structural measurements of §1–§2 are deterministic and need no protocol: they come from a
  scratch instrumentation of the engine's own gather runs (counting ascending runs, comparisons
  through a counting comparator, and distinct keys, immediately before the real sort, behind an
  env var and the `phase-timing` feature). That instrumentation is **not** committed — it puts an
  `env::var_os` and an `O(n log n)` shadow sort in the per-run path.

## 1. The presortedness conjecture, and why it is a constant

### 1.1 The derivation

Keys compare lexicographically over `(x[0..W], z[0..W])` as unsigned words, so all of `x` outranks
all of `z` and bit 63 is the most significant bit of a word. For two keys `u < v`, XOR by a
constant `d` does not change *which* bits differ, so it flips their order exactly when `d` sets the
highest bit at which they differ:

```
u^d < v^d   iff   d has a 0 at hdb(u, v)
runs(stream_d) = 1 + #{ adjacent pairs (u,v) of the sorted bucket : hdb(u,v) ∈ bits(d) }
```

That is exact, and it makes presortedness a function of two things only: the bucket's **hdb
distribution over adjacent pairs**, and **which bits the delta mask sets**.

### 1.2 The measurement: the two distributions do not overlap at all

A two-qubit gate on qubits 0 and 1 sets, at most, bits 0–1 of `x[0]` and of `z[0]` — key positions
62, 63, 190, 191 counting from the most significant. The hdb distribution of a 128-qubit random
sum driven to each channel's fixed point (`--n 200000`, engine bucket policy) is concentrated at
positions **3–12**, the *top* bits of `x[0]`, exactly where `log2(terms)` puts it: adjacent keys in
a sorted bucket first differ near the top of the first word, never near its bottom.

| layer | bucket bits | hdb mode (share of pairs) | hdb mass at 62/63/190/191 |
|---|---:|---:|---:|
| `cnot` | 8 | 8 (20%) | **0** |
| `gu2q` | 10 | 6 (20%) | **0** |
| `su4` | 12 | 8 (20%) | **0** |
| `rotation_zz` | 9 | 8 (20%) | **0** |

Zero, not "small". So no delta of any two-qubit gate can invert any adjacent pair, and the engine's
own gather runs confirm it — `runs / block` is exactly the stream count, on every layer, with no
remainder:

| layer | rows / block | runs / block | rest streams | cmp / row | `log2(k)+1` | rows / distinct key |
|---|---:|---:|---:|---:|---:|---:|
| `rotation_zz` | 488 | **1.00** | 1 | 1.00 | 1.00 | 1.00 |
| `cnot` | 734 | **3.00** | 3 | 2.65 | 2.58 | 1.00 |
| `gu2q` (compacting layer) | 549 | **3.00** | 3 | 2.64 | 2.58 | 1.00 |
| `gu2q` (fanning layer) | 2196 | **3.00** | 3 | 2.66 | 2.58 | 3.00 |
| `su4` | 7212 | **15.00** | 15 | 4.93 | 4.91 | 14.00 |

`runs = k` and `cmp/row` at the `k`-way-merge floor, everywhere. The comparison kernel is already
doing the minimum possible number of comparisons on every layer in the suite, and it is doing the
*same* number per row on the two layers that disagree about which kernel they want.

Two corollaries worth keeping:

- `examples/delta_span_diagnostics.rs`'s `runs` column is a correct instrument that measures a
  quantity with no variance in this engine. Its own doc says driftsort's run detection is "exactly
  what decides" the kernel; that is true of the *mechanism* and false of the *variation*. (The
  example was also checked against the post-`4ee8033` gathers: its mirrored orders still emit the
  same row sequence, because the branchless rewrite changed only how a row is published.)
- Presortedness *does* vary where the keys are not dense random: `tfim_step`'s truncated layers
  reach **9–16 runs in a 117–196-row block** off a *single* rest stream, because a
  coefficient-truncated low-weight sum has nearly constant `x` words, the discrimination falls into
  `z`, and a `ZZ` generator's mask reaches it. It is the one place in the engine where §1.1's rule
  has a nonzero right side, and it is out of the gate's reach anyway: one rest stream is below both
  arms. Anyone re-opening the presortedness idea should start here rather than at a dense random
  sum.
- A correction to `2026-09-10-constant-recalibration.md` §1's table: `rotation_zz` is listed there
  as a `Rotation` plan, "not this path". It is not — the probe's generator is a weight-2 `ZZ`, and
  `PauliRotation::prepare` tabulates any generator of weight `≤ MAX_LOCAL_SUPPORT`, so the layer is
  a `Local` plan with 1 rest stream that does go through `fill_coset`'s sort. The row's conclusion
  (radix off, 1 stream) is unaffected; only the plan label is.

## 2. What actually differs: nanoseconds per comparison

Sort cost per row sorted, from the probe's own `sort_ns / rows_sorted`, both kernels:

| layer | comparison kernel | radix kernel | cmp / row | **ns / comparison** |
|---|---:|---:|---:|---:|
| `cnot` | **11.21** | 8.68 | 2.65 | **4.23** |
| `gu2q` | **7.25** | 9.67 | 2.65 | **2.74** |

Same algorithm, same comparison count, 54% apart. `perf stat` over 10 layers, radix minus
comparison, says what it is:

| layer | Δ cycles | Δ instructions | Δ branch-misses | Δ misses / row | Δ misses / **comparison** |
|---|---:|---:|---:|---:|---:|
| `cnot` | −68M | +526M (+4.9%) | **−10.97M** | −1.46 | **−0.55** |
| `gu2q` | +1102M | +4935M (+13.9%) | −15.79M | −0.28 | −0.11 |

`cnot`'s `k`-way merge mispredicts on **55% of its comparisons** — a coin flip, which is what a
merge of unrelated key sets is. `gu2q`'s mispredicts on 11%. The radix kernel pays a flat ~70–88
extra instructions per row in both (its passes are counting sorts, with no data-dependent branch to
miss), and on `cnot` that buys more than it costs: instructions up 4.9%, cycles down, wall down
5.7%.

The mechanism is structural, not statistical. `cnot` is a key **permutation**: each source row
produces exactly one output row, so the three rest streams are disjoint key sets drawn from three
*different* source buckets, and their interleaving in the merged output is as random as the hash.
`gu2q` fans out: in its steady state every output key is produced by all three rest deltas, so the
three streams are near-copies of one another, the merge advances them round-robin, and the
predictor learns the pattern in a handful of iterations.

## 3. The predictor: `rest_rows_per_key`

Under the locally-closed sum a steady-state layer sees, output local pattern `o` receives a rest
row from entry `e` exactly when `amp_e[o ^ local_delta_e] != 0` — `amp[s]` being the weight of
`s -> s ^ local_delta`. So

```
rest_rows_per_key = ( Σ_o #{e ∈ rest : amp_e[o ^ ld_e] ≠ 0} )
                  / #{ o : that count is ≥ 1 }
```

is the mean number of rest rows landing on one output key, which is precisely `rows_sorted /
distinct keys` in a gather run. `1.0` means the streams are pairwise disjoint. It costs `|D| · 4^k`
amplitude tests — 240 for the densest two-qubit channel — once per layer, against a layer that is
tens of milliseconds, and it is evaluated only when the first arm has already declined. It reads
only the prepared PTM, so it adds **no per-sum state**: nothing to maintain across `rebucket` /
`refine` / `coarsen`, and nothing on the MPI wire.

**One trap, and it is not hypothetical.** `partitioned::layer` hands `DeltaPlan::new` a PTM cut
down by `retain_entries` to the deltas that stay local, which is why
`LayerKnobs::rest_streams` already exists to override the *count* with the channel's total.
Dropping entries can only make the remainder look more disjoint — restrict `sqrt(SWAP)` to one rest
delta and it reads **1.00**, `cnot`'s value — so a locally-derived overlap would flip a fan-out
channel onto the radix kernel on the partitions that happen to keep few deltas, and not on the
others. The overlap therefore gets its own `LayerKnobs::rows_per_key`, set from the unrestricted
`prep`, and `bucketed::tests::a_partitioned_plan_reads_the_channel_wide_overlap` pins both halves:
that the override gives the unpartitioned answer, and that the restricted PTM really does mislead
without it.

Predicted against measured, the latter from the engine's instrumented gather runs:

| channel | rest streams | predicted `rest_rows_per_key` | measured `rows / distinct` | kernel |
|---|---:|---:|---:|---|
| `cnot` | 3 | **1.00** | 1.00 | radix (new) |
| `cz` | 3 | **1.00** | — | radix (new) |
| `swap` | 3 | **1.00** | — | radix (new) |
| `gu2q` = `sqrt(SWAP)` | 3 | **3.00** | 3.00 | comparison |
| `su4` = Haar SU(4) | 15 | **14.00** | 14.00 | radix (arm 1) |
| `rotation_zz`, `h`, `s` | 1 | 1.00 | 1.00 | comparison |

Exact on all three measured channels. The gate:

```rust
radix = rest_streams >= RADIX_MIN_REST_STREAMS                 // 8   — many comparisons per row
     || (rest_streams >= RADIX_MIN_DISJOINT_STREAMS            // 3   — ordinary comparison count,
         && rest_rows_per_key(ptm, rest_start)                 //       but every comparison misses
            < RADIX_MAX_REST_ROWS_PER_KEY)                     // 2.0
```

Two arms because there are two mechanisms; each threshold sits in the middle of a gap the built-ins
leave empty (`3 → 15` streams, `1.00 → 3.00` rows per key), and neither is interpolated into.
`RADIX_MIN_DISJOINT_STREAMS = 3` rather than 2 is the one conservative choice: no built-in realizes
exactly two rest deltas, so the `k = 2` case is unmeasurable here, and `log2(2)+1 = 2` comparisons
per row puts the estimate on the comparison kernel's side.

**The second arm fires only where the two kernels are bitwise identical.** `rest_rows_per_key ==
1.0` says the rest stream has no duplicate keys at all, and equal-key order is the *only* thing the
two kernels disagree about (`merge.rs` module doc). So `layer_fingerprints_are_stable` — whose
channel set includes `clifford2q_cnot` and `clifford2q_swap`, both of which change kernel here —
passes unchanged, as do the thread-count and bucket-count byte-identity tests. No literal was
regenerated and nothing was demoted; that is a *consequence* of which channels the arm selects, not
a constraint anyone imposed on it.

## 4. Alternatives considered

- **Sampling a few hundred adjacent pairs per layer.** Cheaper than a maintained histogram and
  accurate enough — for the wrong quantity. §1 says any estimate of presortedness, however good,
  returns `k` for both `cnot` and `gu2q`. Rejected on the derivation, not on cost.
- **A maintained per-sum hdb histogram** (`2·64·W` counters, rebuilt at `rebucket`/`refine`). Same
  objection, plus it is the only candidate that would have added state to `PauliSum` and therefore
  touched `refine`/`coarsen` and the MPI wire format. Rejected twice over. The histogram was
  nonetheless *built* offline for §1.2's table — that is how the "zero mass at 62/63/190/191" row
  was established, and it is the evidence that killed the whole family.
- **A static property of the delta masks alone.** This is the cheapest thing that could have
  worked, and §1.1 shows it is the right shape for presortedness: the masks of any two-qubit gate
  sit far below the hdb band, so the answer is "never inverts" for every channel and every mask.
  True, free, and useless as a discriminator. What shipped *is* a static property of the PTM — the
  amplitude support rather than the masks.
- **Counting runs during the gather and choosing the kernel after.** Would measure the constant of
  §1, at the cost of a compare-and-branch in a loop from which this week removed instructions for
  −4 to −11% each (`4ee8033`, `2b949e6`). Rejected on both counts.
- **A fitted two-term cost model** — predicted `ns/row = (log2(k)+1) · f(rows_per_key)` against the
  radix's flat cost — fits all four cells and would pick the same kernels. Rejected as
  over-specification: it bakes two host-specific curves into a decision with four known inputs, and
  `benchmarks/PROFILING.md`'s discipline is to threshold on the measured gap, not to interpolate
  across it.

## 5. Gates and results

Pre-registered before measuring: `su4` stays on radix, `rotation_zz` stays on the comparison
kernel, `cnot` moves to radix, `gu2q` stays on the comparison kernel; a direction-consistent
end-to-end win on `cnot` with no consistent regression on the other three, plus `tfim_step` and
`heavyhex_step` as realistic-shape controls.

Kernel selection, by construction and pinned by
`bucketed::tests::radix_sort_kernel_is_selected_only_for_dense_ptms`: `su4` radix (arm 1),
`rotation_zz` comparison (1 stream, below both arms), `cnot` radix (arm 2), `gu2q` comparison
(3.00 rows per key), `tfim_step` / `heavyhex_step` comparison (1 stream). All four gates met at the
plan level; the sort kernel of every layer but `cnot`, `cz` and `swap` is the one it had.

`a_r8` → `c_new`, 7 pairs each:

| layer | binary | kernel | wall Δ% | sort Δ% |
|---|---|---|---:|---:|
| **`cnot`** | `d_final` | comparison → **radix** | **−5.26 (14/14)** | **−21.67 (14/14)** |
| `gu2q` | `d_final` | comparison (unchanged) | −0.08 (8/14 neg) — ns | +0.03 — ns |
| `cnot` | `c_new` | " | −5.30 (14/14) | −22.20 (14/14) |
| `gu2q` | `c_new` | " | +0.10 (5/14 neg) — ns | +0.23 — ns |
| `rotation_zz` | `c_new` | comparison (unchanged) | +0.32 (5/14 neg) — ns | +0.55 — ns |
| `su4` | `c_new` | radix (unchanged) | +0.10 (6/14 neg) — ns | +0.08 — ns |
| `tfim_step` | `c_new` | comparison (unchanged) | −0.42 (13/14 neg) — ns | +0.88 (14/14) |
| `heavyhex_step` | `c_new` | comparison (unchanged) | −0.07 (8/14 neg) — ns | +2.34 (14/14) |

`cnot`'s `gather_ns` and `merge_ns` are clean nulls on both binaries, so the
−5.3% is the sort phase and nothing else; the two independent `cnot` campaigns
agree to 0.04 points of wall and 0.5 of sort. The five control
layers are also the **layout control** for this edit: `bucketed.rs` and
`merge.rs` are the modules whose code layout this repo used to have to argue
about, and on the padded build every untouched layer lands inside ±0.5% with
inconsistent sign.

The two kicked-Ising cells are the ones worth reading twice, because both show a
direction-consistent *phase* move under a flat total. `tfim_step`: `sort_ns`
**+0.88% (14/14)** against `merge_ns` **−0.81% (14/14)**, wall −0.42% and
sign-inconsistent — a redistribution between two adjacent timers.
`heavyhex_step`: `sort_ns` **+2.34% (14/14)** with wall −0.07% at 8/14 negative
and `gather_ns` flat. Neither layer's kernel changed, so both are code layout on
an untouched path, and both sort phases are small enough relative to their layer
that a 1–2% move on them does not reach the total. This is the `4e1eea2` lesson
applied: a consistent phase delta is not an effect when the total is flat.

The forced-radix reference measured separately on the same two binaries (`a_r8` → `b_r2`, i.e. the
recalibration note's sweep re-taken): `cnot` **−5.66% wall / −22.58% sort (14/14)**, `gu2q`
**+12.43% wall / +33.65% sort (14/14)**. Both reproduce the recalibration note's −5.48 / +13.20 to
within the noise of two protocols, which is the third independent confirmation of each.

## 6. Negatives, in full

- **Presortedness as a predictor: dead.** Not weak — constant. `runs = k` exactly on every built-in
  `Local` layer, and `cmp/row` at the merge floor on all of them. Any sampled, maintained, or
  gather-time estimate of it returns the same number for `cnot` and `gu2q`. This retires
  `2026-09-10-constant-recalibration.md` §6.1 as stated.
- **The coset-coordinate story in that §6.1 is also wrong.** It proposed that `cnot`'s rest deltas
  share a coset coordinate and therefore interleave row by row under the input-major gather. They
  do not: at `r = 2` three distinct nonzero bucket deltas span the whole 2-dimensional coset space,
  so `cnot`'s coords are `{1, 2, 3}` and `gu2q`'s are `{1, 2, 3}` — each delta owning a coordinate
  in *both* channels. The two plans are structurally identical down to the coordinate multiset.
- **`RADIX_MIN_REST_STREAMS` is still 8.** Nothing here re-opens it: the second arm is additive,
  arm 1's decision set is unchanged, and the recalibration note's sweep over its four distinct
  settings stands as taken.
- **`RADIX_MIN_DISJOINT_STREAMS = 2` is untested**, because no built-in realizes two rest deltas.
  Set to 3 and documented as such.
- **`cz` and `swap` move kernel on the mechanism, not on a measurement.** Both are key permutations
  with 3 rest streams and `rest_rows_per_key == 1.00`, i.e. `cnot`'s cell in every respect the gate
  can see, and neither is a probe layer. Measuring them needs a probe layer that does not exist;
  the alternative was to special-case `cnot`, which is not a predictor.

## 7. What to do next

1. **Multi-thread the second arm.** Everything here is one thread, as §3 of the recalibration note
   was. The radix kernel's scratch grows 16 B/row and its win shrinks toward the write ceiling
   (`2026-09-01-sort-kernel.md` §6 risk 2), and `cnot`'s runs are 734 rows against `su4`'s 7212, so
   the arm most likely to survive scaling is the new one — worth confirming rather than assuming.
2. **A `cz` or `swap` probe layer.** Two channels now change kernel with no cell to measure them
   in. A Clifford2Q layer parameterized by gate would close that and cost nothing else.
3. **The remaining `Vec::push` sites in the rotation gather** — unchanged from the recalibration
   note's §6.2, and still the largest untaken item in this module (`rotation_zz`'s gather is 45% of
   its layer).
4. **Do not re-attempt a presortedness predictor** without first re-running §1.2's histogram. The
   overlap between a two-qubit gate's mask bits and a bucket's hdb distribution is zero on a dense
   random sum, and the one regime where it is not — coefficient-truncated low-weight sums, where
   the discrimination falls into `z` and a `ZZ` mask reaches it — has a single rest stream, which
   is below both arms of the gate. If a many-stream channel ever meets a low-weight sum, that is
   the cell where the quantity would finally carry signal.

## 8. Reproduction

```bash
# The shipped change, against its parent.
env -u RUSTFLAGS scripts/ab-compare.sh radix-arm2 --a <parent> --b <this commit> \
  --pairs 7 --order abba --features phase-timing \
  --probe '--n 1000000 --threads 1 --reps 20 --layers cnot --qubits 128'

# The forced-radix reference: build RADIX_MIN_REST_STREAMS = 2 into its own
# CARGO_TARGET_DIR, copy the binary out, alternate abba against an unmodified
# build. Cells as in §0.

# §1–§2's structural numbers need the scratch instrumentation described in §0:
# an ascending-run count, a counting-comparator sort and a distinct-key count
# over `run.x` / `run.z` immediately before the kernel call in `fill_coset`,
# accumulated into atomics and drained per layer in `engine/mod.rs`. Behind
# `#[cfg(feature = "phase-timing")]` and an env var, and reverted after use.

# The predictor's own values need no instrumentation:
cargo test -p paulistrings --lib radix_sort_kernel -- --nocapture
```

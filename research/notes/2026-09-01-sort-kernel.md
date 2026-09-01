# The dense-PTM sort has headroom, and it is not in the comparison count

Experiment **E2** of the large-`m` campaign (`expt/sort-kernel`), scoped by
`research/notes/2026-09-01-large-m-campaign-log.md` §"Phase 2 slate" to two questions:

- **(a)** can the dense-PTM per-run sort — **58–60 % of an `su4` layer's wall time**
  (`2026-09-01-large-m-phase-breakdown.md` §1) — be beaten by a radix replacement?
- **(b)** what is the `W = 1` sort defect (§3.2's "1.9× slower per row for half the key bytes")?

**Answer to (a): yes, and the reason is worth more than the number.** E4 established that at full delta-span
rank the sort is already at `log₂ 15 + 1 = 4.9` comparisons per row — *"at full rank this sort has no headroom
left"* — and predicted a run-oblivious radix would therefore lose exactly there. It does not. The floor is a
floor on **comparison count**, and each of those comparisons is a *dependent indexed load* through the `Vec<u32>`
permutation into a 100–400 KiB key column, ~10–13 cycles. Two 8-bit radix passes over 8-byte records read
sequentially do the same job for ~2 cycles each. Measured in situ at full rank, `W = 2`, `m = 9.9 × 10⁵`:
**sort −24.9 %, layer −15.2 %, 3/3 paired**. The headroom is in cost per comparison, not in the number of them.

**Answer to (b): E4 is right that the 1.9× is mostly a rank effect, and the ≈1.35× residual it isolates is a
codegen pathology in `<[u64; 1] as Ord>::cmp`.** With *identical key content* — a `[u64; 2]` column whose word 1
is all zeros, i.e. a 64-qubit key padded to `W = 2` — and an x-only comparator, the same comparison sequence
costs **16.3 ns/row through `[u64; 1]` columns and 7.5 ns/row through `[u64; 2]`**. LLVM compiles the one-word
comparator into driftsort's fully branchless cmov merge (22 instructions, loop-carried dependency straight
through the compare) and the two-word comparator into a branchy variant (39 instructions) whose predicted branch
lets the next iteration's loads issue early. **The radix kernel removes the residual as a side effect** — it has
no key comparator on the hot path — and in situ the `W = 1` sort lands at 11.10 ns/row against `W = 2`'s
12.11, i.e. the width penalty inverts from 1.35× to 0.92×.

**Landed on this branch, not proposed:** the kernel, gated to dense two-qubit PTMs only. It is a merge candidate
pending the campaign's serialized `ab-compare` pass (§5). No default constant, no `#[inline]` attribute and no
sparse-PTM code path was changed.

---

## 0. Provenance and what these numbers are worth

Host ccqlin038 (reference host), 2026-09-01, rustc 1.94.0, branch `expt/sort-kernel` off campaign tip
`f592c43`. **The box was not quiet**: sibling E-agents were building and testing throughout (load average
0.1–4.4 over the session). Per `benchmarks/PROFILING.md` single-shot noise here is ±5–8 % single-threaded and
±10–26 % at 8+ threads.

Every timing in this note is therefore **NON-AUTHORITATIVE** and labelled. They are reported the way E4 reports
its own: interleaved A/B adjacent in time, 3 pairs per cell, acceptance by **direction consistency across every
pair** with the median Δ% as effect size. They exist to say which way the kernel points at effect sizes far
above the noise floor, not to certify a number. §5 specifies the gate that would.

Two of the four structural claims below rest on nothing timed at all: comparison counts (§1.2) and the
disassembly (§2.2) are exact and reproduce anywhere.

---

## 1. The `W = 1` diagnosis, and how it composes with E4's

### 1.1 The two findings are on different axes and both hold

E4 (`2026-09-01-bucket-cliff.md`) decomposed the fact sheet's 1.9× as: a **hash delta-span rank** effect worth
−29.6 % of the sort (rank 3 at the default seed for `su4`'s support `(0, 1)` at `W = 1`, rank 4 at `W = 2`),
leaving a **≈1.35× genuine width residual** with byte-identical comparison counts at 4.93/row. E2 measured only
the residual, because its synthetic generator *forces* full rank — it reseeds the GF(2) hash until the 16 deltas
map to 16 distinct coset coordinates — and independently landed on the same 1.35–1.65× with the same 4.93
comparisons/row. So:

- E4's *"ns per comparison is flat to ±15 %"* is across the **rank/bits** axis at fixed `W`. True.
- E2's 2.2× is across the **`W`** axis at fixed key content and a fixed comparison sequence. Also true.

Nothing to reconcile: rank sets *how many* comparisons, width sets *what each one costs*.

### 1.2 Localizing the residual

A standalone harness reproducing one `su4` gather run — closed key set, support `(0, 1)`, all 16 local deltas,
full-rank hash, member 0's rest stream assembled the way `gather_local_output_major` assembles it — was stepped
through four controls. All figures ns per sorted row, minimum of three trials, **non-authoritative**:

| control | `W = 1` | `W = 2` | reads |
|---|---|---|---|
| full kernel (sort + 3-column gather) | 20.3 | 12.4 | the defect, reproduced in isolation |
| **permutation sort only** | **18.3** | **10.5** | all of it is the sort |
| 3-column gather only | 1.87 | 1.88 | none of it is the gather — and the gather moves *more* bytes at `W = 2` |
| sort, keys padded to `[u64; 2]` (64 qubits) | — | 10.5 | **zero data dependence**: a padded `W = 1` key behaves exactly like a real `W = 2` one |

and then the storage type varied at fixed content and fixed comparison sequence (`x`-only comparator, so the
comparison count is identical by construction):

| key column storage | ns/row |
|---|---|
| `Vec<[u64; 1]>` | 16.3 |
| `Vec<u64>` (scalar) | 16.6 |
| `Vec<Pad16<[u64;1]>>` (16-byte stride, 1-word compare) | 24.8 |
| **`Vec<[u64; 2]>` (16-byte stride, 2-word compare)** | **7.5** |

The 16-byte-stride row is the control that kills the obvious explanations: it is *slower*, so this is neither
element stride nor cache footprint (the `W = 1` column is half the bytes and loses). What tracks the time is the
number of *words the comparator compares*. Comparison counts across the whole grid: identical, 4.93/row.

### 1.3 Mechanism: LLVM picks a different merge shape for a one-word key

Disassembling driftsort's inlined merge loop for the two monomorphizations (`objdump` of a minimal
`perm.sort_by(|&i,&j| x[i].cmp(&x[j]))` at `W = 1` and `W = 2`):

- **`[u64; 1]` — 22 instructions, fully branchless.** `cmp` → `setb`/`setae` → two `lea`s advance both cursors
  by the flag. The loop-carried chain is `cursor → load index → load key → cmp → setcc → lea → cursor`:
  every iteration's address depends on the previous iteration's comparison result. At ~13 cycles that predicts
  `4.93 × 13 = 64` cycles/row = **17.8 ns/row at 3.6 GHz** against 18.3 measured.
- **`[u64; 2]` — 39 instructions, branchy.** A `jne` on "word 0 differs" (taken with probability
  `1 − 2⁻⁶⁴` except on exact duplicates, so ~perfectly predicted) splits out the equal-word-0 path. More
  instructions, more work per compare, but the predicted branch lets the front end run ahead of the
  serialized flag computation.

So the `W = 1` comparator is *penalized for being simple enough to make branchless*. That is a codegen accident,
not something the source expresses, which is why §1.2's hand-unrolled comparator (explicit early-return branches)
changed nothing: at one word LLVM canonicalizes it straight back to the select chain.

**Consequence for the fix.** Any repair that keeps a key comparator on the hot path is fighting the optimizer for
a 1.35× that a second, dead word already buys. The radix kernel does not have one.

### 1.4 What was *not* done about it

Padding `W = 1` keys to two words, or hand-writing a branchy comparator with an opaque second word
(`black_box`), would both plausibly recover the residual for *every* channel including the sparse path where
the sort is 7 % of the layer. Neither was attempted: both are optimizer-shape bets that LTO can revoke, and both
touch the comparison kernel that `rotation_zz`/`cnot` depend on. §6 lists it as the one follow-up worth a
measurement.

---

## 2. The kernel

`merge::sort_rows_radix_with_scratch`. Same contract as the comparison kernel and nothing more — ascending in
lex `(x, z)`, duplicates allowed, a permutation of the input triples — so the two are interchangeable to
floating-point tolerance (ARCHITECTURE.md §Determinism).

### 2.1 An order-faithful surrogate, found in one pass

The full key is `16·W` bytes, so an LSD radix over all of it is 32 passes at `W = 2` — a non-starter. Instead one
`OR`/`AND` reduction finds the **most significant key word `k` the rows actually disagree on** and the highest
disagreeing bit `hb` inside it, stopping at the first such word (word 0 for the runs this kernel serves, so the
scan touches only the `x` column). Every row then shares the same value on words before `k` and on bits above
`hb`, so writing `word_k = H·2^(hb+1) + L` with `H` **constant across the run**:

```
surrogate = (word_k >> shift) & (2^16 − 1) = L >> shift,      shift = (hb + 1) − 16
```

The mask erases exactly the constant `H`, and `L >> shift` is monotone in `L`, which is monotone in the key.
Hence `key₁ < key₂ ⟹ surrogate₁ ≤ surrogate₂`: sorting by the surrogate can never place two rows in the wrong
order, only leave them tied. `shift` saturates at 0, which is why narrow key spaces and low qubit counts are
correct rather than merely lucky.

Three properties fall out for free:

- **A run whose keys are all equal** has no disagreeing word, so the kernel returns immediately — it is already
  ascending. The comparison kernel sorts it.
- **A run whose window carries fewer than 8 discriminating bits** (a `WeightCutoff`-truncated sum whose `x`
  words are nearly constant; a two-bit key space) cannot be usefully separated, so the kernel *calls the
  comparison kernel* and costs one extra reduction.
- **Low qubit counts are handled by construction, not by a special case.** `q = 36` — the width of the committed
  cross-engine su4 curve — puts every live bit in `x[0]`'s low 36, where a fixed "top 32 bits of `x[0]`"
  surrogate would have discriminated *nothing*. The adaptive window is not a refinement; it is what makes the
  design correct below 64 qubits.

### 2.2 Why 16 bits in two 8-bit digits

The surrogate is not sized to resolve every key — it is sized to resolve *groups*. A dense-PTM run of `n` rows
holds about `n/15` distinct keys, so 65536 surrogate values leave the residual tie groups at the duplicate
groups themselves, which the fixup pass then orders with roughly one full-key comparison per row. Swept
(non-authoritative, su4-shaped runs, Δ vs the comparison kernel):

Δ is on the whole kernel (sort + 3-column gather) against the comparison kernel, over rows/run 900–18 800:

| surrogate / digit | passes | `W = 1` | `W = 2` |
|---|---|---|---|
| 11 / 11 | 1 | −39…−50 % (best ≤ 8 k rows, **decays above**) | −45…**+10 %** |
| **16 / 8** | **2** | **−36…−51 %** | **−10…−46 %** |
| 22 / 11 | 2 | **+15**…−41 % | −11…+3 % |
| 24 / 8 | 3 | −22…−39 % | −37…−2 % |
| 32 / 8 | 4 | −17…−32 % | −28…**+11 %** |

`16/8` is the only row favourable in every cell at both widths. Everything wider than 16 bits pays a whole extra
pass for ties it does not have, and the single 11-bit pass runs out of resolution above ~8 k rows — where the
engine actually lives (bucket target 1024 × fanout ~15 ≈ 15 k rows/run). A size-adaptive "11 bits below 8 k
rows, else 16" was measured and is not better than plain `16/8` there; left as a follow-up, not landed.

### 2.3 The gate, and why it is a gate

`RADIX_MIN_REST_STREAMS = 8`, tested against the plan's *realized* rest-delta count once per layer — so the
choice costs nothing per run and every sparse-PTM layer keeps byte-identical code.

| run shape | rest streams | duplicate density | radix vs comparison (microbench) |
|---|---|---|---|
| dense PTM (`su4`) | 15 | ~15× | **−36…−51 % (`W=1`), −10…−46 % (`W=2`)** |
| dense PTM, single bucket | 15 | ~15× | **−31…−38 % (`W=2`)** |
| **sparse PTM (`rotation_zz`)** | **1** | ~1× | **+133…+165 %** |

The sparse row is the whole reason for the gate: with one nearly-sorted stream the comparison sort costs about
one comparison per row and the radix's fixed passes are pure overhead. `2..8` rest streams — `sqrt(SWAP)`'s
regime, whose sort is 33 % of its layer — is **unmeasured** and deliberately left on the comparison kernel.
`bucketed::tests::radix_sort_kernel_is_selected_only_for_dense_ptms` pins which built-ins clear the gate
(today: the dense SU(4), and nothing else — `sqrt(SWAP)` is the runner-up at 3 streams), because a gate that
silently stopped firing would look like nothing worse than a lost speedup, and one that started firing on
`rotation_zz` would be a large regression.

---

## 3. Smoke timings — **NON-AUTHORITATIVE**

Protocol per §0: two prebuilt `phase_breakdown --features phase-timing` binaries (A = `f592c43`,
B = `expt/sort-kernel` tip), alternated adjacent in time, 3 pairs per cell, `RUST_LOG` unset, `--reps`
auto-scaled to ≳200 ms. `runs/coset` = `2^r` is quoted because it is what E4 showed the comparison sort's cost
tracks.

| cell | `m` | `2^r` | A ns/term | B ns/term | Δ per pair | consistent | **median Δ** | sort ns/row A → B | sort % of layer A → B |
|---|---|---|---|---|---|---|---|---|---|
| `su4` `q=128` 1t | 9 884 | 16 | 413.3 | 316.7 | −23.4 / −25.3 / −22.3 | **3/3** | **−23.4 %** | 18.61 → 11.58 (−37.8 %) | 62.3 → 50.6 % |
| `su4` `q=128` 1t | 63 518 | 16 | 372.7 | 314.0 | −15.7 / −15.6 / −15.2 | **3/3** | **−15.6 %** | 15.60 → 11.51 (−26.2 %) | 58.4 → 51.1 % |
| `su4` `q=128` 1t | 989 268 | 16 | 380.2 | 322.5 | −15.2 / −14.1 / −15.3 | **3/3** | **−15.2 %** | 16.11 → 12.11 (−24.9 %) | 58.8 → 51.7 % |
| `su4` `q=128` **8t** | 989 268 | 16 | 65.95 | 60.25 | −8.6 / −11.7 / −9.1 | **3/3** | **−9.1 %** | 21.98 → 18.13 (−17.5 %) | — |
| `su4` `q=64` 1t | 63 364 | **8** | 574.1 | 377.7 | −34.2 / −34.3 / −35.0 | **3/3** | **−34.3 %** | 30.83 → 16.64 (−46.0 %) | 74.5 → 61.4 % |
| `su4` `q=64` 1t | 991 060 | 16 | 459.4 | 304.5 | −33.7 / −33.7 / −32.8 | **3/3** | **−33.7 %** | 22.12 → 11.10 (−49.8 %) | 67.6 → 50.9 % |
| `rotation_zz` `q=128` 1t | 149 957 | 2 | 29.49 | 27.64 | −6.3 / −6.2 / −6.7 | 3/3 | −6.3 % | 3.53 → 3.60 (+1.8 %) | 7.9 → 8.7 % |
| `rotation_zz` `q=64` 1t | 150 114 | 2 | 23.53 | 23.99 | +1.9 / +1.8 / +3.9 | 3/3 | **+1.9 %** | 3.25 → 3.25 (−0.0 %) | 9.2 → 9.0 % |
| `cnot` `q=128` 1t | 100 000 | 4 | 42.48 | 40.20 | −5.4 / −2.6 / −3.4 | 3/3 | −3.4 % | 12.72 → 12.46 (−2.0 %) | 22.5 → 23.0 % |

### What the table says

**The sort phase is where every dense-PTM change lands**, and the baseline column reproduces the fact sheet's
own numbers independently — 58.4–58.8 % of the layer at `W = 2` against its 58–60 %, 16.11 ns/sorted row against
its 16.1, `rotation_zz` at 7.9 % against its 7 %. That agreement is what licenses reading the rest.

1. **`W = 2`, full rank: sort −25 to −26 %, layer −15 %, at both `m = 6.4 × 10⁴` and `m = 9.9 × 10⁵`.** This is
   the cell E4's floor argument predicted a radix would lose, and the microbench had put at only −7…−16 % on the
   sort. In situ it is −25 %, because the engine's key columns are colder than the microbench's and the
   dependent indexed load costs more, not less, than the harness suggested.
2. **`W = 1`: sort −46 to −50 %, layer −34 %, in *both* rank regimes.** At `m = 6.3 × 10⁴` the default seed puts
   this support at `r = 3` (`2^r = 8`, E4 §1.3) and the baseline sort is 30.83 ns/row; at `m = 9.9 × 10⁵` the
   bucket count lifts it to `r = 4` and the baseline is 22.12 — E4's measured 22.14/22.12/22.28 at that
   configuration, reproduced. The radix lands at 16.64 and 11.10 respectively: **it is favourable in both, and
   by a similar margin, because it is order-oblivious.** It does not repair the rank draw; it removes the
   sensitivity to it.
3. **The width residual is gone.** `W = 1` 11.10 ns/row against `W = 2` 12.11 at the same rank and comparable
   rows/run: 0.92×, against the comparison kernel's 1.37× (22.12 / 16.11) — itself E4's ≈1.35×, reproduced a
   third time. A kernel with no key comparator on the hot path cannot have a comparator pathology.
4. **The mid-`m` cell moves most (−23.4 % at `m = 9884`), without a policy change.** E4 §1.5 explains why the
   baseline is bad there — 15 runs of only ~62 rows, so driftsort loses its adaptivity even at full rank
   (9.72 comparisons/row, not 4.9) — and a fixed-cost radix is indifferent to run length. This is the overlap
   with E4's cliff, reached from the other side; see §6.
5. **8 threads: −9.1 %, consistent, and smaller than 1t.** Expected: the fact sheet has this layer at 63 % of
   the measured write ceiling at 8 threads already, so a compute-side win is partly absorbed. The direction is
   what matters here; a 16-thread number would be a memory-controller measurement (fact sheet §7 item 7) and was
   not taken.
6. **The sparse controls behave exactly like code layout, which is the honest reading.** Their code path is
   byte-identical — the gate cannot fire — and the sort phase confirms it: **+1.8 %, −0.0 %, −2.0 %**. Yet the
   *layers* move −6.3 %, **+1.9 %** and −3.4 %, each 3/3 consistent within its own cell but **disagreeing in
   sign across cells**. That is the signature of LTO placement shifting around ~200 lines of new code in
   `engine/merge.rs`, not of anything the kernel does. It is also the risk §6 item 1 names, and the reason §5
   treats a consistent sparse regression as a module-layout problem rather than a verdict on the kernel.

---

## 4. Tests

- `merge::tests::assert_sort_contract` — one harness over a kernel `fn` pointer, so both kernels are held to
  *exactly* the same bar, parameterized over `W ∈ {1, 2}` and over `num_qubits ∈ {3, 8, 17, 33, 64, 65, 128}`.
  Fixtures: the shapes a real run takes (dense-random, already/reverse sorted, 15 duplicate streams) plus the
  ones a bit-inspecting kernel can trip over (all keys equal, `x` identically zero, a two-bit key space, a
  constant nonzero high bit), plus degenerate lengths 0/1/2.
- `merge::tests::sort_rows_radix_contract_proptest` — random rows × narrow key spaces × a `spread` shift that
  slides the varying bits up and down word 0, exercising every window shift including the saturating one.
- `merge::tests::the_two_sort_kernels_reduce_to_the_same_sum` — the interchangeability claim `bucketed.rs`
  relies on: both kernels feed `merge2_into` and produce the same reduced sum exactly (the fixtures'
  coefficients are small integers, so any summation order is exact).
- `merge::tests::radix_sort_scratch_capacity_stabilizes` — the steady state still allocates nothing.
- `bucketed::tests::differential_against_the_naive_oracle_w1_dense_collisions` / `…_w2_sparse` gained a
  `haar_su4` cell (the same matrix `examples/phase_breakdown.rs` uses, all 16 deltas realized), at `W = 1` and at
  `W = 2` with the support straddling the word boundary. **The net had no dense-PTM cell before this**:
  `general_2q` is `sqrt(SWAP)`, whose PTM is sparse — so nothing in it exercised the run shape the whole
  experiment is about.
- `cargo test --workspace` and `cargo test --workspace --release` green; clippy clean.
- **No fingerprint or byte-identity tripwire moved**, and that is a fact about coverage rather than a
  reassurance: no channel in `layer_fingerprints_are_stable` clears the gate, so the radix path is covered by
  the tolerance-based differential net and not by the bitwise one. That is the right net for it (the two kernels
  agree to tolerance by design, not bitwise), but it means the fingerprints are silent here, not confirming.

---

## 5. Proposed gate

`scripts/ab-compare.sh --a f592c43 --b expt/sort-kernel`, dense-PTM cells at **≤ 8 threads** per the fact
sheet's write-ceiling finding (§7 item 7: at 16+ threads the dense path measures the memory controller).
`su4`'s closure factor is ~14.2×, so `--n` is the pre-closure seed and `m ≈ 14.2 × --n`.

```bash
# --- the win: dense PTM, m near 1e4 / 1e5 / 1e6, one thread
--n 700    --qubits 128 --layers su4 --reps 2000   # m ~ 1.0e4
--n 7000   --qubits 128 --layers su4 --reps 200    # m ~ 1.0e5
--n 70000  --qubits 128 --layers su4 --reps 40     # m ~ 1.0e6
# --- the same three at 8 threads (never 16 or 32 for this layer)
--n 70000  --qubits 128 --layers su4 --reps 40 --threads 8
# --- W = 1: the width residual, and (at the default seed, per E4) the r=3 regime
--n 70000  --qubits 64  --layers su4 --reps 40     # m ~ 1.0e6, W = 1
--n 4480   --qubits 64  --layers su4 --reps 400    # m ~ 6.3e4, W = 1
# --- controls: the sparse path must not move at all (identical code path)
--n 100000 --qubits 128 --layers rotation_zz --reps 40
--n 100000 --qubits 64  --layers rotation_zz --reps 40
--n 100000 --qubits 128 --layers cnot        --reps 40
```

Acceptance:

1. **Dense-PTM cells**: direction-consistent across all pairs, median Δ ≤ −10 % at `W = 2` and ≤ −25 % at
   `W = 1`. Below that, treat as null and keep the branch as a negative result.
2. **Sparse-PTM controls**: the code path is byte-identical, so any consistent movement is an LTO layout
   artefact of adding a second kernel to the module. Read a consistent regression > 5 % as a reason to move
   `sort_rows_radix_with_scratch` into its own module (or to A/B its `#[inline]`), not as a reason to drop it.
3. If E4's bucket-policy change lands too, **re-run cell 1 after it**: both changes attack the mid-`m` cliff by
   different routes (§6) and their effects are not additive.

E4's `phase_breakdown --hash-seed` / `--bucket-bits` knobs (on `expt/bucket-cliff`) would sharpen this: pinning
`--bucket-bits 7` separates the kernel's effect from the bucket-count policy's, and `--hash-seed` separates it
from the rank draw. Worth taking that branch's probe first if the two are gated in sequence.

---

## 6. Overlap with E4, risks, and what was deliberately not done

**Overlap, flagged for the merge order.** The radix kernel is a *second, independent* attack on E4's mid-`m`
cliff, and the two are not additive:

- E4's route is policy — give a small dense-PTM sum more buckets, so `r` reaches 4 and the gather blocks arrive
  ascending, restoring the 4.9-comparison floor. Worth −38.6 % at `m = 980`, but it costs **+46.6 %** on a
  rotation layer at `m = 1497`, so it needs a fanout-aware constant that E4 explicitly did not choose.
- E2's route is the kernel — make the comparisons cheap enough that presortedness stops mattering. The radix is
  order-oblivious, so it removes the cliff's *sensitivity* rather than its cause: microbench −31…−38 % on the
  single-bucket corner, in situ −23.4 % on the layer at `m = 9884`, both without touching a default constant or
  the rotation path.

They should be gated in sequence, not together, and cell 1 of §5 re-run after whichever lands first.

**Merge conflicts to expect.** E4 added `test_support::haar_su4_matrix`; E2 added an equivalent
`bucketed::tests::haar_su4`. Whichever lands second should delete its copy and use the shared fixture.

**Risks.**

1. **The `#[inline]` set.** Untouched — `sort_rows_with_scratch` keeps its hint, `merge2_into` keeps its
   absence, and the new function carries no attribute at all, recorded in its doc as its own A/B. But
   `engine/merge.rs` grew by ~200 lines of new code in a module whose layout is A/B-verified load-bearing in
   both directions (±6 %, +20–34 %). **The sparse-PTM controls in §5 exist to catch exactly that**, and a
   consistent sparse regression is a layout problem with a known remedy (separate module), not a reason to
   reject the kernel.
2. **Scratch growth.** `SortScratch` gains `packed` + `aux`, 16 bytes per row of the largest run seen —
   ~245 KiB per worker at a 15 k-row run, against a run's own 48 B/row ≈ 737 KiB. The fact sheet already puts
   the dense path at 100 % of the measured write ceiling at 16 threads, so this is a real L2/L3 pressure
   increase at high thread counts even though the 8-thread smoke cell was favourable (−9.1 %). It is the reason
   §5 caps the dense cells at 8 threads and the reason a 16-thread cell should quote bandwidth alongside.
3. **`2..8` rest streams is unmeasured.** `sqrt(SWAP)`'s sort is 33 % of its layer and is left entirely alone.
   That is a deliberate omission, not an oversight: the crossover between the two kernels is steep and nothing
   in this experiment locates it.

**Deliberately not done.**

- **No unconditional replacement.** The radix costs +130…+165 % on a sparse-PTM run; a single kernel for both
  regimes is not available.
- **No `std::simd`** (nightly, toolchain pinned 1.94.0) and **no stable intrinsics** — no `_pext_u64` for a
  compacted multi-word surrogate. The portable contiguous window is enough for every measured cell, and PEXT
  would only help runs whose discriminating bits are scattered across words, which none of the dense-PTM shapes
  are.
- **No fusing of the window reduction into the gather.** `GatherRun` could accumulate the `OR`/`AND` as rows are
  pushed and hand the kernel its window for free, saving one pass over the `x` column (~1 ns/row at `W = 2`).
  Rejected for now: it puts work in the gather loop — 46–51 % of the *sparse* layer, which gets nothing back for
  it — and would have to be gated on the plan to stay honest.
- **No repair of the `W = 1` comparator itself** (§1.4), which is what would carry the residual to the sparse
  path too.
- **No authoritative benchmark.** Siblings were building throughout; §5 is the gate.

## 7. Reproduction

```bash
cargo test -p paulistrings --lib engine::merge          # the contract harness, both kernels
cargo test -p paulistrings --lib radix                  # the gate's channel pinning
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown
./target/release/examples/phase_breakdown --n 70000 --qubits 64 --layers su4 --reps 40 --format json
```

The standalone microbench harness of §1.2 and §2.2 (the faithful gather-run generator plus the storage-type and
digit-width sweeps) was scratch tooling and is **not** checked in — §1.2's controls are reproducible from the
description, and everything load-bearing in this note is either a committed test, the disassembly, or a
`phase_breakdown` cell.

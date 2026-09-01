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
sequentially do the same job for ~2 cycles each. Gated at full rank, `W = 2`, `m = 9.9 × 10⁵`:
**sort −25.4 %, layer −15.2 %, 3/3 pairs**. The headroom is in cost per comparison, not in the number of them.

**Answer to (b): E4 is right that the 1.9× is mostly a rank effect, and the ≈1.35× residual it isolates is a
codegen pathology in `<[u64; 1] as Ord>::cmp`.** With *identical key content* — a `[u64; 2]` column whose word 1
is all zeros, i.e. a 64-qubit key padded to `W = 2` — and an x-only comparator, the same comparison sequence
costs **16.3 ns/row through `[u64; 1]` columns and 7.5 ns/row through `[u64; 2]`**. LLVM compiles the one-word
comparator into driftsort's fully branchless cmov merge (22 instructions, loop-carried dependency straight
through the compare) and the two-word comparator into a branchy variant (39 instructions) whose predicted branch
lets the next iteration's loads issue early. **The radix kernel removes the residual as a side effect** — it has
no key comparator on the hot path — and at the gate the `W = 1` sort lands at 11.11 ns/row against `W = 2`'s
12.07 at matched `m` and rank, i.e. the width penalty inverts from 1.36× to 0.92×.

**GATE: PASS**, every cell, 3/3 pairs (§3). Dense-PTM layers −10.5 % to −33.8 %; the sort phase −17 % to −50 %;
sparse-PTM controls unperturbed (§3.2). **Landed on this branch, not proposed:** the kernel, gated to dense
two-qubit PTMs only. No default constant, no `#[inline]` attribute and no sparse-PTM code path was changed.

---

## 0. Provenance, and which numbers here are authoritative

Host ccqlin038 (reference host), 2026-09-01, rustc 1.94.0, powersave governor, branch `expt/sort-kernel`
(`968598c`) against campaign tip `f592c43`. Per `benchmarks/PROFILING.md` single-shot noise here is ±5–8 %
single-threaded and ±10–26 % at 8+ threads, which is why everything below is paired rather than compared as
campaign means.

Two tiers, and they are labelled everywhere:

- **§3 is AUTHORITATIVE.** Run on an exclusively-held box (E1 and E3 gates finished, E4 queued behind),
  `scripts/ab-compare.sh` with two prebuilt binaries alternated adjacent in time, **3 pairs per cell,
  `--order abba`** so a monotone drift in machine state cannot masquerade as a consistent B-is-faster signal,
  `RUST_LOG` unset. Acceptance is **direction consistency across every pair**, median Δ% as effect size.
  Raw artefacts in the gitignored `benchmarks/results/2026-09-01-ccqlin038/e2-*`.
- **§1.2 and §2.2 are NON-AUTHORITATIVE microbench**, taken earlier with siblings building (load 0.1–4.4).
  They chose the design; they do not certify it. Where §3 measures the same quantity, §3 wins.

Two of the load-bearing claims rest on nothing timed at all: comparison counts (§1.2) and the disassembly
(§1.3) are exact and reproduce anywhere.

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

| run shape | rest streams | duplicate density | radix vs comparison |
|---|---|---|---|
| dense PTM (`su4`), **gated in situ** | 15 | ~15× | **sort −45…−50 % (`W=1`), −17…−42 % (`W=2`)** (§3.1) |
| dense PTM, single bucket, microbench | 15 | ~15× | −31…−38 % (`W=2`) |
| **sparse PTM (`rotation_zz`), microbench** | **1** | ~1× | **+133…+165 %** |

The sparse row is the whole reason for the gate: with one nearly-sorted stream the comparison sort costs about
one comparison per row and the radix's fixed passes are pure overhead. `2..8` rest streams — `sqrt(SWAP)`'s
regime, whose sort is 33 % of its layer — is **unmeasured** and deliberately left on the comparison kernel.
`bucketed::tests::radix_sort_kernel_is_selected_only_for_dense_ptms` pins which built-ins clear the gate
(today: the dense SU(4), and nothing else — `sqrt(SWAP)` is the runner-up at 3 streams), because a gate that
silently stopped firing would look like nothing worse than a lost speedup, and one that started firing on
`rotation_zz` would be a large regression.

---

## 3. Gate results — **AUTHORITATIVE**

Protocol and provenance per §0: exclusively-held box, `scripts/ab-compare.sh --a f592c43 --b expt/sort-kernel
--pairs 3 --order abba`, seven invocations (one per `(n, qubits, reps)` triple, since those are per-invocation
while `--layers`/`--threads` are CSV). Δ% is per-term wall, `(B − A)/A`. Phase columns are worker-summed busy
time and do not sum to wall — they explain a wall effect, they are not one.

### 3.1 Dense-PTM cells — the target

| cell | `W` | thr | `m` | A ns/term | B ns/term | Δ per pair | **median Δ** | sort ns/row A → B | sort % of layer A → B |
|---|---|---|---|---|---|---|---|---|---|
| `su4` `q=128` | 2 | 1 | 9 884 | 417.9 | 318.3 | −24.1 / −23.7 / −24.0 | **−24.0 %** | 18.69 → 11.61 (−37.9 %) | 62.4 → 50.9 % |
| `su4` `q=128` | 2 | 1 | 99 386 | 379.4 | 320.5 | −15.9 / −14.5 / −15.9 | **−15.9 %** | 16.03 → 11.87 (−26.0 %) | 58.9 → 51.8 % |
| `su4` `q=128` | 2 | 1 | 989 268 | 383.2 | 325.1 | −15.2 / −14.7 / −15.8 | **−15.2 %** | 16.18 → 12.07 (−25.4 %) | 58.9 → 51.9 % |
| `su4` `q=128` | 2 | 8 | 9 884 | 85.1 | 59.6 | −36.1 / −27.1 / −30.3 | **−30.3 %** | 22.45 → 13.06 (−41.8 %) | — |
| `su4` `q=128` | 2 | 8 | 99 386 | 74.4 | 70.6 | −15.3 / −5.1 / −14.0 | **−14.0 %** | 22.09 → 18.33 (−17.0 %) | — |
| `su4` `q=128` | 2 | 8 | 989 268 | 69.4 | 60.3 | −10.5 / −13.0 / −9.3 | **−10.5 %** | 22.99 → 18.19 (−20.9 %) | — |
| `su4` `q=64` | **1** | 1 | 63 364 | 579.0 | 382.9 | −33.7 / −33.8 / −34.0 | **−33.8 %** | 30.95 → 16.87 (−45.5 %) | 74.5 → 61.4 % |
| `su4` `q=64` | **1** | 1 | 991 060 | 455.3 | 304.3 | −33.0 / −33.4 / −33.2 | **−33.2 %** | 22.08 → 11.11 (−49.7 %) | 67.6 → 51.0 % |

**8/8 direction-consistent, 3/3 pairs each.** Against the acceptance thresholds (median ≤ −10 % at `W = 2`,
≤ −25 % at `W = 1`): every `W = 2` cell clears, the tightest by 0.5 points (8 threads, `m` = 9.9 × 10⁵, −10.5 %);
both `W = 1` cells clear by ~9 points. **PASS.**

Five things the table settles:

1. **The baseline column reproduces the Phase-1 fact sheet independently** — 58.9 % sort share at `W = 2`
   against its 58–60 %, 16.18 ns/sorted row against its 16.1, 22.08 ns/row at `W = 1` against E4's
   22.12–22.28 at `r = 4`. That agreement is what licenses reading the rest as a measurement of the change.
2. **The `W = 2` win is bigger in situ than the microbench predicted** (−25 % on the sort against a predicted
   −10…−16 %), because the engine's key columns are colder than the harness's, so the dependent indexed load
   the radix removes costs *more* in the real layer, not less.
3. **`W = 1` is where the kernel pays most: −33 % on the layer, −45…−50 % on the sort, in both rank regimes**
   (`m` = 63 364 is `r = 3` at the default seed, `m` = 991 060 is `r = 4`; E4 §1.3). Nearly identical Δ in the
   two — the kernel is order-oblivious, so it does not repair a deficient rank draw, it removes the sort's
   sensitivity to one.
4. **The width residual is gone.** `W = 1` 11.11 ns/row against `W = 2` 12.07 at matched `m` and rank: **0.92×**,
   where the comparison kernel is 22.08 / 16.18 = **1.36×** — E4's ≈1.35×, reproduced a third time and then
   removed. A kernel with no key comparator on the hot path cannot have a comparator pathology.
5. **8 threads is favourable but smaller, as predicted.** The fact sheet has this layer at 63 % of the measured
   write ceiling at 8 threads, so a compute-side win is partly absorbed; the sort phase still moves −17…−42 %.
   No 16- or 32-thread cell was run: at 16+ the dense path measures the memory controller (fact sheet §7 item 7).

### 3.2 Sparse-PTM controls — layout, not perturbation

The gate's controls run channels that **cannot** clear the radix gate, so their code path is byte-identical
between the two binaries. The discriminator (orchestrator's calibration from the E1 and E3 gates): a
direction-consistent ±4–7 % wall move with **bit-identical work counts**, concentrated in the motion-sensitive
merge kernel, is LTO code layout rather than semantics.

| control | `m` | Δ per pair | median Δ | sort Δ | gather Δ | **merge Δ** | work counts |
|---|---|---|---|---|---|---|---|
| `rotation_zz` `q=128` | 149 957 | −5.9 / −7.3 / −7.8 | **−7.3 %** | +1.0 % | −1.0 % | **−15.0 %** | identical |
| `cnot` `q=128` | 100 000 | −4.0 / −4.1 / −3.7 | **−4.0 %** | −2.2 % | +0.2 % | **−14.0 %** | identical |
| `rotation_zz` `q=64` | 150 114 | +1.5 / +2.2 / +2.6 | **+2.2 %** | −0.6 % | −1.6 % | **+7.2 %** | identical |

Every criterion of the discriminator is met, and the reading is unambiguous:

- **Work counts are bit-identical** — `terms_in`, `terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`,
  `cosets`, `runs`, every pair, every cell. Nothing semantic changed, which is what the byte-identical source
  path already implied and this confirms empirically.
- **The sort phase — the only phase whose source gained a neighbour — does not move: +1.0 %, −2.2 %, −0.6 %.**
- **The delta is concentrated in `merge2_into`, whose source is byte-identical between the two sides**
  (−15.0 %, −14.0 %, +7.2 %). A ±15 % swing in a function nobody edited, at identical work, is placement.
- **The three cells disagree in sign** (−7.3 %, −4.0 %, **+2.2 %**). A real perturbation from adding a
  never-taken branch would not flip sign between two `rotation_zz` cells that differ only in width.

**No module move.** Per the calibration, surgery is warranted only for a delta clearly outside the ±4–7 % band
or with changed work counts; neither holds. The largest magnitude here (−7.3 %) is a *speedup* on an untouched
path — moving `sort_rows_radix_with_scratch` into its own module to undo an accidental win would be
superstition — and the only regression, +2.2 %, sits comfortably inside the band. `engine/merge.rs` keeps its
current shape and its current `#[inline]` set.

### 3.3 A correctness result the gate gives for free

`terms_out` is bit-identical between the two sides on **every** cell, including the dense-PTM ones the radix
kernel actually serves — 40 layers over 9.9 × 10⁵ terms, and 2000 layers at `m` = 9884. The two kernels are only
required to agree to floating-point tolerance (equal-key summation order is unspecified), so an identical
*surviving term count* at that scale is a differential check far wider than the unit tests reach: no key was
gained, lost, or spuriously cancelled by the reordering.

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

## 5. The gate, exactly as run

Seven `ab-compare.sh` invocations, `--a f592c43 --b expt/sort-kernel --pairs 3 --order abba`, dense-PTM cells at
**≤ 8 threads** per the fact sheet's write-ceiling finding (§7 item 7: at 16+ threads the dense path measures
the memory controller). `su4`'s closure factor is ~14.2×, so `--n` is the pre-closure seed and `m ≈ 14.2 × --n`.
`--reps` is carried over from the smoke run, raised on the controls so their timed region is ~2 s rather than
~0.2 s — the controls are where the layout-vs-perturbation discrimination happens and they need the resolution.

```bash
A=f592c43; B=expt/sort-kernel; unset RUST_LOG
ab() { scripts/ab-compare.sh "$1" --a $A --b $B --pairs 3 --order abba --probe "${*:2}"; }

# dense PTM, W = 2: m near 1e4 / 1e5 / 1e6, threads 1 and 8
ab e2-su4-1e4     --n 700    --qubits 128 --layers su4 --threads 1,8 --reps 2000
ab e2-su4-1e5     --n 7000   --qubits 128 --layers su4 --threads 1,8 --reps 200
ab e2-su4-1e6     --n 70000  --qubits 128 --layers su4 --threads 1,8 --reps 40
# dense PTM, W = 1: the width residual, r = 3 and r = 4 regimes
ab e2-su4-w1-1e6  --n 70000  --qubits 64  --layers su4 --threads 1   --reps 40
ab e2-su4-w1-6e4  --n 4480   --qubits 64  --layers su4 --threads 1   --reps 400
# controls: sparse PTM, byte-identical code path
ab e2-sparse-q128 --n 100000 --qubits 128 --layers rotation_zz,cnot --threads 1 --reps 400
ab e2-sparse-q64  --n 100000 --qubits 64  --layers rotation_zz      --threads 1 --reps 400
```

`--order abba` rather than the default `abab`: with the box exclusively held there was no reason not to take
the stronger protocol, and it removes the one alternative explanation a uniformly-negative result invites
(a monotone drift in machine state that happens to favour whichever side runs second).

**Verdict against the acceptance criteria stated before the run:**

| criterion | result |
|---|---|
| Dense-PTM cells direction-consistent, median Δ ≤ −10 % (`W = 2`) / ≤ −25 % (`W = 1`) | **PASS** — 8/8 cells, 3/3 pairs each; `W = 2` −10.5…−30.3 %, `W = 1` −33.2 / −33.8 % (§3.1) |
| Sparse-PTM controls: consistent regression > 5 % ⇒ move the kernel to its own module | **Not triggered** — work counts bit-identical, sort phase within ±2.2 %, deltas −7.3 / −4.0 / **+2.2 %** disagreeing in sign, concentrated in the untouched `merge2_into`. Layout, not perturbation; no module move (§3.2) |
| If E4's bucket-policy change lands, re-run the `m` = 1e4 cell after it | **Outstanding** — both attack the mid-`m` cliff by different routes and are not additive (§6) |

Not run, and worth saying so: E4's `phase_breakdown --hash-seed` / `--bucket-bits` knobs live on
`expt/bucket-cliff`, so this gate could not pin `bits` independently of the term count. It did not need to —
the `W = 1` pair (`m` = 63 364 at `r = 3`, `m` = 991 060 at `r = 4`) brackets both rank regimes anyway and the Δ
is the same in each — but a post-merge re-run with `--bucket-bits 7` would separate the kernel's effect from the
bucket-count policy's on the same cell, which is the one measurement that would let E4's constant be chosen
against a fixed sort kernel.

---

## 6. Overlap with E4, risks, and what was deliberately not done

**Overlap, flagged for the merge order.** The radix kernel is a *second, independent* attack on E4's mid-`m`
cliff, and the two are not additive:

- E4's route is policy — give a small dense-PTM sum more buckets, so `r` reaches 4 and the gather blocks arrive
  ascending, restoring the 4.9-comparison floor. Worth −38.6 % at `m = 980`, but it costs **+46.6 %** on a
  rotation layer at `m = 1497`, so it needs a fanout-aware constant that E4 explicitly did not choose.
- E2's route is the kernel — make the comparisons cheap enough that presortedness stops mattering. The radix is
  order-oblivious, so it removes the cliff's *sensitivity* rather than its cause: **gated at −24.0 % on the
  layer and −37.9 % on the sort at `m` = 9884** (§3.1), and −33 % at `W = 1` in E4's own `r = 3` regime, all
  without touching a default constant or the rotation path.

They should be gated in sequence, not together, and the `m` = 1e4 cell re-run after whichever lands first. The
sequencing matters in one direction specifically: this kernel has already taken the mid-`m` cell most of the way,
so E4's bucket-policy constant should be chosen against the *post-merge* sort, not against the comparison
kernel's numbers — otherwise it will be tuned to close a gap that no longer exists, at the +46.6 % rotation-layer
cost it carries.

**Merge conflicts to expect.** E4 added `test_support::haar_su4_matrix`; E2 added an equivalent
`bucketed::tests::haar_su4`. Whichever lands second should delete its copy and use the shared fixture.

**Risks.**

1. **The `#[inline]` set — measured, and it held.** Untouched: `sort_rows_with_scratch` keeps its hint,
   `merge2_into` keeps its absence, and the new function carries no attribute at all, recorded in its doc as its
   own A/B. `engine/merge.rs` did grow by ~200 lines in a module whose layout is A/B-verified load-bearing in
   both directions (±6 %, +20–34 %), and §3.2 shows exactly that effect: ±15 % swings in the *untouched*
   `merge2_into` at bit-identical work. But they land as −7.3 / −4.0 / **+2.2 %** on the layer — sign-inconsistent
   and inside the calibrated ±4–7 % band — so **no module move was made**, and the residual risk is now bounded
   by measurement rather than argued. The `#[inline]` hint on the new function remains untested and is the one
   cheap follow-up A/B here.
2. **Scratch growth — still the open one.** `SortScratch` gains `packed` + `aux`, 16 bytes per row of the largest
   run seen — ~245 KiB per worker at a 15 k-row run, against a run's own 48 B/row ≈ 737 KiB. The fact sheet puts
   the dense path at 100 % of the measured write ceiling at 16 threads, and the gate deliberately stopped at 8
   (where it measured −10.5…−30.3 %), so **the 16- and 32-thread behaviour of this kernel is unmeasured**. The
   trend across the three 8-thread cells is mildly discouraging at scale — −30.3 % at `m` = 9884 but only −10.5 %
   at `m` = 9.9 × 10⁵ — consistent with a compute win being progressively absorbed as the layer approaches the
   write ceiling. Anyone taking a 16-thread number must quote measured bandwidth alongside it.
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
- **No merge.** The branch is a gated merge candidate; the orchestrator merges.

## 7. Reproduction

```bash
cargo test -p paulistrings --lib engine::merge          # the contract harness, both kernels
cargo test -p paulistrings --lib radix                  # the gate's channel pinning
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown
./target/release/examples/phase_breakdown --n 70000 --qubits 64 --layers su4 --reps 40 --format json
```

The gate itself is the seven `ab-compare.sh` lines in §5, verbatim. Its archived artefacts — provenance header,
every run's stdout, both prebuilt binaries and the paired report — are in the gitignored
`benchmarks/results/2026-09-01-ccqlin038/e2-*`; the report can be regenerated from the sidecars at any time
without re-running anything:

```bash
python3 scripts/ab-report.py benchmarks/results/2026-09-01-ccqlin038/e2-su4-1e6-{a,b}.probe.jsonl --all-phases
```

The standalone microbench harness of §1.2 and §2.2 (the faithful gather-run generator plus the storage-type and
digit-width sweeps) was scratch tooling and is **not** checked in — §1.2's controls are reproducible from the
description, and everything load-bearing in this note is either a committed test, the disassembly, or a
`phase_breakdown` cell.

# Findings

One entry per experiment: the idea, the verdict, the number that matters, and when to revisit.
Full write-ups are in git history (`git log --diff-filter=D -- research/`); host facts are in `HARDWARE.md`.
Unless stated, CPU numbers are from ccqlin038 at `W = 2` and GPU numbers from its RTX A6000 at 1800 MHz SM / 7601 MHz memory.

## Rejected

### Support-bit bucket concatenation

Idea: bucket on a gate's support bits so concatenation replaces the sort.
Impossible: a four-term `H(0)` counterexample at `W = 1` interleaves buckets, since support bits are never the most-significant key field.
This is why the engine partitions with a GF(2)-linear hash.

### Static coset→worker placement

Idea: a stable coset→worker assignment to recover page locality lost to work-stealing.
Rejected: **1.25–1.9× slower** in 7 of 8 cells at 16/32 threads; stragglers cost more than locality returns.
Revisit only with work-stealing within a socket plus NUMA-aware first touch.

### Recompute-in-merge gather borrow

Idea: drop the materialized identity stream and recompute coefficients inside the merge.
Rejected: the borrowed stream is unfiltered, so `cnot`'s merge scan grew 248,810 → 1,000,000 rows/layer, **+23.7%** wall single-threaded.
Coefficient-only materialization for dense identity streams shipped instead.

### Segment-copy merge

Idea: gallop to the next rest key and bulk-copy the identity segment in `merge2_into`.
Rejected: id/rest ratios (rotation ~1.5, cnot ~1.3, gu2q ~0.4) make segments one or two rows, **+36.3%** wall on `cnot`.
Revisit only for a workload whose rest stream is far sparser.
Settled rotation buckets are such a workload: §Block merge of clean rows (revisits §Segment-copy merge).

### Interleaved transient key layout

Idea: store rest keys as one `[[u64; W]; 2]` instead of separate `x`/`z` columns.
Rejected: `gu2q` **+1.9%** single-threaded (the SoA comparator usually decides on `x` alone), `rotation_zz` +17.1% at 16 threads.

### Reserving a safe upper bound in the merge

Idea: reserve `an + bn` in `merge2_into` and the exact split in `refine_bucket` to remove `Vec`-doubling slack.
Rejected: the bound is loose exactly where the merge deduplicates, so `su4` peak RSS went **2.55–2.62× worse**; no correctness-safe upper bound may be reserved before a reduction in that path.

### SIMD kernels and `target-cpu=x86-64-v3`

Idea: AVX2, then hand-written vector kernels, for the engine (the shipped build is SSE2).
Rejected: AVX2 was net negative on two of three priority layers, and a `#[target_feature]` second path made an **untouched sort 34.66% slower** through fat-LTO layout.
Both results predate JCC padding, so the ISA question is open; a second kernel path still costs more in layout than it buys (§Word-planar layout, and kernels outside fat LTO).

### Presortedness as the radix gate's predictor

Idea: ascending-run count explains why `cnot` and `gu2q` want opposite sort kernels.
Rejected: every built-in layer arrives as exactly `k` runs for `k` rest streams; the separator is ns per comparison (4.23 `cnot` vs 2.74 `gu2q`), read at plan time as `rest_rows_per_key`.

### The `engine/merge.rs` `#[inline]` folklore

Idea: the merge's recorded `#[inline]` constraints matter.
Rejected on the padded build: three of four are codegen no-ops (`.text` byte-identical); `sort_unstable_by` still costs **+44%** on `cnot`.
Attributes in that file are not a performance hazard; the sort algorithm choice is.

### Branchless `merge2_into`

Idea: a `cmov` select for the merge's `take_a` compare.
Rejected: mispredicts relocated to the equal-key drain (56.7M → 58.4M) and the dependency chain lengthened, **+4.83%** on `rotation_zz`, +4.64% on `trotter`.
Branchless pays only when the branch genuinely misses and nothing downstream re-asks the same bit.

### Branch-free rotation merge (revisits §Branchless `merge2_into`)

Idea: a borrow-chain key compare, masked selects and an unconditional store kept by moving `w` in `merge_backward_in_place`, `merge_sparse_in_place` and `count_hits`, and a select-step binary search for the sparse merge's blocks.
Rejected: branch misses fall 24–39% but cycles rise, ccqlin038 one thread at `float64`: heavyhex Clifford **+6%**, heavyhex generic +9%, TFIM +14% (`perf stat` cycles, every pair slower); the per-row loop's next load waits on the compare, as in §Branchless `merge2_into`.
Only the sparse merge's search and step branch-free: TFIM +3–5% and heavyhex Clifford +1–4% cycles; heavyhex generic +5% cycles and +4.3% to +6.1% wall in 4 of 4 interleaved pairs at a steady load of 7.5, branch misses −4%.

### Greedy partition-row selector

Idea: choose partition rows by greedy search over circuit generators.
Rejected and removed: on heavy-hex it halved remote layers but at **1.30 imbalance** against `cut`'s 1.085, and once left a partition empty.
`PartitionRows::cut` is the recommendation wherever the lattice is known.

### Carried key in the fused layer's collision check

Idea: walk contiguous chunks carrying the previous key in registers, halving key gathers in the equal-`g32` check.
Rejected: `su4` at 1.41e7 **+4.18% (5/5)**; chaining the gathers loses more memory-level parallelism than halving them saves.
Redundant key gathers are still the fused kernel's largest known cost; a fix must keep the gathers independent.

### One `BuiltinTruncation` enum for host truncation

Idea: run every host policy as the `BuiltinTruncation` enum, matched per term, instead of one monomorphized engine per policy.
Rejected: **10–14% slower** wall on `cnot` and `rotation_zz`, merge **+31–37%** (job 7228665); the host keeps the generic policies, onto which Python lowers its policy trees (`crates/paulistrings-py/src/truncation_dispatch.rs`).

### Remote rotation's identity scaling folded into the export fill

Idea (branch `rotation-fusion`, `5d579b52`): scale a remote rotation's identity half while the export fill reads the bucket, and merge received partners straight from the exchange block.
Rejected: gather −23% / −57%, but sort +29% / +67%, merge +20% / +26% and export +18% / +13%, net wall **+1.15%** on `heavyhex_step` at 4 MPI ranks (job 7400720) and **+1.96%** on 24-qubit `tfim_step` at P = 2 (job 7400713), 5/5 pairs each.
The reads it removed reappeared cold in the next pass: at saturation moving a pass does not help, only removing one does.

### Out-of-place rotation merge

Idea: merge every rotation bucket into the scratch buffer and swap it in, dropping the in-place merge's closing shift.
Rejected: TFIM at one thread **+22.7% (0/3 faster)**; the shift copies a bucket already in L1/L2, while the swapped buffer is colder.

### Rotation partners as coefficients only (revisits §Skipping small rotation partners)

Idea: the scan keeps only each partner's coefficient, the partner order is computed on row positions from the source keys, and only the new keys above the policy's floor are ever built as rows.
Rejected: one pair each against v0.3.0 at one thread, `float64`: `tfim_11x11` −17.3%, `heavyhex_clifford` +1.6%, `heavyhex_generic` −1.1% cycles, all behind §Fold rotation hits before the merge; reading the partner keys back from the source costs more than writing them, and ordering is only 4–8% of the time (`PartnerOrder` + `Placer`).

### Skipping small rotation partners

Idea: under a coefficient floor `ε` a partner with `|c·a₁| ≤ ε` survives only by summing into a present key (`tfim_11x11`: 84.6% of partners are that small and 27.9% of those hit, so 61% of all partners are built only to be dropped; `heavyhex_clifford`: 43.7% and 10.0%), so the scan built such a partner only after a binary search of the destination bucket's keys found it.
Rejected: exact (counts and ⟨Z⟩ identical) but one thread `tfim_11x11` **+33% (3/3**, 9.8 → 13.1 s), `heavyhex_clifford` +12/+44/+12% (3/3); the search costs ~27 ns per small partner, a binary search on the first key word only the same, while building, ordering and merging a partner that then fails `keep_term` costs a few, so no per-partner membership test can pay.
An opt-in `branch_atol` that drops such partners whatever their key (prototyped as `BranchThreshold`, not shipped) costs one compare when off (neutral, 3/3 mixed signs) but buys nothing on the bench tasks at `coeff:1e-4`, one thread, float64 (exact: TFIM 7.0 s, 1.01e6 strings, ⟨Z⟩ 0.3101; heavyhex Clifford 19.7 s, 4.65e5, 0.1588):

| `branch_atol` / cutoff | TFIM wall, strings, ⟨Z⟩ error | heavyhex Clifford wall, strings, ⟨Z⟩ error |
|---|---|---|
| 0.1 | 7.06 s, 1.03e6, −2.9e-3 | 20.1 s, unchanged, 0 |
| 0.3 | 7.10 s, 1.26e6, −1.2e-2 | 20.2 s, unchanged, 0 |
| 1 | 9.85 s, 2.47e6, +9.1e-2 | 15.3 s, 3.90e5, +5.8e-3 |
| 3 | 1.74 s, 5.1e5, +0.41 | 3.15 s, 8.6e4, +4.8e-2 |

A dropped branch onto a present string is mostly one that would have shrunk it below the cutoff, so TFIM grows rather than shrinks until the tolerance is far above the cutoff, where the error is no longer small; on heavyhex every rotation is `rx(π/4)` or Clifford, so nothing drops below `atol ≈ 0.7·ε`.

## Shipped

### Fold rotation hits before the merge

A partner of an anticommuting string anticommutes too, so it can only land on one of the destination bucket's anticommuting rows, which the scan has already listed: before the merge, one walk of the ordered partners against that list sums each hit into its row (bucket-first, as the merge sums), and drops each new key at or below the policy's `TruncationPolicy::coefficient_floor`, so the merge sees new keys above the floor only and needs no `count_hits` (`rotation_layer/merge.rs`, `fold_hits`).
Measured on ccqlin038 against v0.3.0 at `float64`, interleaved pairs, results identical in every pair: one thread `tfim_11x11` **−20.0% to −21.3% cycles (4/4)**, 18% fewer instructions and 21% fewer branch misses; `heavyhex_generic` −5.1% to −5.8% (5/5 at steady load); `heavyhex_clifford` −1.0% to −1.7% in 5 of 6 pairs and +0.1% in one, its rotations being mostly Clifford; 8 threads `tfim_11x11` −20% to −22% cycles (−10% to −20% wall, 3/3), `heavyhex_clifford` −1.6% to −3.2% (3/3).
Where the time of the dropped partners went on v0.3.0 (one thread, `tfim_11x11` / `heavyhex_clifford`): `merge_partners` 37% / 29%, `scan` 26% / 18%, memmove 9% / 6%, the fallback merge 7% / 3%, `count_hits` 5% / 2%, partner ordering 8% / 4%.

### Direct-apply path for small sums

Small-`m` cost is `Channel::prepare` (4.19–5.71 µs per dense two-qubit gate), not the bucketed pipeline (0.19 µs/layer).
A direct-apply path gains **2.28–2.36×** on kicked-Ising at 2⁻⁴; threshold 2048, behind the opt-in `EngineSelection::Auto`.
The path has since been removed; revisit small `m` by caching the hash-independent part of `prepare` instead.

### Radix sort kernel for dense PTMs

A radix sort beats a comparison sort at its count floor because each comparison is a ~10–13-cycle dependent load: **sort −25.4%, layer −15.2%** at `m = 9.9e5`, −10.5% to −33.8% across dense-PTM layers.
Gated to dense two-qubit PTMs; it also removes the `W = 1` comparator penalty (1.36× → 0.92×).

### The dense-PTM bucket cliff is a delta-span rank effect

The `W = 2` bucket-count cliff is the hash's GF(2) rank on the layer's delta space, not a width effect: re-drawing only the hash seed recovers **−29.6%** of the sort.
The "`W = 1` sort is 1.9× slower" finding reduces to a ≈1.35× per-comparison residual; no default changed.

### Truncation finalize path

`norm_sqr()` for `norm()` plus a pooled magnitude array: **−44 to −45%** wall on `TopN` layers, **−26%** on `CoefficientThreshold`.
Ties are preserved exactly; the exponent-histogram selector (`ApproxTopN`) shipped alongside.

### JCC branch padding, opt-in

The engine's 45.8% DSB residency was the JCC erratum (SKX102); `-Cllvm-args=-x86-branches-within-32B-boundaries` gives **97.9%** DSB and **−7.5 to −12.6%** wall.
It taxes parts without the erratum (`HARDWARE.md §JCC branch-padding cost off Skylake`), so `scripts/jcc-rustflags.sh` opts affected measurement hosts in.

### Branch misprediction: merge loop split and branchless gather filter

Bad speculation is 7–16% of cycles.
Splitting `merge2_into` into a both-live walk plus two drains, and a store-then-conditional-length `push_if` gather filter: **−14.68%** `rotation_zz`, −9.65% `cnot`, −2.77% `trotter`.
The filter retires 22% more instructions on `cnot` and is still 7.9% faster.

### Constant recalibration and the radix gate's second arm

`RADIX_MIN_REST_STREAMS = 8` and `GATHER_OUTPUT_MAJOR_MIN_R = 3` keep their values after re-measurement; built-in plans realize only 1, 3 or 15 rest streams.
`push_if` in the output-major gather: **−4.22%** wall on `su4`; a second gate arm (`rest_streams >= 3`, `rest_rows_per_key < 2`) puts `cnot` on radix for **−5.26%** wall.

### Partitioned engine

Locality is the whole result: an exchange-free dense layer is **−4.1% to −18.2%** at P=2 across four hosts, while a remote layer costs **2–8×** its local time.
MPI weak scaling is flat from 4 to 8 ranks (48.3 → 48.6 ms remote rotation layer).

### Cut partition rows

Rows drawn as a cut of the gate graph leave 4 of 271 heavy-hex layers remote instead of 139: P=2 is **15–21% faster per step** than one process.
Over MPI they beat random rows **2.4× (2 ranks) to 1.9× (8 ranks)**, and **3.4× / 3.2×** at 4 / 8 ranks with a 16-layer bucket-bits all-reduce schedule.

### Collapse-to-one sampling (hybrid PP-MC)

These numbers are for the collapse-to-one rule (one string drawn by `|c|²`, restarted at coefficient 1), which reproduced arXiv:2607.25998's hybrid PP-MC; `CollapseSample` now draws `samples` systematic points by `|c|` instead, and B8 pairs independent trajectories.
56-qubit echo at cache 5e7: **131 s on 8 threads, 3.9 GB** against the paper's ~8 min on 10 cores; cache 3e10 over 16 ranks moves no η outside its standard error.

### Unbiased systematic collapse on the echo

On the 9-qubit B8 patch (`L = 2`, η = 9π/40, one channel per gate, exact echo 0.993 over 262 142 terms), the paired unbiased estimator is unusable below saturation: cache 2e5 with 131 072 samples collapses ~90 times per trajectory and gives **40 ± 22** over 60 pairs, cache 3.3e4 gives ~1e20, and caches ≤ 256 give pairs with no shared string (estimate 0).
Every collapse after a gate loses the cancellations the echo needs, so `Σ|c|` compounds across collapses; on the first 100 gates probed on all nine qubits, with ~15 collapses per trajectory, the same estimator gives 0.429 ± 0.002 against the exact 0.430 over 200 pairs.
The 56-qubit `L = 1` pair at cache 2e4 and 1e4 samples shares no string (estimate 0).

### Partition rows on x-bits for CZ and `rz` circuits

Rows on x-coordinates only (`partition_row_exclude={"z": all}`) leave only `rx` exchanging: on the 56-qubit echo at 4 ranks, non-local layers 77% → 34%, peak RSS/rank 1.73 → 0.75 GB, wall 103–136 → 66 s.

### Python API extensions

The examples-suite capability register is shipped; the Python docstrings are canonical for its signatures.

### GPU layer vs host, first table

One A6000 against the 16-thread host: dense `su4` **11.0× at 1.41e7 terms, 15.9× at 5.65e7**; sparse layers 2–4.7×; `heavyhex_step` 1.9×; `trotter` on ≤ 6.7e4 terms 0.9×.
Table: `HARDWARE.md § ccqlin038 — GPU`.

### Fused layer: block-uniform chunk early-out

Radix passes skip item chunks past `ceil(n_rows / THREADS)`, with a no-skip specialization: `su4` **−11.45% (5/5)**, `cnot` −2.14%.

### Fused layer: no allocation and no re-upload in steady state

Grow-only scan scratch and a position map cached on `(bits, bucket deltas)`: −1.4% to −3.1% wall per layer, all 5/5.

### Fused layer: table load and count-table registers

Skipping the amplitude-table load for rotations and unrolling `k_count` over `MAX_ENTRIES`: `su4` −2.17%, `rotation_zz` −1.12% (5/5).

### Rule: a direction-consistent phase delta is not an effect if the total is flat

`gather_ns` +1.04% (7/7) against `merge_ns` −2.33% with instruction counts flat; let instruction count settle it.

### Rule: suspect any constant tuned by wall-clock A/B before 2026-09-10

Four recorded conclusions dissolved once JCC padding was applied, all the same 32-byte-alignment artifact.
Re-derive any tuning verdict measured on an unpadded build.

### Resolved: the Y-phase convention conflict

Every parser uses the core's Hermitian convention, `Y ↔ (x=1, z=1)` with no phase.

### Resolved: `AmplitudeDamping` was transposed

`apply`/`apply_adjoint` were swapped; caught by the PauliPropagation.jl baseline, now pinned from both sides.

### `Gf2Hash` rows are splitmix64, not xorshift successors

Consecutive xorshift64 rows satisfy `rows_z = M·rows_x`, so 64 weight-~6 deltas hashed to 0 under every seed.
Rows are splitmix64 of `(seed, row, word, x-or-z)` for `Gf2Hash` and `PartitionRows`: on a sum closed under those deltas, **371 empty buckets and max 6144 → none empty, max 1084**; probe layers unchanged.

### Intra-node strong scaling is the socket's copy bandwidth

Genoa (job 7392301, `HARDWARE.md § genoa — thread scaling`): one NUMA domain scales at 0.84–0.93 efficiency to 32 threads and saturates at 40–48, where a layer moves 180–207 GB/s against a 233–243 GB/s copy ceiling (84–89%).
Through 32 threads the coset loop stays 0.96–0.98 busy, the serial share is at most 2.2%, and busy ns/term rises only 8.2 → 9.4 (gather); the rise to 15.9 at 48 (one run) is queueing at that ceiling.
Spreading threads over the CCDs triples membench's read ceiling at 8–16 threads but leaves the layer time unchanged (27.1 vs 27.5 ms at 16), so neither the shared L3 nor the per-CCD link is the limit.
Past one socket a single pool gains to 64 threads and loses at 96, as membench does; interleaving pages changes nothing, and two partitions with cut rows are the answer there.
A lever must cut bytes per term per layer, e.g. fusing commuting layers into one pass or real coefficients for Hermitian observables.

### Coset loop enumerates each coset's buckets from the span

The serial handle passes into coset order and back cost 45–70 ns per bucket on the calling thread.
Each task now swaps its `2^r` members in place (`Gf2Span::representative_at`, `member`): `heavyhex_step` at 1.8e6 terms/layer **−21.9% at 48 threads, −37.8% at 96 (5/5)**, coset loop −13% / −26% (5/5); `rotation_zz` at 3e7 terms no consistent change at 16, 48 or 96 threads (job 7392302).

### Rebucketing coarsens below an eighth of the target

A coefficient-truncated sum that peaks about 10× above its late size kept its peak bucket count, down to ~11 terms per bucket; `rebucketed_bits` now coarsens once buckets average below `target / 8`.
`heavyhex_step`, 12 steps at `coeff:1e-5` (peak 3.57e7 terms), genoa: **one pool at 48 threads −65.6%, two partitions at 96 threads −58.4% (5/5 each)**; sort −87% / −83%, gather −60% / −55%, merge −38% / −27%, and at P=2 export −55% (jobs 7393225–6).
Peak RSS rises 33.2 → 35.8 GB at 48 threads and 49.9 → 52.3 GB at P=2, unattributed.

### Clifford frame of the kicked-Ising step

With `J·dt = π/4` every `ZZ` gate is Clifford, so `C` (the `ZZ` layer) moves onto the `Z`-basis initial state and step `k` keeps only `rx` about `C^k X_q C^-k`: 127 instead of 271 channels per heavy-hex step, the series and term counts identical.
Wall **1.9–2.1×** at every rank count under x-parity rows (genoa, 2 to 32 ranks, jobs 7392303–5), and 2.0× at P=1 on ccqlin038.
A `ZZ` gate costs as much as an `rx` (busy time: gather 67%, merge 28%, sort 3%), because every gate streams the whole sum however few rows it changes (6.6% anticommute on average); only removing passes moves the step, which §Clifford runs as one pass does in the lab frame.
Not adopted in paulistrings-sim; issue #47 tracks it and the hash-function variant.
Under z-only cut rows the frame's generators `X_q Z_N(q)` leave 45.5 of 127 rotations remote per step at 32 ranks, against 67 of 271 gates in the lab frame.

### Clifford runs as one pass

A maximal run of channels that map strings one to one (Clifford-arm rotation tables, `H`, `S`, `CNOT`, `CZ`, `SWAP`, Paulis, key-preserving rescales) is one pass (`engine/bucketed/permutation_run.rs`): each string walks the run, then the sum is scattered to its new buckets and each bucket sorted once.
The walk visits only the steps whose support a string occupies (per-qubit bitsets over 256-step segments); visiting every step cost 89% of the run, and the sparse walk is 3.4× faster (no-LTO probe, one thread, 2.06e6 terms, 144 bonds: walk 4.04 → 1.18 s).
`heavyhex_clifford` against `3d25ccba`, smoke grade (ccqlin038 shared, load 4–16, every pair negative, counts and ⟨Z⟩ identical): one thread **−41.6%** (59.59 → 34.79, 59.70 → 35.05 s), eight **−47.4%** (11.26 → 5.93, 11.26 → 5.91 s); PP.jl 0.9.2 under the same lock 40.1 s (`vector`, one thread) and 13.1 s (`multi`, eight).
Controls without a Clifford gate (`tfim_11x11`, `heavyhex_generic`): no consistent change once the run handling left the inlined layer loop body, +2.4% to +13% on `tfim_11x11` while it sat inside.
A rehashed bucket gets half its length again in spare capacity, so the next rotation layer merges in place: probe step **−3.4%** (3/3), peak RSS +4%.
Rejected: the radix kernel for the rehash sort, no consistent change on the probe step.
A run spans only layers that exchange nothing, which issue no transport call (§Bucket bits agreed on the exchanges), so it runs at any partition count; under x-only rows every `Z`-type Clifford is local.
The shortest worthwhile run is unmeasured (2 to 16 bonds at 1.9e5 terms: no consistent difference inside ±10% noise), so any run of two or more is taken.
Not built: a remote run whose exchange is deferred to its end, and a GPU form.
Quiet-node A/B pending.

### Row choice on heavy-hex: x-parity rows beat the z-only cut

`phase_breakdown --mpi`, `heavyhex_step`, 12 steps at `coeff:1e-5`, genoa, one rank per socket, 48 threads per rank; x-parity rows are `log2(P)` x-only rows, each the parity of `k` qubits nearest the observable's site in breadth-first order; imbalance is the time-weighted max/mean of terms per rank, from every rank's `terms_by_layer`:

| rows | 2 ranks | 8 ranks | 32 ranks | 2 → 32 |
|---|---|---|---|---|
| z-only `cut` | 46.5 s, 4 remote/step, 1.00 | 34.9 s, 25, 2.00 | 16.1 s, 67, 1.77 | **2.9×** |
| x-parity, `k = 2` | 46.0 s, 2, 1.11 | 27.5 s, 6, 1.29 | **4.8 s**, 10, 1.14 | **9.6×** |
| x-parity, `k = 4` | 47.6 s, 4, 1.05 | 17.2 s, 12, 1.02 | 5.3 s, 20, 1.06 | 9.0× |

At 32 ranks x-parity rows are **3.4× faster** than the cut, against the 1.3–1.5× a serial cost model predicted, which priced an export at 0.2 of a layer where it measures 1.3 and left out the per-remote-layer synchronization.
Per layer at 32 ranks, cut against `k = 2`: compute 2.19 against 0.99 ms, export 0.72 against 0.03, transfer and chunk waits 0.32 against 0.03, waiting in the bucket-count collective 1.73 against 0.42 (the slowest rank's compute is 0.94 and 0.39 ms above the mean).
Bandwidth is not the limit: the cut moves 0.72 GB per rank in 16 s.
The z-only cut's imbalance is a conserved quantity: kicked Ising keeps the register's total z-parity, so once the strings occupy few blocks one row combination is nearly constant and half the partitions starve.
Which `k` balances best depends on `P` (`k = 4` at 8 ranks, `k = 2` at 32); rows reading x-bits only suit circuits whose entanglers are `Z`-type, so they are a recommendation for such circuits, not a default.
Measured with phase timers on, which cost 1.6–4.4% of wall here.
The explicit-row API lands with PR #45.

### Remote rotations run on the in-place rotation kernel

A rotation whose generator crosses partitions runs on `fill_received_rotation_bucket` instead of the general coset fill.
Genoa code A/B, P = 2 in process (jobs 7398796–7398798, 5 pairs): `heavyhex_step` under random rows at `coeff:2^-14` **−9.7% / −12.5%** wall at 96 / 48 threads (merge −55%, sort −21%, gather +52% as it now holds the scale and the received append), 24-qubit `tfim_step` at `2^-16` **−5.5% / −6.5%**, all 5/5; VmHWM within ±1%.
The synthetic all-hit `rotation_remote` at 1e6 × 40 reps is +2.2% at 48 threads (5/5, merge +13%) and inconsistent at 96, against a +4% merge shift on the untouched `rotation_local`; untouched `cnot` no consistent change.

### Export fill balanced on scanned terms

The fill split a block's destination positions on emitted rows, but bucket bits can decide the emit test for whole buckets: on `heavyhex_step` at `coeff:2.5e-5` under random rows, 60.5% of the scanned terms sat in buckets that export nothing, and the fill's leaves were busy 25% of its wall at 16 threads per partition.
It now splits at half the scanned terms and skips zero-count positions.
Icelake code A/B against `68952e43`, 5 pairs each, every pair negative: in-process P=2 at 64 threads wall **−45.4%** (7.23 → 3.95 s), export −75.2% (4.06 → 1.00 s); two MPI ranks on one node wall **−42.9%** (7.06 → 4.06 s), export −77.1% (jobs 7399740, 7399742).
Rejected: a smaller row threshold (4096 → 512, no change on ccqlin038), freshly allocated node-local payloads instead of the transport's handed-over ones (fill unchanged), and a pass 1 that loads only the column deciding the emit test (`rx` reads `z`, `rzz` reads `x`): export −5.8% (5/5) at no consistent wall change (job 7399741).
Pass 1 now streams at ~30 of ccqlin038's 39 GB/s socket read ceiling; fusing it into the fill needs a sum-sized staging buffer and was not built.
Across genoa nodes, one rank per socket at 48 threads, 5/5 pairs each: 4 ranks on 2 nodes wall **−51.2%** (4.86 → 2.37 s), export 2.72 → 0.67 s; 8 ranks on 4 nodes wall **−46.9%** (3.34 → 1.77 s), export 1.82 → 0.59 s (jobs 7399800–1, measured at the reverted column-only commit).
There the merge's busy time also falls 41–48% (5/5) though its code is unchanged; unattributed, plausibly the inbound transfer overlapping the merge less once partners send earlier.
Against `main` after §Remote rotations run on the in-place rotation kernel (`d97db49b`), 4 genoa ranks on 2 nodes: wall **−47.5%** (4.57 → 2.40 s), export 2.63 → 0.73 s, 5/5 (job 7400053).
Rejected: sending each chunk as soon as it is filled (branch `stream-sends`, `19f29c19`), wall **+26.9% / +31.5%** at 4 / 8 genoa ranks against this fill (5/5 each, jobs 7400054–5), export +84–87% against an exchange wait −12–20%; the export's extra cost is unattributed.

### `ApproxTopN`'s layer pass fused into the coset loop

At saturation `ApproxTopN`'s layer pass was three passes over the sum (octave histogram, per-bucket boundary count, retain): on ccqlin038 at `atopn:2e6` it was **48%** of wall on 24-qubit `tfim_step` and 28% on `heavyhex_step` at P = 1, 32–35% at P = 2, split histogram 26% / boundary count 22% / retain 52%.
Now the coset loop counts each output bucket into a 64-octave window around the last boundary while the bucket is in cache (`truncation/octave_window.rs`), the pass decides the edge and per-bucket quotas from those counts, and the cut is deferred to the next coset loop, which applies it to each bucket as it swaps it in (`LayerOctaves`); a rescale, a rebucket and the end of a call apply it whole, and an exchanging layer's export reads around it (§Exchanging layers count and mark their rows a layer ahead).
A layer whose agreed bound on the sum's length times its fan-out (one for a Clifford rotation, a permutation or a rescale) stays within `n` takes no counts and skips the pass, collectives included.
Quiet-node code A/B against `6a2e9e4a` (`scripts/slurm/ab-campaign.sbatch`, `4394d8b6`, probe `--n 1e6 --qubits 24` at `atopn:2e6`, 5 ABBA pairs, every pair the same sign): icelake in-process `tfim_step` P = 1 wall **−35.3%** (job 7402418), P = 2 **−15.0%** (7402416), `heavyhex_step` P = 2 **−17.1%** (7402417); two genoa nodes over MPI `tfim_step` **−15.2%** (7402419), `heavyhex_step` **−22.4%** (7402420).
Finalize falls to 1–2% of wall at P = 1 and 4–6% at P = 2 (ccqlin038), where the remaining cost is the exchanging layers' whole-sum cut and the per-layer all-reduce wait.
Counting alone (without the deferral) raised the coset loop 3–11%, which on `heavyhex_step` P = 1 cancelled its gain (its Clifford bond layers counted without cutting); the deferral and the skip removed that.

`PerPartition(ApproxTopN(n))` takes the same pass locally: each partition decides from its own window on a group of one, with no collective, and its skip bound is its own length, forgotten by an exchanging layer.
Code A/B against `23c15ce4` plus the probe's `patopn:` spelling on ccqlin038 (shared, load 1.5–10, abba, 4 pairs each), 24-qubit `tfim_step`: `patopn:1e6` at P = 2 wall **−11.8%** (4.93 → 4.35 ms/layer, finalize 27% → 0.8% of wall), `patopn:2e6` at P = 1 **−43.7%** (6.45 → 3.63 ms/layer, finalize 48% → 1.6%); the untouched `atopn:2e6` at P = 2 moved −3.2% to +1.0%, no consistent change.
At P = 2 the barrier wait fell 39% and the exchange wait rose 35%: the partitions no longer meet at the layer pass, so the wait moves to the next exchange.
Quiet-node A/B pending.

The decision itself was a serial pass over every bucket's 66 window counts, twice, which at 2^14 buckets is 6–9% of a genoa step.
Each coset-loop worker now sums the counts it takes (`WindowTotals`), and a cut keeps one split bucket and its quota (`QuotaSplit`), found by a parallel chunked scan of the boundary slot only when the quota splits the partition's boundary terms.
Code A/B against `727d58b2` on ccqlin038 (shared, load 3–28, abba, 4 pairs each), 24-qubit `tfim_step` at 32 threads: finalize per layer **−72%** at `atopn:1e7` P = 1 (0.57 → 0.16 ms, wall −2.1%, 4/4), **−69%** at `atopn:2e6` P = 1 (0.058 → 0.018 ms, wall no consistent change), **−21%** at `atopn:2e6` P = 2 (0.27 → 0.21 ms, the all-reduce wait remains) and **−76%** at `patopn:2e6` P = 2 (0.058 → 0.014 ms); the coset loop shows no consistent change in any cell.
Quiet-node A/B pending.

### Exchanging layers count and mark their rows a layer ahead

An exchanging layer made three passes over the sum before its coset loop: the deferred `ApproxTopN` cut applied whole, the export's count pass, and a fill scanning every bucket that exports anything.
Now the export reads around the pending cut (`BoundaryCut::filter` on a branch-free `boundary_filter`) and the coset loop applies it as on a local layer; when the next layer has exactly one remote delta, the coset loop also counts the terms it emits for per bucket and octave slot and marks them one bit per term (`engine/bucketed/export_ahead.rs`), so the count pass reads those counts and the fill visits only the marked rows, except in the one bucket a boundary quota splits.
`paulistrings-sim` `weak_scaling` `ccqlin038_ab` (16×16 TFIM, `approx_topn` 2e6 per rank, 2 MPI ranks on ccqlin038, code A/B against `ad9d358d`, base/new/new/base): x-only rows **−23.2%** (13.82 → 10.61 s/step, 2/2 pairs), export 2.24 → 0.33 s, exchange 0.65 → 0.06 s, coset loop 11.24 → 9.96 s; cut rows −1.5% (10.43 → 10.28 s); one rank +0.5% (8.98 → 9.03 s).
By step on the x-only cell: reading around the cut alone −8.2% (13.92 → 12.79 s), with the count pass replaced −16.0% (13.88 → 11.65 s), with the marked fill −23.2%.
The 24-qubit `tfim_step` probe at `atopn:2e6`, P = 2 random rows (half an exporting bucket's terms emit, `W = 1`): wall **−7.1%** median (1791 → 1661 ms, 5/6 pairs, load 16–21), export −13%, coset loop +14% for the counting and marking; reading around the cut alone was flat there (export +100–140 ms against the 180 ms whole-sum cut, 4/6 pairs negative).
Rejected: `boundary_filter` and the count with branches, export +35% and wall +2–4% on the probe.
Quiet-node code A/B against `ad9d358d` (`9e8f5c03`, same campaign and probe, 5 ABBA pairs): icelake in-process P = 2 `tfim_step` wall **−11.9%** (5/5, job 7402421); two genoa nodes over MPI `tfim_step` −0.03% (no consistent sign, 7402422) with export **+28.7%** (5/5), `heavyhex_step` **+1.65%** (5/5, 7402423) with export **+23.8%** (5/5).
On genoa weak-scaling sweeps (16×16 TFIM, 1e7 per rank, jobs 7402414 `ad9d358d` against 7402372 `9e8f5c03`, the latter noisy between cells), steady s/step at 2 ranks on one node: x-only 19.9 → 17.1, random rows 27.3 → 14.3, `avoid:10` 25.0 → 13.8; at 8 nodes random rows 41.5 → 21.0, x-only 28.5 → 18.4.
So the large-sum weak-scaling workload gains roughly 2×, and the small probe across genoa nodes is flat to slightly worse; there the export absorbs the read of the terms the cut drops, which the whole-sum cut used to take outside the export phase, and the counting and marking cost is not repaid; the MPI export increase is open.

### Marked export fills split by rows

A marked fill (§Exchanging layers count and mark their rows a layer ahead) reaches each row at a cache miss, ~145 ns CPU per row at 2e6 terms per partition and ~290 at 1e7, but the split weighed a marked bucket at one scanned term per row, so the fill formed few leaves.
In-process 16×16 TFIM probe (`partition_approx_topn`, P = 2, 16 threads per partition, x-only rows), per exchanging layer: at 2e6 1.15 ms, of which pass 1 0.09, the per-position loop 0.07 and the fill 0.93 over 8 leaves, the longest 95% of the fill; at 1e7 5.2 ms (0.26 / 0.63 / 4.15, 32 leaves, 12 busy threads).
A marked row now weighs 32 scanned terms (`MARKED_ROW_WORK`; 16 and 64 measure the same) and pass 1 records each bucket's fill work, so the per-position loop only gathers it: 128 leaves at 2e6, fill −25..−30% at 2e6 and −15..−19% at 1e7.
Code A/B against `3d25ccba` on ccqlin038 (abba, other sessions' builds loading it to 30–48 on 32 cores): on the probe, export per remote layer **−26.3%** at x-only 2e6 (6/6, wall −1.2%, 5/6), −27.3% at random 2e6 (4/6; the other two are B runs whose coset loop and collective also rose ~40%, wall no consistent change), −19.4% at x-only 1e7 (3/4) and −6% at random 1e7 (2/4), wall no consistent change at 1e7 (step walls spread ±30%).
`phase_breakdown` at `atopn:2e6`, P = 2, 32 threads: 24-qubit `tfim_step` random rows wall +0.4% median (3/8 negative) and `heavyhex_step` −6.7% (6/8; a first 4-pair run at +22%, 4/4, in the exchange and barrier waits did not reproduce), no consistent change; `tfim_step` cut rows export −36.7% and wall −28% (4/4 each).
Rejected: software prefetch of the marked rows, fill −9% at 2e6 and flat at 1e7.
Quiet-node A/B pending.

### Bucket bits agreed within exchange groups

Under `partition_approx_topn` a local layer issued no collective but the scheduled bucket-count all-reduce (the opening 16 layers and every 16th), and on cut rows the partitions waited there for per-layer work imbalance that moves between them.
Now only a remote layer agrees the count, by an `exchange` of the proposals within its exchange group just before the layer's exchange, a local layer rebuckets to its own proposal, and one all-reduce at the end of a call evens the counts (`engine/driver.rs`).
Code A/B against `1abdcaa8`, 4–5 abba pairs per cell: ccqlin038 in process (shared, load 15–31; `paulistrings-sim` 16×16 TFIM step, timed steps 3–4 of 4) P = 4 cut rows **−9.8%** (4/4), P = 8 cut **−9.4%** (4/4), x-only rows no consistent change; two genoa nodes over MPI, 4 ranks, 24-qubit `tfim_step` cut rows at `patopn:1e6` wall **−5.7%** (5/5, barrier −18%, job 7409289).
Rejected: carrying the proposals for the next remote layer on each remote layer's exchange instead of a round of their own, **+28.5%** on the same MPI cell (5/5, exchange +403%, export +38%, job 7409291); it adds proposal-only payloads to every member of the next group that is not a partner, and in process the separate round's wait only moves into the exchange.

### Operator products skip disjoint supports

A Liouvillian pair whose supports miss always commutes, and on a nested-commutator probe (Heisenberg chain `L = 40` with a `0.3` Z field, 157 generator terms, `O` from `Z_20` thresholded at `1e-12` per step, 1.9e6 terms at step 13, `W = 1`) only 30% of pairs meet and 16.5% anticommute.
`SupportIndex` keeps per bucket one bitset over its terms per qubit, and a generator term visits the OR of its qubits' rows: **−30% at 1 thread (3.48 → 2.46 s), −31% at 32 (215 → 149 ms)**, 4/4 pairs each, on ccqlin038.
Without the index the visit cost 7.5 of 9.7 ns per pair and the inserts 2.2; with it the visits still cost 5.6 of 8.2, see §Operator products re-read each input bucket once per generator delta.
About a third of `O`'s terms are rounding residues (below `1e-14` of the largest) when no threshold is applied, since nested commutators of a symmetric generator cancel to rounding rather than to exact zero.

### Partitioned Liouvillian: exchange volume against balance

`liouvillian_share` ships each partner the operand terms a crossing generator term anticommutes with; on the §Operator products skip disjoint supports probe at step 13 (1.9e6 terms, `O` grown from `Z_20`), ccqlin038, 32 threads in all, best of 3:

| rows | partners per rank | shipped per step | largest share | wall |
|---|---|---|---|---|
| one process | — | — | — | 123 ms |
| P=2 seeded | 1 | 1.00 × operand | 1.00 × mean | 179 ms |
| P=2 x-cut at bond (19, 20) | 1 | 0.75 × | 1.09 × | 150 ms |
| P=4 seeded | 3 | 2.99 × | 1.00 × | 208 ms |
| P=4 x-cut, blocks of 10 in Gray order | 2 | 1.04 × | 1.92 × | 196 ms |

The halo is thin only while the light cone stays inside a block: strings averaging weight 10 almost all touch a cut through it.
Cut rows trade volume for balance as they do for circuits (§Partition rows without a known lattice); in one process the exchange is a copy, so balance decides, and across nodes the 3× volume of seeded rows is unmeasured.
Rows reading only x-bits keep a `Z` field local; `PartitionRows::cut` reads z-bits, so every field term crosses.
Selecting the sends in parallel over buckets and reusing the send index for the local job took P=2 seeded from 219 to 179 ms.

### Rotation circuits against PauliPropagation.jl 0.9.2

On kicked-Ising heavy-hex (Clifford and `J = 0.2`) and 11×11 TFIM at `coeff:1e-4`, 1e5–3e6 strings, `3d25ccba` spent 12.5 ns per string per gate on one ccqlin038 core against PP.jl's 7.4: scan 70%, merge 25%, partner order 3%.
A sum above the L3 streamed whole from DRAM every gate, yet more than half the gates touch no string (anticommuting fraction per gate median 0, mean 4–7%, 90th percentile 16–28%).
The four entries below remove that; together, single runs on the quiet host, one thread against PP.jl `vector`: heavyhex Clifford 63.7 → 28.9 s (PP.jl 38.2), TFIM 19.3 → 7.7 s (11.3), heavyhex generic 29.0 → 11.6 s (18.6); eight threads against PP.jl `multi`: 11.3 → 4.3 s (13.0), 2.9 → 1.14 s (3.3), 5.8 → 1.72 s (6.0).
Every A/B in them is smoke grade (shared workstation, every pair the same sign unless stated, step counts and ⟨Z⟩ identical); quiet-node A/B pending.

### The rotation scan reads keys first

A key-only pass lists each anticommuting row with its support pattern, then a dense pass over the listed rows reads and writes their coefficients (§In-place rotation layer: faster rotation steps at 5–8% more peak memory rejected the branchy store this avoids); the Clifford arm zeroes moved strings for the merge to drop.
TFIM at one thread **−16.7% (3/3)**.

### Consecutive rotation tables run as one coset loop

Alone and under a policy without a layer pass, up to 8 consecutive rotation tables whose bucket deltas span at most 3 dimensions run as one coset loop, each coset (8 buckets, within L2) taking them in order, so the sum streams from memory once per run.
One thread: TFIM **−14.4%**, heavyhex Clifford **−20.7%** (3/3 each); dimension 4 against 3 neutral, 2 against 3 slower (2/2).
Rejected: also capping the span so a run leaves at least 8 cosets per pool thread (perf/fused-rotations, whose 8-dimensional groups at ~2^10 buckets left 4 cosets for 8 threads and ran 2.6–2.8× slower, with 12 MB cosets): `tfim_11x11` at 8 threads **+3.5%** (3/3, 1.13 → 1.17 s), and at 16 threads on 8 cores TFIM at `coeff:3e-4` (1.5e5 strings, one coset per thread uncapped) **+30%** and at `coeff:1e-3` (1.8e4) **+18%** (3/3 each), where the cap removes the run; other cells flat, load 3–5.
At 96 threads a dimension-3 run at 1e6 strings leaves about one coset per thread; unmeasured.
Traces and debug logs keep the layers apart, so `propagate_with_stats` measures the unfused path.

### Bucket supports skip what a rotation cannot touch

Each bucket's `x` and `z` columns, ORed, ride from one rotation layer to the next within a call (exact from the scan, widened by partners, forgotten at call boundaries, on every rebucket, after a layer pass and by every other layer kind).
A bucket with no `x` bit under the generator's `z` and no `z` bit under its `x` is not read, and one with no anticommuting row and no partners is not merged, since every string already passed the call's `keep_term`, which must be pure.
One thread: TFIM **−37.1%**, heavyhex Clifford **−26.2%**, heavyhex generic **−35.7%** (3/3 each); `x` and `z` apart rather than ORed: no consistent change.

### Block merge of clean rows (revisits §Segment-copy merge)

With known supports, rows outside the scan's list keep their coefficients and pass `keep_term`, so when partners and changed rows are at most an eighth of the bucket the in-place merge moves the clean rows between them as blocks; here a partner has 15–1000 clean rows, not §Segment-copy merge's one or two.
One thread against the per-row merge: heavyhex Clifford **−7.0% (3/3**, quiet), instructions −13.2%; TFIM −9.3% / −9.6% with one pair +2.7%, heavyhex generic −6.5% / −8.0% with one pair under a load burst.

### Real coefficients pay once the scan stops streaming

`--dtype float64` against complex128 with bucket supports: heavyhex Clifford **−14.2%**, TFIM **−12.2%** (2/2 each), identical counts and ⟨Z⟩; before them the same switch showed no clear change (heavyhex Clifford, noisy).
The kernels are generic over the coefficient, so the choice left is the API's: a Hermitian observable under Hermiticity-preserving channels could default to `float64`.

## GPU spike

Numbers are the second application of a gate on the saturated sum, CUDA events, medians of 5 warm repetitions, against `phase_breakdown` at 16 threads.

### GPU fused layer clears the spike gate at the threshold

`su4` at 5.65e7 terms: **318 ms of kernels against 3293 ms on the host, 10.4×**, peak 22.6 GB; go, at the threshold.

### Segmented sum must be a block scan, not a head-serial walk

A head-serial walk idles 15 of 16 lanes on `su4`; a warp-shuffle segmented scan takes the layer **604 → 318 ms**.
On length-one runs the scan is 15–20% slower, so the reduction could be chosen per prepared table.

### Records are index-sorted, not record-sorted

Two 64 KB record buffers exceed sm_86's 99 KB block limit; a 16-bit index sort over 11-byte records fits at 95 KB with `CAP = 8192`.
The 12-bit tag caps a source bucket at 4096 rows, which the device refine enforces.

### Device refine is one multi-bit pass

A four-bit refine in one counting pass: **12.4 ms at 1.41e7 terms, 53 ms at 5.65e7**; steady layers pay nothing since the bucket count is grow-only.

### Compaction stays; the loose CSR is rejected

Compaction is **2.5 ms of a 144 ms layer**; skipping it holds 11.8 GB resident against 0.8 GB, peak 34 GB.

### Occupancy is not the fused kernel's lever

512-thread blocks or `--maxrregcount=32`: all variants within 3%; the time is in the reduction and global loads.

### Pinned staging is 4.8× the pageable download

Pinned D2H **12.9 GB/s** against 2.7 GB/s pageable; allocating the pinned buffer took 4.8 s, so it is pooled.

### Exported blocks stage through a pinned pool

A pinned pool beat pageable D2H by **−23%** on `su4` remote layers; superseded by device payloads and NCCL, which never stage through the host.

### A dense remote layer on device partitions is staging-bound

Host-staged `su4` at P=2 on one device was **1760 ms/layer against 68.5 ms at P=1**, ~80% of it host copies.
Led to device-resident payloads (next entry).

### Device-resident payloads remove the staging

Blocks stay on the device and move device-to-device: `su4` at P=2 **1784 → 127 ms/layer (−92%)**, `rotation_remote` −85%; the remote penalty against P=1 is +86% instead of +2500%.
Cost: a partition holds one export plus one receive volume on its device.
Open: zero-copy adoption for the one-partner case is unmeasured.

### Sender-side merge of a partner's exported rows

Summing a partner's rows by key before the exchange ships `su4` **7.5× fewer rows** (`gu2q` 2.8×): −15% wall same-device, −80% wherever bytes really move; `gu2q` same-device is +18–22%.
Two partitions at 5.65e7 terms now fit a 48 GB card (428 ms/layer against 274 at P=1).
Revisit: a gate on the expected merge ratio for same-device groups is unmeasured.

### Chunked device receive

`exchange_bytes` cuts the receive into power-of-two position chunks: four chunks take **~5.5 GB (−20%)** off peak device memory at unchanged wall (`su4`, 5.65e7 terms, P=2).
The send side stays whole, the default is unbounded, and the NCCL form is unmeasured at two ranks.

### GPU single-device profile

Sparse fused layers are neither memory- nor compute-bound: ~50% barrier stalls over eight radix passes, ~50% of shared-memory wavefronts are bank-conflict replays, DRAM 22–26%.
At 1e4 terms about 0.08 ms of a 0.15 ms layer is host issue and round trips.
Shipped from this profile: the Clifford permutation path, the rotation join (§Rotation layers join instead of sorting) and the on-disk NVRTC cache.
Open levers, by estimated gain: fewer host round trips, then a captured graph (~2× at 1e4); disabling cudarc event tracking on single-stream sums (−15–18% at 1e4, needs a multi-stream audit); geometric over-reservation of arena and output columns (`trotter`, unmeasured); chunked or pinned first `to_host`/`from_host`; fused-layer bank conflicts.

### GPU Clifford permutation path

A permutation table (one emitting entry per pattern, no colliding outputs, no received entries) scatters rows directly (`k_permute_*`), with no sort, arena or compaction.
**−62.7% to −66.9%** wall on `cnot` from 1e5 to 1.6e7 (5/5); `heavyhex_step` (rotations only) is flat as the control.
Open: Clifford layers with received entries, and an emitter count replacing `k_permute_count`'s survivor pass.

### Rotation layers join instead of sorting

A two-entry table with one identity-mask entry (every rotation, tabulated or wide) and no received entry runs `k_join`: per output bucket a shared open-addressing table over the base rows a partner can land on, probed by the partner rows; every other row is written as it is read, so there is no sort, no fallback and no segmented sum (`JoinTable` in `engine/gpu/prepared.rs`).
A/B on ccqlin038's A6000 against the radix fused layer, same binary (`GpuLayerOptions::join`), `sampling_bench` release with the JCC flag, f64, SM 1800–1890 MHz, memory 7601 MHz:

| cell | radix wall | join wall | Δ wall | rotation layer mean, radix → join |
|---|---|---|---|---|
| kicked Ising `--approx-topn 1e6` (2 pairs) | 3.264 / 3.381 s | 2.679 / 2.839 s | **−17.9% / −16.0%** | 911 → 696 µs |
| kicked Ising `--approx-topn 1e7` (2 pairs) | 25.82 / 25.98 s | 20.62 / 20.67 s | **−20.1% / −20.4%** | 7739 → 5695 µs |
| kicked Ising `--threshold 2.5e-5` (2 pairs) | 38.23 / 38.70 s | 28.81 / 28.63 s | **−24.6% / −26.0%** | 11162 → 7421 µs |
| TFIM 16×16 `--approx-topn 1e6` (2 pairs) | 15.00 / 15.01 s | 13.43 / 13.36 s | **−10.4% / −11.0%** | 1476 → 1322 µs |
| TFIM 16×16 `--approx-topn 1e7` (1 pair) | 125.69 s | 123.77 s | −1.5% | 12809 → 12654 µs |

Under the threshold both paths give the same counts and ⟨Z₆₂⟩ = 0.171746373755 to 12 digits; under `ApproxTopN` the readouts differ in the fourth digit, as the boundary octave keeps terms in bucket order, which the join's atomic slot claims do not fix.
Phase timing at kicked Ising 1e7: `merge` (fused layer plus Clifford scatter) 16338 → 10822 ms; a saturated rotation layer's coset loop 9273 → 6332 µs, of which `gather` and `compact` (≈2.5 ms together) are unchanged.
TFIM at 1e6: `merge` 6649 → 5077 ms (−24%), while `compact` (2.8 s) and the `ApproxTopN` pass (3.5 s) stay.
Open: the TFIM 1e7 cell (W = 4) gains almost nothing, unexplained without a phase profile; writing the join's rows straight to the output columns at the pre-dedup offsets, which drops the compaction; and fused runs of commuting rotations, whose outputs can collide across every non-empty entry subset and so need insert-or-accumulate slots rather than a base-only table.

### Kernel keys stay in registers

ncu (2025.1 runs on ccqlin038; nsys does not) on steady-state launches put `k_join` at 87% of DRAM peak, but a third of its traffic was local memory: `support_bits` indexed a key word at run time (`k.x[q >> 6]`), which spills the whole key to local memory, at 1.4 useful bytes of 32 per sector (`k_join` W = 2: 160 B stack frame, W = 4: 320 B and 84 registers; `k_count` 32/64 B; the permutation run's `run_bit` select too, as ptxas folds a select chain back into an index).
`key_bit` masks the word out of every word instead: no stack frame left on `k_join` (W = 2), `k_count` or `k_permute_run_*`, and `k_join` W = 4 at 64 registers.
A/B (`sampling_bench` release, JCC flag, f64, ABAB, SM 1800–1875 MHz, memory 7601 MHz): kicked Ising `--approx-topn 1e7` **15.03 / 15.26 → 11.36 / 11.55 s (−24.4% / −24.3%)**, TFIM 16×16 `--approx-topn 1e6` **14.81 / 14.13 / 13.39 → 12.18 / 11.45 / 11.73 s (−17.7% / −19.0% / −12.4%)**.

### Warp-per-bucket kernels narrow their blocks at few buckets

TFIM 16×16 and kicked Ising at `--approx-topn 1e6` run at 1024 buckets, where eight-warp blocks make a 128-block grid: 44 of 84 SMs hold two blocks and 40 hold one, so `k_retain`, `k_count`, `k_boundary_counts` ran at 23–26% achieved occupancy and `k_retain` at 505 GB/s (ncu).
`warp_per_bucket` keeps at least 1024 blocks by narrowing to one warp a block: `k_retain` 324 → 265 µs (617 GB/s), `k_count` 130 → 103 µs.
Wall (ABAB, 3 rounds, SM 1770–1890 MHz, memory 7601 MHz): TFIM `--approx-topn 1e6` **−4.8% / −3.7% / −3.9%**, kicked Ising 1e6 −0.9% / −1.9% / +0.2% (not consistent), kicked Ising 1e7 (16384 buckets, unchanged shape) flat as the control.

### Keys move as 16-byte vectors

ncu found `k_join`'s global loads using 14.5 and its stores 16.4 of 32 bytes per sector per request, one 8-byte word per lane per instruction; `load_key`/`store_key` move an even W's words as `ulonglong2` in the join, compaction, retain and permutation-run scatter.
Same DRAM bytes, half the LSU requests: TFIM `--approx-topn 1e6` −2.6 / −1.8 / −2.0 / −3.9 / −2.5 / −2.1% (6/6), kicked Ising 1e7 −0.9 / −0.8 / −1.0 / −0.9 / −0.5% (5/5), threshold 2.5e-5 −0.3 / −0.2%, kicked Ising 1e6 mixed (−3.3 to +1.6%).

### A shared-memory carveout hint on `k_join` is flat

ncu showed `k_join` W = 4 limited to one block per SM by shared memory under the driver's default carveout; `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 100` lifts that to five, but wall moved +1.3% / 0.0% / +1.0% on TFIM `--approx-topn 1e6` and +0.2% / +0.8% / +0.3% on kicked Ising 1e7 (3 rounds), since the join is at 83–87% of DRAM peak either way; rejected.

### NCCL exchange between MPI ranks

NCCL against host staging at two ranks on one A100 node: `su4` **3.5×**, `cnot` 6.3×, `rotation_remote` 6.8× (`HARDWARE.md § gpu cluster nodes — NCCL exchange between MPI ranks`).
Verdict: NCCL is the only exchange of an MPI device group; host staging was removed.
A four-rank hang on workergpu047 is attributed to that node; every job step is bounded by `scripts/slurm/bounded.sh`; two-node NCCL is unmeasured.

## Open

### Channels above `MAX_LOCAL_SUPPORT = 2`

Non-rotation channels on more than two qubits panic in `propagate`.
Sketch: a heap `DeltaEntry::amp` variant for `k > 2`, `O(16^k)` probe cost, practical ceiling `k ≈ 4–5`.

### Partition rows without a known lattice

`cut` needs a hand-bisectable lattice and the greedy selector was removed for imbalance; a choice scoring balance and remote weight together is open.

### The small per-rank distributed regime

At ~1e5 terms per rank remote layers are latency-bound and export runs 10× slower per term than at 6e6; fewer, larger chunks and export parallelism with few buckets are unmeasured.

### A post-hoc memory trim

`shrink_to_fit` once `m` stabilizes is untried; a returned sum keeps its peak bucket capacity (holding one PP-MC result doubled the next run's peak, 34 → 67 GB).
The bucket count was grow-only too: a sum that peaks 10× above its late size kept 262,144 buckets for 3e6 terms per rank, ~11 terms per bucket, where per-coset fixed costs (~0.12 ns/term at 1024 terms per bucket) approach the per-term work.
In the 2-rank cells of §Row choice on heavy-hex: x-parity rows beat the z-only cut (jobs 7392303–5, before §Coset loop enumerates each coset's buckets from the span) the serial permute and unpermute at that count took 9 ms of a 14 ms layer.
`rebucketed_bits` now coarsens once buckets average below an eighth of the target, agreed by the same all-reduce as refining (§Rebucketing coarsens below an eighth of the target).

### Partitioned memory overhead

At cache 2e7 on the 56-qubit echo: one process 97 B per term, four partitions 221 B, four MPI ranks 345 B; grow-only exchange pools account for ~64 B, the rest is unattributed, and peak RSS ratchets across collapses.

### Word-planar layout, and kernels outside fat LTO

The planar-layout gate (`merge2_into` within +2% at `W = 2`) was never reached; first establish whether kernels can live outside the fat-LTO unit.

### The merge's irreducible per-row decision

About one bit of entropy per output row survives both merge forms (≈4% of cycles on `rotation_zz`); only algorithmic changes remain (emit both and compact, or a non-random stream layout).

### Multi-thread confirmation of the front-end campaign

The front-end campaign is single-threaded; its 16-thread arms are "no consistent change", and the radix gate's second arm wants a quiet-box multi-thread cell (its win shrinks from −30.3% to −10.5% with `m` at 8 threads).

### Sparse layers are per-block-overhead bound at a 256-term bucket target

Under the 4096 records-per-block policy sparse layers run at 1.0–1.2 ns per term, 2–4.7× the host; the floor is eight radix passes over a ~1000-record block.
Levers: merging positions into one block, or a shorter sort for short runs.

### CPU/GPU crossover is below 1e4 terms for a second layer

The GPU wins at every measured size: `cnot` 2.3× at 1e4, 7.5× at 1e5, 12.4× at 1e6 against in-process `propagate_with`.
Unexplained: in-process `propagate` ran 1.6–2.9× slower than the probe on the same cell.

### In-place rotation layer: faster rotation steps at 5–8% more peak memory

At saturation (`heavyhex_step` at `coeff:2^-15`, 24-qubit `tfim_step` at `2^-16`) an active rotation gate changes 20–70% of the strings, most partners are new keys, and a bucket's first change falls at its front, so the in-place merge rewrites nearly every bucket it touches, within L2.
Genoa code A/B against `6723e31e` (jobs 7396289–7396291, 5 pairs, every rotation cell 5/5): `heavyhex_step` −16.9% / −8.0% at 16 / 48 threads, `tfim_step` −5.1% / −5.7%, `rotation_zz` −9.6% / −6.5%, `rotation_local` −8.9% / −5.5%, untouched `cnot` no consistent change; VmHWM +4.6–6.4%.
The VmHWM cost is held capacity, not fragmentation (same ordering under `MALLOC_MMAP_THRESHOLD_=16384`): a bucket merged in place keeps its high-water capacity, while the general path's buffers circulate through scratch (#51).
Rejected: a coefficient store only for anticommuting strings (+15%, mispredicts), merging without counting hits (+12–15%), reserving `len + partners` (VmHWM +15%), and growing a short bucket by exactly its new keys (VmHWM +20–26%, the TFIM gain lost).
`PartnerOrder` against the stable `sort_by`: 15% faster on `heavyhex_step` at 8 threads, 10% slower on `rotation_zz` at one.
Skipping the hit count when a bucket has room for every partner, and closing the hits' gap with the final shift, took `merge_ns` −22% to −28% on `heavyhex_step` and `tfim_step` and −4% to −12% on the all-hit `rotation_zz` and `rotation_local`; wall −9.1% / −1.5% (16 / 48 threads, the second not consistent) on `heavyhex_step` and −16.6% / −13.4% on `tfim_step`, VmHWM +0.3–1.6% (jobs 7396538–7396540 against `d7bbc753`; untouched `cnot` +2.2% at 16 threads, so wall effects under ~3% there are layout).

### Operator products re-read each input bucket once per generator delta

Output bucket `β′` reads the operand's buckets `β′ ⊕ h(s)` for every distinct generator bucket delta, so each input bucket is streamed once per delta (~152 distinct deltas on the §Operator products skip disjoint supports probe), and at 32 threads the visits run at ccqlin038's ~48 GB/s read ceiling.
No loop order fixes this under dense hash rows: a tile `c ⊕ S` of output buckets reads `c ⊕ S ⊕ Γ`, and random deltas `Γ` spread that over every input bucket at any tile size that fits in cache.
The layout does: hash rows on the coordinates `x_q`, `z_q` of the 6 central qubits leave 130 of the 157 generator terms at delta 0 and 26 distinct deltas, and cut the visits 39% at 1 thread (1450 → 891 ms) and the call **−26% at 1 thread (2160 → 1590 ms), −15% at 32 (122 → 104 ms)**, inserts unchanged; the largest bucket is 14.8× the mean, which is what holds back 32 threads.
Open: choosing such rows from a generator and an operand, balanced, and keeping them across a recursion (the output inherits the operand's hash).
Measured and not kept (branch `multiply-gemm-wip`, `0576def7`): a bit-sliced ℤ₄ micro-kernel, each generator run transposed into per-qubit X/Y/Z planes so one operand term's phases against the whole run come out of two bit-plane counters, −6% at 1 thread under coordinate rows and nothing under dense rows; `POPCNT` multiversioning of the per-bucket pass, −3–4% at 1 thread, flat at 32; both inside ccqlin038's ±10% noise.
After the layout change a call splits about evenly between visits, inserts with their drain sort, and the singleton runs of the generator terms on the hashed qubits.

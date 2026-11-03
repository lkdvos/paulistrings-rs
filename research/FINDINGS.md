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

## Shipped

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

`CollapseSample` reproduces arXiv:2607.25998's hybrid PP-MC, collectively over partitions and ranks.
56-qubit echo at cache 5e7: **131 s on 8 threads, 3.9 GB** against the paper's ~8 min on 10 cores; cache 3e10 over 16 ranks moves no η outside its standard error.

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
A `ZZ` gate costs as much as an `rx` (busy time: gather 67%, merge 28%, sort 3%), because every gate streams the whole sum however few rows it changes (6.6% anticommute on average); only removing passes moves the step, so a CPU permutation path for Clifford tables, which can drop the sort and the duplicate check, was not built.
Not adopted in paulistrings-sim; issue #47 tracks it and the hash-function variant.
Under z-only cut rows the frame's generators `X_q Z_N(q)` leave 45.5 of 127 rotations remote per step at 32 ranks, against 67 of 271 gates in the lab frame.

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
Workers still spend 15.4 s (4 ranks) and 7.8 s (8 ranks) of busy time waiting for a received chunk: sending each chunk as soon as it is filled, rather than after the whole export, is the next lever and is unmeasured.

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
Shipped from this profile: the Clifford permutation path and the on-disk NVRTC cache.
Open levers, by estimated gain: rotation dedup as a shared-memory hash join instead of eight radix passes (fused layer −50–65%); fewer host round trips, then a captured graph (~2× at 1e4); disabling cudarc event tracking on single-stream sums (−15–18% at 1e4, needs a multi-stream audit); geometric over-reservation of arena and output columns (`trotter`, unmeasured); chunked or pinned first `to_host`/`from_host`; fused-layer bank conflicts.

### GPU Clifford permutation path

A permutation table (one emitting entry per pattern, no colliding outputs, no received entries) scatters rows directly (`k_permute_*`), with no sort, arena or compaction.
**−62.7% to −66.9%** wall on `cnot` from 1e5 to 1.6e7 (5/5); `heavyhex_step` (rotations only) is flat as the control.
Open: Clifford layers with received entries, and an emitter count replacing `k_permute_count`'s survivor pass.

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

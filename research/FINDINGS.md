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

## Shipped

### Direct-apply path for small sums

Small-`m` cost is `Channel::prepare` (4.19–5.71 µs per dense two-qubit gate), not the bucketed pipeline (0.19 µs/layer).
A direct-apply path gains **2.28–2.36×** on kicked-Ising at 2⁻⁴; threshold 2048, behind the opt-in `EngineSelection::Auto`.

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

Skipping the amplitude-table load for rotations and unrolling K1 over `MAX_ENTRIES`: `su4` −2.17%, `rotation_zz` −1.12% (5/5).

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
Open levers, by estimated gain: rotation dedup as a shared-memory hash join instead of eight radix passes (K3 −50–65%); fewer host round trips, then a captured graph (~2× at 1e4); disabling cudarc event tracking on single-stream sums (−15–18% at 1e4, needs a multi-stream audit); geometric over-reservation of arena and output columns (`trotter`, unmeasured); chunked or pinned first `to_host`/`from_host`; K3 bank conflicts.

### GPU Clifford permutation path

A permutation table (one emitting entry per pattern, no colliding outputs, no received entries) scatters rows directly (K12–K14), with no sort, arena or K4.
**−62.7% to −66.9%** wall on `cnot` from 1e5 to 1.6e7 (5/5); `heavyhex_step` (rotations only) is flat as the control.
Open: Clifford layers with received entries, and an emitter count replacing K12's survivor pass.

### NCCL exchange between MPI ranks

NCCL against host staging at two ranks on one A100 node: `su4` **3.5×**, `cnot` 6.3×, `rotation_remote` 6.8× (`HARDWARE.md § gpu cluster nodes — NCCL exchange between MPI ranks`).
Verdict: NCCL is the only exchange of an MPI device group; host staging was removed.
A four-rank hang on workergpu047 is attributed to that node; every job step is bounded by `scripts/slurm/bounded.sh`; two-node NCCL is unmeasured.

## Open

### Channels above `MAX_LOCAL_SUPPORT = 2`

Non-rotation channels on more than two qubits panic in `propagate`.
Sketch: a heap `DeltaEntry::amp` variant for `k > 2`, `O(16^k)` probe cost, practical ceiling `k ≈ 4–5`; it would also first exercise `GATHER_OUTPUT_MAJOR_MIN_R`'s output-major branch.

### Partition rows without a known lattice

`cut` needs a hand-bisectable lattice and the greedy selector was removed for imbalance; a choice scoring balance and remote weight together is open.

### The small per-rank distributed regime

At ~1e5 terms per rank remote layers are latency-bound and export runs 10× slower per term than at 6e6; fewer, larger chunks and export parallelism with few buckets are unmeasured.

### A post-hoc memory trim

`shrink_to_fit` once `m` stabilizes is untried; a returned sum keeps its peak bucket capacity (holding one PP-MC result doubled the next run's peak, 34 → 67 GB).

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

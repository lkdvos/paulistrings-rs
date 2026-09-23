# Findings

One entry per experiment: the question, the verdict, the number that matters.
Full write-ups — protocols, raw A/B tables, reproduction commands — are in git history under the deleted `research/README.md`, `research/notes/YYYY-MM-DD-*.md` and `research/plans/YYYY-MM-DD-*.md`; recover them with `git log --diff-filter=D -- research/`.
Measured host facts live in `HARDWARE.md`.

## Rejected

### Support-bit bucket concatenation

Asked whether a sorted input stays sorted after bucketing on a gate's support bits, so concatenation could replace the sort.
It cannot: a four-term `H(0)` counterexample at `W = 1` produces buckets that interleave, because a support qubit's key bits are never the most-significant field of `(x[0..W], z[0..W])`.
This is why the engine partitions with a GF(2)-linear hash instead.

### Static coset→worker placement

Asked whether a stable coset→worker assignment would recover page locality that Rayon work-stealing loses.
It measured **1.25–1.9× slower** than work-stealing in 7 of 8 probe cells at 16/32 threads; stragglers with no stealing cost far more than locality returns.
Never landed; revisit only with work-stealing within a socket plus NUMA-aware first touch.

### Recompute-in-merge gather borrow

Asked whether dropping the materialized identity stream and recomputing each row's coefficient inside the merge would cut DRAM traffic.
The borrowed stream is unfiltered, so `cnot`'s merge a-scan went 248,810 → 1,000,000 rows/layer and wall rose **+23.7%** at one thread; the traffic cut only appears where the identity stream is already dense.
Superseded by coefficient-only materialization for dense identity streams, which kept the gather structure and landed.

### Segment-copy merge

Asked whether galloping to the next rest key and bulk-copying the identity segment between beats the per-row compare loop.
Real workloads' id/rest row ratios (rotation ~1.5, cnot ~1.3, gu2q ~0.4) make the average segment one or two rows, so the gallop calls cost more than the compares they avoid: **+36.3%** wall on `cnot`.
Reverted; only a workload whose rest stream is far sparser than trotter's would change this.

### Interleaved transient key layout

Asked whether storing the run's rest keys as one contiguous `[[u64; W]; 2]` instead of separate `x`/`z` columns speeds the sort.
It did the opposite on the gate cell — `gu2q` **+1.9%** at one thread with its sort +8.4% busy, because the SoA comparator usually decides on the x words alone — and regressed `rotation_zz` +17.1% at 16 threads where the merge straddles both layouts.
Not landed.

### Reserving a safe upper bound in the merge

Asked whether reserving `an + bn` in `merge2_into` and the exact split size in `refine_bucket` removes `Vec`-doubling slack from peak RSS.
The bound is loose exactly where the merge deduplicates heavily, and `Vec` capacity never shrinks, so `su4` peak RSS went **2.55–2.62× worse** (27,368 → 71,908 kB) while `rotation_zz` gained nothing.
Reverted; this rules out reserving any correctness-safe upper bound before a reduction in that path.

### SIMD kernels and `target-cpu=x86-64-v3`

Asked whether the instruction set, and then hand-written vector kernels, buy anything in the engine kernels — the shipped build turned out to be plain SSE2, with **0 `popcnt` instructions** in the release binary.
Enabling AVX2 was net negative on two of three priority layers (`cnot` −3.56%, `rotation_zz` +3.31%, `trotter` +4.53%), and isolating the one win with `#[target_feature]` dispatch made an **untouched sort 34.66% slower**.
Both halves were later shown to be JCC-erratum layout artifacts, so the ISA question is reopened; what stands is that a second kernel path costs more in fat-LTO layout than the arithmetic buys.

### Presortedness as the radix gate's predictor

Asked whether the number of ascending runs a gather run arrives as explains why `cnot` and `gu2q` want opposite sort kernels at the same rest-stream count.
It does not vary at all: every built-in `Local` layer arrives as exactly `k` ascending runs for `k` rest streams, zero inversions, with comparisons per row equal (2.65) on the two 3-stream layers.
The separating quantity is nanoseconds per comparison (4.23 on `cnot` against 2.74 on `gu2q`), readable at plan time as `rest_rows_per_key`.

### The `engine/merge.rs` `#[inline]` folklore

Asked whether the recorded `#[inline]` constraints in the merge survive re-measurement on the branch-padded build.
Three of the four are codegen no-ops — adding or removing the hint leaves `.text` **byte-identical** under `lto = "fat"` + `codegen-units = 1` — and the fourth shrank: `sort_unstable_by` costs **+44%** on `cnot` (not the recorded +77%) and is neutral on `rotation_zz`.
Attribute changes in that file are no longer a performance hazard; the sort *algorithm* choice still is.

### Branchless `merge2_into`

Asked whether replacing the merge's `take_a` compare with a `cmov` select removes its mispredicts.
Conditional branches fell 18% but `br_misp_retired` did not move (56.7M → 58.4M) — 97.9% of the misses relocated to the equal-key drain test at the same ~35% rate — and the cmov chain lengthened the loop-carried dependency: **+4.83%** wall on `rotation_zz`, +4.64% on `trotter`.
Third confirmation that branchless pays only when the branch genuinely misses *and* nothing downstream re-asks the same bit.

### Greedy partition-row selector

Asked whether a greedy search over circuit generators can choose partition rows automatically, replacing a hand-drawn lattice cut.
On the primary heavy-hex workload it buys two remote layers per step instead of four but at **1.30 imbalance** against `cut`'s 1.085, and on the chain its tie order once left a partition empty.
Removed from the crate; `PartitionRows::cut` is the recommendation wherever the lattice is known.

### Carried key in the fused layer's collision check

Asked whether the equal-`g32` collision check, which gathers both keys of every adjacent pair, gets cheaper when each thread walks a contiguous chunk and carries the previous key in registers (one gather per record instead of two).
5 `abab` pairs, wall per layer: `su4` at 1.41e7 **+4.18% (5/5 slower)**, `rotation_zz` −0.57% (5/5), `cnot` no consistent change.
The strided loop issues its `C` independent gathers at once; the carried-key walk chains them through the register, and on a saturated sum the lost memory-level parallelism outweighs the halved gather count.
Redundant key gathers remain the fused kernel's largest known cost (`product` and `write_row` gather again), but the shape that removes them must keep the gathers independent.

## Shipped

### Direct-apply path for small sums

Asked whether the small-`m` regime, where the engine loses outright to PauliPropagation.jl, is limited by the bucketed serial pipeline.
It is not — that pipeline is 0.19 µs/layer, flat across five channel types and six decades of `m`; the fixed cost is `Channel::prepare`, 4.19–5.71 µs per gate for a dense two-qubit PTM.
A direct-apply path gains **2.28–2.36×** on kicked-Ising at 2⁻⁴ and 1.55–1.68× on XXZ; the default threshold moved 512 → 2048 and `EngineSelection::Auto` stays off by default.

### Radix sort kernel for dense PTMs

Asked whether a run-oblivious radix sort can beat a comparison sort already at its comparison-count floor (4.9 comparisons per row at full delta-span rank).
It can, because the floor is on comparison *count* and each comparison is a dependent indexed load of ~10–13 cycles against ~2 for a radix pass: **sort −25.4%, layer −15.2%, 3/3 pairs** at `W = 2`, `m = 9.9e5`, and −10.5% to −33.8% across dense-PTM layers.
Gated to dense two-qubit PTMs; the kernel also erases the `W = 1` comparator pathology, inverting the width penalty from 1.36× to 0.92×.

### The dense-PTM bucket cliff is a delta-span rank effect

Asked why a one-qubit flip of the probe's support switches the `W = 2` bucket-count cliff on and off.
It is the GF(2) rank of the partitioning hash restricted to the layer's key-delta space, not a width or word-occupancy effect: flipping only the hash seed at fixed `W = 1` recovers **−29.6%** of the sort and −22.5% of the layer.
This dissolved the recorded "`W = 1` sort is 1.9× slower per row" finding down to a ≈1.35× per-comparison residual, and no default bucket-count constant was changed.

### Truncation finalize path

Asked what the `TopN` and `CoefficientThreshold` paths cost beyond the arithmetic they need.
Replacing `norm()` with `norm_sqr()` and pooling the magnitude array (12 MB/layer at `m` = 1.5e6) is worth **−44 to −45%** wall on `TopN` layers and **−26%** on `CoefficientThreshold`, both 3/3 pairs.
Tie semantics are exactly preserved for symmetry multiplets; an exponent-histogram selector shipped as an opt-in policy, with exact `TopN` unchanged and still the default.

### JCC branch padding, opt-in

Asked whether the engine's 45.8% DSB residency — blamed on hot-path code size — is recoverable.
The cause was the JCC erratum (SKX102), and `-Cllvm-args=-x86-branches-within-32B-boundaries` takes DSB residency to **97.9%** for **−7.5..−12.6% wall** on all three priority layers at bit-identical work counters.
It is a tax on parts without the erratum, so the shipped default is portable and `scripts/jcc-rustflags.sh` detects the CPU and opts measurement hosts in.

### Branch misprediction: merge loop split and branchless gather filter

Asked how much bad speculation costs and whether it is addressable — the honest conversion is **7–16% of cycles**, most likely near the low end, with two source lines carrying 78% of `rotation_zz`'s misses.
Splitting `merge2_into` into a both-live walk plus two drains, and replacing the gather's `if a == ZERO { continue }` with a store-then-conditional-length `push_if`, removed ~40% of the misses for **−14.68%** wall on `rotation_zz`, −9.65% on `cnot` and −2.77% on `trotter`.
The filter is the counter-intuitive half: `cnot` retires 22% more instructions and is 7.9% faster.

### Constant recalibration and the radix gate's second arm

Asked whether `RADIX_MIN_REST_STREAMS = 8` and `GATHER_OUTPUT_MAJOR_MIN_R = 3`, both tuned before the padding fix, survive re-measurement — both keep their values.
The structural finding was cheaper than the sweeps: every built-in plan realizes 1, 3 or 15 rest streams, so each constant has **four** distinct settings, not fourteen.
Converting the output-major gather to the same `push_if` filter fell out of the setup and is worth **−4.22% wall / −10.68% gather on `su4`, 11/11 pairs**; a second gate arm (`rest_streams >= 3` and `rest_rows_per_key < 2`) moves `cnot` to the radix kernel for **−5.26% wall / −21.67% sort, 14/14**.

### Partitioned engine

Asked whether splitting the sum across NUMA domains, and later MPI ranks, by designated GF(2) partition rows pays.
Locality is the whole result: an exchange-free dense layer is **−4.1% to −18.2%** at P=2 across four hosts, while any layer with remote deltas costs **2–8×** its local time under the push exchange; three optimization passes took a remote rotation layer from 19× a local one to 10×, 4.6×, then 3.8×.
Shipped as the in-process partitioned engine plus `DistributedSum` over an MPI transport, with weak scaling flat from 4 to 8 ranks (48.3 → 48.6 ms for the remote rotation layer).

### Cut partition rows

Asked whether partition rows drawn as a cut of the gate graph, rather than at random, reduce the exchange enough to make P > 1 a win.
On the heavy-hex kicked-Ising step cut rows leave 4 of 271 layers remote instead of 139 and export 10× fewer rows, making P=2 **15–21% faster per step** than the single-process engine.
Over MPI they beat random rows **2.4× (2 ranks) to 1.9× (8 ranks)**, and moving the bucket-bits all-reduce to a 16-layer schedule took the 4- and 8-rank steps from flat to scaling (125 → 86 → 81 ms), for **3.4× / 3.2×** over random rows.

### Python API extensions

The capability register designed for the examples and benchmarks suite is implemented and shipped.
The Python docstrings are canonical for those signatures and semantics.

### GPU layer vs host, first table

Asked how one A6000 compares with the 16- and 32-thread host on the probe's cells under one protocol (`phase_breakdown --device 0` against `--threads 16,32`, steady-state sum, five timed applications), and whether the fused kernel's 77.8 ms on `su4` at 1.41e7 terms against the spike's 58.8 ms was real.
The dense layer is **11.0× the 16-thread host at 1.41e7 terms (4.65 ns/term) and 15.9× at 5.65e7**; the sparse layers run at 1.0–1.2 ns/term, 2–4.7× the host; `heavyhex_step` at `2^-13` is 1.9×; `trotter`'s 64 layers on a ≤ 6.7e4-term sum are 0.9× (slower than the host).
The alarm was three shipped costs, not one: padding in the radix passes, per-layer allocation and re-upload, and the amplitude-table load, together −14.7% on the layer (76.8 → 65.7 ms wall); K3 is now 62.0 ms against the spike's 58.8, the remainder within the two harnesses' bucket-count difference.
The full table, clocks and load are in `research/HARDWARE.md` § ccqlin038 — GPU.

### Fused layer: block-uniform chunk early-out

Asked whether the fused kernel's launch-wide `n_cap = next_pow2(records_max)` costs the average block real work, since under the records-per-block policy a 2048–4096-record block pads to 8192 and runs its eight radix passes over the padding.
Each radix pass now skips whole item chunks past `ceil(n_rows / THREADS)` (block-uniform), with the `n_it == C` case a separate instantiation so a block with nothing to skip pays no test.
`scripts/ab-report.py` over 5 `abab` pairs on the A6000, wall per layer: `su4` at 1.41e7 **−11.45% (5/5)**, `cnot` at 1e6 −2.14% (5/5), `rotation_zz` +0.16% with pairs disagreeing in sign; the runtime-test-only form cost `rotation_zz` a consistent +1.49%, which the specialization removed.

### Fused layer: no allocation and no re-upload in steady state

Asked what the layer's host side costs when nothing changes between layers: every device scan allocated three buffers, and the count rebuilt and re-uploaded the position map every layer.
The scan buffers are grow-only scratch and the position map is cached on `(bits, bucket deltas)`.
5 `abab` pairs, wall per layer: `su4` −1.43%, `rotation_zz` −2.52%, `cnot` −3.14%, all 5/5.

### Fused layer: table load and count-table registers

Asked whether the 4 KB amplitude table loaded per block in rotation mode, and K1's `c[MAX_ENTRIES]` indexed by a runtime entry (local memory), cost anything.
The load is skipped for a rotation table and the K1 loop is unrolled over `MAX_ENTRIES` with the entry test inside.
5 `abab` pairs, wall per layer: `su4` −2.17% (5/5), `rotation_zz` −1.12% (5/5), `cnot` −0.83% with pairs disagreeing in sign.

### Rule: a direction-consistent phase delta is not an effect if the total is flat

Seen twice in one campaign: `gather_ns` +1.04% at 7/7 against `merge_ns` −2.33% with instruction counts flat.
Let instruction count settle it before claiming a phase moved.

### Rule: suspect any constant tuned by wall-clock A/B before 2026-09-10

Four separately recorded conclusions dissolved on re-measurement once branch padding was applied, all the same 32-byte-alignment artifact.
Re-derive rather than inherit any pre-padding tuning verdict.

### Resolved: the Y-phase convention conflict

The first execution of the Python test suite exposed the parsers disagreeing with the core on the phase of `Y`.
Resolved: every parser now uses the core's Hermitian convention, `Y ↔ (x=1, z=1)` with no phase factor.

### Resolved: `AmplitudeDamping` was transposed

The PauliPropagation.jl cross-engine baseline caught `apply`/`apply_adjoint` swapped relative to every other channel, so `direction="heisenberg"` applied `Φ` instead of its dual `Φ†`.
The two bodies were swapped; the Heisenberg fixture is now bit-exact against jl on all 9 terms, and a unit test pins the orientation from both sides.

### `Gf2Hash` rows are splitmix64, not xorshift successors

Asked whether rows drawn as consecutive xorshift64 outputs cost load balance: they satisfy `rows_z[r] = M·rows_x[r]` word for word, so the 64 deltas `d_j = (row_j(M), e_j)`, mean Pauli weight 6.1, hashed to 0 under every seed and bucket count (192/192 at 20 bits over three seeds) and a 64-row fingerprint collided on 1258 of 18 337 weight-≤2 keys.
Every row word is now splitmix64 of `(seed, row, word, x-or-z)`, for `Gf2Hash` and `PartitionRows` alike: 0/192 kernel deltas hash to 0 and the fingerprint is injective.
On a sum closed under ten of the `d_j` (1024 weight-4 bases × 2^10, `B` = 1024) the old rows left **371 buckets empty and a max of 6144** against a median of 1024; the new rows leave none empty, max 1084.
The probe's layers never contain the family, so their occupancy is unchanged: `su4` median/p95/max 862/913/985 → 861/915/970, `heavyhex_step` 682/692/692 → 695/725/725 at 8 buckets, no empty buckets either way.

## GPU spike

Measured on ccqlin038 (RTX A6000, sm_86, 48 GB, shared box, clocks unlocked at 1800 MHz SM / 7601 MHz memory) with a throwaway `gpu_spike` example; CPU references from `phase_breakdown` at 16 threads with `scripts/jcc-rustflags.sh` sourced.
Every GPU number is the second application of the gate on the saturated sum, CUDA events per kernel, 5 warm repetitions, medians.

### GPU fused layer clears the spike gate at the threshold

Asked whether one A6000 runs a saturated dense two-qubit layer ten times faster than the 16-thread host at 5e7 terms, under 30 GB, with no oversize segment.
`su4` at 5.65e7 steady terms takes **318 ms of kernels (5.6 ns/term) against 3293 ms on the host, 10.4× (10.1× on wall)**, peak 22.6 GB, zero fallbacks and zero oversize segments; at 1.41e7 terms it is 12.2×.
The margin is one measurement's noise wide, so the verdict is "go, at the threshold", and the two levers below are what would widen it.

### Segmented sum must be a block scan, not a head-serial walk

Asked whether the per-run reduction in the fused layer kernel could be one thread per run head walking its duplicates.
On a saturated `su4` sum every key arrives 16 times, so the walk leaves 15 of 16 lanes idle and the kernel costs 0.70 ns per record; a warp-shuffle segmented scan brings it to 0.29 ns and the layer from **604 ms to 318 ms (1.9×)**.
On sparse layers, whose runs have length one, the scan is 15–20% slower than the walk, so the reduction should be chosen per prepared table.

### Records are index-sorted, not record-sorted

Asked whether 8-byte `(g, tag)` records could be radix-sorted in a shared-memory ping-pong at `CAP = 8192`.
Two 64 KB buffers exceed the 99 KB sm_86 block limit; sorting a 16-bit index over `(g_lo32, tag16)` records costs 11 bytes per record and fits at 95 KB with `CAP = 8192`.
The 12-bit tag offset caps a source bucket at 4096 rows, which the device refine enforces.

### Device refine is one multi-bit pass

Asked what a second layer costs when the sum has grown 14× since its partition was chosen.
Without a device `rebucket` every segment overflows `CAP` (43,108 records at 2^9 buckets for 1.4e6 terms); a four-bit refine in one counting pass costs **12.4 ms at 1.41e7 terms and 53 ms at 5.65e7**, bitwise the host's term set.
The steady-state layer then pays nothing, since the bucket count is grow-only.

### Compaction stays; the loose CSR is rejected

Asked whether skipping compaction and keeping the pre-dedup arena as the next layer's `start/len` sum would pay for its memory.
Compaction is **2.5 ms of a 144 ms layer (1.7%)** while the loose sum is 11.8 GB resident against 0.8 GB compact and its peak 34 GB against 11.8.
Not worth a 15× memory footprint; arena batching over contiguous positions at 4 GiB (12 batches at the gate cell) is the design.

### Occupancy is not the fused kernel's lever

Asked whether 512-thread blocks or `--maxrregcount=32` (two blocks per SM instead of one) speed the saturated `su4` layer.
All four variants land within 3% (136.7–140.8 ms at 1.41e7 terms).
The 64-register, one-block-per-SM configuration is fine; the time is in the reduction and the per-record global loads.

### Pinned staging is 4.8× the pageable download

Asked what the 0.8 GB saturated sum costs to bring back and re-sort.
Pinned D2H runs at **12.9 GB/s (61 ms)** against 2.7 GB/s pageable including the `Vec` allocation; the host per-bucket lex re-sort is 77 ms (5.4 ns/term at 16 threads); allocating the pinned buffer itself took 4.8 s, so it must be pooled.

## Open

### Channels above `MAX_LOCAL_SUPPORT = 2`

A channel with support on more than two qubits — other than `PauliRotation`, which overrides `prepare` — makes `propagate` panic, with no fallback path.
The design sketch moves `DeltaEntry::amp` to a heap variant for `k > 2` and leaves the `k ≤ 2` hot path untouched; probe cost is `O(16^k)` per layer, putting the practical ceiling at `k ≈ 4–5`.
Raising the constant is also what first exercises `GATHER_OUTPUT_MAJOR_MIN_R`'s output-major branch, which no built-in reaches and which has never been measured on a real workload.

### Partition rows without a known lattice

`cut` needs a lattice the caller can bisect by hand, and the automatic alternative was removed for imbalance.
A row choice that scores balance as well as remote weight is open research, as is exchange volume for circuits with no obvious geometry.

### The small per-rank distributed regime

At ~1e5 terms per rank the cut-crossing layers are latency-bound 16 MB exchanges and the export pass runs 10× slower per term (14 ns/term) than at 6e6 terms per rank.
Fewer, larger pipeline chunks when a layer is small, and export parallelism with few buckets, are unmeasured.

### A post-hoc memory trim

`shrink_to_fit` once `m` has stabilized never over-reserves, so it does not inherit the reservation experiment's failure mode.
Never attempted.

### Word-planar layout, and kernels outside fat LTO

The decisive gate for a planar layout (`merge2_into` within +2% at `W = 2`) was never reached, and the whole SIMD evaluation rests on measurements taken in the unpadded regime.
The prior question is whether the kernels can live outside the fat-LTO unit without losing more than they gain.

### The merge's irreducible per-row decision

About one bit of real entropy per output row survives both the branchy and the branchless form — ≈22M misses, ≈4% of cycles on `rotation_zz`.
Only algorithmic shapes remain: emit both candidate rows and compact, or change the stream layout so the interleaving stops being random.

### Multi-thread confirmation of the front-end campaign

Every result in the front-end campaign is single-threaded on a shared box; the 16-thread arms are "no consistent change" on wall while their phase deltas hold direction.
The radix kernel's scratch grows 16 B/row and its win shrinks toward the write ceiling (−30.3% at `m` = 9884 down to −10.5% at `m` = 9.9e5 at 8 threads), so the second gate arm in particular wants a quiet-box multi-thread cell.

### Sparse layers are per-block-overhead bound at a 256-term bucket target

Asked how the GPU does on `cnot`, `gu2q` and `rotation_zz` at steady state.
1.3–2.0 ns per steady term, only **1.3–3.6× the 16-thread host**, because a position holds ~300 records padded to a 1024-record block whose eight radix passes and syncs dominate (1.1–1.4 ns per record against 0.26 on dense layers).
Under the shipped records-per-block policy (4096) they run at 1.0–1.2 ns per steady term, 2–4.7× the host, at 0.34–0.76 ns per record against 0.29 on `su4` (`research/HARDWARE.md` § ccqlin038 — GPU); the per-block floor is now the eight passes over a ~1000-record block, and merging positions into one block or a shorter sort for short runs is the open lever.

### CPU/GPU crossover is below 1e4 terms for a second layer

Asked at what size a device layer stops paying for its launches and syncs.
With pooled buffers a layer carries 0.08–0.4 ms of host time over its kernels (5 syncs), and the GPU wins at every measured size: `cnot` 2.3× at 1e4, 7.5× at 1e5, 12.4× at 1e6 against in-process `propagate_with_scratch` on the same partition.
In-process `propagate` ran 1.6–2.9× slower than the probe on the same cell even after re-partitioning to the CPU policy; unexplained.

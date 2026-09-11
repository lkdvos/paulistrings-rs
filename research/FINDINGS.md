# Findings

One entry per experiment: the question, the verdict, the number that matters.
Full write-ups — protocols, raw A/B tables, reproduction commands — are in git history under the deleted `research/README.md`, `research/notes/YYYY-MM-DD-*.md` and `research/plans/YYYY-MM-DD-*.md`, recoverable across `9107bf7..2b95210` (`git log --diff-filter=D -- research/`).
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

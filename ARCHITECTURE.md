# Architecture

This document records the contracts and invariants of `paulistrings-rs` that span modules, and so have no single rustdoc home.
Code cites it by heading (`ARCHITECTURE.md §Engine`), so renaming a `##` or `###` heading means sweeping its citations in the same commit.
Per-item mechanics live in rustdoc, the narrative in the book's [Inside the engine](docs/book/src/manual/propagation/engine.md), measured results in `research/FINDINGS.md`, host facts in `research/HARDWARE.md`, and measurement method in `benchmarks/PROFILING.md`.

## Data-Model

`PauliString<W>` is the symplectic key `(x, z)`: per qubit `I = (0,0)`, `X = (1,0)`, `Z = (0,1)`, `Y = (1,1)`.
`Y` carries no phase, so every parsed string is Hermitian; a phase arises only from a product, which `mul_assign` returns as a `Phase` for the caller to fold into a coefficient when the term enters a `PauliSum` or `BuildAccumulator`.
The derived `Ord`, lexicographic in `x` then `z`, is the key order of every bucket and the trait the engine rests on; `Hash` serves `BuildAccumulator` alone.

`PauliSum<W>` is a list of buckets under one `Gf2Hash` (§Hash), each bucket owning its own `x`, `z` and coefficient columns.

> **Invariant:** every term lives in `buckets[h(term)]`, and within a bucket keys are strictly ascending with no duplicates.
> The canonical order is bucket index, then key.

Buckets own their columns rather than slicing one flat array, so each keeps its capacity across layers and a propagation's steady state allocates nothing.
Two sums combine (`add`, `overlap`) under the first one's partition: the second is rehashed if its rows differ, and refined or coarsened if only its bucket count does.
`BuildAccumulator` is the ingestion path for unsorted input and never appears in the propagation loop.

## Width

`W`, the number of 64-bit words per `x`/`z` part, is a const generic, so the bit operations unroll and `PauliString` stays `Copy`.
The Python bindings instantiate `W ∈ {1, 2, 4, 8, 16}` and dispatch once per call through an enum over the instantiations (§Python-Bindings).

## Bucketing

A key is a vector `v = (x, z) ∈ GF(2)^{2n}`, and multiplying by a Pauli string `d` is `v ⊕ d`, the phase living outside the key.
The bucket of a key is `h(v) = H·v` for `H ∈ GF(2)^{b × 2n}`, giving `B = 2^b` buckets, and linearity is the property everything else follows from:

```
h(v ⊕ d) = h(v) ⊕ h(d)
```

A channel maps an input key `v` to outputs `v ⊕ d`, with `d` drawn from its **delta set** `D`, the realized set `{s ⊕ t : amp[s][t] ≠ 0}` over the support coordinates.
`dim D` is 0 for a key-preserving channel, 1 for `H`, `S`, amplitude damping and a Pauli rotation of any generator weight (`D = {0, P}`), 2 for `CNOT`, `CZ` and `SWAP`, and at most 4 for a two-qubit unitary.

Output bucket `β′` reads only the input buckets `β′ ⊕ h(D)`, an affine set of `2^r` buckets with `r = rank(h(D)) ≤ dim D`, and writes nowhere else: **output buckets are write-disjoint**, which is the fact behind §Parallelism.
Equal keys share a bucket because `h` is a function, so deduplication is bucket-local and nothing in the propagation loop sorts the whole sum.
When `r = dim D`, filling `β′` from `β′ ⊕ δ` uses the one `d ∈ D` with `h(d) = δ`, a term-independent constant; when `r < dim D`, several `d` share a `δ` and the gather walks them all, so correctness never depends on the draw of `H`, only performance does.

`H`'s active rows are a prefix of one fixed seeded matrix, so `h_{b+1}(v) = (h_b(v), row_{b+1}·v)`: refining splits every bucket in two by one row parity with the within-bucket order inherited, and coarsening merges bucket pairs.
That is what makes a partition that persists across layers viable while the term count swings by orders of magnitude.

## Hash

`Gf2Hash<W>` draws all `B_MAX_BITS` rows from its seed up front and keeps an active prefix of `b`; the rows depend on `(num_qubits, seed)` alone, not on `W`, and are masked to the live columns.

**The rows must be dense and random.**
A coordinate projection is GF(2)-linear too, but weight truncation keeps keys low-weight, so chosen coordinates are almost always zero and load balance collapses on exactly the workloads that matter.
A dense random `H` is a universal hash family: bucket loads stay near `m/B` whatever the input structure, and `rank(h(D)) = dim D` with probability at least `1 − 2^{dim D − b}`.
Random also means free of GF(2)-linear relations between row words, which rules out a GF(2)-linear generator (research/FINDINGS.md §`Gf2Hash` rows are splitmix64, not xorshift successors).
A term's hash is computed at ingestion and refinement only; inside a layer every output bucket follows from the deltas.
`h(0) = 0`, so the identity string always sits in bucket 0.

## Bucket-Policy

`desired_bits` targets `DEFAULT_TARGET_BUCKET_LEN = 1024` terms per bucket and, once a sum holds `min_buckets × MIN_TERMS_PER_TASK` terms (8192 by default), floors the count at `DEFAULT_MIN_BUCKETS = 128`.
The floor is fixed rather than derived from the thread count, so the partition is a function of the sum alone, never of the machine (§Determinism).
A sum of at most 1024 terms has one bucket and is therefore plain lex-sorted.
The bucket count also caps the coset dimension a dense layer's sort needs (`research/FINDINGS.md §The dense-PTM bucket cliff is a delta-span rank effect`).

`rebucket` is grow-only: the count is the running maximum of the desired one and only `with_hash` coarsens, so a sum whose size swings across a power-of-two boundary never refines and coarsens on alternate layers.

`PropagateOptions::{target_bucket_len, min_buckets}` are a measurement lever, not a tuning parameter.
The floor clamps the count from below whatever the target asks, so fewer buckets needs both to move, and `validate` requires `min_buckets >= 16` at every entry point, which keeps a sum of at most 1024 terms in one bucket.
Under partitioning the policy applies per partition (§Partitioning).

## Prepared-Channels

The engine prepares each channel once per layer into a `Prepared`, so no layer pays a vtable call, a re-derived table or trigonometry per term:

- `LocalPtm`, for support on at most `MAX_LOCAL_SUPPORT = 2` qubits: the local Pauli-transfer matrix grouped by key delta, one `DeltaEntry` per delta with its bucket delta, full-width XOR masks and an amplitude per input support pattern, the `i^k` phases folded in.
- `PreparedRotation`, for a `PauliRotation` on more than two qubits: the delta set is `{0, P}` at any weight, and the amplitude follows per term from commutation with the generator.

A channel of wider support makes `propagate` panic naming the layer (research/FINDINGS.md §Channels above `MAX_LOCAL_SUPPORT = 2`).
The default preparation probes the channel's own `apply` on the at most 16 local basis Paulis, so a custom channel gets the bucketed engine from `apply` alone; `PtmChannel` builds the same table from its stored PTM, and a test holds the two to bit equality.

**Soundness precondition:** the probe is correct exactly when the channel honours the bounded-support contract, outputs differing from the input only on the support and amplitudes depending on the input only through its support bits.
Debug builds re-probe on an all-ones background to catch a violation, and a randomized test compares every built-in's table against `apply` on full-width inputs.

A layer never calls `apply`: every output row is a table lookup or a rotation's commutation test.
Preparation classifies the identity entry as **dense**, nonzero on every support pattern (rotations, general unitaries, amplitude damping), or **sparse** (the Cliffords), which the gather exploits (§Engine).

## Engine

`propagate` runs one partition of `run_layers` (§Transport composition), which per layer settles the bucket count (§Bucket-Policy), prepares the channel, applies the layer coset by coset, and runs the policy's layer pass.
A key-preserving table (identity delta only) skips the coset machinery as `rescale_in_place`, a coefficient scan with the same zero drop and `keep_term`.

**The unit of work is a coset** of the span of `h(D)` (`Gf2Span`), the span rather than `h(D)` because a custom channel's delta set need not be XOR-closed.
Every output bucket in a coset reads only input buckets of the same coset, so a coset is a closed task.
The layer moves bucket handles into coset-contiguous order (`CosetOrder`) and back, and each coset task:

1. swaps its `2^r` bucket columns into worker-persistent scratch, leaving the emptied, capacity-retaining columns as write destinations, so the layer is in place and peak memory is the sum plus one coset's working set per worker;
2. sizes every output member's gather run exactly from the swapped-out lengths, plus one spare slot for the branchless zero-amplitude filter to discard into;
3. gathers each input term's whole fanout into the runs;
4. per run, sorts the rest stream and merges it with the identity stream straight into the member's live slot.

**Split streams.**
Identity-delta rows keep their keys, so a run's identity stream inherits the source bucket's order and is never sorted.
Under a dense identity that stream is one-to-one with the source bucket, so the gather writes only coefficients and the merge borrows the source's key columns.

**The sort contract.**
Both `SortKernel`s leave a run ascending in `(x, z)` with duplicates allowed, as a permutation of its rows, so they are interchangeable to floating-point tolerance (§Determinism).
The comparison kernel **must remain the standard library's stable, adaptive `sort_by`** wherever a run holds more than one stream: a gather run is a concatenation of sorted per-delta streams whose natural runs driftsort detects and merges nearly for free, which pdqsort (`sort_unstable_by`) does not; stability itself is irrelevant.
The radix kernel is selected per layer for many-stream runs (`SortKernel::for_layer`) and is never a replacement, being far slower on one nearly-sorted stream.

**The merge contract.**
`merge_identity_and_rest` walks both streams, identity first on a key tie, sums equal-key coefficients, and applies `keep_term` to each fully summed coefficient.
**Signed-zero contract:** an exact-zero row, such as a `θ = π/2` rotation's `cos·c = ±0.0` identity row, is summed like any other, and the only zero test is on the final sum.

After the coset loop the handles return to bucket order, the length is recounted, and debug builds assert the invariant of §Data-Model.

## Parallelism

One coset is one Rayon task, and a task reads and writes only its own coset's buckets (§Bucketing): no atomics, no locks, no concurrent maps, and no synchronization inside a layer, only at its boundary.
Load balance comes from the random hash (§Hash) and the bucket floor (§Bucket-Policy), with work-stealing absorbing the rest (`research/FINDINGS.md §Static coset→worker placement`).
Partitions are the outer level of the same decomposition: static across NUMA domains, ranks or devices, work-stealing inside one (§Partitioning).

## Partitioning

A partitioned run splits the sum across `P = 2^p` partitions (NUMA domains in one process, MPI ranks, or CUDA devices) by widening the bucket index.
A global bucket is the pair `(part(v), loc(v))`: `part(v) = P·v` from `p` **partition rows** (`PartitionRows<W>`, `p ≤ P_MAX_BITS = 6`), and `loc(v) = H·v` from an unchanged `Gf2Hash`.
**A partition holds the terms with `part(v) = rank` and nothing else**, so a key lives on exactly one partition, and duplicates can no more straddle partitions than buckets (§Bucketing).

The partition rows are a separate matrix, not a prefix of `H`: `H`'s active rows grow with the term count, and a row that moved would change a term's owner mid-run.
Seeded rows are salted to be independent of `H` at every bucket count, so the global bucket carries `p + b` bits of entropy rather than `max(p, b)`; debug builds check it (`is_independent_of`), and dependence costs load balance, not correctness.

**The classification is per layer, not per term.**
Both maps are GF(2)-linear, so a key delta `d` moves every term by the same partition delta `part(d)` and bucket delta `h(d)`.
A delta with `part(d) = 0` is **local** and runs in the ordinary coset loop; one with `part(d) ≠ 0` is **remote**, every row it produces from local bucket `β` belonging to partition `rank ⊕ part(d)`, bucket `β ⊕ h(d)`.
Remoteness is a property of the delta alone, so every partition reaches the same verdict without communicating, and the partner set is symmetric: who sends to me is who I send to.

**The wire unit is one CSR block per remote delta, laid out in the receiver's coset order.**
Both sides can compute the receiver's `CosetOrder` (§Engine), so the sender writes segment `p` with the rows for the receiver's position `p`, generated from its own bucket `bucket_at(p) ⊕ h(d)`.
A coset's rows are then contiguous, so a prefix of the transfer is a whole unit of the receiver's work, which is what the pipelined receive waits on (`ChunkMap`).
A partner's payload holds its blocks in ascending remote-delta order, walked in lockstep with the receiver's own plan, so a delta with no rows still ships its empty block.
An exported row is computed operation for operation as a local gather would compute it (`DeltaEntry::emit`, `PreparedRotation::emit_generator`).

A layer is **export, exchange, local coset loop**, a push model whose last two steps overlap: `Transport::exchange_layer` lands the block headers and CSR offsets before the coset loop starts, and the loop waits on the bulk columns chunk by chunk, each task at most once (`ChunkWait`).
**Received rows join the rest stream, never the identity stream**, so the merge sees a key's complete sum before `keep_term` runs (§Truncation).
The coset loop runs against the table with its remote entries dropped (`retain_entries`).
A channel whose every non-identity delta is remote leaves an identity-only table, which is key-preserving, and the rescale fast path would silently drop every received row, so the fast path is compiled out for a layer that can receive rows (`ExtraRows::NEEDS_BETA`).
A rotation with a remote generator keeps its `Prepared` and switches the generator pass off instead.

The exchange's memory is one layer's traffic: a partition holds one export volume of send blocks and one of receive payloads, pooled and grow-only, so a steady-state layer neither allocates nor zeroes them again.

**The bucket count is agreed, on a schedule.**
Each partition proposes the bucket count for its own share, the group takes the maximum, and each refines to it; so `P` partitions of `n/P` terms carry the same total bucket count as one partition of `n`, and grow-only survives (§Bucket-Policy) because a maximum of grow-only proposals is monotone.
Agreement happens on any layer whose plan has a remote delta, since both sides index the exchange's blocks by the count, and otherwise on the first `BITS_AGREE_EVERY = 16` layers of a call and every 16th after.
Between agreements a partition keeps its count even when its own proposal is higher: nobody refines off-schedule, so the counts stay equal by construction.
The schedule must be computable identically by every partition without communicating, which is why it is a function of the layer index and the plan, and why "my own share grew" cannot trigger it.
`P = 1` has no group, so it skips the reduction and refines every layer, which is what keeps it bitwise `propagate`.

**The per-layer collective schedule** is therefore: an `allreduce_max_u8` of the bucket count on the schedule above, then the exchange only if the plan has a remote delta, then the policy's layer pass if `finalizes_layer` is true.
A layer with no remote delta makes no transport call, and off the agreement schedule it makes none at all unless the policy's layer pass is collective.
Every partition issues the same calls in the same order, or the group falls out of step.

**Layer passes are collective.**
`finalize_layer_partitioned` runs on every layer on every partition for a policy whose `finalizes_layer` is true; a collective is well defined only if nobody skips it, and `finalizes_layer` is a property of the policy type, so the group cannot split on it.
Its default runs `finalize_layer` at one partition and panics above it for a policy with a layer pass, so an override must treat one partition as plain `finalize_layer`, and every partitioned driver checks `supports_partitioned` before the first layer (§Truncation).

**Scatter and gather bracket a run, not a layer.**
The scatter is collective: it agrees one bucket count by the **scatter bits rule** `want.max(bits − log2 P).min(bits)`, so a partition sheds at most `log2 P` of the bits the whole sum arrived with and never goes below what its own share wants, and the bucket count summed over partitions equals the unpartitioned one.
In process, `filter_partition` runs on the owning partition's pool, so every column is first touched in the domain that reads it, and `merge_partitions` merges the disjoint shares back.
At `P = 1` scatter, all-reduce and layer are the identity, so **the partitioned engine at `P = 1` is `propagate`, bit for bit**; across partition counts the bar is tolerance (§Determinism).

**Export volume is a property of the row draw.**
A random row set sends a nonzero delta across a boundary with probability `1 − 2^{-p}`, so an exchange-free layer gains and an exporting layer is transfer-bound (`research/HARDWARE.md §Partitioned engine, P=1 vs P=2 in-process`).
Traffic-minimizing rows read the qubits on the boundary of a spatial cut (`PartitionRows::cut`), and choosing them without a known lattice is open (`research/FINDINGS.md §Partition rows without a known lattice`).

### Transport composition

A *partition* is whatever a `Transport` says a peer is: `run_layers` is one function generic over the transport, and over the backend that stores a partition's terms.
The loop and its seams (`engine/{driver,backend,plan,transport}.rs`) serve every partition count; `engine/partitioned/` holds what `P > 1` adds.
Unpartitioned `propagate` is one `HostPartition` on the caller's pool over the no-op `SoloTransport`; alone, the host backend rebuckets before `prepare` (`refine_unprepared`), so a growing layer prepares once.

**Two drivers, because three things above the layer loop do not reconcile.**
`PartitionedSum` holds `P` partitions inside one process and fans out to them per call; `DistributedSum` *is* one partition, its peers other processes (`MpiTransport`, or the in-process transport, which is how the distributed shape is tested without MPI).
They differ in the transport group's lifetime (per call, so a panicking partition drops its endpoints and its partners fail naming it, against one duplicated communicator for the process's life), in scatter and gather (one sum split locally and merged back, against a replicated input and a byte-framed gather to rank 0), and in the consistency check (only the distributed driver can be handed different circuits per rank, so only it pays for `check_consistency`).
Everything else about a call is one `DriverFrame::propagate`.

**Backend composition.**
Where a partition's terms live is a second axis, orthogonal to how its peers are reached: `run_layers` touches storage only through `PartitionStorage` and `PartitionBackend`, so it is generic over the backend exactly as over the transport.
Everything collective stays in the loop (the bucket-count schedule, the exchange decision from `PartitionPlan`, the policy's layer pass, the trace row), and a backend must issue exactly the transport calls the host layer issues, in the same order; the device backend's one addition is §Device exchange's vote.
The drivers check `supports_partitioned` before the loop, so a backend cannot widen what a partitioned run accepts.
`HostPartition` is the host backend and `DevicePartition` the device one (§GPU).

**The runtime is one pinned Rayon pool per partition, with work-stealing inside a partition only** (§Parallelism).
A layer runs inside its pool's `install`, so the thread issuing a transport call is a pool worker, which fixes the MPI thread level at `MPI_THREAD_SERIALIZED` rather than `FUNNELED`.
A distributed rank holds exactly one partition, placed by the launcher's binding rather than by the engine.

**In process, a moved payload and shared-memory collectives.**
`InProcessTransport` moves payloads through a `P × P` matrix of channels and meets for collectives at a mutex-and-condvar rendezvous; each call records `(generation, kind)`, so a collective-order violation panics naming both partitions in every build rather than hanging, and a dropped endpoint wakes the others so a waiter fails fast.

**The library never initializes MPI.**
`MPI_Init` is a process-global one-shot, so the application owns it: `MpiTransport::from_communicator` and `from_raw_handle` duplicate a communicator the caller already has, so the engine's tags cannot collide with the application's traffic.

**Wire framing, per partner, in three streams.**
A self-describing header carrying the wire version, the source rank and every part's byte length, then the early parts under their own tag, then the bulk parts, each column cut at the chunks' destination-position edges and posted chunk-major.
The header is the only message whose size the receiver cannot predict; everything after it is a posted receive of known length straight into the columns the coset loop reads, with no decode pass.
MPI's non-overtaking guarantee then matches sends to receives in posting order, which is why both sides walk partners, parts and chunks in the same ascending order, and why the chunk count is a constant (`PAULISTRINGS_EXCHANGE_CHUNKS`) rather than a thread count, two ranks possibly running different pool widths.
Raw host bytes on the wire mean a run is homogeneous: the same architecture and `W` on every rank.

**The pipeline's mutual exclusion is the MPI thread level, not a lock on the data.**
Rayon workers reach `wait_chunk` together; the first whose chunk is incomplete claims the driver role, under a small mutex never held across an MPI call, and sits in `MPI_Waitsome` over every outstanding request of the rank until its own chunk completes, notifying a condvar whenever a chunk completes and when it releases the role; the others sleep on that condvar, so a waiter never steals the cores the copy runs on.
One role holder at a time is exactly `MPI_THREAD_SERIALIZED`, and it cannot deadlock: a waiter sleeps only while the role is held and wakes when it is released, so a rank with an incomplete chunk always has a thread in `MPI_Waitsome`; every send of a call is posted before its first receive, a waiting rank services its partner, and a rank whose coset loop never asks for a chunk still reaches the closing wait.

## Determinism

The correctness bar is agreement to floating-point tolerance (`test_support::assert_terms_close`), not bitwise equality.
Equal-key summation order is unspecified, so a different bucket count, hash seed, partition count or a device may change output bits.

Reproducible as a property of the implementation, not a promise: at a fixed bucket count and seed, output is bitwise identical across thread counts and repeat runs, because cosets are write-disjoint and sequential inside; one partition's output is likewise identical across its pool sizes; `P = 1` partitioned is bitwise `propagate`; and a device run is bitwise reproducible on one device.
`TopN` and `ApproxTopN` fill from their boundary in bucket order (§Truncation), so runs at different bucket counts, partition counts or on a device keep different boundary terms and agree as operators (`test_support::assert_runs_agree`, or `assert_runs_agree_count_only` on `heavy_cut`).

A test pinning exact bits is a tripwire: when it trips under a change that is correct to tolerance, regenerate its literals or demote it to `assert_terms_close` in the same commit.
Never design, constrain or reject an optimization to keep output bits stable.

## Truncation

`keep_term` runs inside the merge on every output, so it must inline to nanoseconds; it sees the coefficient summed over every contribution to the key, received rows included (§Partitioning).
`finalize_layer` runs once per layer and may be non-local.
Above one partition a layer pass must be collective, `finalize_layer_partitioned` in lock-step on every partition (§Partitioning).
`Or` combines per-term filters only; an operand with a layer pass is rejected at construction (`BuiltinTruncation::try_or`, Python's `|`) or by the generic `Or`'s `finalizes_layer`, which every driver calls before the first layer.
`BuiltinTruncation` holds every built-in as one runtime value and is what the device drivers lower, so a custom policy has no device form.

**Magnitudes are compared as `|c|²`, never as `|c|`.**
`norm()` is a libm `hypot`, and squaring is strictly increasing on `[0, ∞)`, so the ordering is exact.
The tie grouping is too: members of a symmetry multiplet differ by a sign or a power of `i`, under which `re² + im²` is bitwise invariant.
What squaring loses is the band below `|c| ≈ 1.6e-162`, whose squares underflow to `0.0` and tie.

**`TopN(n)` and `ApproxTopN(n)` keep exactly `n` terms whenever the sum holds more.**
Each has a boundary, `TopN`'s the `n`-th largest `|c|²` and `ApproxTopN`'s the octave of `|c|²` where the running count from the top first exceeds `n`.
Every term above it is kept, and the boundary contributes its first `n − kept_above` terms in bucket order, rank order first across partitions; which boundary terms survive is not a semantic promise (§Determinism).
The quota is deterministic: count boundary terms per bucket, hand the quota out over buckets in order, then retain each bucket in parallel with its own share (`retain_with_boundary_quota`).

`ApproxTopN` is **partition-exact**: the global octave histogram is the sum of the per-partition ones, so one `allreduce_sum_u64` of `[len, histogram…]` gives every partition the same boundary, and at a boundary one more of a length-`P` vector of boundary counts gives each partition its quota by a prefix over the lower ranks.
On a device the same rule runs on integer counts with an order-preserving retain, so the device run stays bitwise reproducible.
Exact `TopN` is a distributed `k`-th selection with no collective form yet, so it runs at one partition only (`supports_partitioned` is `false`).

## Channels

A custom channel implements `support`, `apply` and `apply_adjoint`; `apply_adjoint` has no default, since `apply` is right only for a self-adjoint channel.
`OutputBuffer` is push-only and grows, so a channel declares no fanout.
The engine form comes from the hidden `prepare_for_layer`, whose `PrepareToken` argument and `Prepared` result cannot be named outside the crate, so the method can be neither called nor overridden there; `PtmChannel` and `PauliRotation` override it, and the engine calls it through the crate-private `Prepare`.
`Circuit::push` asserts that the channel's support lies below `num_qubits`.

## Python-Bindings

The extension module `paulistrings._paulistrings` (abi3) is a thin layer over enums holding the monomorphized widths (`PauliSumImpl`, `CircuitImpl`, and `GpuPauliSumImpl` under `cuda`; §Width): every method dispatches once and calls the core code Rust users call.
`python/paulistrings/` re-exports it.
Truncation factories return a handle over a `BuiltinTruncation`, composed with `&` and `|`.
A device run takes the tree as it is; a host run lowers a tree with no `Or` whose `And`-reachable filters are at most one coefficient and one weight cut onto a generic `CoefficientThreshold` or `And<CoefficientThreshold, WeightCutoff>`, so the merge's `keep_term` is monomorphized while the layer passes still run through the tree (`truncation_dispatch.rs`).

The Python `Circuit` keeps a width-erased `ChannelSpec` per pushed channel beside the core `Circuit<W>`, because the core stores channels it cannot describe back; the two are appended together and never diverge, and the spec list serves gate introspection (`Circuit.gates`, in the task-JSON gate vocabulary), slicing, concatenation and `adjoint()`.

## GPU

`DevicePartition` (feature `cuda`) is a second backend under `run_layers` (§Transport composition), and the device drivers are the host's two driver types over it: `GpuPartitionedSum`, which also serves one device, and `GpuDistributedSum`.

**The device sum.**
`GpuSum` holds flat `x`, `z` and coefficient columns with a CSR per bucket under the host's `Gf2Hash`, plus a 64-bit GF(2)-linear fingerprint `g(v) = G·v` per term, so `g(v ⊕ d) = g(v) ⊕ g(d)` is one XOR per delta.
Within a bucket the device keeps unique keys in no particular order, and `to_host` restores the invariant of §Data-Model.

**The fused layer** runs one block per output position of the coset-contiguous order (§Engine): it builds a record per emitted row, sorts the records by fingerprint, falls back to the full key when fingerprints collide so that equal keys always end adjacent, and reduces each run of equal keys under `keep_term`.
Correctness never depends on the fingerprint being collision-free, only performance does.
A block holds at most `LAYER_CAP` records and a source bucket at most `MAX_BUCKET_LEN` rows, both checked from the count table before the launch: a lone partition refines past a violation up to `GpuLayerOptions::max_bits`, while a member of a group runs at exactly the agreed bucket count and reports `Unsupported`, since nobody refines off-schedule (§Partitioning).
A table under which no two input keys emit one output key (`DevicePrepared::permutation`, every Clifford) needs no sort and no dedup and takes the permutation layer instead, unless the layer has received rows.

**Bucket policy on device.**
`proposed_bits` is the larger of the host formula (§Bucket-Policy) and the smallest count whose blocks hold `GpuLayerOptions::records_per_block` records at the channel's fanout, with a factor of two of headroom in a group, since a group member cannot refine on its own.

**Errors.**
The seam under `run_layers` is infallible, so `DevicePartition` records its first `GpuError`, skips every later layer, and the driver returns the error after the loop, agreed over the group.
A lone partition keeps the last completed layer's output and resumes from it; a group of more than one is poisoned until the caller scatters again.
`refine` advances only a host mirror of the hash, which the next layer brings the device up to and which is re-synced to the device whenever the driver reads the error.

Every device count is an integer, so a device run is bitwise reproducible on one device (§Determinism).

### Device exchange

A device partition exports the same wire unit as the host (§Partitioning), CSR blocks in the receiver's position order, and its fused layer reads received segment `p` exactly as it reads local bucket `bucket_at(p) ⊕ h(d)`.
Exchange rows never leave device memory, so a group is all device partitions or all host ones.

**The vote is the one call beyond the host layer's.**
A remote layer sends the block headers and CSR offsets through `Transport::exchange`, then one `allreduce_sum_u64` votes on going ahead and on the chunk count, then on a unanimous yes moves the bulk columns over a device wire straight into the receivers' columns.
The extra collective is legal only because every partition of a group runs the same backend.
The wire is `PeerWire` in one process (device-to-device copies on the receiver's stream) and NCCL across MPI ranks (features `cuda` and `mpi`), started at scatter.

**Failure never strands a peer.**
A partition that cannot receive, has a failed wire or has already failed votes no; nobody then posts, the ready partitions take every block as empty, and the agreement after the loop names the rank that voted no.
A partition whose layer fails before its exchange still makes the call, with one empty skeleton per remote delta and a no vote.
After a yes every partition posts every chunk's transfer even if its own layer fails mid-way, discarding what arrives; only a failed wire stops the posting, and its peers' bounded waits then fail too.

**The receive moves in chunks of destination positions.**
`GpuLayerOptions::exchange_bytes` picks the fewest power-of-two chunks whose rows fit it; a power of two because such a cut refines every coarser one, so the group can agree on the largest count anyone asked for without growing anyone's chunks.
The send side is not chunked, so a partition holds one export volume and one receive chunk on its device during a remote layer.

**The sender merges one partner's rows by key before the exchange.**
Two remote deltas to one partner can emit one key only into one receiver position, so the merge is position-local: it sums equal keys and drops an exact-zero sum but truncates nothing, since only the receiver sees a key's complete sum.
The merged rows are split over the partner's blocks without exceeding any block's unmerged count, so the wire format and the receive path are unchanged.

**A cross-device copy needs peer access granted twice**: the destination context's access to the source, and the source memory pool's access list (`try_enable_peer_access`), without which a peer copy of a pooled allocation silently stages through the host.

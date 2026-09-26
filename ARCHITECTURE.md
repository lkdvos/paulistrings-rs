# Architecture

This document is the design reference for `paulistrings-rs`.
Code comments cite it by section name (`ARCHITECTURE.md §Engine`), so the `##` headings are a stable anchor vocabulary — do not rename one without sweeping the citations in `crates/` and `python/`.
It records contracts and invariants only.
Measured results live in `research/FINDINGS.md`, host facts in `research/HARDWARE.md`, measurement method in `benchmarks/PROFILING.md`, and reader-facing narrative on the docs site (`docs/book/src/manual/propagation/engine.md`).

## Overview

The library implements **Pauli propagation**: classical simulation by evolving operators in the Pauli basis under gates and noise channels.
A weighted sum of Pauli strings is pushed through a circuit layer by layer — forward, or in the Heisenberg picture by applying adjoints in reverse — with truncation keeping the sum tractable.

Design pillars, in priority order: correctness of the core algebra; performance at 10⁶–10⁸ terms; extensibility (custom channels and truncation policies without forking); GPU-readiness.

Non-goals: state-vector, tensor-network, stabilizer and matrix-product-state simulation, and anything of a quantum SDK — no transpilation, no hardware control.
Circuits come from upstream tooling.

## Data-Model

**`PauliString<const W: usize>`** uses the symplectic encoding: each qubit's Pauli is a bit pair with `I = (0,0)`, `X = (1,0)`, `Z = (0,1)`, `Y = (1,1)`, stored as `x: [u64; W]`, `z: [u64; W]`.
One word covers 64 qubits.
The type is `Copy + Pod + Zeroable` and `#[repr(C)]` with no padding — `16·W` bytes, directly serializable and GPU-uploadable.

Multiplication is bitwise XOR of the `(x, z)` parts plus a phase `i^k`; `mul_assign` returns `k` as a `u8` in `0..4` and stores no phase.
Callers fold the phase into a `Complex64` coefficient at the boundary — the moment a string enters a `PauliSum` or `BuildAccumulator`.

The load-bearing trait is **`Ord`** (lexicographic over the concatenated `(x, z)` words), not `Hash`: the engine is sort- and partition-based.
`Hash` exists for the ingestion path only (§Ingestion).

**`PauliSum<const W: usize>`** is a bucketed structure-of-arrays: per-bucket column triples (`Vec<[u64; W]>` for `x` and `z`, `Vec<Complex64>` for coefficients) partitioned by a GF(2)-linear hash (§Hash), plus `num_qubits` and a cached length.

> **Invariant:** every term lives in `buckets[h(term)]`; within each bucket keys are strictly ascending in lex `(x, z)` order with no duplicates.
> The canonical order — promised publicly — is bucket index, then key.

A single-bucket sum is automatically in plain lex order, because `h(v)` is constant over it; sums below the parallelism threshold (§Bucket-Policy) have one bucket.

Buckets own their columns rather than slicing one flat SoA, so every bucket retains its capacity across layers and the steady state of a propagation loop allocates nothing.
SoA keeps coefficient-only scans (truncation, expectation values) and key-only scans (weight, commutation) cache-friendly, and each column maps directly to a GPU device buffer (§GPU-Readiness).

## Width

`W` is a const generic: monomorphization eliminates indirection, fully unrolls the bit operations, and keeps `PauliString` `Copy`.
Python supplies `num_qubits` at runtime, so the binding layer instantiates the fixed width set `{1, 2, 4, 8, 16}` (64–1024 qubits) and dispatches once, outside any hot loop, via an enum over the instantiations (§Python-Bindings).
Rust users call the core crate with any `W` they like.

## Bucketing

The sum carries no global sorted order; it is partitioned by a GF(2)-linear hash instead.
The partition is persistent across layers, makes a channel's output buckets statically predictable, and makes deduplication bucket-local — so there is no global sort anywhere in the propagation loop.

**Keys form a vector space.**
Under the symplectic encoding a key is `v = (x, z) ∈ GF(2)^{2n}` and Pauli multiplication is `⊕` (XOR); the phase lives outside the key entirely.

**The bucket function.**
Fix `H ∈ GF(2)^{b × 2n}` and define `h(v) = H·v`, giving `B = 2^b` buckets.
Linearity yields the property everything else follows from:

```
h(v ⊕ d) = h(v) ⊕ h(d)
```

**Channels act by a bounded delta set.**
A channel with support `S`, `|S| = k`, maps an input key to outputs differing only inside the `2k` support coordinates: `v_out = v ⊕ d` with `d` drawn from the channel's **delta set** `D ⊆ GF(2)^{2k}`.
For the built-ins `dim D` is 0 for key-preserving channels (identity, depolarizing, dephasing, Pauli gates), 1 for `H`/`S`/amplitude damping and for a Pauli rotation of **any** generator weight (the delta is the fixed generator, so a weight-`w` rotation needs 2 buckets, not `4^w`), 2 for `CNOT`/`CZ`/`SWAP`, and bounded by `2k` for general unitaries.
`D` is the *realized* set `{s ⊕ t : amp[s][t] ≠ 0}`, so a sparse unitary reads fewer buckets than the bound.

**Bucket prediction, and its inverse.**

```
forward:   h(v_out) ∈ h(v_in) ⊕ h(D)
inverse:   inputs contributing to output bucket β′ live in β′ ⊕ h(D)
```

`h(D)` spans a subspace of dimension `r = rank(H|_D) ≤ dim D`, so each output bucket reads an affine set of exactly `2^r` input buckets — at most 2 for rotations, 4 for two-qubit Cliffords, 16 for a dense two-qubit unitary — and writes nowhere else.
**Output buckets are write-disjoint**, which is the structural fact behind the parallel decomposition (§Parallelism).

**Dedup is bucket-local.**
`h` is a function, so equal keys land in the same bucket and duplicates can never straddle buckets.
Deduplication therefore needs only a canonical order *within* a bucket.

**The per-(input, output) delta is a constant.**
Filling output bucket `β′` from input bucket `β = β′ ⊕ δ` uses the `d ∈ D` with `H·d = δ`; when `rank(H|_D) = dim D` — the common case for a random `H` — that `d` is unique and term-independent.
The inner loop is: extract the ≤ `2k` support bits, one table lookup (phase already folded in), skip if the amplitude is zero, XOR with a precomputed full-width mask, one complex multiply.
When `rank(H|_D) < dim D`, several `d` share a `δ` and are iterated as a short member list — correctness never depends on `H` being well-chosen, only performance does.

**Refinement is one parity pass.**
`H`'s active rows are a prefix of a fixed seeded matrix, so `h_{b+1}(v) = (h_b(v), row_{b+1}·v)`: doubling `B` splits each bucket in two with within-bucket order inherited, an `O(n)` single-row-parity pass with no re-sorting.
Halving merges bucket pairs with a two-way merge.
This is what makes a *persistent* partition viable while `n` swings by orders of magnitude across a run.

## Hash

`Gf2Hash<W>` stores `b_max` rows as `(rows_x, rows_z)` word masks, an active prefix length `b`, and the seed that generated the rows; each row word is splitmix64 of `(seed, row, word, x-or-z)`, so the rows are reproducible with no added dependency and the same at every `W`.
`bucket_of(x, z)` sets result bit `i` to `parity(x & rows_x[i]) ^ parity(z & rows_z[i])`; `row_parity` evaluates a single row for the refinement pass, making refine `O(n)` rather than `O(n·b)`.
Columns beyond `2·num_qubits` are masked to zero at construction.
The hash is stored with the sum; two sums combine only if they share it.
`PartitionRows<W>` holds additional rows of the same kind, drawn from a salted seed so they are independent of this prefix at every bucket count (§Partitioning).

**The rows must be dense and random.**
A coordinate projection is also GF(2)-linear, but weight-based truncation keeps sums low-weight, so chosen coordinates are almost always zero and load balance collapses exactly on the workloads that matter.
A dense random `H` is a universal hash family on the key space: maximum bucket load is `m/B + O(√(m log B / B))` with high probability *independent of input structure*, and `rank(H|_D) = dim D` holds with probability `≥ 1 − 2^{dim D − b}`.
Random must also mean free of GF(2)-linear relations between row words: consecutive outputs of a GF(2)-linear generator such as xorshift satisfy `rows_z = M·rows_x`, which puts a fixed family of weight-≈6 keys in the kernel of `H` under every seed.
The `b × 2W` popcount cost per term is paid only at ingestion and rehash, never in the layer loop.
Known wart: `h(0) = 0`, so the identity string always sits in bucket 0.

## Bucket-Policy

The bucket count targets `DEFAULT_TARGET_BUCKET_LEN = 1024` terms per bucket, so a bucket and its scratch sit in L2.
The floor is a fixed `DEFAULT_MIN_BUCKETS = 128` — deliberately **not** derived from the thread count, so the partition is a deterministic function of the sum alone, not of the machine.
A sum only leaves the single-bucket regime above `DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK` (= 8192) terms; below that one bucket keeps the plain lex order (§Data-Model).
Under partitioning the floor applies per partition, so `P` partitions carry `P × DEFAULT_MIN_BUCKETS` buckets between them (§Partitioning).

The bucket count also fixes the engine's coset dimension `r = min(rank(h(D)), bits)` (§Engine), and a sort reaches its `log2(fanout)` comparison floor only at full delta rank — `r = 4` for a two-qubit channel.
Below 8192 terms `bits ≤ 3`, so a dense-PTM layer cannot reach full rank; a short rank also happens by *draw* at `B = 128`.
Either costs sort time and nothing else (`research/FINDINGS.md`).

`rebucket` is **grow-only**: `B` is the running maximum of the desired bucket count over the sum's history, and only an explicit `with_hash` shrinks it.
Coarsening would let a sum whose size swings across a power-of-two boundary refine and coarsen at `O(n)` on alternate layers.
A hysteresis band was tried instead and rejected.
Refine and coarsen parallelize per bucket (pair) above the same 8192-term threshold.

`PropagateOptions::{target_bucket_len, min_buckets}` expose both values per call.
They are a **measurement lever, not a tuning parameter**: the defaults are the measured optimum.
Both have to move together — above the floor, `desired_bits` clamps the count at `min_buckets` whatever the target asks for — and `min_buckets` must stay `>= 16` or the "worth splitting" gate goes non-monotone.
`rebucket` being grow-only, lowering either mid-run never coarsens a partition already grown.
The small-sum direct path (`engine::direct`) sizes its partition from the defaults regardless.
Pinned by `crates/paulistrings/tests/bucket_knob.rs`.

## Prepared-Channels

The engine **prepares** a channel once per layer into one of two forms, so no layer pays a vtable call, a re-derived table or trig per term:

```rust
pub enum Prepared<const W: usize> {
    Local(LocalPtm<W>),        // support on ≤ MAX_LOCAL_SUPPORT qubits
    Rotation(RotationPrep<W>), // exp(-iθP/2), any generator weight
}
```

`LocalPtm` is the channel's local Pauli-transfer matrix over its support: a list of `DeltaEntry`s, each carrying the bucket delta `δ = H·d`, the delta in local support coordinates, full-width XOR masks, and an amplitude per input support pattern (`amp[s]` takes pattern `s` to `s ⊕ d`; exact zero means "no output").
The `i^k` phase is folded into `amp` at prepare time.
`MAX_LOCAL_SUPPORT = 2` bounds the dense table at `16 × 16` amplitudes, 4 KB per layer; a support-3 table would be 64 KB with a 1 KB amplitude row inlined per entry, which is why wider supports take a different route.

`Channel::prepare` has a **default implementation that is automatic and complete for any channel with support on ≤ 2 qubits**: `derive_local` calls the channel's own `apply` on each of the ≤ 16 local basis Paulis and reads the PTM off the results.
A custom channel that implements `apply` gets the bucketed engine for free, and the derivation doubles as a cross-check between the two representations.
`PauliRotation` overrides `prepare` and returns `Prepared::Rotation` at any generator weight — its delta set is `{0, gen}` regardless of weight, with the amplitude computed per term from commutation with the generator.

**Soundness precondition:** `derive_local` is correct exactly when the channel honors the bounded-support contract — output amplitudes may depend on the input only through its support bits.
This is a documented trait requirement, pinned by a property test comparing each derived table against `apply` on randomized full-width inputs.

**Identity-stream density.**
Every built-in's delta set contains the identity delta.
Preparation classifies it as **dense** — amplitude nonzero on every active support pattern (all rotations, general unitaries, amplitude damping) — or **sparse** (Cliffords: `CNOT` keeps 4 of 16 patterns, `H` 2 of 4).
The engine exploits density to avoid materializing identity-stream keys at all (§Engine).

**Declined preparation is an error.**
A channel whose support exceeds `MAX_LOCAL_SUPPORT` without overriding `prepare`, or one that writes outside its declared support, makes `propagate` panic with a message naming the layer and the reason.
No built-in can reach this.
The extension path for genuinely wide custom channels is a heap-backed `LocalPtm` variant; composing from 1- and 2-qubit channels covers the rest.

Channel fanout (`max_fanout`) sizes the `OutputBuffer` for direct `apply` calls and is not an engine concern: the gather emits at most one output per (term, delta entry), sized exactly from bucket lengths before any work begins.

## Engine

`propagate` (and `propagate_with_scratch`, which it wraps) iterates the circuit's channels — in order for forward propagation, in reverse with adjoints for Heisenberg — and per layer runs:

```
rebucket → prepare → apply layer over cosets → policy.finalize_layer
```

Key-preserving channels (identity delta only) bypass the whole pipeline via `rescale_in_place`, a parallel coefficient scan that touches no keys.

**The unit of work is a coset.**
The engine works with the span of `h(D)` (`Gf2Span`) — the span rather than `h(D)` itself, because a custom channel's delta set need not be XOR-closed.
Cosets of the span partition the bucket index space, and every output bucket in a coset reads only input buckets in that same coset: a coset is a closed task.
Bucket *handles* are permuted into coset-contiguous order once per layer (two `O(B)` handle moves bracket the layer), then each coset task, independently:

1. **Swap** its `2^r` bucket columns into worker-persistent scratch, leaving empty, capacity-retaining columns as write destinations — the layer is in-place, so peak memory is `n` plus per-worker scratch of one coset's working set, not a second full-size copy.
2. **Size** each per-member gather run exactly from the swapped-out lengths, plus one spare slot per column for the gather's branchless zero-amplitude filter to discard into.
3. **Gather input-major**: each term is loaded once and its whole fanout scattered to runs via the O(1) index identity `member(i) ⊕ δ = member(i ⊕ coord(δ))`, so the gather visits each input term exactly once with no read amplification.
   Rows whose PTM amplitude is exactly zero are filtered **branchlessly** — always materialized, published only by `len += (amp != 0)` — because which entries vanish depends on the term's support pattern.
   (An output-major variant guards rank ≥ 3 custom channels, selected by `GATHER_OUTPUT_MAJOR_MIN_R`; no built-in reaches it.)
4. Per run, **sort the rest stream and merge**, straight into the member's live slot.

**Split streams.**
A gather run keeps the identity-delta stream separate from the rest.
Identity rows keep their keys, so the id stream inherits the source bucket's strictly-ascending unique order and is **never sorted**; only the rest stream is.
When the identity amplitude is dense (§Prepared-Channels) the id stream is 1:1 with the source bucket, so the gather materializes only the 16-byte coefficients and the merge borrows the key columns from the source bucket in place.
Sparse identity streams materialize pre-filtered keys and coefficients.

**The sort.**
Two kernels live in `engine/merge.rs` alongside their shared worker-persistent `SortScratch`, and the layer picks between them *once*, from its plan's realized rest-delta count.
Both satisfy one contract and nothing more: the output is ascending in lex `(x, z)` with duplicates allowed, and is a permutation of the input triples, so they are interchangeable to floating-point tolerance (§Determinism) and never bitwise.
`merge::tests::assert_sort_contract` holds both to it.

`sort_rows_with_scratch` — the default, and the only kernel a sparse-PTM layer ever sees — is a permutation sort over the run.

> Its comparison sort **must remain the standard library's stable adaptive `sort_by`** wherever a gather run holds more than one stream.
> A gather run is a concatenation of per-delta streams, each drawn from one sorted bucket — piecewise-sorted data whose natural runs the adaptive driftsort detects and merges nearly for free, and which `sort_unstable_by` (pdqsort, no run detection) does not.
> Stability per se is irrelevant; adaptivity is the point.
> Recorded on the function's doc — do not "simplify" it.

`sort_rows_radix_with_scratch` serves the dense-PTM path, where the run arrives as many ascending blocks with heavily duplicated keys.
Adaptivity already puts that at the information-theoretic comparison floor, so the win is not in the comparison *count* but in what one costs: each is a dependent indexed load through the permutation into a several-hundred-KiB key column.
The radix kernel finds the most significant key word the rows actually disagree on, extracts an order-faithful 16-bit surrogate from it (every row shares the bits above, so the shifted masked window is monotone in the key), sorts `(surrogate, row index)` records with two 8-bit passes of sequential reads, and orders the residual ties on the full key at ~1 comparison per row.
Runs whose keys are all equal return immediately; runs whose window cannot discriminate delegate to the comparison kernel.

> This kernel is **selected, never a replacement**: on a single nearly-sorted stream it is far slower.
> `RADIX_MIN_REST_STREAMS` gates it at 8, so today only a dense two-qubit PTM reaches it and every rotation/Clifford layer keeps byte-identical code.
> It is order-*oblivious*, so it does not repair a deficient delta-span rank draw (§Hash) — it removes the sort's sensitivity to one.

**The merge.**
`merge2_into` fuses the two-stream merge with the segmented reduction: a two-pointer walk over id + rest, id-first on key ties, summing equal-key coefficients, dropping exact zeros, and applying the policy's `keep_term` to the fully summed coefficient.
Exact-zero coefficient rows (a θ = π/2 rotation emits `cos·c = ±0.0` id rows) flow through to the accumulator — the only zero test is on the final sum, the **signed-zero contract**, pinned by test.
The walk is written as three loops — both streams live, then one drain each — so the hot loop tests neither stream's bound and a channel with no identity delta runs a single-stream reduction with no `a`-side test at all.
A segment-copy variant and a branchless comparison were both measured and rejected (`research/FINDINGS.md`); the rejections are recorded on the function's doc.

After the coset loop the handles are un-permuted, the length recounted, and invariants asserted (debug builds).

## Parallelism

One coset per Rayon task.
By construction (§Bucketing) a task reads and writes only its own coset's buckets, so there are no atomics, no locks, no concurrent maps, and no synchronization inside a layer at all — only the layer boundary.
Load balance comes from the random hash (uniform bucket loads) plus the bucket floor (§Bucket-Policy), with Rayon work-stealing absorbing residual variation; a layer parallelizes once it has at least `MIN_COSETS_FOR_PARALLEL` cosets.

Work-stealing is a measured choice, not a default: static coset→worker assignment ran materially slower, because stragglers with no stealing cost more than page locality recovers (`research/FINDINGS.md`).

Partitions (§Partitioning) are the outer level of the same decomposition: the split across NUMA domains or MPI ranks is static, and stealing runs unchanged inside one.

## Partitioning

A partitioned run splits the sum across `P = 2^p` independent partitions — one NUMA domain in-process, one MPI rank distributed — by widening the bucket index.
A global bucket is the pair `(part(v), loc(v))`: `part(v) = P·v` from `p` designated **partition rows** (`PartitionRows<W>`, `P_MAX_BITS = 6`, so `P ≤ 64`), and `loc(v) = H·v` from an unchanged `Gf2Hash`.
**A partition holds the terms with `part(v) = rank` and nothing else**, so a key lives on exactly one partition and duplicates can no more straddle partitions than buckets (§Bucketing).

The partition rows are a separate matrix, not a prefix of `H`: `H`'s active rows grow with the term count, and a row that moved would change a term's owner mid-run.
`from_seed` draws them from a salted seed so they are independent of the refinement stream at every bucket count; `from_rows` is the hook for choosing them deliberately.
`is_independent_of(hash)` checks at scatter that the joint row set has full rank, so the global bucket carries `p + b` bits of entropy rather than `max(p, b)`; dependence costs load balance, not correctness.

**The classification is per layer, not per term.**
Both maps are GF(2)-linear, so a prepared channel's key delta `d` moves every term by the same partition delta `pd = part(d)` and bucket delta `bd = h(d)`.
A delta with `pd = 0` is **local** — the ordinary coset loop handles it with no communication.
A delta with `pd ≠ 0` is **remote**: every row it produces from local bucket `β` belongs to partition `R ⊕ pd`, bucket `β ⊕ bd`, one partner and one offset known before a term is touched.
The identity delta has mask `0`, so a partition never ships to itself, and remoteness is a property of the mask alone, so every partition reaches the same verdict without a vote — which is what lets the transport pair calls positionally.

**The wire unit is one CSR block per remote delta, indexed by the receiver's destination position.**
`Gf2Span::perm_index` renumbers the bucket index so a receiver's coset occupies a contiguous run of *positions* (§Engine); both sides can compute that renumbering, so the **sender** lays the block out in it: segment `p` holds the rows for the receiver's position `p`, generated from the sender's own bucket `bucket_at(p) ⊕ bd`, and the receiver filling output bucket `β′` reads `segment(position_of(β′))` through a table.
A coset's rows are then **contiguous**, so a prefix of the transfer is a whole unit of the receiver's work — what the pipelined receive waits on (`ChunkMap` carries both the order and the chunk edges).
A `PartnerPayload` is that partner's blocks in ascending remote-delta index, walked in lockstep with the receiver's own plan, so a delta with no rows still ships its empty block.

A layer is **export → exchange → local coset loop**, a push model, and the last two overlap:

1. Two passes over the local buckets build one block per remote delta — count rows per (delta, source bucket), then fill each block's CSR segments in destination-position order.
   The row arithmetic is the engine's own gather at row granularity (`DeltaEntry::emit`, `RotationPrep::emit_gen`), so an exported row is bitwise the row a local gather would have produced.
2. One all-to-all `Transport::exchange_layer`, which is **two-phase**: the *early* parts (block headers and CSR offsets) are waited out before the caller's body runs, while the *bulk* parts (key and coefficient columns) are cut at the chunk edges, posted chunk-major, and still in flight while the coset loop runs inside the call.
   `ExtraRows::count` needs only the offsets, so a gather run can be sized before a row has landed.
3. The bucketed coset loop (§Engine) over the *local* deltas only, with the received rows entering each output bucket's gather run through `ExtraRows`.
   A task calls `ChunkWait::wait_chunk` at the top of `append_into` — once per task, a whole coset being inside one chunk — and blocks only if its own chunk has not landed.

`exchange_layer` is the transport trait's one **required** method; a transport with nothing to overlap completes the transfer first and hands the body a no-op `ChunkWait`, which is what `InProcessTransport` does, and the blocking `Transport::exchange` is the provided method of that shape.

**The pipeline's mutual exclusion is the MPI thread level, not a lock on the data.**
Rayon workers reach `wait_chunk` together; whichever takes the pipeline's mutex is inside `MPI_Waitsome` over *every* outstanding request of the rank, driving its own receives and its partner's rendezvous at once, and the others back off on the per-chunk counters it publishes — spin, then yield, then short sleeps, so a waiter never steals the cores the copy runs on.
One thread inside MPI at a time is exactly the `MPI_THREAD_SERIALIZED` the transport requires, and it cannot deadlock: every send of a call is posted before the call's first receive, a waiting rank services its partner, and a rank whose coset loop never asks for a chunk still reaches the closing wait.

**The exchange's memory is one layer's traffic, reused across layers.**
A partition holds one export volume of send blocks plus one of receive payloads, both from a per-partition pool and both grow-only (`header.rows`, never `x.len()`, says how much of a column is live), so a steady-state layer neither allocates nor zeroes its megabytes again.

**Received rows join the rest stream, never the id stream.**
The rest stream is sorted anyway, so a received row may duplicate a local key and `merge2_into` still sees the key's complete sum before `keep_term` runs (§Truncation).

The coset loop runs against a **retained delta table** — `retain_entries` rebuilds `LocalPtm` with the remote entries dropped — plus `LayerKnobs` carrying the local bucket deltas and the rest-stream count.
Retention has one sharp edge the engine guards: a channel whose every non-identity delta is remote leaves an **identity-only table, for which `is_key_preserving` is true**, and the key-preserving `rescale_in_place` fast path would silently drop every received row, so that fast path is gated on the absence of extra rows.
A rotation with a remote generator keeps its `Prepared` and switches the generator pass off instead, exporting its anticommuting rows.

**The bucket count is agreed, on a schedule.**
Each partition proposes `desired_bits` for its own share, the group takes the maximum, and each refines to it; per-partition rather than global, so `P` partitions of `n/P` terms carry the same *total* bucket count as one partition of `n`, and the grow-only rule (§Bucket-Policy) survives because a maximum of grow-only proposals is itself monotone.
Agreement happens on any layer whose plan has a remote delta (both sides index the exchange's blocks by the count, so they must agree first) and otherwise every `BITS_AGREE_EVERY = 16` layers, preceded by an opening ramp of the same length.
Between agreements a partition keeps the count it has even when its own `desired_bits` is higher — nobody refines off-schedule, so the counts stay equal by construction.
The schedule must be computable identically by every partition without communicating, which is why it is a function of the layer index and the plan and why a "my own share grew" trigger does not qualify.
`P = 1` has no group, so it skips the reduction and refines every layer, which is what keeps it bit for bit `propagate`.
**A layer with no remote delta makes no transport call at all**, and with the bits agreement off its schedule it makes no call of any kind.

Layer finalization is collective, so the policy bound is `PartitionedTruncation`, and `finalize_layer_partitioned` runs on every layer on every partition for a policy whose `finalizes_layer` is true — a collective is well defined only if nobody skips it, and `finalizes_layer` is a property of the policy *type*, so the group cannot split on it.
A policy with no layer pass costs nothing per layer; one that has a collective form must report `finalizes_layer`.
`ApproxTopN` is **partition-exact**: the global octave histogram is the sum of the per-partition histograms, so one all-reduce has every partition choose the same edge and the union of the retained sets is the single-partition answer (§Truncation) — at the price of one collective per layer whatever the partition rows do.
`And` runs both sides; `Or` runs neither, because its unpartitioned `finalize_layer` is the trait's no-op default rather than either child's, and the two must agree.
Exact `TopN` is a distributed `k`-th selection, not a sum, and is **rejected at compile time** by the trait bound rather than approximated.

**The runtime is one pinned Rayon pool per partition, with work-stealing inside a partition only.**
The split is static at the outer level because first touch needs a stable domain-level split, and stealing is untouched at the inner one because it is what beats a static assignment (§Parallelism).
Partition 0 drives on the calling thread and partitions `1..P` get scoped threads that pin themselves to their slot and enter their pool.
The transport group is built **per call** and moved into the partitions, so a partition that panics drops its endpoints and its partners fail naming its rank instead of blocking forever.
A layer runs inside `ThreadPool::install`, so the thread issuing a transport call is a pool worker — which fixes the MPI thread level at `SERIALIZED` rather than `FUNNELED`.

Scatter and gather bracket a run, not a layer: `filter_partition` runs on the owning partition's own pool, so every column is first-touched in the domain that will read it, and `merge_partitions` merges the disjoint runs back.
The **scatter bits rule** is `want.max(bits − pbits).min(bits)`: a partition sheds at most `log2 P` of the bits the whole sum arrived with, never going below what its own share wants, so the bucket count summed over partitions equals the unpartitioned one.
At `P = 1` it is the identity and the round trip is bitwise.

`PartitionTrace` is the opt-in per-layer record (bucket bits, remote-delta count, collectives issued, terms in and out, rows and bytes sent `[from][to]`, rows received, imbalance), and under `phase-timing` the same run adds the export and exchange laps — the field list is contract (a) in `benchmarks/PROFILING.md`.
`chunk_wait_ns` is the part of the transfer the coset loop failed to hide, and is read together with `exchange_ns`, which the overlap makes small by construction.

What the split guarantees: **at `P = 1` the partitioned engine is `propagate`, bit for bit** — the scatter is the identity, the all-reduce is the identity, and the layer takes its unpartitioned branch.
Across partition counts the bar is floating-point tolerance (§Determinism); within a partition output stays byte-identical across pool sizes, because received rows are appended in the plan's fixed order.

**The cost model is locality.**
A random partition row set sends a nonzero delta across a boundary with probability `1 − 2^{-p}`, so roughly half of a dense two-qubit gate's deltas are remote at `P = 2`, and a rotation whose generator crosses exports one row per anticommuting term (`bytes ≈ rows × 48`).
Export volume is therefore a property of the row *draw*, not of the circuit alone: an exchange-free layer gains, an exporting layer pays, and a remote layer is transfer-bound (`research/HARDWARE.md`).
Traffic-minimizing rows are cut-like, reading the qubits on the boundary of a spatial cut, the extreme being a conserved quantity of the circuit, which produces no traffic at all; row tuning is open research.

### Transport composition

The engine's *partition* is not a thread and not a process: it is whatever a `Transport` says a peer is, and `run_layers` is one function generic over the transport, with `scatter_local`, `PartitionWork` and `apply_layer_partitioned` shared below it.

**Two drivers, because three things above the layer loop do not reconcile.**
`PartitionedSum` holds `P` partitions inside one process and fans out to them per call; `DistributedSum` *is* one partition, and its peers are other processes (`MpiTransport`, behind the off-by-default `mpi` feature — or the in-process transport, which is how the distributed shape is tested with no MPI in the picture).
They differ in the transport group's lifetime (per call, against one endpoint for the process's whole life, because an `MPI_Comm` is not something to duplicate per layer), in scatter and gather (one sum split locally and merged back bitwise, against a replicated input and a byte-framed gather to rank 0), and in the consistency check (one process cannot hand its own partitions different circuits, so only the distributed driver pays for it).

**Backend composition.**
Where a partition's terms live is a second axis, orthogonal to how its peers are reached: `run_layers` touches a partition's storage only through two crate-private traits, the policy-free `PartitionStorage` (`len`, `hash`, `refine`, `detach`, `stats`) and the layer itself, `PartitionBackend<W, T>: PartitionStorage` (`apply_layer`, `finalize_layer`), so it is generic over the backend exactly as it is over the transport.
Everything collective stays in the loop — the bucket-count schedule, the exchange decision from `PartitionPlan`, the counted policy finalization, the trace row — and a backend must issue exactly the transport calls the host layer issues, in the same order.
The loop keeps the `PartitionedTruncation` bound, so a backend cannot widen what a partitioned run accepts, and exact `TopN` stays a compile-time rejection.
`HostPartition` (a `PauliSum` plus its layer and export scratch) is the host backend; `PartitionedSum` holds `P` of them and `DistributedSum<W, X, B = HostPartition<W>>` holds one.
`DevicePartition` (the `cuda` feature) is the device backend and a full peer: its K10 export lays out the same CSR blocks in the receiver's position order, its fused layer reads a received entry's rows from segment `p` of the block exactly as a local entry's from bucket `bucket_at(p) ⊕ bd`, and a host partition and a device partition interoperate in one group.
An in-process device group exchanges device-resident payloads — the transport moves the block through the channel without reading its bytes, and the receiver adopts it with a device-to-device or peer copy into its own pooled columns — while the host wire format above stays in force for an MPI rank and for a group mixing host and device partitions.
**A device receive moves in chunks of destination positions.**
The receiver keeps every received row's CSR offsets on the device but only one chunk's rows: chunk `c` of `2^j` equal position ranges is copied (in-process) or received (NCCL) into the receive columns just before the fused layer's first batch in it, batches never straddle a chunk, and the fused layer finds a row at `base[k] + off[k][p]` with `base[k]` rebased per chunk in wrapping `u32` arithmetic.
`GpuLayerOptions::exchange_bytes` (`PAULISTRINGS_GPU_EXCHANGE_BYTES`) picks the fewest power-of-two chunks whose rows fit it, unbounded and so one chunk by default; a power of two because such a cut refines every coarser one, which is what lets an NCCL group agree on the largest count any rank asked for without growing anyone's chunk.
The send side is not chunked: its export volume stays resident until the last chunk moved.
A device group over a byte transport (`GpuDistributedSum`, feature `nccl`) agrees one exchange mode for the whole group at scatter, NCCL only when every rank can start it on a device of its own, and otherwise the host wire format, so the mode is group-uniform and such a group never contains a host partition.
Under NCCL a remote layer sends only the block headers and CSR offsets through `Transport::exchange`, then one `allreduce_sum_u64` vote on going ahead and on the chunk count, then, on a unanimous yes, one NCCL group per chunk, chunk-major, that moves the chunk's `x`/`z`/`coeff` columns straight into the receiver's receive columns, where the receiver computes the fingerprints; this vote is the one call beyond the host layer's, legal only because the mode is group-uniform.
After a yes every rank posts every chunk's group even if its own layer fails mid-way, discarding what arrives, so a local failure never strands a peer's receive; only a failed group itself stops the posting, and its peers' bounded waits then fail too.
A rank that cannot receive (a received segment past the tag, a failed allocation) or has already failed votes no, nobody posts, the ready ranks take every block as empty, and the after-loop agreement names the rank that voted no.
**A device sender merges one partner's rows by key before the exchange.**
Two remote deltas to one partner can emit one key only into one receiver position, so the merge is position-local: K3 runs over the partner's sub-table under the keep-everything program, which sums equal keys and drops an exact-zero sum but truncates nothing, since the receiver alone sees a key's complete sum.
The merged rows of a position are split over the partner's blocks greedily in entry order, never more than a block's unmerged count there, so the wire format, `ExtraRows` and the receive path are untouched and no received segment outgrows the tag.
It runs for a partner whose remote entries share an output support pattern (a Clifford's never do, a rotation has one remote entry), both payload forms and the MPI rank alike; `GpuLayerOptions::premerge` and `PAULISTRINGS_GPU_PREMERGE=off` switch it off.
The memory consequence is that a partition holds one export volume and one receive chunk on its device on top of its sum during a remote layer, where a host partition's equivalent volumes sit in system RAM.
A backend proposes the bucket count through `PartitionStorage::proposed_bits`, the host formula by default; the device raises it to its records-per-block target, with a factor of two of headroom in a group, because nobody refines off-schedule: a group member runs every layer at exactly the agreed count and reports `Unsupported` rather than refine when a block or a received segment exceeds the fused kernel's cap.
A partition whose layer fails before its exchange still makes the call, with one empty block per remote delta the plan names under the real chunk map, so its partners finish the run and the error surfaces after the loop; the device driver then refuses every later call on that split until it is scattered again.

**In-process: a moved payload, shared-memory collectives.**
`InProcessTransport` moves its payload through a `P × P` matrix of `mpsc` channels — there is nothing to encode and nothing to overlap — but the **collectives are shared atomics with a spin wait**: each rank numbers its own transport calls and publishes `(generation, kind)` plus its contribution into its own cache-line-padded slot, and a waiter spins, then yields, then sleeps briefly, with a dropped endpoint as the fail-fast signal.
The partition threads are pinned and dedicated for the whole call, so a spin wait rather than a futex is what makes an unconditional per-layer collective affordable.
The published `(generation, kind)` pair also makes a collective-order violation a panic naming both partitions, in every build, rather than a hang.

**`D = 1`: one rank per NUMA domain, no hybrid.**
A distributed rank's runtime holds exactly one partition, and its placement comes from the launcher rather than from the engine: `mpirun --map-by ppr:1:numa --bind-to numa` or `srun --cpu-bind=ldoms` leaves the process an affinity mask of one domain, and `Placement::Auto` over that mask resolves to a single slot covering it.

**The library never initializes MPI.**
`MPI_Init` is a process-global one-shot, so the application owns it: `MpiTransport::from_communicator` duplicates a communicator the caller already has, and `from_raw_handle` does the same for a foreign `MPI_Comm` (a Python host's, through `mpi4py`).
The duplicate is the engine's own, so its tags cannot collide with the application's traffic, and the `mpi` crate is re-exported as `paulistrings::mpi::rsmpi` so the caller builds its `Universe` from the version the library links.

**Point-to-point, not all-to-all-v.**
A remote delta moves rank `R`'s rows to `R ⊕ pd` and the same delta on `R ⊕ pd` moves its rows back, so the partner set is symmetric by construction: "who sends to me" *is* "who I send to", known from the rank's own plan with no discovery collective.

**The per-layer collective schedule** is unchanged from the in-process case, and an MPI implementation must not second-guess it: an `allreduce_max_u8` for the bucket count **on the schedule above**, then the layer's exchange **only if the plan has a remote delta**, then the policy's collective finalization if it has one.
On top of that a distributed propagation calls `check_consistency` exactly once, before its first layer: an all-reduce of a fingerprint of the run's shape (channel count, direction, bucket-policy knobs, qubit count, `W`), exact because it reduces the 64 per-bit counts and every count must be 0 or `size`.
Ranks handed different circuits then get a message instead of a deadlock two layers in.

**Wire framing, per partner, in three streams.**
A self-describing header — `u64[2 + n_parts]`, carrying the wire version, the source rank and one byte length per part — then the `Payload::early_parts` under their own tag, then `Payload::bulk_parts`, each column cut at the chunks' destination-position boundaries and posted **chunk-major**.
The header declares *every* part's length, early and bulk alike, which is what sizes the receiving columns; it is also the only message whose size the receiver cannot predict, so it arrives through a matched probe, and everything after it is a posted receive of known length.
MPI's non-overtaking guarantee for a `(source, tag, communicator)` triple then matches sends to receives in posting order, which is why both sides walk partners, parts and chunks in the same ascending order, deriving the chunk edges from the same CSR offsets and cutting at a constant chunk count rather than a thread count, since two ranks may run different pool widths.
Tags pack `epoch:11 | kind:4`, at most 32767 and so inside the guaranteed `MPI_TAG_UB`.
Everything is sent as bytes and chunked again at 1 GiB, MPI's counts being `i32` and a `u64` view of the parts unavailable (the block header is four `u32`s and the CSR `offsets` column a `Vec<u32>`, neither 8-aligned); raw host bytes on the wire means a run is homogeneous, same architecture and same `W` on every rank.
The declared part lengths are enough to size the receiving payload, so `Payload::recv_into` hands MPI mutable byte views of the very columns the coset loop will read and there is no decode pass; `finish_recv` then checks the header against the shape the lengths implied.

**Scatter and gather bracket a distributed run too, with a different contract.**
The input is *replicated* — every rank calls `scatter` with the same sum and keeps `filter_partition(rows, rank)`, the rows drawn from one seed so nobody has to agree by collective.
`local()` is always this rank's share, a valid `PauliSum` under the group's shared hash; `gather()` is collective and returns `Some` on rank 0 only, each rank shipping its bucket lengths and the three columns `to_arrays` concatenates and rank 0 rebuilding them before `merge_partitions`.
`PartitionTrace` stays per rank: `terms_in`, `terms_out` and `rows_received` have one entry, while `rows_sent` and `bytes_sent` are indexed by destination rank over the whole group.

## Determinism

The correctness bar for engine changes is **agreement to floating-point tolerance** (`assert_terms_close`), not bitwise equality.
Equal-key summation order is unspecified and free to change between versions, configurations, and optimizations; floating-point addition is not associative, so a different bucket count or hash seed may legitimately change output bits.

What *is* reproducible, as a property of the current implementation rather than a promise: at a fixed bucket count and hash seed, output is bitwise identical across thread counts and repeat runs, because cosets are write-disjoint and work within one is sequential.
The same holds per partition across pool sizes, and a `P = 1` partitioned run is bitwise `propagate`; across partition counts the bar is tolerance, as it is across bucket counts (§Partitioning).
Tests that pin exact output bits (the fingerprint net, the thread-count byte-identity tests) are **convenience tripwires** for unintended perturbation: when one trips under a change that is correct to tolerance, regenerate its literals or demote it to `assert_terms_close` in the same commit, with a one-line note.
Do not design, constrain, or reject an optimization to keep output bits stable.

## Truncation

Truncation is what keeps Pauli propagation tractable, and it is a composable extension surface:

```rust
pub trait TruncationPolicy<const W: usize>: Send + Sync {
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool { true }
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {}
}
```

The split is performance-critical: `keep_term` runs on every merged output — potentially billions of times — and must inline to nanoseconds; it sees the **summed** coefficient, inside the merge.
`finalize_layer` runs once per layer and may be non-local.

Built-ins: `CoefficientThreshold(eps)` and `WeightCutoff(k)` are per-term filters; `TopN(n)` and `ApproxTopN(n)` are layer finalizations.
Policies compose with `And` / `Or` (Python: `&` / `|`).

**Magnitudes are compared as `|c|²`, never as `|c|`.**
`Complex64::norm()` is `hypot`, a libm call, and `x ↦ x²` is strictly increasing on `[0, ∞)`, so `|c|² > t²` decides the same predicate.
The equivalence is exact for the *ordering* and near-exact for the *tie grouping*: a symmetry multiplet's members differ by a sign or a power of `i`, and `re² + im²` is bitwise invariant under both, so exact ties survive.
What squaring loses is the band below `|c| ≈ 1.57e-162`, whose squares underflow to `0.0` and therefore tie.

**`TopN` never splits a tie group.**
Terms with exactly equal magnitude are typically a symmetry multiplet, and truncation should commute with the symmetry, so the group at the threshold magnitude is kept only if it fits entirely within `n` and discarded whole otherwise.
Consequences, all deliberate: `TopN(n)` retains *at most* `n`; it retains exactly `n` when magnitudes are distinct; and a sum whose coefficients all share one magnitude is wiped to empty.
Implementation: gather squared magnitudes into a per-thread pooled buffer, select the `n`-th largest once globally, decide the tie group from the selection's own partition (the group fits iff nothing after the pivot equals the pivot), then filter each bucket in parallel — per-bucket filtering preserves within-bucket order automatically.

**`ApproxTopN(n)` trades the exact count for the selection.**
It histograms the octave of `|c|²` (the 11-bit `f64` exponent: 2048 bins, 8 KB, L1-resident), walks the bins down to the lowest edge whose cumulative count still fits in `n`, and retains against that edge — two `O(m)` passes, no candidate array, no selection.
It keeps `≤ n` (so the memory bound is exact) and `> n - p`, where `p` is the population of the coarsest octave that did not fit.
Tie groups need no rule here: equal magnitudes share an octave, so a multiplet is always kept or dropped whole — at the price of a wider degenerate case, a sum confined to a single octave of `|c|²` being wiped exactly as an all-tied sum is under `TopN`.
`TopN` remains the default and the choice whenever the retained count itself matters.

Under partitioning the two swap places: `ApproxTopN` is **partition-exact**, while exact `TopN` has no collective form and is rejected at compile time (§Partitioning).

## Channels

```rust
pub trait Channel<const W: usize>: Send + Sync {
    fn support(&self) -> [u64; W];            // bitmask of acted-on qubits
    fn max_fanout(&self) -> usize;            // outputs per input, upper bound
    fn apply(&self, x, z, coeff, out: &mut OutputBuffer<W>);
    fn apply_adjoint(&self, ...);             // default: self-adjoint
    fn prepare(&self, hash, adjoint) -> Option<Prepared<W>>;  // default: derive_local
}
```

Implementing `apply` (plus `support`) is the whole cost of a custom channel; `prepare`'s default derives the engine form automatically (§Prepared-Channels).
`max_fanout` is a method rather than an associated const so `Circuit` can store `Box<dyn Channel<W>>`; concrete impls return literals, so call sites through generics still constant-fold.

Built-ins: `Clifford1Q` / `Clifford2Q` (table-driven from their symplectic action), `PauliRotation` (`exp(-iθP/2)`, any generator weight; support derived from the generator, never caller-supplied), `GeneralUnitary1Q` / `GeneralUnitary2Q` (from a matrix or a Pauli-transfer matrix), and the noise channels `Depolarizing`, `Dephasing` (pure coefficient rescales) and `AmplitudeDamping` (genuine fanout 2).
`IdentityChannel` exists for tests and composition.

## Ingestion

`BuildAccumulator<W>` is a hashmap accumulator (`FxBuildHasher` — Pauli bitstrings are already high-entropy, so SipHash buys nothing) for unsorted input: Hamiltonian parsing, dict construction, custom analyses.
`finalize()` hashes, scatters, and sorts into a canonical `PauliSum`, choosing the bucket count by the standard policy so small sums come out single-bucket.
The accumulator is an ingestion path only — it never appears in the propagation loop.

## Python-Bindings

The Python package is a thin layer over enums (`PauliSumImpl`, `CircuitImpl`) holding the monomorphized widths (§Width); every method dispatches once and calls the same core code Rust users call.
Construction accepts dictionaries and `(string, coefficient)` pairs; bulk export returns NumPy arrays (`to_arrays`).
Expectation values against product states, overlaps, and the identity coefficient are computed in Rust.
Truncation factories return spec objects composed with `&` / `|` and translated to core policies at the boundary.
The extension module is `paulistrings._paulistrings` (abi3), and `python/paulistrings/` re-exports it.

The Python `Circuit` additionally keeps the width-erased `ChannelSpec` of every channel pushed, alongside the materialized `Circuit<W>`: the core stores prepared channels and cannot hand a gate description back out, so that list is what serves gate-list introspection (`Circuit.gates`, emitted in the frozen task-JSON gate vocabulary), slicing, concatenation, and `adjoint()`.
The two are appended to together and never diverge.
Everything those methods do is spec-rewriting outside any hot loop — no core code changes and no per-term work.

## GPU-Readiness

The design decisions a GPU backend needs are already in place: `PauliString` is `Pod` with a defined layout; bucket columns are SoA and flatten to device buffers in one pass; the coset decomposition maps to one block per coset with gather/sort/merge in shared memory, a better CUB fit than any global sort.
The extension to distributed memory is no longer forward-looking: §Partitioning is that exchange, and MPI is the same exchange over ranks.

**The device sum.**
`engine::gpu::GpuSum` holds the flat SoA columns `x`, `z`, `coeff` and a CSR `start`/`lens` per bucket under the same `Gf2Hash` as the host, plus a 64-bit GF(2)-linear fingerprint `g(v) = G·v` per term with `G` drawn from a salted seed, so `g(v ⊕ d) = g(v) ⊕ g(d)` is one XOR per delta.
Within a bucket the device keeps unique keys in no particular order; `to_host` re-sorts each bucket to the host's lexicographic order.

**The fused layer.**
One block per output position `p`, the coset-contiguous renumbering `Gf2Span::perm_index` of a bucket `β`.
For every entry `e` of the prepared table and every row `r` of source bucket `β ⊕ δ_e` that the entry emits (`amp_e[s] ≠ 0` on the table entry, never on the product), the block builds a record `(g_lo32, tag)` in shared memory with `tag = e:4 | r:12`.
A 16-bit index array is radix-sorted by `g_lo32`; adjacent equal-`g_lo32` records with different keys trigger eight more passes over `g_hi32`, and a pair still colliding a full lex-key sort, so equal keys always end adjacent.
A segmented sum over each equal-key run (a warp-shuffle block scan on dense tables, a head-serial walk on sparse ones) gives the coefficient; a row survives if the sum is not exactly zero and `keep_term` accepts it, and its key and coefficient are recomputed from the input at write time.
`CAP = 8192` records per block and the 12-bit offset caps a source bucket at 4096 rows; both are checked from the count table before the launch, and a violation refines the bucket count by one bit and recounts, up to `max_bits`, on a lone partition; a device partition of a group runs at the agreed count and reports `Unsupported` instead (§Partitioning).
A device with less opt-in shared memory loads fewer block variants and runs under a lower cap.
Rows land in a loose arena sized by the exact pre-dedup counts, batched over contiguous position ranges so the arena stays under `arena_bytes`, then compact into the output columns at running offsets; input and output columns ping-pong between layers.
The block width is a per-`W` constant (`THREADS`): 1024 threads at `W ≤ 2`, 512 at `W = 4`, 256 above, register-bound.
The kernels are named K1 count table `cnt[β][e]`, K2 segment sizes and scan, K3 the fused layer, K4 compaction, K5 the key-preserving rescale (an identity-only table, gated exactly like the host's `rescale_in_place`), K6 refine, K7 the octave histogram of `ApproxTopN`, K8 exact `TopN`'s radix-select over the bit pattern of `|c|²` (one device only; `DevicePartition::finalize_layer` reports `Unsupported` above one partition, since the `n`-th largest of a split sum has no collective form), and K11 the device invariant check.

**Bucket policy on device.**
The target is records per block rather than terms per bucket: the bucket count is the smallest `2^b` with `fanout × terms ≤ 4096 × 2^b`, where `fanout` is the number of table entries with any nonzero amplitude, never below the current count, and capped at `B_MAX_BITS`.

**Errors.**
Every device operation returns `GpuError`; the layer loop's seam is infallible, so `DevicePartition` records the first error, skips every later layer, and the driver returns it after the loop with the sum holding the last completed layer's output.
`DevicePartition::refine` advances only a host mirror of the hash; the layer refines the device to the mirror and the bucket policy in one pass, and the mirror is re-synced to the device whenever the driver reads the error.

## Performance-Model

Gather and merge dominate a layer; the sort only matters for dense two-qubit unitaries.
Idle is single digits — load balance is a solved problem, and the cost that grows with thread count is per-row time under memory contention, not imbalance.
The workload is bandwidth-bound at high thread counts, so further wins come from traffic reduction or partitioning rather than from scheduling; the measured ceilings and the roofline denominators are in `research/HARDWARE.md`.

Binding constraints for optimization work — do not rediscover these the hard way:

1. **The determinism policy** (§Determinism): tolerance, not bits, is the bar; byte-exact tests are tripwires to regenerate, never constraints.
2. **The signed-zero contract** (§Engine): exact-zero id rows flow to the accumulator; the only zero test is on the final sum.
3. **The stable adaptive sort** (§Engine): the per-run sort exploits piecewise-sortedness, and the constraint binds wherever a coset gathers ≥ 2 streams.
   It carries no `#[inline]` constraint on `engine/merge.rs`.
4. **Measurement discipline:** release builds only, seeded inputs outside the timed region, and the campaign workflow in `benchmarks/PROFILING.md`.
   Campaign noise exceeds most effects, so anything small needs the interleaved A/B protocol rather than one campaign per build.

Read `research/FINDINGS.md` before re-attempting an optimization: the rejected ideas are recorded there with the evidence that rejected them.

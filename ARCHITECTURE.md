# Architecture

This document is the design reference for `paulistrings-rs`. Code comments cite
it by section name (`ARCHITECTURE.md §Engine`); the section headings below are
therefore a stable anchor vocabulary — do not rename them without sweeping the
citations in `crates/` and `python/`.

Measurements quoted here were taken on the reference host (2× Xeon Gold 6244,
16 cores / 32 threads, 2 NUMA nodes); see `benchmarks/PROFILING.md` for the
measurement methodology and `research/notes/` for the underlying data.

## Overview

The library implements **Pauli propagation**: classical simulation of quantum
systems by evolving operators in the Pauli basis under gates and noise
channels. A weighted sum of Pauli strings is pushed through a circuit layer by
layer — forward, or in the Heisenberg picture by applying adjoints in reverse —
with truncation keeping the sum tractable. This serves observable
backpropagation, density-matrix-style forward evolution, and hybrid uses such
as operator backpropagation for error mitigation.

Four design pillars, in priority order:

1. **Correctness of the core algebra.** Pauli string manipulation is the
   foundation; bugs at this level invalidate everything downstream.
2. **Performance at scale** — sums of 10⁶–10⁸ terms. Memory layout, cache
   behavior, and parallelism are first-class concerns.
3. **Extensibility for research.** Custom channels and custom truncation
   strategies are implementable without forking the library.
4. **GPU-readiness.** A future GPU backend must be addable without
   restructuring the core data types or algorithms.

Non-goals: the library is not a state-vector, tensor-network, stabilizer, or
matrix-product-state simulator, and it is not a quantum SDK — no transpilation,
no hardware control. It expects circuits from upstream tooling.

## Data-Model

**`PauliString<const W: usize>`** uses the symplectic encoding: each qubit's
Pauli is a bit pair with `I = (0,0)`, `X = (1,0)`, `Z = (0,1)`, `Y = (1,1)`,
stored as `x: [u64; W]`, `z: [u64; W]`. One word covers 64 qubits. The type is
`Copy + Pod + Zeroable` and `#[repr(C)]` with no padding — `16·W` bytes,
directly serializable and GPU-uploadable.

Multiplication is bitwise XOR of the `(x, z)` parts plus a phase `i^k`;
`mul_assign` returns `k` as a `u8` in `0..4` and stores no phase. Callers fold
the phase into a `Complex64` coefficient at the boundary — the moment a string
enters a `PauliSum` or `BuildAccumulator`. Storing phase per string would cost
a byte plus padding for a value that is zero everywhere it would be read in
bulk.

The load-bearing trait is **`Ord`** (lexicographic over the concatenated
`(x, z)` words), not `Hash`: the engine is sort- and partition-based. `Hash`
exists for the ingestion path only (§Ingestion).

**`PauliSum<const W: usize>`** is a bucketed structure-of-arrays: per-bucket
column triples (`Vec<[u64; W]>` for `x` and `z`, `Vec<Complex64>` for
coefficients) partitioned by a GF(2)-linear hash (§Hash), plus `num_qubits`
and a cached length.

> **Invariant:** every term lives in `buckets[h(term)]`; within each bucket
> keys are strictly ascending in lex `(x, z)` order with no duplicates. The
> canonical order — promised publicly — is bucket index, then key.

A single-bucket sum is automatically in plain lex order, because `h(v)` is
constant over it; sums below the parallelism threshold (§Bucket-Policy) have
one bucket, so small sums present the familiar globally-sorted order.

Per-bucket owned columns, rather than one flat SoA plus offsets, let every
bucket retain its capacity across layers — the steady state of a propagation
loop allocates nothing. SoA keeps coefficient-only scans (truncation,
expectation values) and key-only scans (weight, commutation) cache-friendly,
and each column maps directly to a GPU device buffer (§GPU-Readiness).

## Width

`W` is a const generic: monomorphization eliminates indirection, fully unrolls
the bit operations, and keeps `PauliString` `Copy`. Python supplies
`num_qubits` at runtime, so the binding layer instantiates a fixed width set
`{1, 2, 4, 8, 16}` (64–1024 qubits) and dispatches once, outside any hot loop,
via an enum over the instantiations (§Python-Bindings). This trades binary
size for speed. Rust users call the core crate with any `W` they like.

## Bucketing

The engine's central idea: **stop maintaining one global sorted order** and
partition the sum by a GF(2)-linear hash instead. The partition is persistent
across layers, commutes with channel action in a way that makes output buckets
statically predictable, and makes deduplication bucket-local — so there is no
global sort anywhere in the propagation loop.

**Keys form a vector space.** Under the symplectic encoding a key is
`v = (x, z) ∈ GF(2)^{2n}` and Pauli multiplication is `⊕` (XOR); the phase
lives outside the key entirely.

**The bucket function.** Fix `H ∈ GF(2)^{b × 2n}` and define `h(v) = H·v`,
giving `B = 2^b` buckets. Linearity yields the property everything else
follows from:

```
h(v ⊕ d) = h(v) ⊕ h(d)
```

**Channels act by a bounded delta set.** A channel with support `S`, `|S| = k`,
maps an input key to outputs differing only inside the `2k` support
coordinates: `v_out = v ⊕ d` with `d` drawn from a small set
`D ⊆ GF(2)^{2k}` — the channel's **delta set**. For the built-ins, `dim D` is
0 for key-preserving channels (identity, depolarizing, dephasing, Pauli
gates), 1 for `H`/`S`/amplitude damping and for a Pauli rotation of **any
generator weight** (the delta is the fixed generator, so a weight-`w` rotation
needs 2 buckets, not `4^w`), 2 for `CNOT`/`CZ`/`SWAP`, and bounded by `2k` for
general unitaries — where `D` is the *realized* set `{s ⊕ t : amp[s][t] ≠ 0}`,
so a sparse unitary (a `T` gate mixes only `X` with `Y`) reads fewer buckets
than the bound.

**Bucket prediction, and its inverse.** Combining the two facts:

```
forward:   h(v_out) ∈ h(v_in) ⊕ h(D)
inverse:   inputs contributing to output bucket β′ live in β′ ⊕ h(D)
```

`h(D)` spans a subspace of dimension `r = rank(H|_D) ≤ dim D`, so each output
bucket reads an affine set of exactly `2^r` input buckets — at most 2 for
rotations, 4 for two-qubit Cliffords, 16 for a dense two-qubit unitary — and
writes nowhere else. **Output buckets are write-disjoint**, which is the
load-bearing structural fact behind the parallel decomposition
(§Parallelism).

**Dedup is bucket-local.** `h` is a function, so equal keys land in the same
bucket — duplicates can never straddle buckets. Deduplication therefore only
ever needs a canonical order *within* a bucket, and every bucket is a small,
cache-resident sum.

**The per-(input, output) delta is a constant.** Filling output bucket `β′`
from input bucket `β = β′ ⊕ δ` uses the `d ∈ D` with `H·d = δ`; when
`rank(H|_D) = dim D` (the overwhelmingly common case for a random `H`) that
`d` is unique and term-independent. The inner loop is: extract the ≤ `2k`
support bits, one table lookup (phase already folded in), skip if the
amplitude is zero, XOR with a precomputed full-width mask, one complex
multiply. No dynamic dispatch, no trig, no phase arithmetic. When
`rank(H|_D) < dim D`, several `d` share a `δ` and are iterated as a short
member list — correctness never depends on `H` being well-chosen, only
performance does.

**Refinement is one parity pass.** `H`'s active rows are a prefix of a fixed
seeded matrix, so `h_{b+1}(v) = (h_b(v), row_{b+1}·v)`: doubling `B` splits
each bucket in two with within-bucket order inherited — an `O(n)`
single-row-parity pass, no re-sorting. Halving merges bucket pairs with a
two-way merge. This incremental rehash is what makes a *persistent* partition
viable while `n` swings by orders of magnitude across a run.

## Hash

`Gf2Hash<W>` stores `b_max` rows as `(rows_x, rows_z)` word masks, an active
prefix length `b`, and the seed that generated the rows (a xorshift64
construction — reproducible with no added dependency). `bucket_of(x, z)` sets
result bit `i` to `parity(x & rows_x[i]) ^ parity(z & rows_z[i])`;
`row_parity` evaluates a single row for the refinement pass, making refine
`O(n)` rather than `O(n·b)`. Columns beyond `2·num_qubits` are masked to zero
at construction. The hash is stored with the sum; two sums combine only if
they share it. `PartitionRows<W>` holds additional rows of the same kind,
drawn from a salted seed so they are independent of this prefix at every
bucket count (§Partitioning).

**Why dense and random.** A coordinate projection (bucket = chosen key bits)
is also GF(2)-linear, but weight-based truncation keeps sums low-weight, so
chosen coordinates are almost always zero and everything lands in bucket 0 —
load balance collapses exactly on the workloads that matter. A dense random
`H` is a universal hash family on the key space: maximum bucket load is
`m/B + O(√(m log B / B))` with high probability *independent of input
structure*, and `rank(H|_D) = dim D` holds with probability `≥ 1 − 2^{dim D − b}`.
The `b × 2W` popcount cost per term is paid only at ingestion and rehash,
never in the layer loop. Known wart: `h(0) = 0`, so the identity string always
sits in bucket 0 — one term, ignored.

## Bucket-Policy

The bucket count targets `DEFAULT_TARGET_BUCKET_LEN = 1024` terms per bucket —
a `W = 2` term is 48 B, so ~48 KB per bucket sits comfortably in a 1 MiB L2
alongside its scratch. A sweep on a rotation layer at 10⁶ terms confirms the
optimum is at this value, flat within 15% over roughly 250–4000 terms per
bucket and sharply worse outside (64× larger buckets cost 4.5×, the per-bucket
sort reasserting itself; 16× smaller costs 1.5× in fixed overhead).

The floor is the fixed `DEFAULT_MIN_BUCKETS = 128` — deliberately **not**
derived from the thread count, so the partition is a deterministic function of
the sum alone, not of the machine; 128 gives Rayon slack to load-balance at
any realistic core count. A sum only leaves the single-bucket regime above
`DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK` (= 8192) terms: below that,
parallelism has nothing to win, and one bucket keeps the plain lex order
(§Data-Model). Under partitioning the floor applies per partition, so `P`
partitions carry `P × DEFAULT_MIN_BUCKETS` buckets between them
(§Partitioning).

That floor has a **cost the sweep above does not see, because the sweep is a
rotation layer**: the bucket count also fixes the engine's coset dimension
`r = min(rank(h(D)), bits)` (§Engine), and the per-run sort's comparison count
collapses to its `log2(fanout)` floor only at *full* delta rank — for a
two-qubit channel, `r = 4`. Below 8192 terms `bits ≤ 3`, so a dense-PTM layer
cannot reach it and its sort costs up to 2.2× its asymptote. A rotation or
Clifford layer does not care (its sort is 7% of the layer); a dense 16×16 PTM's
sort is 58–60% of it. Two further consequences: `rank(h(D)) < 4` also happens
by *draw* — ~10% of two-qubit placements at `B = 128` — costing the same
1.7–1.9× at any term count; and the sum's own steady state is what makes this
bite, since a repeated dense-PTM layer closes to "every off-support pattern ×
all 16 local patterns", exactly the key set a short rank collides into one
bucket. Full mechanism, evidence and the tuning gate:
`research/notes/2026-09-01-bucket-cliff.md`.

`rebucket` is **grow-only**: `B` is the running maximum of the desired bucket
count over the sum's history, and only an explicit `with_hash` shrinks it.
Growing on every upward crossing but never coarsening avoids the oscillation
failure mode — a sum whose size swings across a power-of-two boundary on
alternate layers would otherwise refine and coarsen at `O(n)` each layer, and
this serial cost measured as the dominant share of wall time on
rebucket-heavy workloads. A hysteresis band was tried instead and measured
actively harmful (~10%): it parks the steady state up to 4× above the
per-bucket target, on the wrong side of the sweep above. Refine and coarsen
parallelize per bucket (pair) above the same 8192-term threshold.

`PropagateOptions::{target_bucket_len, min_buckets}` expose both values per
call. They are a **measurement lever, not a tuning parameter**: the defaults
are the optimum above, and the only reason to move them is to measure what a
coarser or finer partition costs. Both have to move together — above the
floor, `desired_bits` clamps the count at `min_buckets` whatever the target
asks for — and `min_buckets` must stay `>= 16` or the "worth splitting" gate
goes non-monotone. `rebucket` being grow-only, lowering either mid-run never
coarsens a partition already grown. The small-sum direct path
(`engine::direct`) still sizes its partition from the defaults; there is
nothing to measure at small `n`. Pinned by
`crates/paulistrings/tests/bucket_knob.rs`.

## Prepared-Channels

Applying a channel through its trait object once per term would pay a vtable
call, re-derived tables, and trig per term. Instead the engine **prepares** a
channel once per layer into one of two forms:

```rust
pub enum Prepared<const W: usize> {
    Local(LocalPtm<W>),      // support on ≤ MAX_LOCAL_SUPPORT qubits
    Rotation(RotationPrep<W>), // exp(-iθP/2), any generator weight
}
```

`LocalPtm` is the channel's local Pauli-transfer matrix over its support: a
list of `DeltaEntry`s, each carrying the bucket delta `δ = H·d`, the delta in
local support coordinates, full-width XOR masks, and an amplitude per input
support pattern (`amp[s]` takes pattern `s` to `s ⊕ d`; exact zero means "no
output"). The `i^k` phase is folded into `amp` at prepare time.
`MAX_LOCAL_SUPPORT = 2` bounds the dense table at `16 × 16` amplitudes — 4 KB
per layer; a support-3 table would be 64 KB and every entry would inline a
1 KB amplitude row, which is why wider supports take a different route (below).

`Channel::prepare` has a **default implementation that is automatic and
complete for any channel with support on ≤ 2 qubits**: `derive_local` calls
the channel's own `apply` on each of the ≤ 16 local basis Paulis and reads the
PTM off the results. A custom channel that implements `apply` gets the
bucketed engine for free, and the derivation doubles as a cross-check between
the two representations. `PauliRotation` overrides `prepare` and returns
`Prepared::Rotation` at any generator weight — its delta set is `{0, gen}`
regardless of weight, with the amplitude computed per term from commutation
with the generator.

**Soundness precondition:** `derive_local` is correct exactly when the channel
honors the bounded-support contract — output amplitudes may depend on the
input only through its support bits. This is a documented trait requirement,
pinned by a property test comparing each derived table against `apply` on
randomized full-width inputs.

**Identity-stream density.** Every built-in's delta set contains the identity
delta. Preparation classifies it as **dense** — amplitude nonzero on every
active support pattern (all rotations, general unitaries, amplitude damping) —
or **sparse** (Cliffords: `CNOT` keeps 4 of 16 patterns, `H` 2 of 4). The
engine exploits density to avoid materializing identity-stream keys at all
(§Engine).

**Declined preparation is an error.** A channel whose support exceeds
`MAX_LOCAL_SUPPORT` without overriding `prepare`, or one that writes outside
its declared support, makes `propagate` panic with a message naming the layer
and the reason. No built-in can reach this: everything ships as `Local` at
`k ≤ 2` or as `Rotation`. The documented extension path for genuinely wide
custom channels is a heap-backed `LocalPtm` variant (design note in
`research/notes/`); composing from 1- and 2-qubit channels covers the rest.

Channel fanout (`max_fanout`) sizes the `OutputBuffer` for direct `apply`
calls — the probe in `derive_local`, the test oracle, user code — and is not
an engine concern: the gather emits at most one output per (term, delta
entry), sized exactly from bucket lengths before any work begins.

## Engine

`propagate` (and `propagate_with_scratch`, which it wraps) iterates the
circuit's channels — in order for forward propagation, in reverse with
adjoints for Heisenberg — and per layer runs:

```
rebucket → prepare → apply layer over cosets → policy.finalize_layer
```

Key-preserving channels (identity delta only: depolarizing, dephasing, Pauli
gates) bypass the whole pipeline via `rescale_in_place` — a parallel
coefficient scan that touches no keys.

**The unit of work is a coset.** The engine works with the span of `h(D)`
(`Gf2Span`) — the span rather than `h(D)` itself because a custom channel's
delta set need not be XOR-closed. Cosets of the span partition the bucket
index space, and every output bucket in a coset reads only input buckets in
that same coset: a coset is a closed task. Bucket *handles* are permuted into
coset-contiguous order once per layer (two `O(B)` handle moves bracket the
layer), then each coset task, independently:

1. **Swap** its `2^r` bucket columns into worker-persistent scratch, leaving
   empty, capacity-retaining columns as write destinations — the layer is
   in-place: peak memory is `n` plus per-worker scratch of one coset's working
   set, not a second full-size copy.
2. **Size** each per-member gather run exactly from the swapped-out lengths.
3. **Gather input-major**: each term is loaded once and its whole fanout
   scattered to runs via the O(1) index identity
   `member(i) ⊕ δ = member(i ⊕ coord(δ))` — so the gather visits each input
   term exactly once, with no read amplification. (An output-major variant
   guards rank ≥ 3 custom channels, selected by `GATHER_OUTPUT_MAJOR_MIN_R`;
   no built-in reaches it.)
4. Per run, **sort the rest stream and merge**, straight into the member's
   live slot.

**Split streams.** A gather run keeps the identity-delta stream separate from
the rest. Identity rows keep their keys, so the id stream inherits the source
bucket's strictly-ascending unique order and is **never sorted**; only the
rest stream is. When the identity amplitude is dense (§Prepared-Channels) the
id stream is 1:1 with the source bucket, so the gather materializes only the
16-byte coefficients and the merge borrows the key columns from the source
bucket in place — id keys are neither written nor re-read. Sparse identity
streams materialize pre-filtered keys and coefficients.

**The sort.** Two kernels live in `engine/merge.rs` alongside their shared
worker-persistent `SortScratch`, and the layer picks between them *once*, from
its plan's realized rest-delta count — so the choice costs nothing per run.
Both satisfy one contract and nothing more: the output is ascending in lex
`(x, z)` with duplicates allowed, and is a permutation of the input triples, so
they are interchangeable to floating-point tolerance (§Determinism) and never
bitwise. `merge::tests::assert_sort_contract` holds both to it.

`sort_rows_with_scratch` — the default, and the only kernel a sparse-PTM layer
ever sees — is a permutation sort over the run.

> Its comparison sort **must remain the standard library's stable adaptive
> `sort_by`**. A gather run is a concatenation of per-delta streams, each
> drawn from one sorted bucket — piecewise-sorted data whose natural runs the
> adaptive driftsort detects and merges nearly for free. Switching to
> `sort_unstable_by` (pdqsort, no run detection) measured **+77%** on a
> rotation layer. Stability per se is irrelevant; adaptivity is the point.
> Recorded on the function's doc — do not "simplify" it.

`sort_rows_radix_with_scratch` serves the dense-PTM path, where the sort is
**58–60% of layer wall time** and the run arrives as ~15 ascending blocks with
~15-fold duplicate keys. Adaptivity already puts that at `log₂ 15 + 1 ≈ 4.9`
comparisons per row — the information-theoretic floor — so the win is not in
the comparison *count* but in what one costs: each is a dependent indexed load
through the permutation into a 100–400 KiB key column. The radix kernel finds
the most significant key word the rows actually disagree on, extracts an
order-faithful 16-bit surrogate from it (every row shares the bits above, so
the shifted masked window is monotone in the key), sorts
`(surrogate, row index)` records with two 8-bit passes of sequential reads, and
orders the residual ties on the full key at ~1 comparison per row. Runs whose
keys are all equal return immediately; runs whose window cannot discriminate
delegate to the comparison kernel.

> This kernel is **selected, never a replacement**: on a single nearly-sorted
> stream it measured **+130–165%**. `RADIX_MIN_REST_STREAMS` gates it at 8, so
> today only a dense two-qubit PTM reaches it and every rotation/Clifford layer
> keeps byte-identical code. It is also order-*oblivious*, so it does not
> repair a deficient delta-span rank draw (§Hash) — it removes the sort's
> sensitivity to one. Measurements, the `W = 1` comparator diagnosis it also
> resolves, and the unmeasured 2–7-stream gap in
> `research/notes/2026-09-01-sort-kernel.md`.

**The merge.** `merge2_into` fuses the two-stream merge with the segmented
reduction: a two-pointer walk over id + rest, id-first on key ties, summing
equal-key coefficients, dropping exact zeros, and applying the policy's
`keep_term` to the fully summed coefficient. Exact-zero coefficient rows
(a θ = π/2 rotation emits `cos·c = ±0.0` id rows) flow through to the
accumulator — the only zero test is on the final sum (the signed-zero
contract, pinned by test). A segment-copy variant (gallop + bulk copy of
id segments) was measured and rejected: real stream densities make the
average segment 1–2 rows, and it cost +20–35% merge time — recorded on the
function's doc.

After the coset loop the handles are un-permuted, the length recounted, and
invariants asserted (debug builds).

## Parallelism

One coset per Rayon task. By construction (§Bucketing) a task reads and
writes only its own coset's buckets, so there are no atomics, no locks, no
concurrent maps, and no cross-thread reconciliation — no synchronization
inside a layer at all, only the layer boundary. Load balance comes from the
random hash (uniform bucket loads) plus the bucket floor (§Bucket-Policy),
with Rayon work-stealing absorbing residual variation; a layer parallelizes
once it has at least `MIN_COSETS_FOR_PARALLEL` cosets.

Work-stealing is a measured choice, not a default: static coset→worker
assignment (a NUMA-affinity experiment) ran 1.25–1.9× *slower* — stragglers
with no stealing cost far more than page locality recovers. See the
static-coset-placement negative-result note in `research/notes/` before
re-attempting placement work.

Partitions (§Partitioning) are the outer level of the same decomposition: the
split across NUMA domains is static, and stealing runs unchanged inside one.

## Partitioning

A partitioned run splits the sum across `P = 2^p` independent partitions — one
NUMA domain today, one MPI rank later — by widening the bucket index. A global
bucket is the pair `(part(v), loc(v))`: `part(v) = P·v` from `p` designated
**partition rows** (`PartitionRows<W>`, `P_MAX_BITS = 4`, so `P ≤ 16`), and
`loc(v) = H·v` from an unchanged `Gf2Hash`. **A partition holds the terms with
`part(v) = rank` and nothing else**, so a key lives on exactly one partition
and duplicates can no more straddle partitions than they can straddle buckets
(§Bucketing).

The partition rows are a separate matrix, not a prefix of `H`. `H`'s active
rows grow and shrink with the term count (§Bucketing), and a row that moved
would change a term's owner mid-run. `from_seed` draws them from a salted seed
so they are independent of the refinement stream at every bucket count;
`from_rows` is the hook for choosing them deliberately.
`is_independent_of(hash)` checks that the joint row set has full rank — the
global bucket then carries `p + b` bits of entropy rather than `max(p, b)`. Its
known limitation: the check runs at scatter, against the rows `H` has *then*,
and a sum that later refines gains rows the check never saw. Dependence costs
load balance, not correctness.

**The classification is per layer, not per term.** Both maps are GF(2)-linear,
so a prepared channel's key delta `d` moves every term by the same partition
delta `pd = part(d)` and bucket delta `bd = h(d)`. A delta with `pd = 0` is
**local** — the ordinary coset loop handles it with no communication. A delta
with `pd ≠ 0` is **remote**: every row it produces from local bucket `β`
belongs to partition `R ⊕ pd`, bucket `β ⊕ bd`, one partner and one offset
known before a term is touched. The identity delta has mask `0`, so it is
always local: a partition never ships to itself. Remoteness is a property of
the mask alone, so every partition reaches the same verdict without a vote —
which is what lets the transport pair calls positionally.

**The wire unit is one CSR block per remote delta, indexed by source bucket.**
`offsets[β]..offsets[β+1]` addresses the rows generated from the sender's
bucket `β`; the sender never permutes, and the receiver filling output bucket
`β′` reads `segment(β′ ⊕ bd)`. A `PartnerPayload` is that partner's blocks in
ascending remote-delta index, walked in lockstep with the receiver's own plan,
so a delta with no rows still ships its empty block. Both sides index by the
same bucket count, which is why the count is agreed collectively below.

A layer is **export → exchange → local coset loop**, a push model:

1. Two passes over the local buckets build one block per remote delta — count
   rows per (delta, source bucket), then fill each block's CSR segments. The
   row arithmetic is the engine's own gather at row granularity
   (`DeltaEntry::emit`, `RotationPrep::emit_gen`), so an exported row is
   bitwise the row a local gather would have produced.
2. One all-to-all `Transport::exchange`.
3. The bucketed coset loop (§Engine) over the *local* deltas only, with the
   received rows entering each output bucket's gather run through `ExtraRows`.

**Received rows join the rest stream, never the id stream.** The rest stream is
sorted anyway, so a received row may duplicate a local key and `merge2_into`
still sees the key's complete sum before `keep_term` runs (§Truncation).
Nothing else in the merge moves: the dense-identity borrowing and the
signed-zero contract are exactly as §Engine describes them.

The coset loop runs against a **retained delta table** — `retain_entries`
rebuilds `LocalPtm` with the remote entries dropped — plus `LayerKnobs`
carrying the local bucket deltas and the rest-stream count. That retention has
one sharp edge, and the engine guards it: a channel whose every non-identity
delta is remote (a Hadamard under a partition row that sees its mask, say)
leaves an **identity-only table, for which `is_key_preserving` is true**.
Taking the key-preserving `rescale_in_place` fast path there would silently
drop every received row, so the fast path is gated on the absence of extra
rows. A rotation with a remote generator keeps its `Prepared` and switches the
generator pass off instead, exporting its anticommuting rows.

**One collective per layer is unconditional: the bucket count.** Each partition
proposes `desired_bits` for its own share, the group takes the maximum, and
each refines to it. Per-partition rather than global, so `P` partitions of
`n/P` terms carry the same *total* bucket count as one partition of `n`; the
grow-only rule (§Bucket-Policy) survives the reduction because a maximum of
grow-only proposals is itself monotone, so the count never falls mid-run and
the two sides of an exchange always agree. Everything else is conditional: **a
layer with no remote delta makes no transport call at all**, so a transport
implementation must not assume a fixed number of calls per layer.

Layer finalization is collective, so the policy bound is
`PartitionedTruncation` and its `finalize_layer_partitioned` runs on every
layer on every partition, whatever `finalizes_layer` reports — a collective is
well defined only if nobody skips it. `ApproxTopN` is **partition-exact**: the
global octave histogram is the sum of the per-partition histograms, so one
all-reduce has every partition choose the same edge and the union of the
retained sets is the single-partition answer (§Truncation). `And` runs both
sides; `Or` runs neither, because its unpartitioned `finalize_layer` is the
trait's no-op default rather than either child's, and the two must agree. Exact
`TopN` is a distributed `k`-th selection, not a sum, and is **rejected at
compile time** by the trait bound rather than approximated.

**The runtime is one pinned Rayon pool per partition, with work-stealing inside
a partition only.** That is the shape the static-placement negative result
points at (§Parallelism): stealing is what beats a static assignment, and a
stable domain-level split is what first touch needs, so the split is static at
the outer level and stealing is untouched at the inner one. Partition 0 drives
on the calling thread — the future MPI funnel, where the calling thread owns
the communicator — and partitions `1..P` get scoped threads that pin themselves
to their slot and enter their pool. The transport group is built **per call**
and moved into the partitions, so a partition that panics drops its endpoints
and its partners fail naming its rank instead of blocking forever.

Scatter and gather bracket a run, not a layer. `filter_partition` runs on the
owning partition's own pool, so every column is first-touched in the domain
that will read it, and `merge_partitions` merges the disjoint runs back. The
**scatter bits rule** is `want.max(bits − pbits).min(bits)`: a partition sheds
at most `log2 P` of the bits the whole sum arrived with, never going below what
its own share wants, so the bucket count summed over partitions equals the
unpartitioned one. At `P = 1` it is the identity, and the round trip is
bitwise.

`PartitionTrace` is the opt-in per-layer record: bucket bits, remote-delta
count, terms in and out per rank, rows and bytes sent `[from][to]`, rows
received, and an imbalance figure per layer. Under `phase-timing` the same run
also reports `export_ns`, `exchange_ns` (which includes the wait for a partner,
so it is the imbalance signal and the traffic cost at once), `collective_ns`,
`rows_exported` and `recv_rows`.

What the split guarantees: **at `P = 1` the partitioned engine is `propagate`,
bit for bit** — the scatter is the identity, the all-reduce is the identity,
and the layer takes its unpartitioned branch. Across partition counts the bar
is floating-point tolerance, exactly as it is across bucket counts
(§Determinism); within a partition output stays byte-identical across pool
sizes, because received rows are appended in the plan's fixed order.

The cost model, qualitatively: a random partition row set sends a nonzero delta
across a boundary with probability `1 − 2^{-p}`, so **roughly half of a dense
two-qubit gate's deltas are remote at `P = 2`**, and a rotation whose generator
crosses exports one row per anticommuting term. Export volume is therefore a
property of the row *draw*, not of the circuit alone — which is what makes the
rows worth tuning rather than merely drawing. The extreme case names the
target: a row that annihilates every delta a circuit uses is a conserved
quantity of that circuit and produces no traffic at all, so the rows that
minimize traffic are cut-like, reading the qubits on the boundary of a spatial
cut. `Transport` is the seam for that work and for the distributed phase: MPI
is the same exchange over ranks, with the same collective-order invariant.

### Transport composition

The engine's *partition* is deliberately not a thread and not a process: it is
whatever a `Transport` says a peer is. Two implementations exist, and the layer
code cannot tell them apart — `run_layers` is one function, generic over the
transport, driven both by `PartitionedSum` (`P` partitions inside one process,
`InProcessTransport`) and by `DistributedSum` (**one partition per process**,
`MpiTransport`, behind the off-by-default `mpi` feature).

**`D = 1`: one rank per NUMA domain, no hybrid.** A distributed rank's runtime
holds exactly one partition, and its placement comes from the launcher rather
than from the engine — `mpirun --map-by ppr:1:numa --bind-to numa` or `srun
--cpu-bind=ldoms` leaves the process an affinity mask of one domain, and
`Placement::Auto` over that mask resolves to a single slot covering it. So the
process placement does the work explicit CPU lists do in-process, and the two
mechanisms never have to be composed. A domains-per-rank hybrid (`D > 1` inside
each rank, MPI between ranks) is a possible later shape, not an implemented
one.

**The library never initializes MPI.** `MPI_Init` is a process-global one-shot,
so the application owns it: `MpiTransport::from_communicator` duplicates a
communicator the caller already has, and `from_raw_handle` does the same for a
foreign `MPI_Comm` (a Python host's, through `mpi4py`). The duplicate is the
engine's own, so its tags cannot collide with the application's traffic. The
`mpi` crate is re-exported as `paulistrings::mpi::rsmpi` for the same reason a
duplicate is taken: the caller must build its `Universe` from the version the
library links. The engine's MPI calls come off a Rayon pool worker (the layer
loop runs inside `ThreadPool::install`), one at a time, so the level to request
is `MPI_THREAD_SERIALIZED` — never `FUNNELED`.

**Point-to-point, not all-to-all-v.** The partner set of a layer's exchange is
symmetric by construction: a remote delta moves rank `R`'s rows to `R ⊕ pd`,
and the same delta on `R ⊕ pd` moves its rows back to `R`. So "who sends to me"
*is* "who I send to", every rank knows the set from its own plan, and no
count-exchange or group-sized collective is needed to discover it.

**The per-layer collective schedule** is unchanged from the in-process case,
and it is what an MPI implementation must not second-guess: one unconditional
`allreduce_max_u8` for the bucket count, then the layer's exchange **only if
the plan has a remote delta** (a key-preserving channel issues no transport
call at all), then whatever the policy's collective finalization runs. On top
of that a distributed propagation calls `check_consistency` exactly once,
before its first layer: an all-reduce of a fingerprint of the run's shape
(channel count, direction, bucket-policy knobs, qubit count, `W`), exact
because it reduces the 64 per-bit counts and every count must be 0 or `size`.
Ranks handed different circuits then get a message instead of a deadlock two
layers in.

**Wire framing.** Per partner, one self-describing header — `u64[2 +
n_parts]`, carrying the wire version, the source rank and one byte length per
part — followed by the payload's `byte_parts` in order. The header is the only
message whose size the receiver cannot predict, so it arrives through a matched
probe; the parts are posted receives of known length, matched to their sends by
MPI's non-overtaking guarantee for a `(source, tag, communicator)` triple,
which is why both sides walk partners, parts and chunks in the same ascending
order. Tags pack `epoch:11 | kind:4`, at most 32767 and so inside the
guaranteed `MPI_TAG_UB`. Everything is sent as bytes and chunked at 1 GiB:
MPI's counts are `i32`, and a `u64` view of the parts — which would raise the
per-message ceiling — is not available, the block header being four `u32`s and
the CSR `offsets` column a `Vec<u32>`, neither 8-aligned. All sends are posted
before the call's first receive, so the rendezvous cannot deadlock. Raw host
bytes on the wire means a run is homogeneous: same architecture, same `W`,
every rank.

**Scatter and gather bracket a distributed run too, with a different
contract.** The input is *replicated* — every rank calls `scatter` with the
same sum and keeps `filter_partition(rows, rank)`, the rows drawn from one seed
so nobody has to agree by collective. `local()` is always this rank's share, a
valid `PauliSum` under the group's shared hash; `gather()` is collective and
returns `Some` on rank 0 only, each rank shipping its bucket lengths and the
three columns `to_arrays` concatenates and rank 0 rebuilding them before
`merge_partitions`. `PartitionTrace` stays per rank: `terms_in`, `terms_out`
and `rows_received` have one entry, while `rows_sent` and `bytes_sent` are
indexed by destination rank over the whole group.

## Determinism

The correctness bar for engine changes is **agreement to floating-point
tolerance** (`assert_terms_close`), not bitwise equality. Equal-key summation
order is unspecified and free to change between versions, configurations, and
optimizations; floating-point addition is not associative, so a different
bucket count or hash seed may legitimately change output bits.

What *is* reproducible, as a property of the current implementation rather
than a promise: at a fixed bucket count and hash seed, output is bitwise
identical across thread counts and repeat runs — cosets are write-disjoint
and work within one is sequential. The same holds per partition, across
pool sizes, and a `P = 1` partitioned run is bitwise `propagate`; across
partition counts the bar is tolerance, as it is across bucket counts
(§Partitioning). Tests that pin exact output bits (the
fingerprint net, thread-count byte-identity tests) are **convenience
tripwires** for unintended perturbation: when one trips under a change that is
correct to tolerance, regenerate its literals or demote it to
`assert_terms_close` in the same commit, with a one-line note. Do not design,
constrain, or reject an optimization to keep output bits stable.

## Truncation

Truncation is what keeps Pauli propagation tractable, and it is a composable
extension surface:

```rust
pub trait TruncationPolicy<const W: usize>: Send + Sync {
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool { true }
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {}
}
```

The split is performance-critical: `keep_term` runs on every merged output —
potentially billions of times — and must inline to nanoseconds; it sees the
**summed** coefficient, inside the merge. `finalize_layer` runs once per layer
and may be non-local.

Built-ins: `CoefficientThreshold(eps)` and `WeightCutoff(k)` are per-term
filters; `TopN(n)` and `ApproxTopN(n)` are layer finalizations. Policies
compose with `And` / `Or` (Python: `&` / `|`).

**Magnitudes are compared as `|c|²`, never as `|c|`.** `Complex64::norm()` is
`hypot`, a libm call, and it was on the hot path twice per candidate; `x ↦ x²`
is strictly increasing on `[0, ∞)`, so `|c|² > t²` decides the same predicate
for a multiply-multiply-add. The equivalence is exact for the *ordering* and
near-exact for the *tie grouping*: a symmetry multiplet's members differ by a
sign or a power of `i`, and `re² + im²` is bitwise invariant under both, so
exact ties survive. What squaring does lose is the band below
`|c| ≈ 1.57e-162`, whose squares underflow to `0.0` and therefore tie — a
coefficient that is numerically zero either way.

**`TopN` never splits a tie group.** Terms with exactly equal magnitude are
typically a symmetry multiplet (lattice symmetries produce exact ties), and
truncation should commute with the symmetry, so the group at the threshold
magnitude is kept only if it fits entirely within `n` and discarded whole
otherwise. Consequences, all deliberate: `TopN(n)` retains *at most* `n`
(discarding is the safe direction for its memory-bounding job); it retains
exactly `n` when magnitudes are distinct; and a sum whose coefficients all
share one magnitude is wiped to empty. Implementation: gather squared
magnitudes into a per-thread pooled buffer, select the `n`-th largest once
globally, decide the tie group from the selection's own partition (the group
fits iff nothing after the pivot equals the pivot), then filter each bucket in
parallel — per-bucket filtering preserves within-bucket order automatically.

**`ApproxTopN(n)` trades the exact count for the selection.** It histograms the
octave of `|c|²` (the 11-bit `f64` exponent: 2048 bins, 8 KB, L1-resident),
walks the bins down to the lowest edge whose cumulative count still fits in
`n`, and retains against that edge — two `O(m)` passes, no candidate array, no
selection. It keeps `≤ n` (so the memory bound is exact) and `> n - p`, where
`p` is the population of the coarsest octave that did not fit, i.e. the
shortfall is bounded by the terms inside one `√2`-wide magnitude band at the
cut. Tie groups need no rule here at all: equal magnitudes share an octave, so
a multiplet is always kept or dropped whole — at the price of a wider
degenerate case, since a sum confined to a single octave of `|c|²` is wiped
exactly as an all-tied sum is under `TopN`. `TopN` remains the default and the
choice whenever the retained count itself matters.

Under partitioning the two swap places: `ApproxTopN` is **partition-exact** —
its histogram all-reduces, so the retained set is the single-partition one —
while exact `TopN` has no collective form and is rejected at compile time
(§Partitioning).

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

Implementing `apply` (plus `support`) is the whole cost of a custom channel;
`prepare`'s default derives the engine form automatically
(§Prepared-Channels). `max_fanout` is a method rather than an associated
const so `Circuit` can store `Box<dyn Channel<W>>` — the channel set is open
to user extension; concrete impls return literals, so call sites through
generics still constant-fold.

Built-ins: `Clifford1Q` / `Clifford2Q` (table-driven from their symplectic
action), `PauliRotation` (`exp(-iθP/2)`, any generator weight; support derived
from the generator, never caller-supplied), `GeneralUnitary1Q` /
`GeneralUnitary2Q` (constructed from a matrix or a Pauli-transfer matrix),
and the noise channels `Depolarizing`, `Dephasing` (pure coefficient
rescales), and `AmplitudeDamping` (genuine fanout 2). `IdentityChannel`
exists for tests and composition.

## Ingestion

`BuildAccumulator<W>` is a hashmap accumulator (`FxBuildHasher` — Pauli
bitstrings are already high-entropy, so SipHash buys nothing) for unsorted
input: Hamiltonian parsing, dict construction, custom analyses. `finalize()`
hashes, scatters, and sorts into a canonical `PauliSum`, choosing the bucket
count by the standard policy so small sums come out single-bucket. The
accumulator is an ingestion path only — it never appears in the propagation
loop.

## Python-Bindings

The Python package is a thin layer over enums (`PauliSumImpl`, `CircuitImpl`)
holding the monomorphized widths (§Width); every method dispatches once and
calls the same core code Rust users call. Construction accepts dictionaries
and `(string, coefficient)` pairs; bulk export returns NumPy arrays
(`to_arrays`). Expectation values against product states, overlaps, and the
identity coefficient are computed in Rust. Truncation factories return spec
objects composed with `&` / `|` and translated to core policies at the
boundary. The extension module is `paulistrings._paulistrings` (abi3), and
`python/paulistrings/` re-exports it.

The Python `Circuit` additionally keeps the width-erased `ChannelSpec` of every
channel pushed, alongside the materialized `Circuit<W>`: the core stores
prepared channels and cannot hand a gate description back out, so that list is
what serves gate-list introspection (`Circuit.gates`, emitted in the frozen
task-JSON gate vocabulary), slicing, concatenation, and `adjoint()`. The two
are appended to together and never diverge; the cost is one spec per channel,
dominated by a 2-qubit unitary's 4×4 matrix and negligible against the prepared
channel's own Pauli-transfer matrix. Everything those methods do is
spec-rewriting outside any hot loop — no core code changes and no per-term work.

## GPU-Readiness

The design decisions a GPU backend needs are already in place: `PauliString`
is `Pod` with a defined layout; bucket columns are SoA and flatten to
device buffers in one pass; the coset decomposition maps to one block per
coset with gather/sort/merge in shared memory — a better CUB fit than any
global sort. The extension to distributed memory is no longer forward-looking:
§Partitioning is that exchange, and MPI is the same exchange over ranks.

## Performance-Model

Where a layer's time goes, at 10⁶ terms: gather + merge dominate (75–92% of
busy time across the built-in workloads; the sort only matters for dense
two-qubit unitaries). Idle is single digits — load balance is a solved
problem; the cost that grows with thread count is per-row time under memory
contention, not imbalance.

Binding constraints for optimization work — do not rediscover these the hard
way:

1. **The determinism policy** (§Determinism): tolerance, not bits, is the bar;
   byte-exact tests are tripwires to regenerate, never constraints.
2. **The signed-zero contract** (§Engine): exact-zero id rows flow to the
   accumulator; the only zero test is on the final sum.
3. **The stable adaptive sort** (§Engine): the per-run sort exploits
   piecewise-sortedness; replacing it with an unstable sort measured +77%.
4. **Measurement discipline:** release builds only, seeded inputs outside the
   timed region, the reference host, and the campaign workflow in
   `benchmarks/PROFILING.md`. Single-shot campaign noise on the reference host
   is ±5–8% single-threaded and ±10–26% at high thread counts — effects below
   that need the interleaved A/B protocol, not one campaign per build.

The memory wall is real and measured: the reference host's usable bandwidth is
~39 GB/s per socket, ~45–49 GB/s across both (2 of 6 memory channels
populated — see `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`,
the denominator for every roofline claim). Trotter-style workloads at 32
threads move ~36 GB/s of attributable DRAM traffic — near the wall — so
further wins there come from traffic reduction or genuine NUMA partitioning
(§Partitioning; measured results pending), not scheduling. Hyperthreads add no bandwidth; the second socket adds only
15–25% under first-touch placement with work-stealing.

Negative results are recorded in `research/notes/` and should be read before
re-attempting the corresponding ideas: static coset→worker placement (slower
than work-stealing), recompute-in-merge id-stream borrowing for sparse
streams, segment-copy merging, and interleaved transient key layouts (all
measured and rejected — see `research/notes/2026-08-31-v0.6-results.md` and
the static-coset-placement note).

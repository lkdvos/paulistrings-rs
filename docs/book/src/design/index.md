# How it works

`paulistrings` evolves a weighted sum of Pauli strings through a circuit layer by layer, forward or
in the Heisenberg picture, with truncation keeping the sum tractable. This page explains the engine
design that makes that loop fast: the sum never maintains a global sorted order, every layer's data
movement is known before any term is touched, and the parallel decomposition needs no locks, no
atomics, and no synchronization inside a layer. The companion [Performance](performance.md) page
puts measured numbers to each claim.

## The propagation loop

A `PauliSum` stores each term as a symplectic key: qubit `i`'s Pauli is a bit pair, packed into
`x: [u64; W]`, `z: [u64; W]` words (`W` words cover `64·W` qubits), with a complex coefficient.
Under this encoding Pauli multiplication is bitwise XOR of keys plus a phase, and the phase is
folded into the coefficient at the boundary, so a key is a plain element of the vector space
`GF(2)^{2n}`.

Each circuit layer applies one channel to every term. A channel with support on `k` qubits maps an
input key `v` to outputs `v ⊕ d`, where `d` ranges over a small **delta set** `D` determined by the
channel alone: a Pauli rotation of any generator weight has `|D| = 2` (identity and the generator),
a two-qubit Clifford at most 4, a dense two-qubit unitary at most 16. Applying a layer means fanning
every term out over `D`, then recombining terms that landed on the same key.

The recombination is the interesting part. Done naively it is a global sort or a concurrent hash
map per layer, at millions to hundreds of millions of terms.

## No global sort

The sum is partitioned into buckets, and its canonical order is bucket index ascending, then
lexicographic key within a bucket, with no duplicate keys anywhere. Deduplication therefore only
ever needs a canonical order *within* one bucket, and every bucket is small and cache-resident.
Nothing in the propagation loop ever sorts the whole sum.

Storage is structure-of-arrays per bucket: parallel `Vec<[u64; W]>` columns for `x` and `z` and a
`Vec<Complex64>` for coefficients. Coefficient-only scans (truncation, expectation values) and
key-only scans (weight, commutation) each stream exactly the bytes they use, and every column maps
directly to a GPU device buffer. Buckets own their columns, so each retains its capacity across
layers: the steady state of a propagation loop allocates nothing.

## The bucket hash is GF(2)-linear

The bucket function is `h(v) = H·v` for a fixed dense random matrix `H` over GF(2), giving
`B = 2^b` buckets. Linearity is the property everything else follows from:

```text
h(v ⊕ d) = h(v) ⊕ h(d)
```

A term in bucket `i` fans out only into buckets `i ⊕ h(d)` for `d ∈ D` — so the layer's entire
data-movement pattern is a function of the channel and the hash, known **before touching a single
term**. Equal keys always share a bucket (h is a function), so duplicates can never straddle
buckets. And because `H` is dense and random, bucket loads stay uniform whatever structure
truncation imposes on the keys, and the map from delta to bucket offset is almost always
one-to-one.

Bucket count follows the sum's size, targeting about a thousand terms per bucket so a bucket and
its scratch fit in L2. Refining the partition when the sum grows is a single parity pass per term,
not a re-sort: adding one row to `H` splits each bucket in two with the within-bucket order
inherited.

## Cosets are closed, write-disjoint tasks

Take the span of `h(D)` in bucket-index space. Its cosets partition the buckets, and every output
bucket in a coset reads only input buckets in that same coset: **a coset is a closed task**. That
is the engine's whole parallel decomposition — one coset per Rayon task, and by construction no two
tasks touch the same bucket, so a layer runs with no atomics, no locks, and no cross-thread
reconciliation of any kind. Load balance comes from the random hash plus work-stealing.

![Bucket space partitioned into cosets](../assets/design/bucket-cosets.svg)

At a fixed bucket count and hash seed this structure also makes the output bitwise identical across
thread counts and repeat runs: cosets are write-disjoint and work within one is sequential. Across
bucket counts or seeds, output agrees to floating-point tolerance — equal-key summation order is
deliberately unspecified.

## A layer runs in place

Each coset task, independently:

1. **Swap** its bucket columns into worker-persistent scratch, leaving emptied, capacity-retaining
   columns behind as the write destinations. The layer is in-place: peak memory is the sum itself
   plus one coset's working set per worker, never a second full-size copy.
2. **Size** every output run exactly from the swapped-out bucket lengths, before any work.
3. **Gather input-major**: each term is loaded exactly once and its whole fanout scattered to
   output runs via an O(1) index identity (`member(i) ⊕ δ = member(i ⊕ coord(δ))`) — no read
   amplification. The inner loop per (term, delta) is a few bit extractions, one table lookup, one
   XOR mask, one complex multiply.
4. Per run, **sort by key and merge** straight into the bucket's live slot, restoring the
   strictly-ascending no-duplicates invariant.

Two structural tricks keep the moved bytes down. The identity delta's stream keeps its source keys,
so it inherits the source bucket's sorted order and is never sorted; and when the identity
amplitude is dense (every rotation and general unitary), the gather materializes only the 16-byte
coefficients and the merge borrows the key columns from the source bucket in place — identity keys
are neither written nor re-read. The rest-stream sort itself exploits that a gather run is a
concatenation of already-sorted per-delta streams; a radix variant takes over for dense two-qubit
PTMs, where the sort dominates the layer. The merge is fused with the segmented reduction: one
two-pointer walk sums equal-key coefficients and applies the truncation filter to each fully summed
coefficient as it passes.

Channels that never change keys at all (depolarizing, dephasing, Pauli gates) bypass the whole
pipeline as a parallel coefficient rescale.

## Channels are prepared once

Applying a channel through its trait object per term would pay a virtual call and re-derived tables
millions of times. Instead the engine prepares each channel once per layer into a local
Pauli-transfer-matrix form: per delta, a bucket offset, full-width XOR masks, and an amplitude
table with all phases pre-folded. The preparation is derived automatically for any channel with
support on at most two qubits by probing the channel's own `apply` on the local Pauli basis — a
custom channel implements `apply` and gets the bucketed engine for free. Pauli rotations prepare
specially at any generator weight, since their delta set is just `{0, generator}`.

## Truncation is what keeps it tractable

Truncation is a trait with two hooks split by cost. `keep_term` runs on every merged output —
potentially billions of times — sees the fully summed coefficient inside the merge, and is
monomorphized into it, so a threshold test inlines to a compare with no call. `finalize_layer` runs
once per layer and may be non-local; `TopN` and its histogram-based approximation live there.
Policies compose with and/or combinators. Magnitude comparisons use `|c|²` rather than `|c|`
(the same ordering, no `hypot` on the hot path), and `TopN` keeps or drops magnitude-tied symmetry
multiplets whole.

## Where to go deeper

[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) is the
maintainer-facing design reference: the same mechanisms with their tuning constants, measured
trade-offs, and the negative results behind each choice. The rustdoc under
[/api/](../api/paulistrings/index.html) documents the public types; the module docs on
`engine::bucketed`, `engine::merge`, and `pauli_sum` carry the precise per-module contracts.

Sources: [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md)
§Data-Model, §Bucketing, §Hash, §Prepared-Channels, §Engine, §Parallelism, §Determinism,
§Truncation; module docs
[`engine/bucketed.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/src/engine/bucketed.rs),
[`engine/merge.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/src/engine/merge.rs),
[`pauli_sum.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/src/pauli_sum.rs).

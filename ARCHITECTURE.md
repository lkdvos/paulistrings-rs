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
they share it.

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
(§Data-Model).

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

**The sort.** `sort_rows_with_scratch` (in `engine/merge.rs`, alongside its
worker-persistent `SortScratch`) is a permutation sort over the run.

> Its comparison sort **must remain the standard library's stable adaptive
> `sort_by`**. A gather run is a concatenation of per-delta streams, each
> drawn from one sorted bucket — piecewise-sorted data whose natural runs the
> adaptive driftsort detects and merges nearly for free. Switching to
> `sort_unstable_by` (pdqsort, no run detection) measured **+77%** on a
> rotation layer. Stability per se is irrelevant; adaptivity is the point.
> Recorded on the function's doc — do not "simplify" it.

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

## Determinism

The correctness bar for engine changes is **agreement to floating-point
tolerance** (`assert_terms_close`), not bitwise equality. Equal-key summation
order is unspecified and free to change between versions, configurations, and
optimizations; floating-point addition is not associative, so a different
bucket count or hash seed may legitimately change output bits.

What *is* reproducible, as a property of the current implementation rather
than a promise: at a fixed bucket count and hash seed, output is bitwise
identical across thread counts and repeat runs — cosets are write-disjoint
and work within one is sequential. Tests that pin exact output bits (the
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
filters; `TopN(n)` is a layer finalization. Policies compose with `And` / `Or`
(Python: `&` / `|`).

**`TopN` never splits a tie group.** Terms with exactly equal magnitude are
typically a symmetry multiplet (lattice symmetries produce exact ties), and
truncation should commute with the symmetry, so the group at the threshold
magnitude is kept only if it fits entirely within `n` and discarded whole
otherwise. Consequences, all deliberate: `TopN(n)` retains *at most* `n`
(discarding is the safe direction for its memory-bounding job); it retains
exactly `n` when magnitudes are distinct; and a sum whose coefficients all
share one magnitude is wiped to empty. Implementation: gather magnitudes,
select the `n`-th largest once globally, filter each bucket in parallel —
per-bucket filtering preserves within-bucket order automatically.

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

## GPU-Readiness

The design decisions a GPU backend needs are already in place: `PauliString`
is `Pod` with a defined layout; bucket columns are SoA and flatten to
device buffers in one pass; the coset decomposition maps to one block per
coset with gather/sort/merge in shared memory — a better CUB fit than any
global sort. The same structure extends to distributed memory: partition on
`h`, and a layer becomes a sparse, statically-known, small-fan-in exchange in
which no key is ever split across ranks (§Bucketing — duplicates cannot
straddle buckets).

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
further wins there come from traffic reduction or genuine NUMA partitioning,
not scheduling. Hyperthreads add no bandwidth; the second socket adds only
15–25% under first-touch placement with work-stealing.

Negative results are recorded in `research/notes/` and should be read before
re-attempting the corresponding ideas: static coset→worker placement (slower
than work-stealing), recompute-in-merge id-stream borrowing for sparse
streams, segment-copy merging, and interleaved transient key layouts (all
measured and rejected — see `research/notes/2026-08-31-v0.6-results.md` and
the static-coset-placement note).

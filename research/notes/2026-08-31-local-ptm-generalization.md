# Generalizing the local PTM beyond `MAX_LOCAL_SUPPORT = 2`

Status: design note, nothing implemented. Written as the pointer target for the
panic `engine::propagate_with_scratch` now raises when a channel declines
`Channel::prepare` (v0.7 Stage 1, which deleted the v0.1 whole-sum sort-merge
fallback that used to absorb that case).

## Why the cap is 2 today

`Prepared::Local` is a dense local Pauli-transfer matrix over the channel's
support: for `k` support qubits the table is `4^k × 4^k` complex amplitudes,
grouped by key delta. Two things in `channel/prepared.rs` are sized by a
compile-time constant rather than by `k`:

- `LocalPtm::qubits: [u32; MAX_LOCAL_SUPPORT]` — the support qubit list, inline.
- `DeltaEntry::amp: [Complex64; LOCAL_DIM]` with `LOCAL_DIM = 4^MAX_LOCAL_SUPPORT
  = 16` — one amplitude per input support pattern, inline in every delta entry.

At `k = 2` that is a 16×16 table, 4 KB of `Complex64` per layer, and the hot
loop's per-row work is one 4-bit index extraction, one array read, one XOR and
one complex multiply — all on data that is trivially in L1 and, because the
arrays are fixed-size, all with the bounds folded away at monomorphization.
`k = 3` would be a 64×64 table (64 KB), `k = 4` a 256×256 one (1 MB); as inline
arrays those stop being reasonable to copy around per delta entry, which is the
real reason the constant is 2 and not something larger.

`derive_local` therefore bails on `k > MAX_LOCAL_SUPPORT` before materializing
anything, `Channel::prepare` returns `None`, and — as of Stage 1 — `propagate`
panics. The one built-in that would exceed the cap, `PauliRotation` at weight
≥ 3, overrides `prepare` and returns `Prepared::Rotation`, a two-delta plan that
computes the `i^k` phase per term instead of tabulating it; so no built-in can
reach the panic.

## The generalization

Move the amplitude row off the stack and keep the `k ≤ 2` path exactly as it is:

- `DeltaEntry::amp` becomes an enum (or the struct gains a variant): `Inline([Complex64;
  16])` for `k ≤ 2`, `Heap(Box<[Complex64]>)` of length `4^k` for `k > 2`.
  Likewise `qubits` becomes `Inline([u32; 2])` / `Heap(Box<[u32]>)`. Nothing on
  the `k ≤ 2` hot path changes shape, so the monomorphized fast path — the one
  every built-in takes — is untouched and needs no re-measurement.
- The engine's `gather_local_*` loops index `amp[s]`; behind an accessor that is
  one bounds-checked slice read either way. The `k > 2` arm pays a pointer
  indirection and loses the folded bound, which is acceptable for a path no
  built-in uses.
- `DeltaPlan::Local`'s dense/sparse identity classification (v0.6 G1d) already
  reads `amp[..dim]` with `dim = 4^k` computed at runtime, so it generalizes
  unchanged.

**Probe cost is the real limit, not storage.** `probe_table` calls
`Channel::apply` once per input support pattern (`4^k` calls) and inspects up to
`max_fanout` outputs each; with a dense channel `max_fanout` is itself `4^k`, so
deriving the table is `O(16^k)` per layer. That is 256 at `k = 2`, 4096 at
`k = 3`, 65 536 at `k = 4`, ~1.05 M at `k = 5`. Against a 10⁶-term layer, `k = 4`
is still a rounding error and `k = 5` is not; the practical ceiling is therefore
`k ≈ 4–5`, and a plausible new constant is 4 (a 256×256 table, 1 MB heap per
layer, ~65 k probe calls). Beyond that a channel wants a *factored* prepared
form — a product of local PTMs, or a rotation-style analytic plan — not a bigger
dense table.

## Interaction with `GATHER_OUTPUT_MAJOR_MIN_R`

`engine/bucketed.rs` has two gather orders and picks between them on the coset
rank `r`: input-major below `GATHER_OUTPUT_MAJOR_MIN_R = 3`, output-major at or
above. Every built-in channel spans rank ≤ 2 (a 2Q Clifford or `GeneralUnitary2Q`
whose delta masks have Pauli structure — sqrt-SWAP's `{XX, ZZ, YY}` — spans
exactly 2; a rotation spans at most 1), so **the output-major branch is dead in
practice and has never been measured on a real workload**. It exists as a guard
for exactly the case this generalization would create: a full-rank `k ≥ 3`
channel, where input-major would keep `2^r ≥ 8` write streams open per task and
the scatter working set would leave L2. Two consequences:

1. Raising `MAX_LOCAL_SUPPORT` is the change that first exercises that branch, so
   it must come with the measurement `GATHER_OUTPUT_MAJOR_MIN_R = 3` was chosen
   on faith. The threshold is a pure performance knob — both orders gather the
   same multiset of rows, agreeing to floating-point tolerance
   (`local_gather_orders_agree_to_fp_tolerance`) — so it can be retuned freely,
   but it should be retuned *with data* at the ranks it newly reaches.
2. Conversely, until then, the branch stays as written: deleting it would remove
   the only thing standing between a future `k = 3` channel and a cliff, and
   keeping it costs one comparison per layer.

## Writing outside declared support stays a hard error

`derive_local` also returns `None` when the debug-mode shadow probe (or, in any
build, the per-output support check in `probe_table`) sees `Channel::apply`
modify a bit outside `Channel::support()`. That is a **contract violation, not a
capability gap**, and the generalization above does not and should not address
it: a channel that misreports its support cannot be tabulated *at any* `k`,
because the whole prepared form — the bucket deltas `δ = H·d`, the write-disjoint
coset partition, the exact per-run sizing — is derived from the claim that the
key transform touches only support bits. Silently routing such a channel through
a slower-but-general path (which is what the deleted v0.1 fallback did) hides a
bug in user code behind a performance mystery. The panic names both causes and
the support weight it measured, so the two are distinguishable at the point of
failure.

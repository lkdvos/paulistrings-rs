# Why the v0.1 §5 bucket phase could not be implemented

*2026-08-26. Context for `research/plans/2026-08-26-v0.2-gf2-bucketing.md`.*

## The claim

`2026-04-30-v0.1-scope.md` §5 specifies an `O(n)` bucket phase, and states the
property it rests on twice — at line 93:

> "If the input sum is sorted by Pauli key, the output sum after gate application
> is *almost sorted*: ordering is preserved across all bit positions outside the
> gate's support, and is perturbed only at the few positions inside it. […]
> Sorting therefore reduces to a single linear-time bucket assignment followed by
> concatenation."

and again at line 99:

> "**Bucket phase.** The output buffer is partitioned into `2^(2|support|)`
> buckets indexed by the bits at the channel's support qubits. Within each bucket
> the relative order is inherited from the input."

`2026-04-30-v0.1-tdd-slices.md` slice 6.2 turns the consequence into a test
contract: *"concatenation of buckets is sorted (when the input was sorted)."*

## The claim is false

The first sentence is true — within a bucket, relative order *is* inherited. The
inference is not: the buckets **interleave**, so no concatenation order is sorted.

Minimal counterexample. `W = 1`, 2 qubits, gate `H(0)`. Keys written as `(x, z)`,
compared as `PauliString::cmp` does (`pauli_string.rs:278`): all of `x` before any
of `z`, low word first.

| input   | key   | `H(0)` output | output key | bucket (support bits at q0) |
|---------|-------|---------------|------------|------------------------------|
| `I`     | (0,0) | `I`           | (0,0)      | 0 |
| `Z₁`    | (0,2) | `Z₁`          | (0,2)      | 0 |
| `X₀`    | (1,0) | `Z₀`          | (0,1)      | 1 |
| `X₀Z₁`  | (1,2) | `Z₀Z₁`        | (0,3)      | 1 |

The input is sorted. Bucket 0 = `[(0,0), (0,2)]`, bucket 1 = `[(0,1), (0,3)]` —
each internally sorted, exactly as §5 says. But the correct output order is
`(0,0), (0,1), (0,2), (0,3)`, which **interleaves** the two buckets. Neither
`0 ++ 1` nor `1 ++ 0` is sorted. Bucketing on *input* support bits gives the same
partition here and fails identically.

## Why

Concatenation-of-buckets is sorted only when the bucket index is the
**most-significant** field of the sort key. It is not, for two independent
reasons:

1. A support qubit `q` occupies bit `q % 64` of word `q / 64`. Within a word,
   qubit 63 is the most significant bit and qubit 0 the least, so a support
   qubit's key bits sit at an arbitrary, generally non-dominant significance.
2. The key is `(x[0..W], z[0..W])` — *all* of `x` before any of `z`. So the x-bit
   and the z-bit of the same qubit are `W` words apart in significance and can
   never jointly form a contiguous most-significant field, whatever `W` is.

There is no cheap repair while "one global sorted order over `(x, z)`" is the
`PauliSum` invariant. This is why slice 6.2 was quietly replaced by an
`O(n log n)` comparison sort (`engine/sort_merge.rs:206-241`, which says so at
lines 213-214) and why `Channel::support()` — added solely to feed the bucket
layout, per §6 line 130 — has never been called by the engine.

## Two ways out

**(a) Keep the global order, replace the sort with a k-way merge.** Every
built-in channel's key map is, per bucket, `key ↦ key ⊕ c` for a constant `c`
(the Clifford tables are bit rewrites at fixed positions; `PauliRotation`'s
second output is literally `input ⊕ gen`, `rotation.rs:76`). XOR by a constant
preserves integer order on any set where the `c`-bits agree, so a layer's output
is a union of at most `fanout × 4^|support|` sorted runs. The sort phase then
becomes a k-way merge at `O(n log k)`, `log k ≤ 2|support| + log₂ fanout` — 2-3
comparisons per element instead of `log₂ n ≈ 20-27`. Same asymptotic family §5
wanted; §5 just skipped the merge. But the result is still one global array, so
the merge is inherently coordinated, and it gives nothing to a distributed
backend.

**(b) Give up the global order; partition with a GF(2)-linear hash.** Because
`h(v) = H·v` is linear and channels act by `v ↦ v ⊕ d`, output buckets are
predictable from input buckets *and the relation inverts*: each output bucket
gathers from a statically-known handful of input buckets. Duplicate keys can
never straddle buckets, so dedup is bucket-local and there is no global sort at
all. This is what v0.2 does; see `2026-08-26-v0.2-gf2-bucketing.md` §1.

(a) survives inside (b) as a within-bucket optimization — see v0.2 §7.2. The
order argument is the same one; it just applies to a bucket instead of the whole
sum.

## Lesson worth keeping

The §5 claim was stated twice in prose and promoted to a test contract without
a worked example. A two-qubit, four-term counterexample would have caught it
before it shaped `Channel::support()` and the `MAX_FANOUT` buffer-sizing
contract. Load-bearing ordering claims get a concrete table in the doc.

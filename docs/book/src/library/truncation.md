# Truncation policies

All five live in the `paulistrings.truncation` module and run after every channel.

| Policy | Keeps |
|---|---|
| `truncation.coeff(eps)` | `\|c\| > eps` — note **strictly** greater; `\|c\| == eps` is dropped |
| `truncation.weight(k)` | Pauli weight `<= k` |
| `truncation.topn(k)` | exactly `k` terms (all of a smaller sum), largest `\|c\|` first; no partitioned form |
| `truncation.approx_topn(k)` | exactly `k` terms, ranked by octave of `\|c\|**2`; partition/comm-safe |
| `truncation.collapse_sample(cache, seed)` | everything up to `cache` terms; above it, one string drawn by `\|c\|**2` with coefficient `1`; partition/comm-safe |

## Combinators

| Expression | Keeps |
|---|---|
| `a & b` | both |
| `a \| b` | either; per-term filters only (`coeff`, `weight` and their compositions), so a `topn`, `approx_topn` or `collapse_sample` on either side raises `ValueError` |

## `topn` vs. `approx_topn`

Both keep every term above a boundary and fill up to `k` from the boundary itself: for `topn` the terms tied at the `k`-th largest magnitude, for `approx_topn` the octave of `\|c\|**2` where the count from the top first exceeds `k`.
Boundary terms are taken in storage order (bucket order, rank order across partitions), so which of them survive depends on the bucketing and partitioning and is not part of the contract; runs that differ in either agree on the count and on every term above the boundary.
`topn` raises `NotImplementedError` under `partitions=`/`comm=`; `approx_topn` is the partitioned/distributed default in that case.

See [Truncation](../manual/propagation/truncation.md#choosing-a-policy) for guidance and [Truncation](../manual/propagation/truncation.md) for the mechanism.

## `collapse_sample`

`collapse_sample(cache, seed)` is one Monte Carlo trajectory: the draw at the `k`-th layer pass is a pure function of `(seed, k)`, independent of the thread count, so the same arguments repeat a trajectory.
The returned object carries the trajectory's position, so propagating with it again (or with a policy composed from it) continues the sequence; call the factory again to restart.
At one partition the trajectory is the unpartitioned one; across partition counts only its distribution agrees.
`PropagationStats.collapses` counts one call's collapses.

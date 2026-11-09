# Truncation policies

All seven live in the `paulistrings.truncation` module and run after every channel.

| Policy | Keeps |
|---|---|
| `truncation.coeff(eps)` | `\|c\| > eps` — note **strictly** greater; `\|c\| == eps` is dropped |
| `truncation.weight(k)` | Pauli weight `<= k` |
| `truncation.topn(k)` | exactly `k` terms (all of a smaller sum), largest `\|c\|` first; no partitioned form |
| `truncation.approx_topn(k)` | exactly `k` terms, ranked by octave of `\|c\|**2`; partition/comm-safe |
| `truncation.partition_topn(k)` | `topn(k)` on each partition separately; partition/comm-safe |
| `truncation.partition_approx_topn(k)` | `approx_topn(k)` on each partition separately; partition/comm-safe |
| `truncation.collapse_sample(cache, samples, seed)` | everything up to `cache` terms; above it, `samples` systematic draws by `\|c\|`, an unbiased estimate of the sum; partition/comm-safe |

## Combinators

| Expression | Keeps |
|---|---|
| `a & b` | both |
| `a \| b` | either; per-term filters only (`coeff`, `weight` and their compositions), so a `topn`, `approx_topn`, their `partition_` forms or `collapse_sample` on either side raises `ValueError` |

## `topn` vs. `approx_topn`

Both keep every term above a boundary and fill up to `k` from the boundary itself: for `topn` the terms tied at the `k`-th largest magnitude, for `approx_topn` the octave of `|c|**2` where the count from the top first exceeds `k`.
Boundary terms are taken in bucket order, and across partitions each partition takes a share of the boundary in proportion to its boundary terms, so which of them survive depends on the bucketing and partitioning and is not part of the contract; runs that differ in either agree on the count and on every term above the boundary.
`topn` raises `NotImplementedError` under `partitions=`/`comm=`; `approx_topn` is the partitioned/distributed default in that case.

See [Truncation](../manual/propagation/truncation.md#choosing-a-policy) for guidance and [Truncation](../manual/propagation/truncation.md) for the mechanism.

## `collapse_sample`

`collapse_sample(cache, samples, seed)` replaces a sum of more than `cache` terms by `samples` systematic draws in proportion to `|c|`.
With `norm = Σ|c|` and the terms laid end to end in storage order, one uniform `u` places the points `(u + k)·norm/samples`, `k < samples`, and a term holding `m` of them survives with coefficient `m·norm/samples·c/|c|`.
Each term is drawn `⌊samples·|c|/norm⌋` or `⌈…⌉` times, so a collapse leaves at most `samples` terms and its expectation is the sum it replaced; the final sum, and any read-out linear in it, is an unbiased estimate, and a quantity quadratic in it is not.
`samples` is required and at least 1; a collapse resamples even when `samples` is at least the length.
One seed is one Monte Carlo trajectory: the draw at the `k`-th layer pass is a pure function of `(seed, k)`, independent of the thread count, so the same arguments repeat a trajectory.
Within one call a run of local layers whose worst-case growth stays within `8 × cache` may check the cache once, at its end, so a seed names a trajectory of that schedule.
The returned object carries the trajectory's position, so propagating with it again (or with a policy composed from it) continues the sequence; call the factory again to restart.
Across partitions the points are laid over the terms in rank order, then bucket order, so the group draws exactly `samples` points; at one partition the trajectory is the unpartitioned one, and across partition counts only its distribution agrees.
`PropagationStats.collapses` counts one call's collapses.

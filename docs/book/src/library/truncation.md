# Truncation policies

All four live in the `paulistrings.truncation` module and run after every channel.

| Policy | Keeps |
|---|---|
| `truncation.coeff(eps)` | `\|c\| > eps` — note **strictly** greater; `\|c\| == eps` is dropped |
| `truncation.weight(k)` | Pauli weight `<= k` |
| `truncation.topn(k)` | at most `k` terms, largest `\|c\|` first; exact, no partitioned form |
| `truncation.approx_topn(k)` | approximately `k` terms; partition/comm-safe |

## Combinators

| Expression | Keeps |
|---|---|
| `a & b` | both |
| `a \| b` | either |

## `topn` vs. `approx_topn`

`topn` never splits a tie group of exactly-equal coefficient magnitudes: the whole group is kept if it fits within `k`, dropped whole otherwise.
`approx_topn` bins by octave of `\|c\|**2` and keeps whole octaves top-down while they fit in `n`: at most `n` is kept, the shortfall is bounded by the coarsest excluded octave's population, and a tie group is always kept whole.
`topn` raises `NotImplementedError` under `partitions=`/`comm=`; `approx_topn` is the partitioned/distributed default in that case.

See [Truncation](../manual/propagation/truncation.md#choosing-a-policy) for guidance and [Truncation](../manual/propagation/truncation.md) for the mechanism.

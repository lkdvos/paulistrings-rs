# NUMA partitions

In a single unpartitioned process, set thread count via `RAYON_NUM_THREADS` **before the interpreter starts** — Rayon builds its global pool at the first `propagate` call and never resizes it, so setting the variable from inside an already-running script doesn't reliably reach it:

```bash
RAYON_NUM_THREADS=32 python my_script.py
```

On a multi-socket machine, run partitioned instead: one pinned pool and one share of the sum per NUMA domain.

<!-- doctest: skip -->
```python
evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    partitions="auto",   # one partition per NUMA node, rounded down to a power of two
)
```

`RAYON_NUM_THREADS` does not reach a partitioned run — the thread count comes from the placement instead.
`partitions` also takes an integer (a power of two, at most 16) or an explicit list of CPU lists, one per partition:

<!-- doctest: skip -->
```python
partitions=["0-7,16-23", "8-15,24-31"]
```

`paulistrings.numa_nodes()` reports what the machine offers, one CPU list per node intersected with this process's affinity mask, so a script can pick a split for itself.
`pin_memory=False` pins the threads but not their allocations; the default pins both.

Exact `truncation.topn` is unavailable under partitioning — use `truncation.approx_topn(n)`, which is partition-exact.

<!-- doctest: skip -->
```python
_, stats = observable.propagate_with_stats(circuit, policy, partitions="auto")
print(stats.partition.rows_exported)   # per-layer bytes/rows sent, None if unpartitioned
```

See [NUMA nodes](../explanation/numa.md) for what this costs and buys, and when the exchange makes it not worth it.

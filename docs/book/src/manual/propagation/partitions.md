# NUMA partitions

A partitioned run splits the sum across NUMA domains instead of sharing one thread pool across the whole box.
Each domain gets its own pinned pool and its own share of the terms, selected by designated rows of the GF(2) hash, and a layer exchanges only the rows that cross a domain boundary.
The mechanism is in [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Partitioning and the engine it runs on is [Inside the engine](engine.md); this page is when to reach for it and how.
To split across *processes* instead, for capacity rather than bandwidth, see [MPI ranks](mpi.md).

## When partitioning pays {#when-it-pays}

**Locality decides everything**, in the sense [Engine performance](../../examples/benchmarks/engine-performance.md#partitioned) measures in full: a dense two-qubit layer with no row crossing runs 4–18% faster at `P = 2` than `P = 1`, while any layer that exports rows runs 2–8.4× slower.

Random partition rows put roughly half of a dense two-qubit gate's deltas across a boundary, so the default draw lands a mixed circuit in the slow case.
**In-process partitioning is worth it when the exchange is rare**, and there are two ways to make it rare.
When the circuit has a known geometry, `partition_row_blocks=` replaces the random draw with an explicit cut — one contiguous block of qubits per partition — so only gates that straddle the cut exchange; the kwarg is described under [MPI ranks](mpi.md#partition-row-selection) and works identically under `partitions=`.
On a heavy-hex kicked-Ising step, cut rows made `P = 2` 15–21% faster per step than the single-process engine ([`research/FINDINGS.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/FINDINGS.md) §Cut partition rows).
Choosing rows automatically for a circuit with no obvious geometry is open research (`research/FINDINGS.md` §Partition rows without a known lattice).

## Python

The snippets below are skipped by the doc checker because they need a multi-node NUMA host to do anything.

<!-- doctest: skip -->
```python
evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    partitions="auto",            # one partition per NUMA node, rounded down to a power of two
)
```

`partitions` takes `"auto"`, an integer (a power of two, at most 64), or an explicit list of CPU lists — one per partition, in the `"0-7,16-23"` spelling — when the automatic split is not the one you want:

<!-- doctest: skip -->
```python
partitions=["0-7,16-23", "8-15,24-31"]
```

`partitions=None` or `partitions=1` is the ordinary unpartitioned run, bit for bit.
On a single-node box `"auto"` also falls back to the unpartitioned path, and an integer larger than the node count this process may run on is a `ValueError` that points at explicit CPU lists.

[`paulistrings.numa_nodes()`](../../library/module-helpers.md) reports what the machine offers, as one CPU list per node intersected with this process's affinity mask, so a script can decide for itself before choosing.
`pin_memory=False` pins the threads but not their allocations; the default binds both, which is the point of the placement.

`propagate_with_stats` fills `PropagationStats.partition` on a partitioned run: per layer, the terms each partition held, the rows and bytes it sent, and the imbalance across partitions.
It is `None` for an unpartitioned run.
`local[k]` says whether layer `k` exchanged at all, which is the figure the cost/benefit split above turns on.

<!-- doctest: skip -->
```python
_, stats = observable.propagate_with_stats(circuit, policy, partitions="auto")
print(stats.partition.rows_exported)   # rows sent across a boundary, per layer
print(stats.partition.local)           # True where a layer exchanged nothing
```

The other fields are under [`PartitionStats`](../../library/propagate.md#partitionstats).

## What changes under partitioning

**`RAYON_NUM_THREADS` is ignored.**
The partitioned engine builds one pool per partition from the placement, not from Rayon's global pool.
An explicit CPU-list placement takes each partition's thread count from its list length; `"auto"` and an integer count take it from the CPUs of the partition's NUMA node inside this process's affinity mask.
The variable still governs an unpartitioned run ([Stats, memory and logging](settings.md#threads)).

**`engine` is ignored.** A partitioned run is always the bucketed engine; the `"auto"`/`"direct"` small-sum paths have no partitioned form.

**Exact `topn` is unavailable.**
Choosing the `n`-th largest magnitude across partitions is a distributed selection, and the engine has no collective form for it, so a partitioned run raises `NotImplementedError` on `truncation.topn`.
Use `truncation.approx_topn(n)`, which is *partition-exact*: its histogram is all-reduced, so the retained set is exactly the set the single-partition run would have kept.
`coeff` and `weight` are per-term filters and need nothing.

## Limits

- **Pinning is Linux-only.** Elsewhere the topology module reports one node and pins nothing, so the run is correct but unplaced.
- **`P` is a power of two, at most 64.** A partition is named by `log2(P)` GF(2) hash rows, and the engine caps that at 6 (`P_MAX_BITS`).
- **Peak memory is above a single-pool run's.** A partitioned run holds `P` shares of the sum plus one layer's exchange buffers, which are pooled and reused rather than freed between layers.
- **Results agree to floating-point tolerance, not bit for bit** — as they do across bucket counts.
  At `P = 1` the output is bitwise the unpartitioned engine's.

## See it in use

[Engine performance](../../examples/benchmarks/engine-performance.md#partitioned) has the `P = 2` versus `P = 1` tables per host and thread count, the per-layer times of an exchange-free layer, the bandwidth counters behind the gain, and the benchmarking protocol for reproducing them.

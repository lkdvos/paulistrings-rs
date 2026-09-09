# Running across NUMA nodes

A partitioned run splits the sum across NUMA domains instead of sharing one pool across the whole
box. Each domain gets its own pinned thread pool and its own share of the terms, and a layer
exchanges only the rows that cross a domain boundary. The mechanism is in
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md)
§Partitioning; this page is when to reach for it and how.

## When it should help

The single-pool engine places pages by first touch and then work-steals across sockets, so about
half of every second socket's reads are remote. That is why the second socket adds only 15–25% of
bandwidth rather than 2× on the reference host, and why the dense two-qubit PTM class sits at the
machine's write ceiling from 16 threads up ([Performance](performance.md)). A bandwidth-bound layer
is the case partitioning targets: each domain reads and writes its own pages, and the only
cross-socket traffic is the exchange itself.

Reach for it when the profile says **bandwidth-bound and flat past one socket**: dense two-qubit
unitaries (`unitary_2q` with a full transfer matrix), large sums, high thread counts.

## When it will not

A latency-bound layer has nothing to recover. Rotation and Clifford circuits already scale to
11–13× at 32 threads with most of their traffic cache-served, and partitioning adds an export pass
and an all-to-all per layer on top. A rotation whose generator crosses a partition boundary is the
worst shape: it exports one row per anticommuting term.

Two structural limits also apply. Pinning is Linux-only — elsewhere the run is correct but
unplaced. And a partitioned run holds `P` shares of the sum plus the exchange blocks, so peak
memory is above a single-pool run's.

## Python

```python
evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    partitions="auto",            # one partition per NUMA node, rounded down to a power of two
)
```

`partitions` takes `"auto"`, an integer (a power of two, at most 16), or an explicit list of CPU
lists — one per partition, in the `"0-7,16-23"` spelling — when the automatic split is not the one
you want:

```python
partitions=["0-7,16-23", "8-15,24-31"]
```

`paulistrings.numa_nodes()` reports what the machine offers, as `(node, cpus)` pairs, so a script
can decide for itself before choosing.

`propagate_with_stats` fills `PropagationStats.partition` on a partitioned run: per layer, the
terms each partition held, the rows and bytes it sent, and the imbalance across partitions. It is
`None` for an unpartitioned run.

To split across *processes* instead of pools — one partition per MPI rank, across sockets or nodes —
see [Running across MPI ranks](mpi.md).

## Rust

```rust
use paulistrings::engine::partitioned::{
    propagate_partitioned, PartitionConfig, Placement,
};

let config = PartitionConfig {
    placement: Placement::Auto { max_partitions: None },
    bind_memory: true,
    partition_row_seed: None,
};
let evolved = propagate_partitioned(
    &circuit, sum, &ApproxTopN(1_000_000), Direction::Heisenberg, &config,
)?;
```

`Placement::Explicit(vec![CpuSet::parse("0-7,16-23")?, CpuSet::parse("8-15,24-31")?])` fixes the
split by hand; `Placement::Unpinned { partitions, threads_per_partition }` gives the shape of a
partitioned run with no pinning at all, which is what the test suite uses.

Hold a `PartitionedSum` instead of calling the one-shot entry point when a driver steps an
observable through many circuits: it scatters once, propagates per step, and gathers once.
`PartitionRuntime` is built from the config and shared behind an `Arc`, so the pinned pools are
paid for once.

## Two things that change

**`RAYON_NUM_THREADS` is ignored.** The partitioned engine builds one pool per partition from the
placement, not from Rayon's global pool, so the thread count comes from the CPU sets. Under
`Placement::Unpinned` it comes from `threads_per_partition`. The variable still governs an
unpartitioned run ([Getting started](../getting-started.md#threads)).

**Exact `topn` is unavailable.** Choosing the `n`-th largest magnitude across partitions is a
distributed selection, and the engine has no collective form for it — a partitioned run rejects
`truncation.topn`. Use `truncation.approx_topn(n)`, which is *partition-exact*: its histogram is
all-reduced, so the retained set is exactly the set the single-partition run would have kept.
`coeff` and `weight` are per-term filters and need nothing.

## Numbers to come

The partitioned engine has landed but has no committed measurement yet. The measurement plan is a
P=1 versus P=N runtime-knob A/B of one binary on a quiet exclusive node
(`scripts/slurm/ab-campaign.sbatch`), with the roofline denominators re-measured on that node. This
page will carry the results when they exist; until then treat partitioning as a facility to try on
a bandwidth-bound workload, not as a documented speedup.

# Running across NUMA nodes

A partitioned run splits the sum across NUMA domains instead of sharing one pool across the whole box.
Each domain gets its own pinned thread pool and its own share of the terms, selected by designated rows of the GF(2) hash, and a layer exchanges only the rows that cross a domain boundary.
The mechanism is in [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Partitioning; this page is when to reach for it and how.

## What it costs and what it buys

**Locality decides everything.** A layer whose gates leave the partition rows fixed runs entirely inside its own domain and gains; a layer that moves any row across a boundary pays for the export pass, the copy and the receiver's larger merge stream.
Measured at `P = 2` against `P = 1` on the same binary, on two-socket hosts from 8 to 48 cores per socket:

| layer class | `P = 2` vs `P = 1` |
|---|---|
| dense two-qubit unitaries, no row crossing | **4–18% faster**, direction-consistent on every host |
| Pauli rotations, no row crossing | coset loop 12–22% faster on two of three hosts |
| any layer that exports rows (random rows) | **2–8× slower**, and worse at higher core counts |

The gain grows with cores per socket, as a memory-system effect should: −4% on an 8-core-per-socket workstation, −18% on a 48-core Genoa at 96 threads.
Load imbalance across partitions stays within 1.000–1.004 with random rows.
Full tables: [`research/HARDWARE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/HARDWARE.md) §Partitioned engine, P=1 vs P=2 in-process.

Random partition rows put roughly half of a dense two-qubit gate's deltas across a boundary, so the default draw lands a mixed circuit in the bottom row of that table.
**In-process partitioning is worth it when the exchange is rare**, and choosing rows that make it rare — rows reading the qubits on the boundary of a spatial cut, so only cut-crossing gates exchange — is open research (`research/FINDINGS.md`).
To split across *processes* instead, for capacity rather than bandwidth, see [Running across MPI ranks](mpi.md).

## Python

<!-- doctest: skip -->
```python
evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    partitions="auto",            # one partition per NUMA node, rounded down to a power of two
)
```

`partitions` takes `"auto"`, an integer (a power of two, at most 16), or an explicit list of CPU lists — one per partition, in the `"0-7,16-23"` spelling — when the automatic split is not the one you want:

<!-- doctest: skip -->
```python
partitions=["0-7,16-23", "8-15,24-31"]
```

`paulistrings.numa_nodes()` reports what the machine offers, as one CPU list per node intersected with this process's affinity mask, so a script can decide for itself before choosing.
`pin_memory=False` pins the threads but not their allocations; the default binds both, which is the point of the placement.

`propagate_with_stats` fills `PropagationStats.partition` on a partitioned run: per layer, the terms each partition held, the rows and bytes it sent, and the imbalance across partitions.
It is `None` for an unpartitioned run.
`local[k]` says whether layer `k` exchanged at all, which is the figure the table above turns on.

## Two things that change

**`RAYON_NUM_THREADS` is ignored.** The partitioned engine builds one pool per partition from the placement, not from Rayon's global pool, so the thread count comes from the CPU sets.
Under `Placement::Unpinned` it comes from `threads_per_partition`.
The variable still governs an unpartitioned run ([Running on NUMA partitions](../how-to/run-on-numa-partitions.md)).

**Exact `topn` is unavailable.** Choosing the `n`-th largest magnitude across partitions is a distributed selection, and the engine has no collective form for it, so a partitioned run rejects `truncation.topn`.
Use `truncation.approx_topn(n)`, which is *partition-exact*: its histogram is all-reduced, so the retained set is exactly the set the single-partition run would have kept.
`coeff` and `weight` are per-term filters and need nothing.

## Limits

- **Pinning is Linux-only.** Elsewhere the topology module reports one node and pins nothing, so the run is correct but unplaced.
- **`P` is a power of two, at most 16.** A partition is named by `log2(P)` GF(2) hash rows.
- **Peak memory is above a single-pool run's.** A partitioned run holds `P` shares of the sum plus one layer's exchange buffers, which are pooled and reused rather than freed between layers.
- **Results agree to floating-point tolerance, not bit for bit** — as they do across bucket counts.
  At `P = 1` the output is bitwise the unpartitioned engine's.

## Measuring it

`P = 1` versus `P = N` is a runtime-knob A/B of one binary, not a code A/B:

```bash
scripts/ab-compare.sh partitions-1v2 --a . --b . \
  --probe   '--n 1000000 --threads 32 --layers su4_local --partitions 1' \
  --probe-b '--n 1000000 --threads 32 --layers su4_local --partitions 2 \
             --partition-cpus "0-15,32-47;16-31,48-63"'
```

`--threads` is the total at every `P`.
A partitioned cell runs under **no** placement prefix: `numactl` and `taskset` both defeat the split the engine is making.
On an exclusive cluster node, `scripts/slurm/ab-campaign.sbatch` builds the CPU lists from the node's own topology.
The protocol and the sidecar fields are in [`benchmarks/PROFILING.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/PROFILING.md).

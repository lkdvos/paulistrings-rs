# propagate / propagate_with_stats

```text
sum.propagate(circuit, policy=None, direction=None, target_bucket_len=1024,
               min_buckets=128, partitions=None, pin_memory=True,
               partition_row_seed=None, partition_row_blocks=None,
               partition_row_exclude=None, partition_rows=None,
               comm=None, result="gather", device=None) -> None

sum.propagate_with_stats(circuit, policy=None, direction=None, target_bucket_len=1024,
                          min_buckets=128, partitions=None, pin_memory=True,
                          partition_row_seed=None, partition_row_blocks=None,
                          partition_row_exclude=None, partition_rows=None,
                          comm=None, result="gather", device=None)
    -> PropagationStats

sum.copy() -> PauliSum
```

Both evolve `sum` in place: the engine takes its terms over instead of copying them, so no second copy of the input is alive during the run.
Propagate a `sum.copy()` (or `copy.copy(sum)`) to keep the input.
A call that raises before the first layer, on a bad keyword or a mismatched circuit, leaves `sum` untouched; one that fails during the run leaves it empty.
`propagate_with_stats` takes the same arguments and evolves `sum` the same way (to floating-point tolerance) as `propagate`, and returns a `PropagationStats` recording per-layer counts.
The GIL is released for the duration of both calls.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `circuit` | `Circuit` | required | must share `num_qubits` with `sum` |
| `policy` | `Truncation \| None` | `None` | `None` applies no per-term filtering beyond the engine's exact-zero drop |
| `direction` | `"forward" \| "heisenberg" \| None` | `None` = `"forward"` | **the default is not the Heisenberg picture most examples here use** — pass it explicitly; see [Direction semantics](direction.md) |
| `target_bucket_len` | `int` | `1024` | engine's per-layer bucket-sizing knob; at least 1, else `ValueError` |
| `min_buckets` | `int` | `128` | engine's per-layer bucket-sizing knob; at least 16, else `ValueError` |
| `partitions` | `None \| "auto" \| int \| list[list[int]]` | `None` | splits the sum across NUMA domains; see below |
| `pin_memory` | `bool` | `True` | binds each partition's allocations to its node |
| `partition_row_seed` | `int \| None` | `None` = the sum's own hash seed | which GF(2) rows decide a term's partition |
| `partition_row_blocks` | `list[list[int]] \| None` | `None` | explicit disjoint qubit blocks, one per partition, instead of a seeded draw |
| `partition_row_exclude` | `dict[str, list[int]] \| None` | `None` | `{"x": [...], "z": [...]}`: the seeded draw with these qubits' x-bits / z-bits read by no row |
| `partition_rows` | `bool array \| None` | `None` | the rows outright, shape `(log2(P), 2 * num_qubits)`, x-bits first, as `PauliSum.hash_rows()` lays them out and [`paulistrings.hashes`](#paulistringshashes) builds them |
| `comm` | `mpi4py.MPI.Comm \| None` | `None` | run one partition per MPI rank instead of `partitions` |
| `result` | `"gather" \| "local"` | `"gather"` | only read under `comm=`: `"gather"` leaves the whole sum on rank 0 and an empty one elsewhere; `"local"`, each rank's own share as a plain `PauliSum`, is deprecated in favour of [`PauliSum.partition`](#partitionedpaulisum) |
| `device` | `None \| int \| list[int] \| "auto"` | `None` | run on CUDA devices instead of the host; see below |

`PauliSum.num_buckets` reads back the realized bucket count, which can differ from `target_bucket_len`/`min_buckets` since bucketing only ever grows.

## `partitions`

| Value | Placement |
|---|---|
| `None` or `1` | unpartitioned |
| `"auto"` | one partition per NUMA node (falls back to unpartitioned on a single-node box) |
| an `int` power of two `>= 2` | caps placement at that many nodes |
| `list[list[int]]` | explicit disjoint CPU lists, one partition per list |

In partitioned mode `RAYON_NUM_THREADS` and `engine` are ignored, and `truncation.topn` raises `NotImplementedError` (use `approx_topn`).

`partition_row_blocks` excludes `partition_row_seed` and `partition_row_exclude`, and `partition_rows` excludes every other row keyword; all but the seed need `partitions=`, `comm=` or a list of several devices.
A partition is named by `log2(P)` rows over the `2 * num_qubits` key coordinates, so `P` is at most `2^min(2 * num_qubits, 31)`.
Rows that leave some partitions unreachable (linearly dependent rows, or an exclusion that leaves no coordinate readable) run with a `RuntimeWarning` naming the shortfall; only a malformed keyword is a `ValueError`.
Rows that overlap the span of the bucket hash's active rows correlate partitions with buckets, so some of their pairs stay empty; that costs balance only once the run refines to the overlapping bucket row, so it is logged as a warning on the `paulistrings` logger at that layer, once per run.
`partition_row_exclude={"x": sites}` keeps strings that differ only in the x-bits of `sites` on one rank; `{"z": sites}` does the same for z-bits.
Under `comm=`, the block count in `partition_row_blocks` or the row count in `partition_rows` must equal the MPI group size, and the blocks or rows must be identical on every rank; a group whose ranks disagree fails on every rank rather than misrouting the exchange.

`comm=` requires `MPI_THREAD_SERIALIZED` set before importing MPI, a power-of-two rank count, and every rank calling with the same replicated input in the same order.
`comm=` and `partitions=` are alternatives — place via the launcher (e.g. `mpirun --map-by ppr:1:numa --bind-to numa`) rather than both. Without the `mpi` feature, `comm=` raises `RuntimeError`.

See [NUMA partitions](../manual/propagation/partitions.md) and [MPI ranks](../manual/propagation/mpi.md) for recipes.

## `device`

| Value | Placement |
|---|---|
| `None` | the host engine |
| an `int` | the whole sum on that CUDA device ordinal; `ValueError` if this process cannot see it |
| `list[int]` | one partition per entry, in partition order; a one-element list is that device |
| `"auto"` | devices `0..k` for the largest power of two `k` visible, so device 0 alone on a one-GPU box |

A list's length must be a power of two, at most `2^min(2 * num_qubits, 31)`, or it is a `ValueError`; an ordinal may repeat, putting several partitions on one device.
Several devices split the sum like `partitions=`: the `partition_row_*` keywords and `partition_rows` pick the rows, the block or row count matching the list length.
`device=` is an alternative to `partitions=`; passing both, or `result="local"` without `comm=`, or `partition_row_blocks=` with a single device, is a `ValueError`.
`engine` is ignored, `truncation.topn` runs on one device and raises `NotImplementedError` on several or under `comm=` (use `approx_topn`), an exhausted device raises `MemoryError`, and without the `cuda` feature or a visible device `device=` raises `RuntimeError`.

With `comm=`, `device=` is this rank's one device: an `int` or `"auto"` (a device near the rank's CPUs, `gpu::local_device_for_comm`), and a list of several ordinals is a `ValueError`.
`result=` and the row kwargs mean what they mean on the host `comm=` path, and a rank that cannot use its device fails the call on every rank.
The pair needs the extension built with both the `cuda` and `mpi` features, and raises `RuntimeError` naming the missing one otherwise.
See [CUDA devices](../manual/propagation/gpu.md).

## `paulistrings.hashes`

```text
hashes.dense(num_qubits, rows, seed, exclude=None) -> bool array (rows, 2 * num_qubits)
hashes.sparse(num_qubits, rows, weight, seed, exclude=None) -> bool array
hashes.projection(num_qubits, rows, seed, exclude=None) -> bool array
hashes.rank(matrix) -> int
hashes.remote_layers(rows, deltas) -> bool array (channels,)
```

Pure-NumPy builders of the GF(2) row matrices `PauliSum.with_hash(rows=)` and `propagate(partition_rows=)` take, laid out as `PauliSum.hash_rows()` returns them: column `c < num_qubits` is the x-bit of qubit `c`, column `num_qubits + c` its z-bit.
`dense` sets each coordinate with probability 1/2; `sparse` reads exactly `weight` coordinates per row; `projection` reads one coordinate per row, distinct until they run out.
Each draws from `numpy.random.default_rng(seed)` and keeps its rows linearly independent as far as the family reaches, redrawing a dependent candidate: `dense` and `projection` up to the readable coordinate count, `sparse` with an even `weight` one short of it (such rows span only the even-weight subspace), `rank` says what was reached.
`exclude={"x": qubits, "z": qubits}` keeps those qubits' x- and z-bits out of every row, as `partition_row_exclude` does for the seeded draw.

## `PartitionedPauliSum`

```text
sum.partition(partitions=None, pin_memory=True, partition_row_seed=None, partition_row_blocks=None,
              partition_row_exclude=None, partition_rows=None, comm=None) -> PartitionedPauliSum
```

A sum kept split between calls, over this process's NUMA partitions (`partitions=`, default `"auto"`, where one partition is a one-pool split) or over the ranks of an MPI group (`comm=`); the row keywords pick the initial partition rows, as for `propagate`.

| member | |
|---|---|
| `.propagate(circuit, policy=None, direction=None, target_bucket_len=1024, min_buckets=128)` | in place; `.propagate_with_stats(...)` returns a `PropagationStats` |
| `.repartition(*, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, partition_rows=None)` | move every term to the partition the new rows name, in one exchange, without gathering |
| `.rehash(rows=None, seed=None)` | rebucket every partition by another hash, as `PauliSum.with_hash`; nothing is exchanged |
| `.expectation(state="x+")`, `len()` | of the whole sum, the same on every rank |
| `.gather()` | the whole sum, on rank 0 under MPI (empty elsewhere) |
| `.local()` | this process's terms: the whole sum in one process, this rank's share under MPI |
| `.partition_rows()`, `.hash_rows()`, `.hash_seed`, `.num_buckets`, `.num_partitions`, `.rank`, `.num_qubits` | the current rows and placement; `rank` is `None` in one process |
| `.dtype` | the coefficient type of the sum it was split from, which `gather()` and `local()` keep |

Under MPI every member that reads or changes the whole sum is collective; `local()`, `repr()` and the properties are not.
Exact `truncation.topn` raises `NotImplementedError`, as for any partitioned run.

`Circuit.deltas()` lists each channel's nonzero key shifts as `(k, 2 * num_qubits)` boolean arrays, and `paulistrings.hashes.remote_layers(rows, deltas)` marks the channels a choice of partition rows makes exchange: exactly the layers `PartitionStats.local` reports as not local.

## `GpuPauliSum`

```text
sum.to_device(device=0) -> GpuPauliSum

resident.propagate(circuit, policy=None, direction=None,
                   target_bucket_len=1024, min_buckets=128) -> None
resident.propagate_with_stats(circuit, policy=None, direction=None,
                              target_bucket_len=1024, min_buckets=128) -> PropagationStats
resident.to_host() -> PauliSum
```

A sum resident on one CUDA device: `propagate` steps it **in place**, `to_host` copies it back and leaves it resident.
The class exists only in a build with the `cuda` feature; elsewhere `to_device` raises `RuntimeError`.
The arguments mean what they mean on `PauliSum.propagate`, with the same errors.
A device error mid-run leaves the sum holding the last completed layer's output.

| Attribute | Type | Meaning |
|---|---|---|
| `len(resident)` | `int` | terms on the device |
| `.num_qubits` | `int` | the sum's qubit count |
| `.device` | `int` | the CUDA device ordinal |
| `.dtype` | `numpy.dtype` | the coefficient type, kept by `to_host` |
| `.num_buckets` | `int` | the device partition's current bucket count, grow-only |

## `PropagationStats`

One entry per layer applied, in application order (reverse circuit order under `direction="heisenberg"`).

A **layer** here is one applied channel — one gate, or one noise channel on one qubit — not a brickwork layer of parallel gates in the field-standard sense.
A broadcast call such as `circuit.depolarize(p, [0, 1])` contributes two layers, and `len(circuit) == stats.layers`.

| Field | Type | Meaning |
|---|---|---|
| `.layers` | `int` | number of layers (channels) applied |
| `.terms_in` | `list[int]` | term count before each layer; `terms_in[k+1] == terms_out[k]` |
| `.terms_out` | `list[int]` | term count after each layer's truncation |
| `.peak_terms` | `int` | peak resident term count between layers (not the transient in-layer expansion) |
| `.final_terms` | `int` | `terms_out[-1]`, or the input's count for a zero-layer circuit |
| `.circuit_index` | `list[int]` | this layer's position in the circuit as written, independent of `direction` |
| `.application_index` | `list[int]` | this layer's position in the propagation loop, `0..layers` regardless of `direction` |
| `.gate_name` | `list[str]` | the applied channel's debug name per layer: the `PtmChannel` constructor name (`hadamard`, `cnot`, `depolarizing`, …; `phase_gate_adjoint` for `sdg`) or `PauliRotation` |
| `.nanos` | `list[int]` | elapsed wall-clock nanoseconds per layer; for a partitioned/distributed run, the max over partitions/ranks |
| `.partition` | `PartitionStats` | per-partition detail; one partition for an unpartitioned call |
| `.collapses` | `int \| None` | collapses a `truncation.collapse_sample` made during this call, `None` without one; under `comm=` rank 0's count, all-reduced so every rank reports it |

## `PartitionStats`

Per-layer, per-partition detail (`PropagationStats.partition`), one entry per layer in application order.

| Field | Type | Meaning |
|---|---|---|
| `.partitions` | `int` | partition count (power of two); the MPI group size under `comm=` |
| `.rank` | `int \| None` | this process's rank under `comm=`, else `None` |
| `.size` | `int \| None` | the `comm=` group size, else `None` |
| `.devices` | `list[int] \| None` | the CUDA device of each partition for a `device=` run (this rank's one device under `comm=`), else `None` |
| `.local` | `list[bool]` | whether each layer moved no row across a partition boundary (`rows_exported[k] == 0`) |
| `.rows_exported` | `list[int]` | rows sent across partition boundaries per layer, summed over sender/receiver pairs |
| `.bytes_exported` | `list[int]` | wire bytes behind `rows_exported`, including per-block headers |
| `.terms_in` | `list[list[int]]` | `terms_in[k][r]`: terms partition `r` held before layer `k` |
| `.terms_out` | `list[list[int]]` | terms partition `r` held after layer `k`'s truncation |
| `.imbalance` | `list[float]` | max/mean of `terms_in` per layer; `1.0` is perfect balance, `partitions` is worst case |
| `.nanos` | `list[list[int]]` | each partition's own elapsed wall time per layer; `PropagationStats.nanos[k]` is `max(nanos[k])` |

Under `comm=`, a distributed run's per-layer lists (`terms_in`, `terms_out`, `rows_exported`, `bytes_exported`, `nanos`) hold this rank's own entry only — nothing gathers the group's counters, since that would add a collective per layer for a diagnostic. Reduce over `comm` for the group's picture; `imbalance` is always `1.0` in that case.

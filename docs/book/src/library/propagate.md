# propagate / propagate_with_stats

```text
sum.propagate(circuit, policy=None, direction=None, engine=None,
               small_sum_threshold=None, target_bucket_len=None, min_buckets=None,
               partitions=None, pin_memory=True, partition_row_seed=None,
               partition_row_blocks=None, comm=None, result="gather",
               device=None) -> PauliSum

sum.propagate_with_stats(circuit, policy=None, direction=None, engine=None,
                          small_sum_threshold=None, target_bucket_len=None, min_buckets=None,
                          partitions=None, pin_memory=True, partition_row_seed=None,
                          partition_row_blocks=None, comm=None, result="gather",
                          device=None)
    -> (PauliSum, PropagationStats)
```

`propagate_with_stats` takes the same arguments and returns the same evolved sum (to floating-point tolerance) as `propagate`, plus a `PropagationStats` recording per-layer counts.
The GIL is released for the duration of both calls.

## Parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `circuit` | `Circuit` | required | must share `num_qubits` with `sum` |
| `policy` | `Truncation \| None` | `None` | `None` applies no per-term filtering beyond the engine's exact-zero drop |
| `direction` | `"forward" \| "heisenberg" \| None` | `None` = `"forward"` | **the default is not the Heisenberg picture most examples here use** — pass it explicitly; see [Direction semantics](direction.md) |
| `engine` | `"sorted" \| "auto" \| "direct" \| None` | `None` = `"sorted"` | `"sorted"`: always bucketed. `"auto"`: a term-by-term hash-map path below `small_sum_threshold`, unless the policy has a layer pass (e.g. `topn`). `"direct"`: same threshold, always. All three agree to floating-point tolerance |
| `small_sum_threshold` | `int \| None` | `None` = `paulistrings.DEFAULT_SMALL_SUM_THRESHOLD` | term-count cutoff for `"auto"`/`"direct"` |
| `target_bucket_len` | `int \| None` | `None` = `1024` | sorting engine's per-layer bucket-sizing knob |
| `min_buckets` | `int \| None` | `None` = `128` | sorting engine's per-layer bucket-sizing knob |
| `partitions` | `None \| "auto" \| int \| list[list[int]]` | `None` | splits the sum across NUMA domains; see below |
| `pin_memory` | `bool` | `True` | binds each partition's allocations to its node |
| `partition_row_seed` | `int \| None` | `None` = the sum's own hash seed | which GF(2) rows decide a term's partition |
| `partition_row_blocks` | `list[list[int]] \| None` | `None` | explicit disjoint qubit blocks, one per partition, instead of a seeded draw |
| `comm` | `mpi4py.MPI.Comm \| None` | `None` | run one partition per MPI rank instead of `partitions` |
| `result` | `"gather" \| "local"` | `"gather"` | only read under `comm=`: `"gather"` returns the whole sum on rank 0 and empty elsewhere, `"local"` returns each rank's own disjoint share |
| `device` | `None \| int \| list[int] \| "auto"` | `None` | run on one CUDA device instead of the host; see below |

`PauliSum.num_buckets` reads back the realized bucket count, which can differ from `target_bucket_len`/`min_buckets` since bucketing only ever grows.

## `partitions`

| Value | Placement |
|---|---|
| `None` or `1` | unpartitioned — bit-for-bit today's path |
| `"auto"` | one partition per NUMA node (falls back to unpartitioned on a single-node box) |
| an `int` power of two `>= 2`, at most 64 | caps placement at that many nodes |
| `list[list[int]]` | explicit disjoint CPU lists, one partition per list |

In partitioned mode `RAYON_NUM_THREADS` and `engine` are ignored, and `truncation.topn` raises `NotImplementedError` (use `approx_topn`).

`partition_row_seed` and `partition_row_blocks` are mutually exclusive; either needs `partitions=` or `comm=`.
Under `comm=`, the block count in `partition_row_blocks` must equal the MPI group size, and the blocks must be identical on every rank.

`comm=` requires `MPI_THREAD_SERIALIZED` set before importing MPI, a power-of-two rank count, and every rank calling with the same replicated input in the same order.
`comm=` and `partitions=` are alternatives — place via the launcher (e.g. `mpirun --map-by ppr:1:numa --bind-to numa`) rather than both. Without the `mpi` feature, `comm=` raises `RuntimeError`.

See [NUMA partitions](../manual/propagation/partitions.md) and [MPI ranks](../manual/propagation/mpi.md) for recipes.

## `device`

| Value | Placement |
|---|---|
| `None` | the host engine |
| an `int` | that CUDA device ordinal; `ValueError` if this process cannot see it |
| `"auto"` | device 0 when exactly one device is visible |
| `list[int]` | one ordinal in a list is that device |

A list of several ordinals, or `"auto"` with more than one device visible, raises `NotImplementedError`: one device per process for now.
`device=` is an alternative to `partitions=` and `comm=`; passing it with either, with `result="local"` or with `partition_row_blocks=` is a `ValueError`.
`engine` is ignored, `truncation.topn` raises `NotImplementedError` (use `approx_topn`), an exhausted device raises `MemoryError`, and without the `cuda` feature or a visible device `device=` raises `RuntimeError`.
See [CUDA devices](../manual/propagation/gpu.md).

## `GpuPauliSum`

```text
sum.to_device(device=0) -> GpuPauliSum

resident.propagate(circuit, policy=None, direction=None,
                   target_bucket_len=None, min_buckets=None) -> None
resident.propagate_with_stats(circuit, policy=None, direction=None,
                              target_bucket_len=None, min_buckets=None) -> PropagationStats
resident.to_host() -> PauliSum
```

A sum resident on one CUDA device: `propagate` steps it **in place**, `to_host` copies it back and leaves it resident.
The arguments mean what they mean on `PauliSum.propagate`, with the same errors.
A device error mid-run leaves the sum holding the last completed layer's output.

| Attribute | Type | Meaning |
|---|---|---|
| `len(resident)` | `int` | terms on the device |
| `.num_qubits` | `int` | the sum's qubit count |
| `.device` | `int` | the CUDA device ordinal |
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
| `.gate_name` | `list[str]` | the applied channel's debug name per layer |
| `.nanos` | `list[int]` | elapsed wall-clock nanoseconds per layer; for a partitioned/distributed run, the max over partitions/ranks |
| `.partition` | `PartitionStats \| None` | per-partition detail for a `partitions=`/`comm=`/`device=` call, else `None` |

## `PartitionStats`

Per-layer, per-partition detail (`PropagationStats.partition`), one entry per layer in application order.

| Field | Type | Meaning |
|---|---|---|
| `.partitions` | `int` | partition count (power of two); the MPI group size under `comm=` |
| `.rank` | `int \| None` | this process's rank under `comm=`, else `None` |
| `.size` | `int \| None` | the `comm=` group size, else `None` |
| `.devices` | `list[int] \| None` | the CUDA device of each partition for a `device=` run (`partitions == 1`), else `None` |
| `.local` | `list[bool]` | whether each layer moved no row across a partition boundary (`rows_exported[k] == 0`) |
| `.rows_exported` | `list[int]` | rows sent across partition boundaries per layer, summed over sender/receiver pairs |
| `.bytes_exported` | `list[int]` | wire bytes behind `rows_exported`, including per-block headers |
| `.terms_in` | `list[list[int]]` | `terms_in[k][r]`: terms partition `r` held before layer `k` |
| `.terms_out` | `list[list[int]]` | terms partition `r` held after layer `k`'s truncation |
| `.imbalance` | `list[float]` | max/mean of `terms_in` per layer; `1.0` is perfect balance, `partitions` is worst case |
| `.nanos` | `list[list[int]]` | each partition's own elapsed wall time per layer; `PropagationStats.nanos[k]` is `max(nanos[k])` |

Under `comm=`, a distributed run's per-layer lists (`terms_in`, `terms_out`, `rows_exported`, `bytes_exported`, `nanos`) hold this rank's own entry only — nothing gathers the group's counters, since that would add a collective per layer for a diagnostic. Reduce over `comm` for the group's picture; `imbalance` is always `1.0` in that case.

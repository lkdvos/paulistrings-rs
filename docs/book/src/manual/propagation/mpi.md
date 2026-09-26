# MPI ranks

A distributed run is a [partitioned run](partitions.md) with one partition per MPI process: each rank holds a disjoint share of the terms, selected by designated rows of the GF(2) hash, and a layer exchanges only the rows that cross a rank boundary.
Everything on the NUMA page still applies — the same layer loop, the same truncation rules, the same things that change.
What is new is the transport (point-to-point MPI instead of in-process channels) and the launch ([`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Partitioning, subsection "Transport composition").

## Why: capacity

**The reason to reach for it is capacity.**
A sum that does not fit one node's memory fits `P` of them, and the per-rank overhead is bounded and does not grow with the rank count.
Measured at one rank per NUMA domain on Icelake `ccq` nodes, a layer with no row crossing costs about the same at 2, 4 and 8 ranks, and a Pauli rotation whose generator crosses costs 3.5× that intra-node and 4.4–4.7× inter-node — see [Engine performance](../../examples/benchmarks/engine-performance.md#distributed) for the full table.
A remote layer is **transfer-bound**, so the lever is fewer bytes (locality rows, a lower truncation), not more threads.

It is an off-by-default build option the released wheel omits, so the default wheel cannot do it.
[`paulistrings.mpi_available()`](../../library/module-helpers.md) says whether this build can, and `comm=` in a build without it raises `RuntimeError`.

## Building from source

The `mpi` feature needs an MPI installation (for `mpicc`, which rsmpi's build script probes) and a `libclang` for its bindgen; the pip build against a cluster's loaded MPI module is in [Installation](../../installation.md).

`mpi4py` must be built against the *same* MPI the extension links; the boundary checks the width of `MPI_Comm` and refuses a mismatch rather than corrupting a handle.
On a Flatiron host the `python-mpi` module provides a matching `mpi4py`, so a venv created with `--system-site-packages` from that interpreter sees it:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9
export LIBCLANG_PATH=$(llvm-config --libdir)

python3 -m venv --system-site-packages .venv-mpi   # so the module's mpi4py is visible
.venv-mpi/bin/pip install maturin pytest numpy
VIRTUAL_ENV=$PWD/.venv-mpi .venv-mpi/bin/maturin develop --release --features mpi \
    -m crates/paulistrings-py/Cargo.toml
```

The built extension carries an rpath to that MPI's library directory, so `import paulistrings` works from a shell with no modules loaded.
`scripts/mpi-test.sh --ranks 2,4 --python` builds the extension into that venv and runs `python/paulistrings/tests/test_mpi.py` under `mpirun` at each rank count.

## Python

The snippets on this page are skipped by the doc checker because they need an `mpi` build and an MPI launcher.

<!-- doctest: skip -->
```python
import mpi4py
mpi4py.rc.thread_level = "serialized"      # before mpi4py.MPI is imported
from mpi4py import MPI

import paulistrings
from paulistrings import truncation

comm = MPI.COMM_WORLD
observable = build_observable(seed=20260908)   # the SAME terms on every rank
circuit = build_circuit()

evolved = observable.propagate(
    circuit,
    truncation.approx_topn(1_000_000),
    direction="heisenberg",
    comm=comm,
)
if comm.Get_rank() == 0:
    print(len(evolved), evolved.expectation("z+"))
```

`comm` and `partitions` are alternatives — passing both is a `ValueError`.
A distributed run places one partition per *process*, so the placement is the launcher's job ([Launching](#launching)).

## Partition row selection

By default the rows are a GF(2)-random draw from the sum's own hash seed, which spreads a gate's deltas over every rank whatever the qubits' geometry.
`partition_row_seed=` picks the draw; `partition_row_blocks=` replaces it with an explicit locality cut, one contiguous list of qubit indices per rank, so a gate whose qubits share a block never exchanges at all:

<!-- doctest: skip -->
```python
half = observable.num_qubits // 2
blocks = [list(range(half)), list(range(half, observable.num_qubits))]
evolved = observable.propagate(circuit, policy, comm=comm, partition_row_blocks=blocks)
```

Both kwargs work the same way under `partitions=`, and the two are alternatives to each other.
The block count must equal the rank count, the blocks must be disjoint, and they must be identical on every rank — the split is a local filter each rank computes for itself.
A term's rank is then the XOR of the blocks it has odd Z-weight in, so a term supported inside one block belongs to that block's rank.

## result="gather" versus result="local"

| `result` | what each rank gets back |
|---|---|
| `"gather"` (default) | rank 0 the whole evolved sum; every other rank an **empty** `PauliSum` of the same `num_qubits`, so downstream code still type-checks |
| `"local"` | this rank's own share |

The shares are disjoint and cover the result, so reductions over `comm` are exact:

<!-- doctest: skip -->
```python
local = observable.propagate(circuit, policy, comm=comm, result="local")
terms = comm.allreduce(len(local))
value = comm.allreduce(local.expectation("z+"))
```

Prefer `"local"` when the answer is a scalar: gathering materializes the whole sum on rank 0, and that rank transiently holds one extra rank's copy on top of it.

## Launching

One rank per NUMA domain, bound to it — that is the placement the in-process engine would have made with explicit CPU lists, done by the launcher instead:

```bash
mpirun -n 4 --map-by ppr:1:numa --bind-to numa python script.py
srun --ntasks-per-node=2 -c 32 --cpu-bind=ldoms --mpi=pmix python script.py
```

`--ntasks-per-node` is the node's NUMA-domain count and `-c` its cores per domain.
Each rank's Rayon pool sizes itself from the CPUs the launcher left in its affinity mask, so an unbound launch gives every rank a pool over the whole node and they fight.
`RAYON_NUM_THREADS` is ignored, as in any partitioned run.

On Rusty, `scripts/slurm/mpi-ranks.sbatch` does the arithmetic: it reads the node's domain count, rounds `nodes × domains` down to a power of two, and runs the differential net and then the probe at that rank count.
See [`scripts/slurm/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/scripts/slurm/README.md).

## One GPU per rank {#gpu-per-rank}

Built with both `mpi` and `cuda`, the Rust driver `gpu::MpiGpuSum` holds each rank's share on one CUDA device, with the same contract as `MpiSum`: replicated input, collective calls, rank 0 gathers.
`gpu::local_device_for_comm` picks the device collectively over the ranks sharing a node: each rank gets a device on a NUMA node its CPUs are on where it can, ranks on one node get distinct devices while there are enough, and without readable NUMA facts it falls back to `gpu::local_device_for_rank`, the launcher's node-local rank (`OMPI_COMM_WORLD_LOCAL_RANK`, `MV2_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`) modulo the visible devices:

<!-- doctest: skip -->
```rust
use paulistrings::engine::partitioned::{Collectives, PartitionRowPolicy};
use paulistrings::gpu::{local_device_for_comm, MpiGpuSum};
use paulistrings::mpi::{rsmpi, MpiTransport};

let (universe, _) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized).unwrap();
let world = universe.world();
let device = local_device_for_comm(&world)?;
let transport = MpiTransport::from_communicator(&world);
let mut split = MpiGpuSum::<2>::scatter(observable, transport, device, &PartitionRowPolicy::Seeded(None))?;
split.propagate(&circuit, &ApproxTopN(10_000_000), Direction::Heisenberg)?;
if let Some(evolved) = split.gather()? {
    println!("{} terms", evolved.len());
}
```

`gpu::propagate_mpi_gpu` is the one-shot form.
A device failure on any rank fails the call on every rank, with `GpuError::Poisoned` naming the failing rank on its peers, so the group never falls out of step.
Launch with one visible device per task (`srun --gpus-per-task=1 --mpi=pmix`, as `scripts/slurm/mpi-gpu-ranks.sbatch` does) or with every node GPU visible to every task (`--gpus-per-node`, as `scripts/slurm/mpi-gpu-nccl.sbatch` does, which the NCCL exchange needs), where the locality pick matters.

Built with the `nccl` feature ([Installation](../../installation.md#gpu-and-mpi-features)) on top of `cuda,mpi`, a remote layer's columns move device to device over NCCL instead of the host wire format ([CUDA devices](gpu.md#exchange)).
The mode is agreed once, collectively, at the scatter that builds the split: NCCL only when the group has more than one rank, every rank can start a communicator, and no two ranks share a device — otherwise the host format, silently unless `PAULISTRINGS_GPU_EXCHANGE=nccl` asks for it, which then errs on every rank instead.
`--gpus-per-node` (rather than `--gpus-per-task=1`) is what lets NCCL find every rank's device and use NVLink between them; `scripts/slurm/mpi-gpu-nccl.sbatch` launches that way and its preamble, `scripts/slurm/check-gpu-links.sh`, stops the job unless every visible GPU pair is NVLink-joined.

From Python, `propagate(..., comm=comm, device="auto")` is the same run, in an extension built with `--features cuda,mpi`: `device=` takes this rank's ordinal or `"auto"`, which is `local_device_for_comm`, and `result=`, `partition_row_seed=` and `partition_row_blocks=` keep their host meanings ([CUDA devices](gpu.md#comm-device)).
`scripts/mpi-test.sh --ranks 2,4 --python --cuda` builds that extension and runs `test_mpi.py`'s device cases, which skip on every rank unless every rank sees a device.

## Requirements

**Thread level at least `MPI_THREAD_SERIALIZED`.**
The layer loop runs inside a pinned Rayon pool, so MPI is called from a pool worker rather than the process's main thread — and not necessarily the same worker on every layer.
Only ever one at a time, which is exactly `SERIALIZED`; `FUNNELED` would be a false claim.
Set `mpi4py.rc.thread_level` *before* `from mpi4py import MPI` (mpi4py's default, `"multiple"`, is also fine).
A weaker level is a `RuntimeError` at the boundary.

**A power-of-two rank count.**
A partition is named by `log2(P)` GF(2) hash rows, so `mpirun -n 3` is a `ValueError`.

**A replicated input.**
Every rank must call `propagate` with the same observable, circuit, policy, direction and options.
The scatter is a *local filter* of a sum every rank already holds, not a distribution of rank 0's copy — build the terms from the same seed, or load the same file, on every rank.
Nothing checks the terms; the run's *shape* is checked, and a group that disagrees about the circuit aborts with a fingerprint mismatch rather than hanging.

`propagate` is collective: every rank of `comm` must reach it, in the same order.
A rank that skips one hangs the rest.

## Stats

`propagate_with_stats(..., comm=comm)` fills `PropagationStats.partition` with a `PartitionStats` whose `rank` and `size` name the group (both are `None` for a run that is not distributed).
Its per-layer lists hold **this rank's entry only** — gathering the group's counters would mean a collective per layer for a diagnostic — so `terms_in[k]` is a one-element list, `rows_exported[k]` is what this rank sent, and `imbalance[k]` is always `1.0`.
Reduce over `comm` for the group's picture:

<!-- doctest: skip -->
```python
_, stats = observable.propagate_with_stats(circuit, policy, comm=comm)
exported = comm.allreduce(sum(stats.partition.rows_exported))
```

The field list is under [`PartitionStats`](../../library/propagate.md#partitionstats).

## Limits

- **Exact `topn` is unavailable**, as in any partitioned run: choosing the `n`-th largest magnitude across ranks is a distributed selection.
  `truncation.approx_topn(n)` all-reduces its octave histogram and retains exactly the set a single-process run would have.
- **A replicated input caps the sum at what one rank can build.** Reaching the capacity the ranks together have needs a driver that ingests already distributed; the scatter itself is a local filter and costs nothing extra.
- **Homogeneous groups only.** The wire carries raw host bytes, so a mixed-architecture job, or one mixing compile-time width tiers across ranks, is silently wrong.
- **One partition per rank.** The hybrid — several NUMA domains inside one rank — is not implemented.
  Intra-node, that is what would take the 2-rank remote layer below the 3.5× in [Engine performance](../../examples/benchmarks/engine-performance.md#distributed).
- **The library never initializes MPI.** The application owns `MPI_Init` and `MPI_Finalize`.

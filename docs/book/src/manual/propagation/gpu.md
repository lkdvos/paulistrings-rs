# CUDA devices

A device run holds the sum on CUDA GPUs and applies every layer there, with the same layer loop, truncation rules and trace as a [partitioned run](partitions.md): one partition on one device, one partition per listed device, or one device per MPI rank.
Each output bucket is one thread block that builds, sorts and merges its incoming rows in shared memory, so a layer never materializes a gather row or runs a global sort.
The mechanism is in [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §GPU-Readiness; this page is when to reach for it and how.

## When a device pays {#when-it-pays}

**Dense two-qubit layers at a million terms and up are where the device wins.**
On an RTX A6000 a saturated `su4` layer runs 11× faster on wall than the 16-thread host engine at 1.41e7 terms, and 15× at 5.65e7 ([`research/HARDWARE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/HARDWARE.md) § ccqlin038 — GPU).
Sparse layers (`cnot`, a Pauli rotation) gain 2–5×, because each output bucket receives few rows and a block's fixed sort and synchronization cost weighs more.
Every layer carries a fixed cost of a few launches and synchronizations that the host engine does not pay, so the advantage shrinks below about 1e4 terms.

The upload and download are paid once per `propagate(device=...)` call.
A loop of many short calls — a Trotter time series, a cutoff sweep over the same prefix — should hold the sum on the device with [`PauliSum.to_device`](#resident-sums) instead.

The feature is off by default and the released wheels omit it.
[`paulistrings.cuda_available()`](../../library/module-helpers.md) says whether this build can run on a device and one is visible, and `device=` in a build without the feature raises `RuntimeError`.

## Building from source

The `cuda` feature needs no CUDA toolkit at build time: the kernels are CUDA C++ compiled by NVRTC when a width is first used, and the driver and NVRTC libraries are loaded at runtime.

```bash
maturin develop --release --features cuda -m crates/paulistrings-py/Cargo.toml
```

At runtime the process needs the NVIDIA driver's `libcuda` and a CUDA 12 `libnvrtc` on the library search path.
On a Flatiron host `module load cuda/12.8.0` provides the latter; elsewhere `pip install nvidia-cuda-nvrtc-cu12` does, with its `lib` directory added to `LD_LIBRARY_PATH`:

```bash
pip install nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_nvrtc as m; print(list(m.__path__)[0])')/lib:$LD_LIBRARY_PATH"
```

A missing library makes `cuda_available()` return `False` rather than fail at import.
The first propagation at each width compiles its kernels, which takes a few seconds; later calls in the process reuse them.

## Python

The snippets on this page are skipped by the doc checker because they need a CUDA device.

<!-- doctest: skip -->
```python
import paulistrings
from paulistrings import truncation

if paulistrings.cuda_available():
    evolved = observable.propagate(
        circuit,
        truncation.approx_topn(10_000_000),
        direction="heisenberg",
        device=0,
    )
```

`device` takes a device ordinal, a list of ordinals, or `"auto"`.
`device=` is an alternative to `partitions=`, and passing both, or `result="local"` without `comm=`, is a `ValueError`.
`target_bucket_len` and `min_buckets` still set the host-side bucket schedule the device refines on top of; `engine` is ignored.

`propagate_with_stats(..., device=0)` fills `PropagationStats.partition` as a one-partition run, and `PartitionStats.devices` names the device:

<!-- doctest: skip -->
```python
_, stats = observable.propagate_with_stats(circuit, policy, device=0)
print(stats.partition.devices)   # [0]
print(stats.nanos)               # wall time per layer, on the device
```

### Resident sums {#resident-sums}

`PauliSum.to_device(device)` uploads a sum and returns a `GpuPauliSum`, which `propagate` steps **in place** and `to_host` copies back:

<!-- doctest: skip -->
```python
resident = observable.to_device(0)
values = []
for _ in range(steps):
    resident.propagate(trotter_step, policy, direction="heisenberg")
    values.append(resident.to_host().expectation("z+"))
```

`GpuPauliSum.propagate` and `propagate_with_stats` take `policy`, `direction`, `target_bucket_len` and `min_buckets` with the meanings above; `len()`, `num_qubits`, `device` and `num_buckets` read the resident sum without a download.
A device error mid-run leaves the resident sum holding the last completed layer's output, and a later call resumes from it.

### Several devices {#multi-device}

A list places one partition on each entry, split by GF(2) partition rows and exchanged between layers as in a [partitioned run](partitions.md):

<!-- doctest: skip -->
```python
evolved, stats = observable.propagate_with_stats(
    circuit, truncation.approx_topn(10_000_000), direction="heisenberg", device=[0, 1, 2, 3]
)
print(stats.partition.devices)   # [0, 1, 2, 3]
```

The list length must be a power of two, since a partition is named by `log2(P)` rows; `device=[0, 1, 2]` is a `ValueError`.
An ordinal may repeat, `device=[0, 0]` putting two partitions on device 0, which runs the exchange on one GPU.
`"auto"` takes devices `0..k` for the largest power of two `k` visible, and is the one-device run when only one is.
`partition_row_seed=` and `partition_row_blocks=` choose the rows as they do for `partitions=`, the block count equal to the list length.
`stats.partition.partitions` is the list length, `devices` the list, and the per-layer lists carry one entry per partition.
The call scatters, propagates and gathers every time; a resident multi-device sum is Rust-only (below).

### One device per MPI rank {#comm-device}

With `comm=`, `device=` names this rank's one device: an ordinal, or `"auto"` for a device near the rank's CPUs, distinct per rank on a node while there are enough (see [MPI ranks](mpi.md#gpu-per-rank)):

<!-- doctest: skip -->
```python
evolved = observable.propagate(circuit, policy, direction="heisenberg", comm=comm, device="auto")
```

Everything else is the host `comm=` contract: replicated input, collective calls, `result="gather"` or `"local"`, and `stats.partition` holding this rank's entry, with `devices` this rank's device.
It needs the extension built with both features (`maturin develop --release --features cuda,mpi`); a build lacking either raises `RuntimeError` naming it.
A list of several ordinals under `comm=` is a `ValueError`.
A rank that cannot use its device fails the call on every rank, the peers raising the same exception type naming it.

## Rust: several devices, and one device per MPI rank {#rust-multi-device}

`GpuPartitionedSum` splits a sum across the device partitions of a `Placement::Devices` runtime, one partition per listed device, and holds it across calls:

<!-- doctest: skip -->
```rust
use paulistrings::engine::partitioned::{PartitionConfig, PartitionRuntime, Placement};
use paulistrings::gpu::GpuPartitionedSum;

let config = PartitionConfig {
    placement: Placement::Devices { devices: vec![0, 1, 2, 3], per_device: 1 },
    bind_memory: false,
    partition_row_seed: None,
};
let runtime = PartitionRuntime::new(&config)?;
let mut split = GpuPartitionedSum::scatter(observable, runtime, &config)?;
split.propagate(&circuit, &ApproxTopN(10_000_000), Direction::Heisenberg)?;
let evolved = split.gather()?;
```

An in-process group exchanges device-resident payloads: the columns move device to device (or, without peer access, staged through the host) rather than through the host wire format.
`per_device > 1` puts several partitions on one device, which is how the exchange is tested on a single GPU — **give each partition its own GPU in practice**; sharing one is a testing configuration, not a performance one.
With features `mpi` and `cuda`, `gpu::MpiGpuSum` is the [MPI ranks](mpi.md#gpu-per-rank) driver with each rank's share on its own device.

## Exchange mode, merge and chunking {#exchange}

`PAULISTRINGS_GPU_EXCHANGE` picks how a remote layer's rows travel, read once per process by an in-process group and at every scatter by an MPI group:

| value | in-process group (`GpuPartitionedSum`) | MPI group (`gpu::MpiGpuSum`) |
|---|---|---|
| unset | device-to-device (or peer) copy | NCCL if built with `nccl` and every rank can start it on a distinct device, else host |
| `host` | staged through host `PartnerPayload` columns | staged through host `PartnerPayload` columns |
| `device` | device-to-device (or peer) copy | same as unset (an MPI group reads `device` and `nccl` apart) |
| `nccl` | same as unset (in-process peer copies are already device to device) | NCCL required — every rank errs if any rank cannot start it |

The MPI mode is agreed **once, collectively, at scatter**, never per layer: with the `nccl` feature, more than one rank, and every rank seeing a distinct CUDA device it can start a communicator on, the group moves columns straight into the receiver's device memory over NCCL; a group with a host member, or ranks sharing a device, always uses the host format.
`PAULISTRINGS_NCCL_TIMEOUT_S` (default 300) bounds every wait on the communicator; a timeout or an NCCL error aborts it and the split, surfacing as `GpuError::Timeout` / `GpuError::Nccl` (`RuntimeError` from Python).
Building and running it needs the `nccl` feature and the `nccl/2.23.4-1` module — see [Installation](../../installation.md#gpu-and-mpi-features) and [MPI ranks](mpi.md#gpu-per-rank).

**A device sender merges one partner's rows by key before the exchange** (`GpuLayerOptions::premerge`, `PAULISTRINGS_GPU_PREMERGE=off`), so two remote deltas that land on the same receiver key ship as one row.
It runs for a dense two-qubit unitary, whose remote entries share an output support pattern, and never for a Pauli rotation or a Clifford, which have at most one remote entry each.
It costs a second fused pass on the sender that a same-device exchange (two virtual partitions on one card) does not repay at low merge ratios; a real deployment, one partition per GPU, is where it pays.

**A device receive moves in chunks of destination positions** rather than all at once (`GpuLayerOptions::exchange_bytes`, `PAULISTRINGS_GPU_EXCHANGE_BYTES`, `K`/`M`/`G` suffixes), capping the receive volume resident on the device at once; unbounded by default, one chunk.
The send side is never chunked: its export volume stays resident until the layer's last chunk moved.

## Peer access and NVLink {#peer-access}

`gpu::peer_access(dst, src)` lets device `dst` reach device `src`'s memory directly, so a copy from `src` into `dst` goes over NVLink or PCIe peer-to-peer, and reports the outcome (`PeerAccess::Enabled`, `Unsupported`, `Failed`, or `SameDevice`), granting both the driver's peer-context access and the source's memory-pool access list a pooled allocation needs to be reachable from a peer at all.
A device exchange calls it before its first cross-device copy; without it, or on a pair the driver reports as unsupported, the copy stages through the host instead of faulting.
`examples/gpu_peer.rs` times the same-device, peer and host-staged paths and reports the driver's P2P attributes.
`scripts/slurm/check-gpu-links.sh` fails a multi-GPU job unless every visible pair is joined by NVLink, so a measurement never silently falls back to PCIe.

## What changes on a device

**Results agree to floating-point tolerance, not bit for bit.**
The device sums equal keys in a different order from the host, as a different bucket count does ([`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Determinism).
No reduction uses a floating-point atomic, so two runs on the same device give the same bits.

**Storage order is the host's canonical order, but for the device's own bucket count.**
The download re-sorts each bucket to the host's lexicographic order, and the device chooses its bucket count by rows per block rather than terms per bucket, so `x_array()` of a device result can list the terms in a different order from a host run's.
Compare two results by key, never by position.

**Device failures are exceptions, not aborts**: an exhausted device raises `MemoryError`, a request the backend does not implement `NotImplementedError`, a device placement that does not resolve `ValueError`, and every other device failure `RuntimeError`.
Under `comm=`, a failure on one rank's device fails the call on every rank, its peers raising a `RuntimeError` that names the failing partition and layer.

## Limits

- **The resident `GpuPauliSum` is one device.** A multi-device or per-rank run from Python scatters and gathers on every call; the Rust API above keeps the split resident.
- **Exact `topn` runs only on one device.** `truncation.topn` matches the host term for term on a lone `device=<int>`, `to_device`, or a `device=` that resolves to one ordinal; a device list of more than one entry or `comm=` with `device=` raises `NotImplementedError`, since the `n`-th largest of a split sum has no collective form — use `truncation.approx_topn(n)` there instead.
- **Only the built-in policies run on a device.** Every `truncation` factory and its `&`/`|` compositions lower to the device; a custom Rust `TruncationPolicy` without a `device_policy` is refused before the first layer.
- **Memory caps the sum at about 5e7 terms per 48 GB card at 128 qubits**, since a layer holds its input, its output and a staging arena at once.
- **Widths `W ≥ 8` (more than 256 qubits) are correct but untuned.**
- **A multi-device or MPI group trades staging time for device memory.** Exchanging device-resident payloads keeps one export volume and one receive volume resident on a partition's device during a remote layer, on top of its sum, so two virtual partitions on one card can run out of memory at a term count the host-staged path (or a single device) still fits.
- **A device partition in a group cannot refine mid-run.** It runs every remote layer at the group's agreed bucket count and raises rather than growing the count when a block or a received segment exceeds the fused kernel's cap.
- **The sender-side merge (`premerge`) does not pay on a same-device exchange at low merge ratios.** Two virtual partitions sharing one card are a testing configuration for that reason; give each partition its own GPU.
- **The cross-device peer copy is untested on a real multi-GPU node.** `gpu::peer_access` grants both peer-context access and the source device's memory-pool access a peer copy needs to land directly rather than stage through the host, and today's multi-device numbers come from virtual partitions sharing one card.
- **The chunked NCCL receive has run only over the in-process loopback test wire**, not a real communicator, and one chunk's transfer does not overlap the fused layer of the chunk before it.

## See it in use

`python/paulistrings/tests/test_cuda.py` runs every policy against the host engine at three widths, and `crates/paulistrings/tests/propagate_gpu.rs` is the Rust differential net under the same channels.

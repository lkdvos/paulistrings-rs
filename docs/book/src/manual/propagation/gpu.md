# CUDA devices

A device run holds the whole sum on one CUDA GPU and applies every layer there, with the same layer loop, truncation rules and trace as a one-partition [partitioned run](partitions.md).
Each output bucket is one thread block that builds, sorts and merges its incoming rows in shared memory, so a layer never materializes a gather row or runs a global sort.
The mechanism is in [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §GPU-Readiness; this page is when to reach for it and how.

## When a device pays {#when-it-pays}

**Dense two-qubit layers at a million terms and up are where the device wins.**
On an RTX A6000 a saturated `su4` layer at 5.65e7 terms runs 10.1× faster on wall than the 16-thread host engine, and 12.2× at 1.41e7 terms ([`research/FINDINGS.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/FINDINGS.md) §GPU spike).
Sparse layers (`cnot`, a Pauli rotation) gain less, because each output bucket receives few rows and a block's fixed sort and synchronization cost weighs more.
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

`device` takes a device ordinal, or `"auto"` for the only visible device.
`device=` is an alternative to `partitions=` and `comm=`, and passing it with either, or with `result="local"`, is a `ValueError`.
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

## What changes on a device

**Results agree to floating-point tolerance, not bit for bit.**
The device sums equal keys in a different order from the host, as a different bucket count does ([`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Determinism).
No reduction uses a floating-point atomic, so two runs on the same device give the same bits.

**Storage order is the host's canonical order, but for the device's own bucket count.**
The download re-sorts each bucket to the host's lexicographic order, and the device chooses its bucket count by rows per block rather than terms per bucket, so `x_array()` of a device result can list the terms in a different order from a host run's.
Compare two results by key, never by position.

**Device failures are exceptions, not aborts**: an exhausted device raises `MemoryError`, a request the backend does not implement `NotImplementedError`, and every other device failure `RuntimeError`.

## Limits

- **One device per process.** A list of several ordinals, or `"auto"` with more than one device visible, raises `NotImplementedError`.
- **Exact `topn` is unavailable**, as in a partitioned run: `truncation.topn` raises `NotImplementedError`, and `truncation.approx_topn(n)` retains exactly the set the host would.
- **Only the built-in policies run on a device.** Every `truncation` factory and its `&`/`|` compositions lower to the device; a custom Rust `TruncationPolicy` without a `device_policy` is refused before the first layer.
- **Memory caps the sum at about 5e7 terms per 48 GB card at 128 qubits**, since a layer holds its input, its output and a staging arena at once.
- **Widths `W ≥ 8` (more than 256 qubits) are correct but untuned.**

## See it in use

`python/paulistrings/tests/test_cuda.py` runs every policy against the host engine at three widths, and `crates/paulistrings/tests/propagate_gpu.rs` is the Rust differential net under the same channels.

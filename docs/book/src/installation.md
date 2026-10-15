# Installation

```bash
pip install paulistrings-rs        # or: uv add paulistrings-rs
```

The distribution is `paulistrings-rs` and the import is `import paulistrings`.
Wheels cover Linux x86_64 and aarch64 (`manylinux_2_28`, glibc 2.28 or newer, which includes every Rusty node), macOS x86_64 and arm64, and Windows x86_64, and one abi3 wheel per platform serves every Python 3.9 or newer.
Anywhere else pip builds the source distribution, which needs a Rust toolchain from [rustup](https://rustup.rs).

Core propagation needs nothing but `numpy`.
The `examples` extra adds `stim` and `qiskit`, which `interop.circuit_from_stim`, `interop.circuit_from_qiskit` and `interop.stabilizers_from_stim` import on first use, plus `qiskit-aer` and `matplotlib` for the example suite:

```bash
pip install "paulistrings-rs[examples]"
```

With the package installed, the [Manual](manual/index.md) is where to go next; it opens with a four-line run and explains each part in turn.
[First propagation](examples/first-propagation.md) carries that same run further, into term inspection and a validation sweep.

## GPUs

The Linux and Windows wheels include the CUDA backend, and need no CUDA toolkit to install.
At runtime a device needs the NVIDIA driver's `libcuda` and a CUDA 12 `libnvrtc` on the library search path: `module load cuda/12.8.0` on a Flatiron host, `pip install nvidia-cuda-nvrtc-cu12` elsewhere ([CUDA devices](manual/propagation/gpu.md#runtime-libraries)).
Without them `paulistrings.cuda_available()` is `False` and everything else works.

## MPI

The `mpi` feature is never in a wheel, since no MPI build is portable across clusters, so it is a source build against the cluster's loaded MPI module.
`mpi4py` must be built against that same MPI, so it is installed from source first:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH=$(llvm-config --libdir)
pip install mpi4py --no-binary mpi4py
pip install paulistrings-rs --no-binary paulistrings-rs --config-settings=build-args="--features cuda,mpi"
```

A source build replaces the wheel, so `--features` names every feature the build keeps; `cuda` is the one to drop on a machine without NVIDIA devices.
Running across ranks is covered in [MPI ranks](manual/propagation/mpi.md).

### GPU and MPI features {#gpu-and-mpi-features}

The build above with `cuda,mpi` gives one GPU per MPI rank ([MPI ranks](manual/propagation/mpi.md#gpu-per-rank)), whose exchange runs over NCCL.
NCCL is loaded at runtime like `libcuda` and `libnvrtc`, so it needs no toolkit to build; at runtime it needs `module load cuda/12.8.0 nccl/2.23.4-1` on a Flatiron host.

## Rust

```bash
cargo add paulistrings
```

The crate is the engine the Python package wraps, with its features off by default: `cuda`, `mpi` (which needs an MPI installation and `libclang` to build) and `phase-timing`.
Its [API documentation](https://docs.rs/paulistrings) is a developer guide that states each item's contracts.
Performance numbers in these docs are measured with `lto = "fat"` and `codegen-units = 1` in the release profile.

## Developing

A checkout builds with [uv](https://docs.astral.sh/uv/), which creates `./.venv` from the locked dependencies and builds the extension into it:

```bash
git clone https://github.com/lkdvos/paulistrings-rs
cd paulistrings-rs
./scripts/setup.sh                        # uv sync, plus the examples extra
uv run pytest python/paulistrings/tests
```

The Python version comes from `.python-version` and the Rust toolchain is pinned in `rust-toolchain.toml`, so neither needs choosing.
`uv run` rebuilds the extension whenever a Rust source changed, so there is no separate rebuild step; from an activated `.venv`, `uv sync` does the same.
`uv sync` removes whatever an extra it is not given had installed, so name every extra each time (`uv sync --extra examples --extra bench`); `uv run` only adds.
Feature builds of a checkout are in [CUDA devices](manual/propagation/gpu.md#building-from-source) and [MPI ranks](manual/propagation/mpi.md#building-from-source).

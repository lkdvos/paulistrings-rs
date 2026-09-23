# Installation

**`paulistrings` is not published to PyPI**, so `pip install paulistrings` will not find it.
Released wheels (manylinux x86_64, macOS x86_64/arm64) are attached to [GitHub Releases](https://github.com/lkdvos/paulistrings-rs/releases) instead — no Rust toolchain needed.

With the [GitHub CLI](https://cli.github.com/), which resolves the latest release for you:

```bash
gh release download --repo lkdvos/paulistrings-rs --pattern '*manylinux_2_28_x86_64.whl'
pip install ./paulistrings-*.whl
```

Substitute the platform tag for your machine: `manylinux_2_28_x86_64` (Linux, including Rusty/Popeye), `macosx_11_0_arm64` (Apple silicon) or `macosx_10_12_x86_64` (Intel Mac).
One abi3 wheel per platform serves every Python >= 3.9.

Without `gh`, take the `.whl` asset for your platform from the [latest release](https://github.com/lkdvos/paulistrings-rs/releases/latest) and `pip install ./paulistrings-*.whl`.

These wheels cover the default engine only.
The `mpi` feature is never bundled into a wheel — no MPI implementation is portable across clusters/vendors — so it stays a from-source pip install against the cluster's loaded MPI module:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH=$(llvm-config --libdir)
pip install "paulistrings[dev] @ git+https://github.com/lkdvos/paulistrings-rs" \
  --config-settings=build-args="--features mpi"
```

The `cuda` feature is also a from-source build, but it needs no CUDA toolkit to compile: the kernels are compiled by NVRTC at runtime.

```bash
pip install "paulistrings[dev] @ git+https://github.com/lkdvos/paulistrings-rs" \
  --config-settings=build-args="--features cuda"
```

At runtime it needs the NVIDIA driver's `libcuda` and a CUDA 12 `libnvrtc` on the library search path: `module load cuda/12.8.0` on a Flatiron host, or `pip install nvidia-cuda-nvrtc-cu12` with its `lib` directory on `LD_LIBRARY_PATH` elsewhere.
Without them `paulistrings.cuda_available()` is `False` and everything else works as in the default build.

Building from source (for contributors, or platforms without a release wheel) uses a setup script that creates `./.venv` and builds the extension into it:

```bash
git clone https://github.com/lkdvos/paulistrings-rs
cd paulistrings-rs
./scripts/setup.sh
source .venv/bin/activate
```

`scripts/setup.sh` expects a Python 3.11 at `/usr/bin/python3.11`; point it
elsewhere with `PYTHON=$(which python3.11) ./scripts/setup.sh`. The Rust
toolchain is pinned in `rust-toolchain.toml` (1.94.0), so no toolchain choice is
needed. After any change to the Rust sources, rebuild:

```bash
maturin develop --release -m crates/paulistrings-py/Cargo.toml
```

Build `--release`. The release profile uses `lto = "fat"` and
`codegen-units = 1`; a debug build of this workload is dramatically slower, and
never worth benchmarking.

Core propagation needs nothing but `numpy`.
The optional extras matter for the example suite and the cross-library
benchmarks, and also cover `stim`/`qiskit`, which `interop.circuit_from_stim`,
`interop.circuit_from_qiskit` and `interop.stabilizers_from_stim` import lazily
on first use — install one of these extras before reaching for those:

```bash
pip install -e ".[examples]"   # matplotlib, stim, qiskit, qiskit-aer — the oracles and plots
pip install -e ".[bench]"      # pytest-benchmark, qiskit, openfermion, stim
```

With the package installed, the [Manual](manual/index.md) is where to go next; it opens with a four-line run and explains each part in turn.
[First propagation](examples/first-propagation.md) carries that same run further, into term inspection and a validation sweep.
Building for MPI is covered separately in [MPI ranks](manual/propagation/mpi.md#building-from-source), since it needs the extra build step above plus a launcher, and building for a GPU in [CUDA devices](manual/propagation/gpu.md#building-from-source).

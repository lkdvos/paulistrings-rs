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

The optional extras matter only for the example suite and the cross-library
benchmarks — the library itself needs nothing but `numpy`:

```bash
pip install -e ".[examples]"   # matplotlib, stim, qiskit, qiskit-aer — the oracles and plots
pip install -e ".[bench]"      # pytest-benchmark, qiskit, openfermion, stim
```

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

## MPI and multi-node runs

The `mpi` feature is never in a wheel, since no MPI build is portable across clusters, so it is a source build against the cluster's MPI.
The commands below are for Rusty; on another cluster, substitute its MPI, LLVM and Python modules.

### Where to build

- **On a Rocky 9 workstation or login node.** The build fetches crates from crates.io, and the `modules/2.4-20250724` stack is Rocky 9 only.
- **With a Rust toolchain from [rustup](https://rustup.rs).** It is needed at build time only; the nodes run the compiled extension.
- **Into a venv on the shared filesystem** (`/mnt/home` or `/mnt/ceph`), which every node mounts. Local `/home` and `/tmp` are invisible to the nodes.

### Build

`mpi4py` must be built against the same MPI as the extension (its pip wheel bundles a different one), so both are built from source:

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9
export LIBCLANG_PATH=$(llvm-config --libdir)    # rsmpi's bindgen needs libclang

python3 -m venv ~/venvs/paulistrings-mpi
source ~/venvs/paulistrings-mpi/bin/activate
pip install "mpi4py>=4,<5" --no-binary mpi4py
pip install paulistrings-rs --no-binary paulistrings-rs --config-settings=build-args="--features mpi"
python -c "import paulistrings; print(paulistrings.mpi_available())"    # True
```

A source build replaces the wheel, so `--features` names every feature the build keeps: `--features cuda,mpi` for [one GPU per rank](#gpu-and-mpi-features).
The extension records the MPI library directory in its rpath, so `import paulistrings` works without the modules loaded; load them in the job anyway, since `srun` and `mpi4py` need the same MPI.

### Check it

Save this as `check_mpi.py`; it evolves `Σ Z` under a layer of Hadamards and reads `⟨+|Σ X|+⟩ = n` back from the ranks' shares:

<!-- doctest: skip -->
```python
import mpi4py

mpi4py.rc.thread_level = "serialized"  # before mpi4py.MPI is imported
from mpi4py import MPI

import paulistrings
from paulistrings import Circuit, PauliSum, truncation

comm = MPI.COMM_WORLD
n = 64
observable = PauliSum.from_strings({"I" * q + "Z" + "I" * (n - q - 1): 1.0 for q in range(n)}, num_qubits=n)
circuit = Circuit(n)
for q in range(n):
    circuit.h(q)
local = observable.propagate(circuit, truncation.approx_topn(1000), direction="heisenberg", comm=comm, result="local")
value = comm.allreduce(local.expectation("x+"))
if comm.Get_rank() == 0:
    print(f"paulistrings {paulistrings.__version__}: {comm.Get_size()} ranks, <+|sum X|+> = {value.real:g} (expect {n})")
```

On the build workstation, `mpirun -n 2 python check_mpi.py` runs it on two ranks, and the job below runs it across nodes.

### Run across nodes

One rank per NUMA domain, bound to it, and a power-of-two rank count; the Rocky 9 constraint keeps the job on nodes that can load a Rocky 9 build.
Rusty's CPU nodes have two sockets, each one NUMA domain unless the node reports more (`lscpu | grep 'NUMA node(s)'` inside a job), so two icelake nodes give four ranks of 32 cores:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --constraint=icelake&rocky9
#SBATCH --time=01:00:00

module load modules/2.4-20250724 openmpi/5.0.6 python-mpi/3.12.9
source ~/venvs/paulistrings-mpi/bin/activate
srun --ntasks-per-node=2 --cpus-per-task=32 --cpu-bind=ldoms --mpi=pmix python check_mpi.py
```

For rome nodes use `--cpus-per-task=64`, for genoa `48`.
Without `--cpu-bind=ldoms` every rank's thread pool spans the whole node and the ranks compete for cores.
Every rank must build the same observable and circuit, since each keeps its own share of a sum it holds whole; [MPI ranks](manual/propagation/mpi.md#requirements) lists the requirements and the [launch options](manual/propagation/mpi.md#launching).

### GPU and MPI features {#gpu-and-mpi-features}

Built with `--features cuda,mpi`, each rank runs its share on one GPU and the ranks exchange over NCCL ([MPI ranks](manual/propagation/mpi.md#gpu-per-rank)).
The job loads the CUDA and NCCL libraries, which the extension opens at runtime, and makes every GPU of a node visible to every rank on it, which `--gpus-per-node` does and `--gpus-per-task` does not; without that, NCCL falls back from NVLink to copies through host memory:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --time=01:00:00

module load modules/2.4-20250724 openmpi/5.0.6 python-mpi/3.12.9 cuda/12.8.0 nccl/2.23.4-1
source ~/venvs/paulistrings-mpi/bin/activate
srun --cpu-bind=cores --mpi=pmix python script.py    # propagate(..., comm=comm, device="auto")
```

`--constraint=h100-sxm5` selects the H100 nodes instead.
Above one node, the NCCL exchange is not yet tested ([#40](https://github.com/lkdvos/paulistrings-rs/issues/40)).
Every rank holds the whole input in host memory before the scatter ([#35](https://github.com/lkdvos/paulistrings-rs/issues/35)), so size `--mem-per-cpu` for that.

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

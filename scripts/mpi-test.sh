#!/usr/bin/env bash
# Run the MPI nets at several rank counts under `mpirun`: the Rust transport's
# differential net (`tests/mpi_ranks.rs`) and, with --python, the bindings'
# (`python/paulistrings/tests/test_mpi.py`).
#
# The Rust net runs twice per rank count: at the default exchange chunk count
# and with `PAULISTRINGS_EXCHANGE_CHUNKS=64`, which forces the pipelined receive
# into as many batches as the layer has cosets.
#
#   scripts/mpi-test.sh                        # 2 and 4 ranks, debug profile
#   scripts/mpi-test.sh --ranks 2,4,8          # more
#   scripts/mpi-test.sh --oversubscribe        # more ranks than cores (CI, a busy box)
#   scripts/mpi-test.sh --release              # the shipping codegen
#   scripts/mpi-test.sh --python               # also build the extension and run its net
#   scripts/mpi-test.sh --python --no-rust     # only the Python net
#   scripts/mpi-test.sh --cuda                 # both nets built with `mpi,cuda`: adds the one-GPU-per-rank cases
#   scripts/mpi-test.sh --nccl                 # both nets built with `mpi,cuda,nccl`: adds the NCCL device-exchange cases
#
# --cuda puts $CUDA_ROOT/lib64 (or $CUDA_HOME/lib64) on LD_LIBRARY_PATH for
# libnvrtc and forwards it to every rank; ranks share a device when there are
# fewer devices than ranks, and the device cases skip on every rank unless
# every rank sees one.
#
# --nccl implies --cuda's LD_LIBRARY_PATH setup and additionally needs
# libnccl on it (module load nccl/2.23.4-1); NCCL refuses two ranks of one
# communicator on one GPU, so this workstation-only invocation only exercises
# the size == 1 world (Host, no NCCL call) — the real multi-rank net is
# `scripts/slurm/mpi-gpu-nccl.sbatch`, which the user submits.
#
# The rank count must be a power of two: a partition is named by log2(P) GF(2)
# rows (ARCHITECTURE.md §Partitioning), and the test binary refuses anything
# else with exit 2.
#
# --python builds `_paulistrings` with `--features mpi` (`mpi,cuda` under --cuda) into $VIRTUAL_ENV, or
# ./.venv-mpi if that is unset, and runs pytest under mpirun. That venv needs
# maturin, pytest, numpy and an importable mpi4py built against the *same* MPI:
#
#   python3 -m venv --system-site-packages .venv-mpi   # from an interpreter with mpi4py
#   .venv-mpi/bin/pip install maturin pytest numpy
#
# This script loads no modules. It needs `mpicc` (for rsmpi's build probe),
# `LIBCLANG_PATH` (for its bindgen) and `mpirun` on PATH, and says how to get
# them if they are missing. On a Slurm node use `scripts/slurm/mpi-ranks.sbatch`
# instead, which drives the same binary through `srun --mpi=pmix`.
set -euo pipefail
cd "$(dirname "$0")/.."

ranks="2,4"
profile=""
oversubscribe=0
python_net=0
rust_net=1
features="mpi"
while [ $# -gt 0 ]; do
    case "$1" in
        --ranks) ranks="$2"; shift 2 ;;
        --ranks=*) ranks="${1#*=}"; shift ;;
        --oversubscribe) oversubscribe=1; shift ;;
        --release) profile="--release"; shift ;;
        --python) python_net=1; shift ;;
        --no-rust) rust_net=0; shift ;;
        --cuda) features="mpi,cuda"; shift ;;
        --nccl) features="mpi,cuda,nccl"; shift ;;
        -h|--help) sed -n '2,43p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
if [ "$rust_net" -eq 0 ] && [ "$python_net" -eq 0 ]; then
    echo "--no-rust without --python leaves nothing to run" >&2
    exit 64
fi

missing=0
command -v mpicc  >/dev/null 2>&1 || { echo "mpicc not found (rsmpi's build script probes it)" >&2; missing=1; }
command -v mpirun >/dev/null 2>&1 || { echo "mpirun not found" >&2; missing=1; }
if [ -z "${LIBCLANG_PATH:-}" ] && ! ldconfig -p 2>/dev/null | grep -q libclang; then
    echo "LIBCLANG_PATH is unset and no libclang is on the loader path (rsmpi runs bindgen)" >&2
    missing=1
fi
if [ "$missing" -ne 0 ]; then
    cat >&2 <<'EOF'

On a Flatiron host:
    module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
    export LIBCLANG_PATH=$(llvm-config --libdir)
Elsewhere: install an MPI (libopenmpi-dev / openmpi-bin) and libclang-dev.
EOF
    exit 2
fi

cuda_x=""
case "$features" in
    mpi,cuda|mpi,cuda,nccl)
        cuda_root="${CUDA_ROOT:-${CUDA_HOME:-}}"
        if [ -n "$cuda_root" ] && [ -d "$cuda_root/lib64" ]; then
            export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        if ! ldconfig -p 2>/dev/null | grep -q libnvrtc \
           && ! ls ${LD_LIBRARY_PATH//:/ } 2>/dev/null | grep -q '^libnvrtc'; then
            echo "warning: no libnvrtc on the loader path (module load cuda/12.8.0, or set CUDA_ROOT); the device cases will skip" >&2
        fi
        if [ "$features" = "mpi,cuda,nccl" ]; then
            nccl_lib="${NCCL_LIB:-}"
            if [ -n "$nccl_lib" ] && [ -d "$nccl_lib" ]; then
                export LD_LIBRARY_PATH="$nccl_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
            if ! ldconfig -p 2>/dev/null | grep -q libnccl \
               && ! ls ${LD_LIBRARY_PATH//:/ } 2>/dev/null | grep -q '^libnccl'; then
                echo "warning: no libnccl on the loader path (module load nccl/2.23.4-1, or set NCCL_LIB to its lib dir); NCCL init will fail" >&2
            fi
        fi
        cuda_x="-x LD_LIBRARY_PATH"
        ;;
esac

if [ "$rust_net" -eq 1 ]; then
    echo "== building the mpi_ranks test binary ${profile:-(debug)}, features $features"
    bin=$(cargo test -p paulistrings --features "$features" --test mpi_ranks --no-run \
            ${profile:+$profile} --message-format=json 2>/dev/null \
          | python3 -c 'import json, sys
for line in sys.stdin:
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if rec.get("reason") == "compiler-artifact" \
       and rec["target"]["name"] == "mpi_ranks" and rec.get("executable"):
        print(rec["executable"])')
    if [ -z "$bin" ]; then
        echo "could not find the mpi_ranks executable; rebuilding with output:" >&2
        cargo test -p paulistrings --features "$features" --test mpi_ranks --no-run ${profile:+$profile} >&2 || true
        exit 1
    fi
    echo "   $bin"
fi

if [ "$python_net" -eq 1 ]; then
    venv="${VIRTUAL_ENV:-$PWD/.venv-mpi}"
    if [ ! -x "$venv/bin/python" ]; then
        echo "no virtualenv at $venv (set VIRTUAL_ENV, or create ./.venv-mpi)" >&2
        sed -n '27,32p' "$0" >&2
        exit 2
    fi
    if ! "$venv/bin/python" -c 'import mpi4py' 2>/dev/null; then
        echo "$venv has no importable mpi4py; the Python net needs one built against this MPI" >&2
        exit 2
    fi
    maturin="$venv/bin/maturin"
    if [ ! -x "$maturin" ]; then
        maturin=$(command -v maturin || true)
    fi
    if [ -z "$maturin" ]; then
        echo "no maturin in $venv (or on PATH): pip install maturin into it" >&2
        exit 2
    fi
    echo "== building _paulistrings --features $features into $venv"
    # `maturin develop` installs into the *active* venv, so name it explicitly
    # rather than relying on the caller's shell.
    VIRTUAL_ENV="$venv" "$maturin" develop ${profile:+$profile} \
        --features "$features" -m crates/paulistrings-py/Cargo.toml
fi

# Shared-memory and oversubscription knobs. `vader_single_copy_mechanism=none`
# turns off CMA/XPMEM cross-process copies, which container runtimes and
# hardened kernels forbid (`ptrace_scope`), so the run falls back to a copy
# through shared memory; `rmaps_base_oversubscribe` lets a CI runner with two
# cores host four ranks. Harmless on a workstation, load-bearing on a runner.
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl_base_warn_component_unused=0

flags=""
[ "$oversubscribe" -eq 1 ] && flags="--oversubscribe"

status=0
for n in ${ranks//,/ }; do
    if [ "$rust_net" -eq 1 ]; then
        # Twice: at the default exchange chunk count, and forced high so the
        # pipelined receive runs many small batches (the count is clamped to
        # the coset count, so "64" is "as many batches as the layer has").
        # Both sides of an exchange must agree on it, hence `-x`.
        for chunks in default 64; do
            unset PAULISTRINGS_EXCHANGE_CHUNKS
            xflag=""
            if [ "$chunks" != default ]; then
                export PAULISTRINGS_EXCHANGE_CHUNKS="$chunks"
                xflag="-x PAULISTRINGS_EXCHANGE_CHUNKS"
            fi
            echo "== mpirun -n $n $flags (mpi_ranks, chunks=$chunks)"
            if mpirun -n "$n" $flags $xflag $cuda_x "$bin"; then
                echo "== $n ranks, mpi_ranks chunks=$chunks: ok"
            else
                echo "== $n ranks, mpi_ranks chunks=$chunks: FAILED" >&2
                status=1
            fi
        done
        unset PAULISTRINGS_EXCHANGE_CHUNKS
    fi
    if [ "$python_net" -eq 1 ]; then
        # `-p no:randomly` because the cases are collective: a plugin that
        # shuffled them would shuffle each rank independently and deadlock.
        # `-p no:cacheprovider` because every rank would write the same
        # .pytest_cache.
        echo "== mpirun -n $n $flags (test_mpi.py)"
        if mpirun -n "$n" $flags $cuda_x "$venv/bin/python" -m pytest \
                python/paulistrings/tests/test_mpi.py -q \
                -p no:cacheprovider -p no:randomly; then
            echo "== $n ranks, test_mpi.py: ok"
        else
            echo "== $n ranks, test_mpi.py: FAILED" >&2
            status=1
        fi
    fi
done
exit "$status"

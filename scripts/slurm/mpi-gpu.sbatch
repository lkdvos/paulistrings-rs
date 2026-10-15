#!/usr/bin/env bash
#SBATCH --job-name=ps-mpi-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --exclude=workergpu062
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# One CUDA device per MPI rank (`gpu::MpiGpuSum`, features `cuda,mpi`, exchanging over NCCL): the differential net, then the probe's cells.
# `--gpus-per-node` rather than `--gpus-per-task=1`: with only its own GPU visible a task's NCCL falls back from NVLink to shared memory.
#
# Node shapes (rank count must be a power of two):
#   1 node x 2 ranks   : env -u SBATCH_RESERVATION sbatch --ntasks-per-node=2 --gpus-per-node=2 scripts/slurm/mpi-gpu.sbatch
#   1 node x 4 ranks   : env -u SBATCH_RESERVATION sbatch scripts/slurm/mpi-gpu.sbatch
#   2 nodes x 8 ranks  : env -u SBATCH_RESERVATION sbatch --nodes=2 scripts/slurm/mpi-gpu.sbatch
#   H100-SXM5, 1 x 4   : env -u SBATCH_RESERVATION sbatch --constraint=h100-sxm5 scripts/slurm/mpi-gpu.sbatch
#
# Environment knobs (all optional):
#   LAYERS   probe layers (default rotation_zz,cnot,gu2q,su4,rotation_remote; heavyhex_step runs as its own cell)
#   N        terms per rank (default 4000000); the probe's --n is N x ranks
#   REPS     --reps (default 5)
#   PROBE=0  run the differential net alone, skip the probe
#   PS_REV   commit to build (default: HEAD when the job starts), in a private git worktree
#   NET_BOUND, PROBE_BOUND   seconds the differential net / one probe step may take (default 1200 / 1500)
#   WIRE_TIMEOUT_S           PAULISTRINGS_WIRE_TIMEOUT_S for the differential net (default 60)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh
. scripts/slurm/bounded.sh

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 cuda/12.8.0 nccl/2.23.4-1
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"
use_shared_rust
# Every rank execs the binary, so the build lives on the shared filesystem, not node-local /tmp.
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
checkout_rev "$PS_REV" "$CARGO_TARGET_DIR/src"
cd "$CARGO_TARGET_DIR/src"
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

nodes=${SLURM_JOB_NUM_NODES:-1}
ranks=${SLURM_NTASKS:-1}
cpus=${SLURM_CPUS_PER_TASK:-8}
echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
echo "== $ranks ranks over $nodes nodes, $cpus CPUs each"
if [ $(( ranks & (ranks - 1) )) -ne 0 ]; then
    echo "the rank count $ranks is not a power of two (a partition is named by log2(P) GF(2) rows)" >&2
    exit 2
fi

# What NCCL sees before it picks a transport.
bounded 300 srun --ntasks-per-node=1 bash -c '
    echo "-- $(hostname -s)"
    nvidia-smi -L
    nvidia-smi topo -m
    ibv_devinfo -l 2>&1 || echo "(no ibv_devinfo)"
    lsmod | grep -E "nvidia_peermem|nv_peer_mem" || echo "(no nvidia_peermem / nv_peer_mem module)"
    ip -br addr
'

bin=$(test_binary cuda,mpi,test-utils mpi_ranks)
[ -n "$bin" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
echo "== mpi_ranks: $bin"

# The net runs under a short NCCL wait bound with NCCL_DEBUG on, so the chosen transport is on record.
failed=0
bounded "${NET_BOUND:-1200}" env PAULISTRINGS_WIRE_TIMEOUT_S="${WIRE_TIMEOUT_S:-60}" \
    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET \
    srun --mpi=pmix --cpu-bind=cores --export=ALL \
    stdbuf -oL -eL "$bin" || { failed=$?; echo "== mpi_ranks FAILED (exit $failed)"; leftover_gpu_processes; }

if [ "$failed" != 0 ] && [ "${PROBE:-1}" = "1" ]; then
    echo "== the differential net failed; skipping the probe"
fi
if [ "$failed" = 0 ] && [ "${PROBE:-1}" = "1" ]; then
    cargo build --offline --release --features phase-timing,mpi,cuda -p paulistrings \
        --example phase_breakdown
    probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"
    out="$repo/benchmarks/results/$(date +%Y-%m-%d)-mpi-gpu"
    mkdir -p "$out"
    n=$(( ${N:-4000000} * ranks ))
    layers=${LAYERS:-rotation_zz,cnot,gu2q,su4,rotation_remote}
    sidecar="$out/mpi-gpu-${SLURM_JOB_ID:-local}-r$ranks.jsonl"
    run_probe() {
        bounded "${PROBE_BOUND:-1500}" srun --mpi=pmix --cpu-bind=cores "$probe" --mpi --device auto \
            --threads "$cpus" --json-out "$sidecar" "$@" \
            || { failed=$?; echo "== probe $* FAILED (exit $failed)"; leftover_gpu_processes; }
    }
    echo "== probe: $ranks ranks, --n $n (replicated), layers $layers -> $sidecar.rank*"
    run_probe --layers "$layers" --n "$n" --reps "${REPS:-5}"
    echo "== probe: heavyhex_step, five steps under coeff:2^-13"
    run_probe --layers heavyhex_step --reps 5 --truncation coeff:0.0001220703125
fi
exit "$failed"

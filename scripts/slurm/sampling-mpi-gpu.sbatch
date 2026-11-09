#!/usr/bin/env bash
#SBATCH --job-name=ps-sampling-mpi-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --exclude=workergpu062
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# One sampled run split over one GPU per MPI rank (`--backend mpi-gpu`, MpiGpuSum, NCCL), against a one-GPU reference (README.md §The sampled multi-GPU jobs).
# `--gpus-per-node` rather than `--gpus-per-task=1`, as in mpi-gpu.sbatch: with only its own GPU visible a task's NCCL falls back from NVLink to shared memory.
# Cells: scripts/slurm/sampling-cells.sh; rank 0 writes one JSON line each into benchmarks/results/<date>-<node>/sampling-mpi-gpu-<job>-r<ranks>.jsonl.
#
#   1 node x 4 ranks  : env -u SBATCH_RESERVATION sbatch scripts/slurm/sampling-mpi-gpu.sbatch
#   2 nodes x 8 ranks : env -u SBATCH_RESERVATION sbatch --nodes=2 scripts/slurm/sampling-mpi-gpu.sbatch
#
# Environment knobs (all optional):
#   CACHES       split-run caches, samples = cache / 4 (default "1e8 4e8", plus 8e8 at 8 ranks or more)
#   REF_CACHE    the one-GPU reference cache and approx-topn cap (default 1e8); the approx-topn splits run at CACHES
#   THRESHOLDS   kicked-Ising coefficient thresholds, each on one GPU and split (default "2.5e-5 1e-5")
#   NET=0        skip the mpi_ranks differential net that gates the sweep
#   PT=0         skip the phase-timing rerun of the largest kicked-Ising cell (export/exchange split)
#   CELL_BOUND, NET_BOUND   seconds one cell / the net may take (default 2400 / 1200)
#   PS_REV       commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh
. scripts/slurm/bounded.sh

module purge
module load modules/2.5-20261027 openmpi/5.0.10 llvm/20.1.8 cuda/12.9.1 nccl/2.29.7-1
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"
use_shared_rust
# Every rank execs the binary, so the build lives on the shared filesystem, not node-local /tmp.
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
checkout_rev "$PS_REV" "$CARGO_TARGET_DIR/src"
out=$(results_dir)
cd "$CARGO_TARGET_DIR/src"
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

tag="${SLURM_JOB_ID:-local}"
nodes=${SLURM_JOB_NUM_NODES:-1}
ranks=${SLURM_NTASKS:-1}
echo "== job $tag on ${SLURM_JOB_NODELIST:-$(hostname -s)}, rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
echo "== $ranks ranks over $nodes nodes"
if [ $(( ranks & (ranks - 1) )) -ne 0 ]; then
    echo "the rank count $ranks is not a power of two (a partition is named by log2(P) GF(2) rows)" >&2
    exit 2
fi
[ "$ranks" -ge 8 ] && : "${CACHES:=1e8 4e8 8e8}"
bounded 300 srun --ntasks-per-node=1 bash -c 'echo "-- $(hostname -s)"; nvidia-smi -L; nvidia-smi topo -m'

failed=0
if [ "${NET:-1}" = "1" ]; then
    net=$(test_binary cuda,mpi,test-utils mpi_ranks)
    [ -n "$net" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
    bounded "${NET_BOUND:-1200}" env PAULISTRINGS_WIRE_TIMEOUT_S=60 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
        srun --mpi=pmix --cpu-bind=cores --export=ALL stdbuf -oL -eL "$net" \
        || { echo "== mpi_ranks FAILED (exit $?); skipping the sweep"; leftover_gpu_processes; exit 1; }
fi

cargo build --offline --release -p paulistrings --features cuda,mpi --example sampling_bench
bin="$CARGO_TARGET_DIR/release/examples/sampling_bench"
sidecar="$out/sampling-mpi-gpu-$tag-r$ranks.jsonl"
launch_one=(srun --ntasks=1 --nodes=1 --cpu-bind=cores "$bin" --backend gpu --device 0)
launch_all=(srun --mpi=pmix --cpu-bind=cores "$bin" --backend mpi-gpu --device auto)
. scripts/slurm/sampling-cells.sh
echo "== sweep on $ranks ranks, caches $CACHES, reference $REF_CACHE -> $sidecar"
run_sweep

phase_timing_cell cuda,mpi,phase-timing "$out/sampling-mpi-gpu-$tag-r$ranks-pt.jsonl"
echo "results: $out/sampling-mpi-gpu-$tag-r$ranks*.jsonl"
exit "$failed"

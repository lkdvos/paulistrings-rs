#!/usr/bin/env bash
#SBATCH --job-name=ps-mpi-gpu-nccl
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The NCCL device exchange for `gpu::MpiGpuSum` (feature `nccl`, ARCHITECTURE.md §Partitioning), then a host-vs-nccl A/B on one binary.
# `--gpus-per-node` rather than `--gpus-per-task=1`: with only its own GPU visible a task's NCCL falls back from NVLink to shared memory.
#
# Node shapes (rank count must be a power of two):
#   1 node x 2 ranks   : env -u SBATCH_RESERVATION sbatch --ntasks-per-node=2 --gpus-per-node=2 scripts/slurm/mpi-gpu-nccl.sbatch
#   1 node x 4 ranks   : env -u SBATCH_RESERVATION sbatch scripts/slurm/mpi-gpu-nccl.sbatch
#   2 nodes x 8 ranks  : env -u SBATCH_RESERVATION sbatch --nodes=2 --ntasks-per-node=4 scripts/slurm/mpi-gpu-nccl.sbatch
#   H100-SXM5, 1 x 4   : env -u SBATCH_RESERVATION sbatch --constraint=h100-sxm5 scripts/slurm/mpi-gpu-nccl.sbatch
#
# Environment knobs (all optional):
#   LAYERS   probe layers for the A/B (default su4,cnot,rotation_remote,rotation_zz)
#   N        terms per rank (default 4000000); the probe's --n is N x ranks
#   REPS     --reps (default 5)
#   PAIRS    alternating host/nccl pairs, >= 5 for direction-consistency (default 5)
#   PROBE=0  run the differential net alone, skip the A/B
#   PS_REV   commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 cuda/12.8.0 nccl/2.23.4-1
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}
# Every rank execs the binary, so the build lives on the shared filesystem, not node-local /tmp.
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
mkdir -p "$CARGO_TARGET_DIR"
unset RUST_LOG RAYON_NUM_THREADS
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null; rm -rf "$CARGO_TARGET_DIR"' EXIT
[ -f "$repo/Cargo.lock" ] && cp "$repo/Cargo.lock" "$src/Cargo.lock"
cd "$src"
. scripts/jcc-rustflags.sh
if [ -n "$JCC_RUSTFLAGS" ]; then export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$JCC_RUSTFLAGS"; fi

nodes=${SLURM_JOB_NUM_NODES:-1}
ranks=${SLURM_NTASKS:-1}
cpus=${SLURM_CPUS_PER_TASK:-8}
echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
echo "== $ranks ranks over $nodes nodes, every node GPU visible to every task, $cpus CPUs each"
if [ $(( ranks & (ranks - 1) )) -ne 0 ]; then
    echo "the rank count $ranks is not a power of two (a partition is named by log2(P) GF(2) rows)" >&2
    exit 2
fi

# Diagnostics preamble: what NCCL sees before it picks a transport.
srun --ntasks-per-node=1 bash -c '
    echo "-- $(hostname -s)"
    nvidia-smi -L
    nvidia-smi topo -m
    ibv_devinfo -l 2>&1 || echo "(no ibv_devinfo)"
    lsmod | grep -E "nvidia_peermem|nv_peer_mem" || echo "(no nvidia_peermem / nv_peer_mem module)"
    ip -br addr
'

bin=$(cargo test --offline --release --features nccl,test-utils -p paulistrings --test mpi_ranks --no-run \
      --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: r=json.loads(l)
    except ValueError: continue
    if r.get("reason")=="compiler-artifact" and r["target"]["name"]=="mpi_ranks" and r.get("executable"): print(r["executable"])')
[ -n "$bin" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
echo "== mpi_ranks: $bin"

echo "== mpi_ranks under NCCL (NCCL_DEBUG on, so the chosen transport is on record)"
PAULISTRINGS_GPU_EXCHANGE=nccl NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,NET \
    srun --mpi=pmix --cpu-bind=cores \
    --export=ALL,PAULISTRINGS_GPU_EXCHANGE,NCCL_DEBUG,NCCL_DEBUG_SUBSYS "$bin"

echo "== mpi_ranks under host staging (the pre-NCCL baseline)"
PAULISTRINGS_GPU_EXCHANGE=host \
    srun --mpi=pmix --cpu-bind=cores --export=ALL,PAULISTRINGS_GPU_EXCHANGE "$bin"

if [ "${PROBE:-1}" = "1" ]; then
    cargo build --offline --release --features phase-timing,mpi,cuda,nccl -p paulistrings \
        --example phase_breakdown
    probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"
    out="$repo/benchmarks/results/$(date +%Y-%m-%d)-mpi-gpu-nccl"
    mkdir -p "$out"
    n=$(( ${N:-4000000} * ranks ))
    layers=${LAYERS:-su4,cnot,rotation_remote,rotation_zz}
    pairs=${PAIRS:-5}
    echo "== A/B: --gpu-exchange host vs nccl, $pairs alternating (abab) pair(s), $ranks ranks, --n $n, layers $layers"
    for ((i = 1; i <= pairs; i++)); do
        for mode in host nccl; do
            sidecar="$out/mpi-gpu-nccl-${SLURM_JOB_ID:-local}-r$ranks-$mode-p$i.jsonl"
            echo "== pair $i/$pairs, --gpu-exchange $mode -> $sidecar.rank*"
            srun --mpi=pmix --cpu-bind=cores "$probe" --mpi --device auto --threads "$cpus" \
                --gpu-exchange "$mode" --layers "$layers" --n "$n" --reps "${REPS:-5}" \
                --format json --json-out "$sidecar"
        done
    done
fi

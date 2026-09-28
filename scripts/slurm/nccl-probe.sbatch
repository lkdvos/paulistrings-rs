#!/usr/bin/env bash
#SBATCH --job-name=ps-nccl-probe
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --exclude=workergpu062
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:20:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The NCCL bring-up probe (`crates/paulistrings/examples/nccl_probe.rs`): the communicator init and warm-up the device exchange performs at scatter, one variant per srun step, each under a hard bound, so a hanging variant costs its bound and not the allocation.
# Variants: the engine's own shape (non-blocking init, one group to every peer, device by local rank), every GPU pair at two ranks, the NCCL knobs (P2P off, CUMEM off, Simple protocol, connections at init), a blocking communicator, one group per peer, a ring, and the CPU-locality device pick.
#
#   env -u SBATCH_RESERVATION sbatch scripts/slurm/nccl-probe.sbatch
#   env -u SBATCH_RESERVATION sbatch --nodelist=workergpu047 scripts/slurm/nccl-probe.sbatch
#
# Environment knobs (all optional):
#   STEP_BOUND      seconds one variant may take (default 120)
#   NCCL_TIMEOUT_S  the probe's wait bound, PAULISTRINGS_NCCL_TIMEOUT_S (default 40)
#   PAIRS=0         skip the two-rank GPU-pair matrix
#   EXTRA_ENV       "NCCL_X=1 NCCL_Y=0": one more variant of the engine's shape under these variables
#   PS_REV          commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/bounded.sh

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

echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
echo "== NCCL_DEBUG=INFO (INIT,P2P) goes to this log line-buffered, so a killed rank loses nothing; grep '^nccl_probe' for the verdict lines"

# What the node looks like before anything touches NCCL: links, link errors, ECC and reset state, leftover processes.
bounded 300 srun --ntasks-per-node=1 bash -c '
    echo "-- $(hostname -s)"
    nvidia-smi -L
    nvidia-smi topo -m
    scripts/slurm/check-gpu-links.sh ${GPU_LINKS_WARN:+--warn} || exit 4
    echo "-- nvlink status"; nvidia-smi nvlink --status || true
    echo "-- nvlink error counters"; nvidia-smi nvlink --errorcounters || true
    echo "-- ecc, row remapper"; nvidia-smi -q -d ECC,ROW_REMAPPER | grep -vE "^\s*$" || true
    echo "-- gpu state"; nvidia-smi --query-gpu=index,name,pci.bus_id,compute_mode,persistence_mode,clocks.sm,clocks.mem,temperature.gpu --format=csv || true
    echo "-- compute processes"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
    echo "-- numa"; cat /sys/devices/system/node/node*/cpulist
'

# A debug build: the probe waits on NCCL, not on the engine, and a release LTO build would take most of the job.
cargo build --offline --features nccl,test-utils -p paulistrings --example nccl_probe --example gpu_peer
probe="$CARGO_TARGET_DIR/debug/examples/nccl_probe"
peer="$CARGO_TARGET_DIR/debug/examples/gpu_peer"

echo "== gpu_peer 64 MiB: CUDA-level peer access and copy bandwidth per device pair, one process, both directions"
bounded 300 srun --ntasks=1 "$peer" 64 || echo "== gpu_peer FAILED (exit $?)"

export PAULISTRINGS_NCCL_TIMEOUT_S=${NCCL_TIMEOUT_S:-40}
failed=()
# variant <name> <ranks> [VAR=value ...] [probe args ...]
variant() {
    local name=$1 ranks=$2
    shift 2
    local envs=()
    while [ $# -gt 0 ] && [[ $1 == *=* ]]; do
        envs+=("$1")
        shift
    done
    echo "== variant $name: $ranks ranks, env ${envs[*]:-none}, args $*, $(date +%T)"
    local t0=$SECONDS status=0
    bounded "${STEP_BOUND:-120}" env "${envs[@]}" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P \
        srun --mpi=pmix --cpu-bind=cores --ntasks="$ranks" --export=ALL \
        stdbuf -oL -eL "$probe" --variant "$name" "$@" || status=$?
    echo "== variant $name: exit $status after $((SECONDS - t0))s"
    if [ "$status" != 0 ]; then
        failed+=("$name")
        leftover_gpu_processes
    fi
}

variant engine 4 --shape all
if [ "${PAIRS:-1}" != 0 ]; then
    for pair in 0,1 2,3 0,2 1,3 0,3 1,2; do
        variant "pair-${pair/,/-}" 2 --device "$pair" --shape all
    done
fi
variant p2p-off 4 NCCL_P2P_DISABLE=1 --shape all
variant cumem-off 4 NCCL_CUMEM_ENABLE=0 --shape all
variant proto-simple 4 NCCL_PROTO=Simple --shape all
variant connect-init 4 NCCL_RUNTIME_CONNECT=0 --shape all
variant blocking 4 --blocking --shape all
variant per-peer 4 --shape per-peer
variant ring 4 --shape ring
variant device-comm 4 --device comm --shape all
# shellcheck disable=SC2086
[ -n "${EXTRA_ENV:-}" ] && variant extra 4 $EXTRA_ENV --shape all
variant engine-again 4 --shape all

echo "== done $(date +%T); failed variants: ${failed[*]:-none}"
grep -h "^nccl_probe" "$repo/benchmarks/results/slurm-${SLURM_JOB_ID:-local}.out" 2>/dev/null | grep -E "summary|teardown=stuck|warmup=(Timeout|Nccl)|init=(Timeout|Nccl)" || true
[ "${#failed[@]}" = 0 ]

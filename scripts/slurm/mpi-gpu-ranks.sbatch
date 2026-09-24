#!/usr/bin/env bash
#SBATCH --job-name=ps-mpi-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# One CUDA device per MPI rank (`gpu::MpiGpuSum`, features `mpi,cuda`) across two 4-GPU nodes:
# the multi-rank differential net (`tests/mpi_ranks.rs`, whose device cases run when every rank
# sees a device), then `phase_breakdown --mpi --device auto` at replicated input `--n N x ranks`.
# `--gpus-per-task=1` gives each task one visible device, so `auto` resolves to ordinal 0 on
# every rank; the node type is the constraint (`sbatch --constraint=h100-sxm5 ...` for the
# 4 x H100-SXM5 genoa nodes). The rank count (nodes x tasks per node) must be a power of two.
#
# Environment knobs (all optional):
#   LAYERS   probe layers (default rotation_zz,cnot,gu2q,su4,rotation_remote; heavyhex_step runs as its own cell)
#   N        terms per rank (default 4000000); the probe's --n is N x ranks
#   REPS     --reps (default 5)
#   PROBE=0  run the differential net alone
#   PS_REV   commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 cuda/12.8.0
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
echo "== $ranks ranks over $nodes nodes, one GPU and $cpus CPUs each"
if [ $(( ranks & (ranks - 1) )) -ne 0 ]; then
    echo "the rank count $ranks is not a power of two (a partition is named by log2(P) GF(2) rows)" >&2
    exit 2
fi
srun --ntasks-per-node=1 bash -c 'echo "-- $(hostname -s)"; nvidia-smi -L; nvidia-smi topo -m'

cargo test --offline --release --features mpi,cuda -p paulistrings --test mpi_ranks --no-run
bin=$(cargo test --offline --release --features mpi,cuda -p paulistrings --test mpi_ranks --no-run \
      --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: r=json.loads(l)
    except ValueError: continue
    if r.get("reason")=="compiler-artifact" and r["target"]["name"]=="mpi_ranks" and r.get("executable"): print(r["executable"])')
[ -n "$bin" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
echo "== mpi_ranks: $bin"
srun --mpi=pmix --cpu-bind=cores "$bin"

if [ "${PROBE:-1}" = "1" ]; then
    cargo build --offline --release --features phase-timing,mpi,cuda -p paulistrings \
        --example phase_breakdown
    probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"
    out="$repo/benchmarks/results/$(date +%Y-%m-%d)-mpi-gpu"
    mkdir -p "$out"
    n=$(( ${N:-4000000} * ranks ))
    sidecar="$out/mpi-gpu-${SLURM_JOB_ID:-local}-r$ranks.jsonl"
    echo "== probe: $ranks ranks, --n $n (replicated), --device auto -> $sidecar.rank*"
    srun --mpi=pmix --cpu-bind=cores "$probe" --mpi --device auto --threads "$cpus" \
        --layers "${LAYERS:-rotation_zz,cnot,gu2q,su4,rotation_remote}" --n "$n" \
        --reps "${REPS:-5}" --format json --json-out "$sidecar"
    echo "== probe: heavyhex_step, five steps under coeff:2^-13"
    srun --mpi=pmix --cpu-bind=cores "$probe" --mpi --device auto --threads "$cpus" \
        --layers heavyhex_step --reps 5 --truncation coeff:0.0001220703125 --format json \
        --json-out "$sidecar"
fi

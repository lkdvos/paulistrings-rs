#!/usr/bin/env bash
#SBATCH --job-name=ps-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The CUDA backend on one 4-GPU node (see README.md): the single-device test net, the
# multi-device partitioned net at one partition per GPU, then the probe on one device and on all
# four, JSON sidecars into benchmarks/results/<date>-<node>/.
#
# Environment knobs (all optional):
#   LAYERS   probe layers, comma separated (default rotation_zz,cnot,gu2q,su4; heavyhex_step and
#            rotation_remote always run as their own cells, see below)
#   N        --n (default 4000000)
#   REPS     --reps (default 5)
#   DEVICES  the multi-device list (default: every GPU the job sees, e.g. 0,1,2,3)
#   TESTS=0  skip the two test nets
#   PS_REV   commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 cuda/12.8.0
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
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

node=$(hostname -s)
out="$repo/benchmarks/results/$(date +%Y-%m-%d)-$node"
mkdir -p "$out"
echo "== job ${SLURM_JOB_ID:-?} on $node, rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
nvidia-smi -L
nvidia-smi topo -m | tee "$out/gpu-${SLURM_JOB_ID:-local}-topo.txt"
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE > "$out/gpu-${SLURM_JOB_ID:-local}-clocks-start.txt"

ngpu=$(nvidia-smi -L | wc -l)
: "${DEVICES:=$(seq -s, 0 $((ngpu - 1)))}"
: "${LAYERS:=rotation_zz,cnot,gu2q,su4}"
: "${N:=4000000}"
: "${REPS:=5}"
echo "gpus: $ngpu, multi-device list: $DEVICES"

cargo build --offline --release -p paulistrings --features cuda,phase-timing,test-utils --example phase_breakdown
probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"

if [ "${TESTS:-1}" = "1" ]; then
    echo "== cargo test --features cuda (single device)"
    cargo test --offline -p paulistrings --features cuda
    echo "== propagate_gpu_partitioned, one partition per device ($DEVICES)"
    PAULISTRINGS_GPU_TEST_DEVICES="$DEVICES" \
        cargo test --offline -p paulistrings --features cuda --test propagate_gpu_partitioned
fi

# One device, then all of them. rotation_remote needs a remote delta, so it only runs multi-device;
# heavyhex_step is five Trotter steps under the coefficient cutoff the ccqlin038 table used.
tag="${SLURM_JOB_ID:-local}"
cells() {
    local devices=$1 name=$2 layers=$3
    echo "== probe --device $devices: $layers"
    "$probe" --device "$devices" --n "$N" --reps "$REPS" --layers "$layers" --format json \
        --json-out "$out/gpu-$tag-$name.jsonl"
    echo "== probe --device $devices: heavyhex_step"
    "$probe" --device "$devices" --reps 5 --layers heavyhex_step --truncation coeff:0.0001220703125 \
        --format json --json-out "$out/gpu-$tag-$name.jsonl"
}
cells 0 dev0 "$LAYERS"
cells "$DEVICES" "dev${DEVICES//,/}" "$LAYERS,rotation_remote"

nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE > "$out/gpu-$tag-clocks-end.txt"
echo "results: $out/gpu-$tag-*.jsonl"

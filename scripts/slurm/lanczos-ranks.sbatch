#!/usr/bin/env bash
#SBATCH --job-name=ps-lanczos
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Operator Lanczos across nodes (`crates/paulistrings/examples/lanczos_probe.rs`, one in-place `add_liouvillian` per step): per step the exchange volume and time, the share balance and the peak RSS.
# Per model: one rank on a whole node as the baseline, then one rank per NUMA domain on each node count in NODE_COUNTS, for every choice of partition rows; `scripts/lanczos-report.py` tabulates the cells at the end.
#
# Environment knobs (all optional):
#   MODELS       model:steps pairs (default "chain:40:17 square:6x6:9")
#   ROWS         partition rows per cell (default "seeded xblocks")
#   NODE_COUNTS  node counts to run on, each a power of two within the allocation (default 1 2 4 ... up to it)
#   THRESHOLD    coefficient threshold per step (default 1e-12)
#   PS_REV       commit to build (default: HEAD when the job starts)
#   SMOKE=1      the pre-submit check, run locally as `SMOKE=1 bash scripts/slurm/lanczos-ranks.sbatch`: toy models under `mpirun -n 2`
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh

if [ "${SMOKE:-0}" = 1 ]; then
    : "${MODELS:=chain:24:8 square:4x4:5}" "${NODE_COUNTS:=1}"
fi
MODELS=${MODELS:-chain:40:17 square:6x6:9}
ROWS=${ROWS:-seeded xblocks}
THRESHOLD=${THRESHOLD:-1e-12}

if command -v module >/dev/null 2>&1; then
    module purge
    module load modules/2.5-20261027 openmpi/5.0.10 llvm/20.1.8
fi
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"   # bindgen (rsmpi) at build time
use_shared_rust
# Every rank execs the binary from its cwd, so the build and its checkout live on the shared filesystem, not node-local /tmp.
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
checkout_rev "$PS_REV" "$CARGO_TARGET_DIR/src"
out=$(results_dir)
cd "$CARGO_TARGET_DIR/src"
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
cargo build --offline --release --features mpi -p paulistrings --example lanczos_probe >/dev/null
probe="$CARGO_TARGET_DIR/release/examples/lanczos_probe"

job=${SLURM_JOB_ID:-local}
echo "== job $job on ${SLURM_JOB_NODELIST:-$(hostname -s)}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
numa=$(ls -d /sys/devices/system/node/node* | wc -l)
per_node=1
while [ $(( per_node * 2 )) -le "$numa" ]; do per_node=$(( per_node * 2 )); done
cpus=${SLURM_CPUS_ON_NODE:-$(nproc)}
cpus_per_rank=$(( cpus / per_node ))
nodes=${SLURM_JOB_NUM_NODES:-1}
if [ -z "${NODE_COUNTS:-}" ]; then
    NODE_COUNTS=1
    n=2
    while [ "$n" -le "$nodes" ]; do NODE_COUNTS="$NODE_COUNTS $n"; n=$(( n * 2 )); done
fi
echo "== $per_node rank(s) per node, $cpus_per_rank CPUs each; node counts: $NODE_COUNTS"

# run <nodes> <ranks> <threads per rank> <cell name> <probe args...>
run() {
    local n=$1 ranks=$2 threads=$3 cell=$4
    shift 4
    echo "== $cell: $ranks rank(s) on $n node(s), $threads threads each"
    if [ "${SMOKE:-0}" = 1 ]; then
        mpirun -n "$ranks" --oversubscribe "$probe" --mpi --unpinned --threads 2 "$@" \
            --json-out "$out/$cell.jsonl" --label "$cell"
    elif [ "$ranks" = 1 ]; then
        srun --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --cpu-bind=none --mpi=pmix \
            "$probe" --mpi --threads "$threads" "$@" --json-out "$out/$cell.jsonl" --label "$cell"
    else
        srun --nodes="$n" --ntasks="$ranks" --ntasks-per-node=$(( ranks / n )) \
            --cpus-per-task="$cpus_per_rank" --cpu-bind=ldoms --mpi=pmix \
            "$probe" --mpi --threads "$threads" "$@" --json-out "$out/$cell.jsonl" --label "$cell"
    fi
}

cells=()
for spec in $MODELS; do
    model=${spec%:*}
    steps=${spec##*:}
    args=(--model "$model" --steps "$steps" --threshold "$THRESHOLD")
    tag=lanczos-$job-${model/:/-}
    if [ "${SMOKE:-0}" = 1 ]; then
        run 1 1 2 "$tag-whole-node" "${args[@]}"
    else
        run 1 1 "$cpus" "$tag-whole-node" "${args[@]}"
    fi
    cells+=("$out/$tag-whole-node.jsonl")
    for rows in $ROWS; do
        for n in $NODE_COUNTS; do
            ranks=$(( n * per_node ))
            [ "${SMOKE:-0}" = 1 ] && ranks=2
            cell="$tag-$rows-r$ranks"
            run "$n" "$ranks" "$cpus_per_rank" "$cell" "${args[@]}" --rows "$rows"
            cells+=("$out/$cell.jsonl")
        done
    done
done

echo "== summary ($out)"
python3 scripts/lanczos-report.py "${cells[@]}"

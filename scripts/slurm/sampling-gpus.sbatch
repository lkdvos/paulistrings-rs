#!/usr/bin/env bash
#SBATCH --job-name=ps-sampling-gpus
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb&rocky9
#SBATCH --exclude=workergpu062
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# One sampled run split over every GPU of one node in one process (`--backend gpus`, GpuPartitionedSum, peer copies), against a one-GPU reference (README.md §The sampled multi-GPU jobs).
# Cells: scripts/slurm/sampling-cells.sh; one JSON line each into benchmarks/results/<date>-<node>/sampling-gpus-<job>.jsonl.
#
# Environment knobs (all optional):
#   CACHES       split-run caches, samples = cache / 4 (default "1e8 4e8")
#   REF_CACHE    the one-GPU reference cache and approx-topn cap (default 1e8); the approx-topn splits run at CACHES
#   THRESHOLDS   kicked-Ising coefficient thresholds, each on one GPU and split (default "2.5e-5 1e-5")
#   DEVICES      the device list (default: every GPU the job sees)
#   SWEEP=rows   the row-choice sweep instead (run_rows_sweep in sampling-cells.sh), with ROWS (default "cut quadrant xparity:2") and P2P_LEVELS (NCCL_P2P_LEVEL values, default none)
#   PT=0         skip the phase-timing rerun of the largest kicked-Ising cell (export/exchange split)
#   CELL_BOUND   seconds one cell may take (default 2400)
#   PS_REV       commit to build (default: HEAD when the job starts), in a private git worktree
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh
. scripts/slurm/bounded.sh

module purge
module load modules/2.5-20261027 cuda/12.9.1
use_shared_rust
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
checkout_rev "$PS_REV" "$CARGO_TARGET_DIR/src"
out=$(results_dir)
cd "$CARGO_TARGET_DIR/src"
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

tag="${SLURM_JOB_ID:-local}"
echo "== job $tag on $(hostname -s), rev $PS_REV ($(git log -1 --format=%s "$PS_REV")), $(date -Is)"
nvidia-smi -L
nvidia-smi topo -m | tee "$out/sampling-gpus-$tag-topo.txt"
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE > "$out/sampling-gpus-$tag-clocks-start.txt"
ngpu=$(nvidia-smi -L | wc -l)
: "${DEVICES:=$(seq -s, 0 $((ngpu - 1)))}"

cargo build --offline --release -p paulistrings --features cuda --example sampling_bench
bin="$CARGO_TARGET_DIR/release/examples/sampling_bench"
sidecar="$out/sampling-gpus-$tag.jsonl"
launch_one=("$bin" --backend gpu --device 0)
launch_all=("$bin" --backend gpus --devices "$DEVICES")
failed=0
. scripts/slurm/sampling-cells.sh
echo "== sweep on devices $DEVICES, caches $CACHES, reference $REF_CACHE -> $sidecar"
if [ "${SWEEP:-}" = rows ]; then run_rows_sweep "$(tr , '\n' <<< "$DEVICES" | wc -l)"; else run_sweep; fi

phase_timing_cell cuda,phase-timing "$out/sampling-gpus-$tag-pt.jsonl"

nvidia-smi -q -d CLOCK,POWER,TEMPERATURE,PERFORMANCE > "$out/sampling-gpus-$tag-clocks-end.txt"
echo "results: $out/sampling-gpus-$tag*.jsonl"
exit "$failed"

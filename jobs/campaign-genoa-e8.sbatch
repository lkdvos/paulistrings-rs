#!/usr/bin/env bash
#SBATCH --job-name=quera-e8
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's E8 (communication-aware
# hash): the in-process partitioned engine, same canonical task as
# campaign-genoa.sbatch's C1/C2, run once under `partition_row_policy=random`
# and once under `partition_row_policy=cut` so `analysis/normalize.py::
# hash_communication` has a real, hardware-valid pair to compare (decisions.md
# #21 built the mechanism and validated it with a preflight-monkeypatched
# local test only -- this is the real allocation that mechanism was waiting
# on). The distributed (`comm=`) path still only supports "random"
# (campaign-genoa-distributed.sbatch), so this uses the in-process partitioned
# engine on one node instead -- no MPI needed, and much cheaper than a
# multi-node job.
#
# Reuses campaign-genoa.sbatch's worktree/venv/maturin preamble verbatim.
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-e8.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree.
#   PARTITIONS      number of in-process partitions (default: 2, one per NUMA domain on genoa).
#   MIN_ABS_COEFF   ONE dyadic value (default: 2^-16, the point campaign-genoa.sbatch already
#                    trusts, so this is an apples-to-apples E6-style overlap point, not a blind
#                    leap to an untested tolerance).
#   TROTTER_STEPS   headline depth (default: 20, contract.md).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   REPS            repetitions per (policy, cell) pair (default: 1 -- this is a mechanism
#                    comparison, not a statistics run; bump once the cost is known).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors campaign-genoa.sbatch) ------------------------------
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}

module purge
module load modules/2.4-20250724 python/3.11.11
PYTHON_BIN=$(command -v python3.11)

# shellcheck disable=SC1091
source scripts/jcc-rustflags.sh   # exports JCC_RUSTFLAGS; append, never assign (see that file)
export RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }${JCC_RUSTFLAGS}"

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep -E 'Thread\(s\) per core'   # contract.md: verify, never trust Slurm's ThreadsPerCore alone

# --- worktree build (mirrors campaign-genoa.sbatch's PS_REV pattern) --------
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-e8-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null' EXIT

venv="$CARGO_TARGET_DIR/venv"
"$PYTHON_BIN" -m venv "$venv"
# shellcheck disable=SC1091
source "$venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet 'maturin>=1.5,<2.0' numpy
(cd "$src" && maturin develop --release -m crates/paulistrings-py/Cargo.toml)

# --- matrix ------------------------------------------------------------
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
: "${PARTITIONS:=2}"
: "${MIN_ABS_COEFF:=1.5258789e-05}"   # 2^-16
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"    # 7*pi/32
: "${REPS:=1}"
threads_per_partition=$(( phys / PARTITIONS ))
echo "physical cores=$phys partitions=$PARTITIONS threads_per_partition=$threads_per_partition min_abs_coeff=$MIN_ABS_COEFF reps=$REPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-e8"
mkdir -p "$out_dir"

cell_json="$CARGO_TARGET_DIR/cell.json"
for policy in random cut; do
  # Shared config_id across policies (only partition_row_policy differs) so
  # figures/make_compact_figures.py::make_hash_communication_figure groups
  # the random/cut pair into one x-axis category instead of two singletons.
  config_id="genoa-e8-eps${MIN_ABS_COEFF}-p${PARTITIONS}"
  for rep in $(seq 0 $((REPS - 1))); do
    cat > "$cell_json" <<EOF
{
  "variant_id": "bucketed_current",
  "theta_h": $THETA_H,
  "trotter_steps": $TROTTER_STEPS,
  "min_abs_coeff": $MIN_ABS_COEFF,
  "direction": "heisenberg",
  "state": "z+",
  "observable": "canonical_z_127",
  "threads": $threads_per_partition,
  "partitions": $PARTITIONS,
  "repetition_index": $rep,
  "task_id": "T02-canonical",
  "config_id": "$config_id",
  "partition_row_policy": "$policy"
}
EOF
    echo "== cell $config_id rep=$rep (policy=$policy)"
    RAYON_NUM_THREADS=$threads_per_partition python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
        "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
  done
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

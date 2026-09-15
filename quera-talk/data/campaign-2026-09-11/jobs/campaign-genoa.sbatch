#!/usr/bin/env bash
#SBATCH --job-name=quera-c1c2
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11 (T07): C1 smoke + C2
# single/multi-thread tolerance sweep on the canonical 127-qubit task,
# `variant_id=bucketed_current` only (T01's historical revisions are
# deferred — see decisions.md and the T07 report).
#
# Reuses the worktree-per-revision / shared-toolchain / JCC-rustflags
# conventions of scripts/slurm/ab-campaign.sbatch rather than reinventing
# them (see scripts/slurm/README.md). Does NOT call sbatch anywhere —
# submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree
#                    (never the live checkout, per ab-campaign.sbatch's note on job 7004970).
#   MIN_ABS_COEFF   space-separated dyadic grid, loosest to tightest (default: a short one).
#   THREADS         space-separated thread counts (default: "1 <physical cores on this node>").
#   TROTTER_STEPS   headline depth for this stage (default: 20; contract.md also names 9 as
#                    the published-anchor rung, not swept here — a single-cell override).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   REPS            repetitions per cell (default: 3 — C2 is a tolerance sweep, not the full
#                    campaign's statistics; bump for a tighter noise floor).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors ab-campaign.sbatch) ---------------------------------
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

# --- one-time verification: Slurm's ThreadsPerCore is NOT trusted ----------
# contract.md: "Slurm reports ThreadsPerCore=1 ... still must be verified
# with lscpu on an actual allocation before the first timed run". This is
# that verification (preflight.py repeats the equivalent check per process,
# but this is the human-readable job-log record of it).
lscpu | grep -E 'Thread\(s\) per core'

# --- worktree build (mirrors ab-campaign.sbatch's PS_REV pattern) ----------
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-target-${SLURM_JOB_ID:-local}"
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
: "${THREADS:=1 $phys}"
: "${MIN_ABS_COEFF:=1.5258789e-05 3.8146973e-06 9.5367432e-07}"   # 2^-16, 2^-18, 2^-20
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"                                # 7*pi/32
: "${REPS:=3}"
echo "physical cores=$phys threads='$THREADS' min_abs_coeff='$MIN_ABS_COEFF' reps=$REPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}"
mkdir -p "$out_dir"

cell_json="$CARGO_TARGET_DIR/cell.json"
config_index=0
for eps in $MIN_ABS_COEFF; do
  for t in $THREADS; do
    config_id="genoa-c2-eps${eps}-t${t}"
    for rep in $(seq 0 $((REPS - 1))); do
      cat > "$cell_json" <<EOF
{
  "variant_id": "bucketed_current",
  "theta_h": $THETA_H,
  "trotter_steps": $TROTTER_STEPS,
  "min_abs_coeff": $eps,
  "direction": "heisenberg",
  "state": "z+",
  "observable": "canonical_z_127",
  "threads": $t,
  "partitions": null,
  "repetition_index": $rep,
  "task_id": "T02-canonical",
  "config_id": "$config_id"
}
EOF
      echo "== cell $config_id rep=$rep (config #$config_index)"
      RAYON_NUM_THREADS=$t python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
          "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
    done
    config_index=$((config_index + 1))
  done
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

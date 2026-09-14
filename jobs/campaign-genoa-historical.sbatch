#!/usr/bin/env bash
#SBATCH --job-name=quera-historical
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's E1/E2 prep: the four
# historical Rust-revision variants (`jobs/run_cell_historical.py`) plus one
# `bucketed_current` cell at the same reduced scale (`jobs/run_cell.py`), all
# written to the same out_dir/runs.jsonl so they are directly comparable.
#
# This is NOT the frozen 127-qubit canonical task -- each historical commit's
# API differs too much from `examples/common/` to build that circuit, so
# every cell here uses run_cell_historical.py's small fixed reduced scale
# (n_qubits=12, 2 Trotter steps, theta_h=7pi/32, theta_zz=-pi/2) -- see that
# module's docstring and decisions.md for the full rationale and the
# per-variant feasibility table. `direct_small_sum_path` runs its default
# (sorted) engine only (no engine= kwarg backport applied, decisions.md).
#
# Reuses campaign-genoa.sbatch's worktree/venv/maturin preamble for the
# `bucketed_current` cell; run_cell_historical.py builds its own four
# throwaway worktrees internally (never the live checkout, and always
# cleaned up -- see that module). Does NOT call sbatch anywhere --
# submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-historical.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build for the bucketed_current cell (default: HEAD).
#   MIN_ABS_COEFF   ONE value shared by every variant, historical and current (default: 1e-6,
#                    loose enough that naive_baseline's unbucketed serial engine finishes in
#                    well under a second at this scale -- decisions.md).
#   REPS            repetitions per variant (default: 1 -- a mechanism/feasibility comparison,
#                    not a statistics run).
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

: "${MIN_ABS_COEFF:=1e-6}"
: "${REPS:=1}"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-historical"
mkdir -p "$out_dir"

# --- four historical variants (run_cell_historical.py builds its own
# throwaway worktree per variant under CARGO_TARGET_DIR, never the live
# checkout) ------------------------------------------------------------
hist_scratch="${TMPDIR:-/tmp}/quera-campaign-historical-${SLURM_JOB_ID:-local}"
mkdir -p "$hist_scratch"
cell_json="$hist_scratch/cell.json"
for variant in naive_baseline direct_small_sum_path bucketed_engine_serial bucketed_engine_parallel; do
  for rep in $(seq 0 $((REPS - 1))); do
    cat > "$cell_json" <<EOF
{
  "variant_id": "$variant",
  "min_abs_coeff": $MIN_ABS_COEFF,
  "direction": "heisenberg",
  "repetition_index": $rep,
  "config_id": "genoa-historical-eps${MIN_ABS_COEFF}"
}
EOF
    echo "== historical cell variant=$variant rep=$rep"
    "$PYTHON_BIN" quera-talk-data/campaign-2026-09-11/jobs/run_cell_historical.py \
        "$cell_json" --out-dir "$out_dir" --scratch-root "$hist_scratch" \
        || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
  done
done

# --- one bucketed_current cell at the same reduced scale, for a fair
# same-out_dir comparison (mirrors campaign-genoa.sbatch's worktree/venv/
# maturin preamble) ------------------------------------------------------
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-historical-current-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building bucketed_current rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
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

for rep in $(seq 0 $((REPS - 1))); do
  cat > "$cell_json" <<EOF
{
  "variant_id": "bucketed_current",
  "theta_h": 0.6872233929727672,
  "trotter_steps": 2,
  "min_abs_coeff": $MIN_ABS_COEFF,
  "direction": "heisenberg",
  "state": "z+",
  "observable": "debug_single_z",
  "threads": 1,
  "partitions": null,
  "repetition_index": $rep,
  "task_id": "T02-canonical-historical-reduced",
  "config_id": "genoa-historical-eps${MIN_ABS_COEFF}",
  "n_qubits": 12
}
EOF
  echo "== bucketed_current cell rep=$rep"
  RAYON_NUM_THREADS=1 python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
      "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

#!/usr/bin/env bash
#SBATCH --job-name=quera-historical
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's recurring-figure
# refinement (E1/E2/E4/E5-historical): the four historical Rust-revision
# variants (`jobs/run_cell_historical.py`) plus one `bucketed_current` leg
# (`jobs/run_cell.py`), all written to the same out_dir/runs.jsonl.
#
# 2026-09-14 full-scale extension (decisions.md, "quera-talk full-scale
# historical sweep"): every variant now runs at n_qubits=127 and
# trotter_steps=10 on the same linear-chain-scaled-up construction the
# original toy comparison used (NOT the heavy-hex topology -- three of the
# four historical commits cannot build that circuit at all, see
# run_cell_historical.py's docstring). Real local feasibility timing (non-genoa
# workstation) showed n_qubits does not move the needle at any depth tested
# (a local observable's backward light cone stays local after only a few
# Trotter layers), so 127 qubits is free for every variant; trotter_steps=10
# is what actually produces a cutoff-sensitive term-count curve. Each variant
# gets its OWN tolerance list from the shared 4-point campaign grid
# {2^-12, 2^-14, 2^-16, 2^-18} -- naive_baseline (unbucketed, serial, no
# truncation-aware engine) already costs ~130s locally for its single loosest
# point at this depth, so it gets ONLY that one point; the other three
# variants and bucketed_current handle the full grid in well under a minute
# per point locally and get all four. This is a deliberate cost-saving
# measure AND the intended narrative device for the recurring figure's cost
# panel (`figures/make_recurring_figure.py::_draw_cost_panel`): a variant
# whose line simply stops short at the tighter cutoffs shows those regimes
# didn't exist until the engine improved -- no failed/OOM record needed for
# the untried points, just don't attempt them.
#
# `direct_small_sum_path` runs its default (sorted) engine only (no engine=
# kwarg backport applied, decisions.md).
#
# Reuses campaign-genoa.sbatch's worktree/venv/maturin preamble for the
# `bucketed_current` cell; run_cell_historical.py builds its own throwaway
# worktree per variant internally (never the live checkout, always cleaned
# up -- see that module), once per variant regardless of how many cutoffs are
# swept. Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-historical.sbatch
#
# Environment knobs (all optional):
#   REPS   repetitions per variant/cutoff (default: 1 -- a mechanism/feasibility
#           comparison, not a statistics run).
#   PS_REV commit to build for the bucketed_current cell (default: HEAD).
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

: "${REPS:=1}"

# Shared campaign cutoff grid (jobs/campaign-genoa-convergence.sbatch,
# jobs/campaign-genoa-e8.sbatch): 2^-12, 2^-14, 2^-16, 2^-18.
FULL_GRID='[2.44140625e-04, 6.103515625e-05, 1.5258789e-05, 3.8146973e-06]'
LOOSEST_ONLY='[2.44140625e-04]'

# Per-variant (n_qubits, trotter_steps, cutoff-grid) plan -- see the header
# comment and decisions.md for the real local timing that justified it.
FULL_SCALE_N_QUBITS=127
FULL_SCALE_TROTTER_STEPS=10

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-historical"
mkdir -p "$out_dir"

# --- four historical variants (run_cell_historical.py builds its own
# throwaway worktree per variant under CARGO_TARGET_DIR, never the live
# checkout, ONCE per variant regardless of how many cutoffs it sweeps)
# ------------------------------------------------------------
hist_scratch="${TMPDIR:-/tmp}/quera-campaign-historical-${SLURM_JOB_ID:-local}"
mkdir -p "$hist_scratch"
cell_json="$hist_scratch/cell.json"

declare -A VARIANT_GRID=(
  [naive_baseline]="$LOOSEST_ONLY"
  [direct_small_sum_path]="$FULL_GRID"
  [bucketed_engine_serial]="$FULL_GRID"
  [bucketed_engine_parallel]="$FULL_GRID"
)

for variant in naive_baseline direct_small_sum_path bucketed_engine_serial bucketed_engine_parallel; do
  grid="${VARIANT_GRID[$variant]}"
  for rep in $(seq 0 $((REPS - 1))); do
    cat > "$cell_json" <<EOF
{
  "variant_id": "$variant",
  "n_qubits": $FULL_SCALE_N_QUBITS,
  "trotter_steps": $FULL_SCALE_TROTTER_STEPS,
  "min_abs_coeff": $grid,
  "direction": "heisenberg",
  "repetition_index": $rep,
  "config_id": "genoa-historical-fullscale"
}
EOF
    echo "== historical cell variant=$variant rep=$rep grid=$grid"
    "$PYTHON_BIN" quera-talk-data/campaign-2026-09-11/jobs/run_cell_historical.py \
        "$cell_json" --out-dir "$out_dir" --scratch-root "$hist_scratch" \
        || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
  done
done

# --- bucketed_current at the same (n_qubits, trotter_steps) and the same
# tolerance grid as the strongest historical variant (bucketed_engine_parallel),
# for a fair current-vs-historical overlay. run_cell.py takes one cutoff per
# invocation, so this loops in bash (mirrors campaign-genoa.sbatch's
# worktree/venv/maturin preamble) ------------------------------------------
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

CURRENT_GRID=(2.44140625e-04 6.103515625e-05 1.5258789e-05 3.8146973e-06)
for min_abs_coeff in "${CURRENT_GRID[@]}"; do
  for rep in $(seq 0 $((REPS - 1))); do
    cat > "$cell_json" <<EOF
{
  "variant_id": "bucketed_current",
  "theta_h": 0.6872233929727672,
  "trotter_steps": $FULL_SCALE_TROTTER_STEPS,
  "min_abs_coeff": $min_abs_coeff,
  "direction": "heisenberg",
  "state": "z+",
  "observable": "debug_single_z",
  "threads": 1,
  "partitions": null,
  "repetition_index": $rep,
  "task_id": "T02-canonical-historical-reduced",
  "config_id": "genoa-historical-fullscale",
  "n_qubits": $FULL_SCALE_N_QUBITS
}
EOF
    echo "== bucketed_current cell rep=$rep eps=$min_abs_coeff"
    RAYON_NUM_THREADS=1 python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
        "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
  done
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

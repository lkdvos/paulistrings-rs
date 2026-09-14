#!/usr/bin/env bash
#SBATCH --job-name=quera-convergence
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11: the observable's
# expectation value at every Trotter step, for a small grid of min_abs_coeff
# cutoffs (2^-12 .. 2^-18), via `jobs/run_convergence_sweep.py`. Replaces a
# single-point accuracy comparison (decisions.md #26-27) with a real
# convergence-trajectory figure -- one line per cutoff.
#
# Reuses campaign-genoa.sbatch's worktree/venv/maturin preamble verbatim.
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-convergence.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree.
#   MIN_ABS_COEFF   space-separated dyadic grid (default: 2^-12, 2^-14, 2^-16, 2^-18 -- kept
#                    loose per the user's time-budget request; each cutoff re-propagates every
#                    circuit prefix up to trotter_steps, so cost grows fast with tighter eps).
#   TROTTER_STEPS   headline depth (default: 20, contract.md).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   THREADS         thread count for the sweep (default: physical cores on this node).
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
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-convergence-target-${SLURM_JOB_ID:-local}"
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

# --- sweep -------------------------------------------------------------
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
: "${THREADS:=$phys}"
: "${MIN_ABS_COEFF:=2.44140625e-04 6.103515625e-05 1.5258789e-05 3.8146973e-06}"   # 2^-12, 2^-14, 2^-16, 2^-18
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"    # 7*pi/32
echo "physical cores=$phys threads=$THREADS min_abs_coeff='$MIN_ABS_COEFF' trotter_steps=$TROTTER_STEPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-convergence"
mkdir -p "$out_dir"

echo "== convergence sweep, eps in {$MIN_ABS_COEFF}"
RAYON_NUM_THREADS=$THREADS python quera-talk-data/campaign-2026-09-11/jobs/run_convergence_sweep.py \
    --out-dir "$out_dir" --min-abs-coeff $MIN_ABS_COEFF --trotter-steps "$TROTTER_STEPS" --theta-h "$THETA_H" \
    || echo "  sweep failed/invalid — see $out_dir/convergence.jsonl status field"

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

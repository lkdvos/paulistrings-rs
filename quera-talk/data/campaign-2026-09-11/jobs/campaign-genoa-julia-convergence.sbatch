#!/usr/bin/env bash
#SBATCH --job-name=quera-julia-convergence
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's real Julia convergence
# trajectory (E9 overlay): PauliPropagation.jl's per-Trotter-step <O>, via
# runner.jl's new PP_LAYER_EXPECTATION diagnostic, for the same canonical
# 127-qubit task as campaign-genoa-convergence.sbatch's Rust-side sweep.
#
# WHY THIS IS SLOW AND WHY THE DEFAULT GRID STOPS AT eps=2^-16: growing a
# Heisenberg-picture circuit prefix prepends new gates at the INNERMOST
# conjugation position, so a longer prefix's result cannot be built
# incrementally from a shorter prefix's -- confirmed against
# PauliPropagation.jl 0.8.2's own propagate!/propagate!(::
# AbstractPauliPropagationCache, ...) source (runner.jl's module header has
# the full derivation), not merely assumed from an API gap. Every one of the
# 20 prefixes is therefore a genuinely fresh, from-scratch propagation, and
# Julia runs this SINGLE-THREADED (decisions.md #10/#27's precedent): the
# Rust engine's convergence sweep only absorbed this same redundancy cheaply
# because it ran at 96 threads. Real Rust term-count data
# (raw/2026-09-14-worker7172-convergence/convergence.jsonl) shows the
# trajectory saturating well before step 20 at eps in {2^-12, 2^-14, 2^-16},
# so the total sweep cost for those cutoffs is roughly a 6-9x multiple of one
# full single-threaded run -- estimated at ~15 min, ~1 hour, and ~9 hours
# respectively (decisions.md's Julia-convergence entry has the full
# derivation and its assumptions). eps=2^-18 is EXCLUDED from the default
# grid: even the RUST engine's own single-thread full run at that cutoff has
# twice failed to complete inside an 8h wall cap (job-ledger.jsonl
# 7030090/7031789), and Julia is measurably slower per gate single-threaded
# (decisions.md #10/#27) -- attempting it here would almost certainly repeat
# that failure at higher cost. Do not add it to MIN_ABS_COEFF without a much
# larger --time budget and a real willingness to eat a probable second
# multi-day timeout.
#
# Reuses campaign-genoa-julia.sbatch's worktree/venv/maturin/Julia-project
# preamble verbatim (this leg still builds the circuit/observable through
# paulistrings, per jobs/run_convergence_sweep_julia.py's module docstring --
# only the propagation itself is external).
#
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-julia-convergence.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree.
#   MIN_ABS_COEFF   space-separated dyadic grid (default: 2^-12, 2^-14, 2^-16 -- deliberately
#                    NOT 2^-18, see above). Each cutoff is ONE runner.jl process covering all
#                    TROTTER_STEPS prefixes internally (jobs/run_convergence_sweep_julia.py),
#                    not one process per step.
#   TROTTER_STEPS   headline depth (default: 20, contract.md).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   JULIA_SWEEP_TIMEOUT  seconds before giving up on one cutoff's runner.jl subprocess
#                    (default: run_convergence_sweep_julia.py's own, 86400s/24h -- deliberately
#                    generous, since a single cutoff's WHOLE sweep is one subprocess call; the
#                    job's own --time is still the real backstop).
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
module load modules/2.4-20250724 python/3.11.11 julia/1.11.2
PYTHON_BIN=$(command -v python3.11)
command -v julia >/dev/null || { echo "julia not on PATH after module load julia/1.11.2" >&2; exit 3; }

# shellcheck disable=SC1091
source scripts/jcc-rustflags.sh   # exports JCC_RUSTFLAGS; append, never assign (see that file)
export RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }${JCC_RUSTFLAGS}"

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
echo "julia: $(julia --version)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# --- worktree build, same convention as campaign-genoa.sbatch ---------------
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-julia-convergence-target-${SLURM_JOB_ID:-local}"
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

# --- Julia project, on the shared filesystem so a re-run does not re-resolve ---
julia --project=benchmarks/julia -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmarks/julia -e 'using Pkg; Pkg.status()'

# --- sweep -------------------------------------------------------------
: "${MIN_ABS_COEFF:=2.44140625e-04 6.103515625e-05 1.5258789e-05}"   # 2^-12, 2^-14, 2^-16 (NOT 2^-18, see header)
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"    # 7*pi/32
: "${JULIA_SWEEP_TIMEOUT:=}"
echo "min_abs_coeff='$MIN_ABS_COEFF' trotter_steps=$TROTTER_STEPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-julia-convergence"
mkdir -p "$out_dir"

timeout_flag=()
if [ -n "$JULIA_SWEEP_TIMEOUT" ]; then
  timeout_flag=(--timeout "$JULIA_SWEEP_TIMEOUT")
fi

echo "== julia convergence sweep, eps in {$MIN_ABS_COEFF}"
python quera-talk-data/campaign-2026-09-11/jobs/run_convergence_sweep_julia.py \
    --out-dir "$out_dir" --min-abs-coeff $MIN_ABS_COEFF --trotter-steps "$TROTTER_STEPS" \
    --theta-h "$THETA_H" "${timeout_flag[@]}" \
    || echo "  sweep failed/invalid — see $out_dir/convergence_julia.jsonl status field"

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

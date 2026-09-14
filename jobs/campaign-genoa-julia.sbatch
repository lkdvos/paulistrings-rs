#!/usr/bin/env bash
#SBATCH --job-name=quera-c1c2-julia
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's E0 (external baseline):
# the same canonical 127-qubit task as campaign-genoa.sbatch's C1/C2, run
# through PauliPropagation.jl (jobs/run_cell_julia.py) instead of the Rust
# engine, at the same hardware class for a fair wall-clock comparison
# (contract.md#hardware-contract).
#
# A separate script rather than folded into campaign-genoa.sbatch: both need
# the Rust worktree build (run_cell_julia.py still builds the circuit/
# observable through paulistrings, per its module docstring — only the
# propagation itself is external), so the preamble below duplicates that
# script's build steps. This mirrors the already-established precedent of
# campaign-genoa-distributed.sbatch (decisions.md #19), which duplicates the
# same preamble for the same reason rather than branching one script two ways.
#
# Does NOT call sbatch anywhere — submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-julia.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree.
#   MIN_ABS_COEFF   space-separated dyadic grid, loosest to tightest (default: matches
#                    campaign-genoa.sbatch's C2 grid, so cells line up 1:1 for comparison).
#   TROTTER_STEPS   headline depth (default: 20, contract.md).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   REPS            repetitions per cell (default: 1 — runner.jl's own PP_WARM_REPEATS=3
#                    already gives an in-process warm-time estimate per cell; this is for
#                    cross-process noise, not the primary noise floor).
#   JULIA_WARM_REPEATS  overrides runner.jl's PP_WARM_REPEATS (default: runner.jl's own, 3).
#   JULIA_CELL_TIMEOUT  seconds before run_cell_julia.py gives up on one cell's julia
#                        subprocess (default: run_cell_julia.py's own, 3600s). Single-thread
#                        Julia at the full 20-step/127-qubit depth is slower per gate than the
#                        Rust engine (decisions.md #10), and PP_WARM_REPEATS=3 means 1 cold + 3
#                        warm full propagations per cell by default -- 3600s is too tight for
#                        anything past the loosest eps and silently wastes the rest of the
#                        allocation looping over cells that will each hit the same timeout
#                        (job 7032188's real failure). Set this close to the wall-time budget
#                        actually left for cells (this job's --time minus ~5 min of build time),
#                        and/or set JULIA_WARM_REPEATS=0 or 1 and MIN_ABS_COEFF to a single,
#                        loosest value per job rather than looping over the whole grid.
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
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-julia-target-${SLURM_JOB_ID:-local}"
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

# --- matrix (mirrors campaign-genoa.sbatch's C2 grid for a 1:1 comparison) --
: "${MIN_ABS_COEFF:=1.5258789e-05 3.8146973e-06 9.5367432e-07}"   # 2^-16, 2^-18, 2^-20
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"                                # 7*pi/32
: "${REPS:=1}"
: "${JULIA_WARM_REPEATS:=}"
: "${JULIA_CELL_TIMEOUT:=}"
echo "min_abs_coeff='$MIN_ABS_COEFF' reps=$REPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-julia"
mkdir -p "$out_dir"

warm_repeats_flag=()
if [ -n "$JULIA_WARM_REPEATS" ]; then
  warm_repeats_flag=(--warm-repeats "$JULIA_WARM_REPEATS")
fi
timeout_flag=()
if [ -n "$JULIA_CELL_TIMEOUT" ]; then
  timeout_flag=(--timeout "$JULIA_CELL_TIMEOUT")
fi

cell_json="$CARGO_TARGET_DIR/cell.json"
for eps in $MIN_ABS_COEFF; do
  config_id="genoa-c2-julia-eps${eps}"
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
  "threads": 1,
  "partitions": null,
  "repetition_index": $rep,
  "task_id": "T02-canonical",
  "config_id": "$config_id"
}
EOF
    echo "== cell $config_id rep=$rep"
    python quera-talk-data/campaign-2026-09-11/jobs/run_cell_julia.py \
        "$cell_json" --out-dir "$out_dir" "${warm_repeats_flag[@]}" "${timeout_flag[@]}" \
        || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
  done
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

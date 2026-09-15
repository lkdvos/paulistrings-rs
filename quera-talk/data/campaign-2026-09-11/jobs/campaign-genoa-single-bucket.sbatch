#!/usr/bin/env bash
#SBATCH --job-name=quera-single-bucket
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11: one `bucketed_current`
# cell at the canonical 127-qubit task, `threads=1`, with `min_buckets`/
# `target_bucket_len` forced to the lowest achievable bucket count
# (`PauliSum.propagate`'s new `target_bucket_len=`/`min_buckets=` kwargs,
# `crates/paulistrings-py/src/sum.rs`).
#
# This is the missing third point for the reframed "baseline" slide
# (deck page 16): the *current* engine's core sort-merge algorithm at
# effectively no bucket-level parallelism, as a fairer "before this
# optimization work" reference than the historical Rust variants — paired
# with the two already-real Julia points (job 7033031 single-thread `dict`,
# job 7034021 96-thread `vector`) at the same (n_qubits, theta_h,
# trotter_steps, min_abs_coeff) config.
#
# `min_buckets=1` (not the documented `>= 16` contract) plus a
# `target_bucket_len` far above any term count this run reaches: verified
# locally that nothing in `desired_bits`/`Gf2Hash::new`/`refine`
# (`bucket/sum.rs:21-35`, `bucket/hash.rs`) enforces that floor in code — it
# is advisory, and `min_buckets=1` realizes a literal single bucket
# (`PauliSum.num_buckets == 1`) throughout the run, not just "minimal".
# The realized count is recorded in each run record's `extra.num_buckets`
# regardless, so the figure caption can cite what actually happened rather
# than the request.
#
# Reuses campaign-genoa.sbatch's worktree/venv/maturin preamble verbatim.
# Does NOT call sbatch anywhere — submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-single-bucket.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree
#                    (never the live checkout, per ab-campaign.sbatch's note on job 7004970).
#   MIN_ABS_COEFF   truncation threshold (default: 2^-16, matching the two Julia reference points).
#   TROTTER_STEPS   circuit depth (default: 20, matching the two Julia reference points).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
#   REPS            repetitions for this cell (default: 1 — this is a single reference point,
#                    not a statistics sweep).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors ab-campaign.sbatch / campaign-genoa.sbatch) ---------
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
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-single-bucket-target-${SLURM_JOB_ID:-local}"
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

# --- one cell ------------------------------------------------------------
: "${MIN_ABS_COEFF:=1.5258789e-05}"   # 2^-16, matching jobs 7033031/7034021
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"    # 7*pi/32
: "${REPS:=1}"
echo "min_abs_coeff=$MIN_ABS_COEFF trotter_steps=$TROTTER_STEPS theta_h=$THETA_H reps=$REPS"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-single-bucket"
mkdir -p "$out_dir"

cell_json="$CARGO_TARGET_DIR/cell.json"
config_id="genoa-single-bucket-eps${MIN_ABS_COEFF}"
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
  "threads": 1,
  "partitions": null,
  "repetition_index": $rep,
  "task_id": "T02-canonical",
  "config_id": "$config_id",
  "target_bucket_len": 1000000000,
  "min_buckets": 1
}
EOF
  echo "== cell $config_id rep=$rep"
  RAYON_NUM_THREADS=1 python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
      "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

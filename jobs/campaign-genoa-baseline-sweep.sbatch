#!/usr/bin/env bash
#SBATCH --job-name=quera-baseline-sweep
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Supplemental stage for the "baseline" asset (deck page 16): a real,
# multi-point wall-time-vs-tolerance sweep for naive_baseline, extending
# campaign-genoa-historical.sbatch's job 7033945, which deliberately ran
# naive_baseline at only its single loosest point (2^-12) to control cost
# (see that script's header comment: naive_baseline already costs ~130s
# locally for that one point, being the unbucketed/serial/non-truncation-aware
# historical variant).
#
# This script does NOT re-run that single point or any other variant --
# it only adds MORE points, all >= 2^-12 (i.e. looser/cheaper than the one
# already-measured point), so the added wall-time is bounded by roughly what
# was already shown to be affordable, never venturing into the tighter
# cutoffs that made a full grid infeasible for this variant.
#
# Writes to its own raw/ subdirectory; the baseline figure script merges this
# with job 7033945's naive_baseline point and the existing bucketed_current /
# external-library reference points, per figures/real/MANIFEST.md's "baseline"
# entry.
#
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-baseline-sweep.sbatch
#
# Environment knobs (all optional):
#   REPS   repetitions per cutoff (default: 1, mirrors campaign-genoa-historical.sbatch).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors campaign-genoa-historical.sbatch) -------------------
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

# Extended loose-only grid for naive_baseline: 2^-4 through 2^-12, all
# looser than or equal to the one already-measured, known-affordable point
# (2^-12, job 7033945). Never adds a tighter cutoff than that point.
LOOSE_SWEEP='[6.25e-02, 3.125e-02, 1.5625e-02, 7.8125e-03, 3.90625e-03, 1.953125e-03, 9.765625e-04, 4.8828125e-04, 2.44140625e-04]'

FULL_SCALE_N_QUBITS=127
FULL_SCALE_TROTTER_STEPS=10

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-baseline-sweep"
mkdir -p "$out_dir"

hist_scratch="${TMPDIR:-/tmp}/quera-campaign-baseline-sweep-${SLURM_JOB_ID:-local}"
mkdir -p "$hist_scratch"
cell_json="$hist_scratch/cell.json"

for rep in $(seq 0 $((REPS - 1))); do
  cat > "$cell_json" <<EOF
{
  "variant_id": "naive_baseline",
  "n_qubits": $FULL_SCALE_N_QUBITS,
  "trotter_steps": $FULL_SCALE_TROTTER_STEPS,
  "min_abs_coeff": $LOOSE_SWEEP,
  "direction": "heisenberg",
  "repetition_index": $rep,
  "config_id": "genoa-baseline-loose-sweep"
}
EOF
  echo "== naive_baseline loose sweep rep=$rep grid=$LOOSE_SWEEP"
  "$PYTHON_BIN" quera-talk-data/campaign-2026-09-11/jobs/run_cell_historical.py \
      "$cell_json" --out-dir "$out_dir" --scratch-root "$hist_scratch" \
      || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

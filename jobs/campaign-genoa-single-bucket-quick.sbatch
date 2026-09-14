#!/usr/bin/env bash
#SBATCH --job-name=quera-single-bucket-quick
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Quick companion to campaign-genoa-single-bucket.sbatch (job 7036526, still
# running as of this writing): the same canonical circuit and single-thread
# setup, but at a much LOOSER tolerance (2^-8, not 2^-16) so both cells --
# default bucket config AND min_buckets=1 -- finish fast, giving an early,
# cheap yes/no signal on whether bucket count matters at all, while the slow
# eps=2^-16 pair (this job's real target for the deck) keeps running.
#
# NOT a replacement for the eps=2^-16 point -- that's still needed for a fair
# comparison against the two real Julia points at that same eps. This is a
# fast, independent, same-circuit check of the underlying bucket-count claim.
#
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-single-bucket-quick.sbatch
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts).
#   MIN_ABS_COEFF   truncation threshold (default: 2^-8 -- deliberately loose/cheap,
#                    NOT the campaign's 2^-16 headline point).
#   TROTTER_STEPS   circuit depth (default: 20, matching the real single-bucket job).
#   THETA_H         kick angle (default: 7pi/32, contract.md primary point).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors campaign-genoa-single-bucket.sbatch) ----------------
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
source scripts/jcc-rustflags.sh
export RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }${JCC_RUSTFLAGS}"

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
lscpu | grep -E 'Thread\(s\) per core'
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-single-bucket-quick-target-${SLURM_JOB_ID:-local}"
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

: "${MIN_ABS_COEFF:=3.90625e-03}"   # 2^-8, deliberately loose/cheap
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"  # 7*pi/32
echo "min_abs_coeff=$MIN_ABS_COEFF trotter_steps=$TROTTER_STEPS theta_h=$THETA_H"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
out_dir="quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-single-bucket-quick"
mkdir -p "$out_dir"

cell_json="$CARGO_TARGET_DIR/cell.json"

for label_bucket in "default:0:0" "onebucket:1000000000:1"; do
  IFS=':' read -r label target_bucket_len min_buckets <<< "$label_bucket"
  bucket_json_fields=""
  if [ "$label" = "onebucket" ]; then
    bucket_json_fields="\"target_bucket_len\": $target_bucket_len, \"min_buckets\": $min_buckets,"
  fi
  config_id="genoa-single-bucket-quick-${label}-eps${MIN_ABS_COEFF}"
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
  "repetition_index": 0,
  "task_id": "T02-canonical",
  "config_id": "$config_id",
  $bucket_json_fields
  "n_qubits": 127
}
EOF
  echo "== cell $config_id ($label)"
  RAYON_NUM_THREADS=1 python quera-talk-data/campaign-2026-09-11/jobs/run_cell.py \
      "$cell_json" --out-dir "$out_dir" || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"

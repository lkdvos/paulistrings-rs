#!/usr/bin/env bash
#SBATCH --job-name=quera-bucketsize-dryrun
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Cheap dry run for the topN-based bucket-size sweep (see decisions.md's
# scoping entry): ONE config (target_bucket_len=1024, the library default),
# --reps 3 (not the full 20), --truncation topn:1000000. Purpose is NOT to
# produce a usable bucket-size point -- 3 reps is very unlikely to even reach
# the 1,000,000-term cap (small-scale tfim_step data suggested heavyhex_step
# might cross it around rep 3-6, but that estimate has real, stated
# uncertainty and may be off by 2x+) -- purpose is to get a REAL 127-qubit
# timing anchor, since the small-scale (16-24 qubit tfim_step) extrapolation
# used to estimate the full sweep's cost was flagged as unreliable (per-
# channel overhead at 127q/W=2/271-channels doesn't extrapolate cleanly from
# a much smaller lattice).
#
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-bucketsize-dryrun.sbatch
#
# Environment knobs (all optional):
#   REPS   Trotter steps for this dry run (default: 3 -- deliberately small).
#   TOPN   topn:<N> truncation cap (default: 1000000, per the scoping entry's
#           recommendation: N must clear min_buckets*target_bucket_len =
#           128*4096 ~ 524288 with margin for the sweep's largest grid point).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}

module purge
module load modules/2.4-20250724
# shellcheck disable=SC1091
source scripts/jcc-rustflags.sh
export RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }${JCC_RUSTFLAGS}"

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
lscpu | grep -E 'Thread\(s\) per core'
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-bucketsize-dryrun-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null' EXIT
cd "$src"

: "${REPS:=3}"
: "${TOPN:=1000000}"
echo "reps=$REPS topn=$TOPN target_bucket_len=1024 (library default)"

cargo run --release --features phase-timing --example phase_breakdown -- \
    --layers heavyhex_step --qubits 127 --threads 1 --reps "$REPS" \
    --truncation "topn:$TOPN" --initial z0 --min-buckets 128 \
    --target-bucket-len 1024 --occupancy-at $((REPS - 1)) --format table

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: printed above (dry run, no raw/ harvest -- this is a cost probe, not a campaign point)"

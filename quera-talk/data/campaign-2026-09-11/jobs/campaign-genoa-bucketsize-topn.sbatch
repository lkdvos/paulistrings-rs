#!/usr/bin/env bash
#SBATCH --job-name=quera-bucketsize-topn
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Re-sweep of campaign-genoa-bucketsize.sbatch's page-30 data (bucket-size
# vs. throughput/occupancy), fixing a real problem the user identified in
# the original coeff:2^-12 sweep: at that tolerance the final term count was
# only ~770-900, far too few for target_bucket_len in {256..4096} to produce
# meaningfully different bucket occupancy -- most buckets ended up near-empty
# regardless of the target.
#
# Fix: `--truncation topn:1000000` caps the sum at a FIXED 1,000,000-term
# population (TopN, exact selection) instead of a coefficient tolerance, so
# every target_bucket_len value gets the same, much larger, realistic
# population to actually partition. 1,000,000 was chosen so even the
# largest target_bucket_len (4096) clears the min_buckets=128 floor with
# margin (128*4096 ~= 524,288 terms needed just for that point to be
# target-driven rather than floor-driven) -- see decisions.md's scoping
# entry for the full derivation.
#
# Real cost anchors (NOT extrapolated from a smaller/different lattice, both
# real 127-qubit dry runs, see decisions.md): --reps 3 reached n=10,545 in
# 10.7ms; --reps 7 reached the exact 1,000,000-term cap in 12,987.8ms
# (13.0s), with per-rep steady-state cost ~= n/throughput ~= 1,000,000 terms
# * 271 gates / 6.94e7 strings/s ~= 3.9s/rep once capped. Extrapolated full
# 20-rep single-config cost: ~13s (ramp-up) + 13 steady-state reps * ~3.9s
# ~= 65s. Five configs: on the order of 5-10 minutes total, not the
# hours-scale estimate the original (smaller-lattice) scoping pass gave.
#
# Same grow-only-bucket-count caveat as campaign-genoa-bucketsize.sbatch:
# each target_bucket_len value gets its own fresh `cargo run` invocation via
# scripts/bench-campaign.sh's `probe:` item, never a shared/reused process.
#
# Same THETA_H caveat as campaign-genoa-bucketsize.sbatch / campaign-genoa-
# memory.sbatch: phase_breakdown's heavyhex_step hard-codes THETA_H=5*pi/16,
# not contract.md's primary 7*pi/32 -- caption accordingly.
#
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-bucketsize-topn.sbatch
#
# Environment knobs (all optional):
#   PS_REV              commit to build (default: HEAD when the job starts).
#   TARGET_BUCKET_LENS   space-separated --target-bucket-len values (default: "256 512 1024 2048 4096").
#   MIN_BUCKETS          fixed --min-buckets floor for every cell (default: 128).
#   TOPN                 topn:<N> truncation cap (default: 1000000).
#   TROTTER_STEPS        headline depth (default: 20, contract.md's primary point).
#   OCCUPANCY_AT         0-based rep index to sample bucket occupancy at (default: TROTTER_STEPS - 1).
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

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
lscpu | grep -E 'Thread\(s\) per core'
# Real L2/L3 cache size for THIS node, not a substituted general Zen4 spec --
# research/HARDWARE.md has no recorded L2 size for genoa, and past job logs
# only ever captured a filtered lscpu subset. lscpu -C is the authoritative,
# node-local source; /sys is a fallback if -C is unsupported on this util-linux.
echo "== cache sizes (real, this node)"
lscpu -C 2>&1 || true
for f in /sys/devices/system/cpu/cpu0/cache/index*/{level,type,size}; do
    [ -f "$f" ] && echo "$f: $(cat "$f")"
done
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-bucketsize-topn-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null' EXIT
cd "$src"

: "${TARGET_BUCKET_LENS:=256 512 1024 2048 4096}"
: "${MIN_BUCKETS:=128}"
: "${TOPN:=1000000}"
: "${TROTTER_STEPS:=20}"
: "${OCCUPANCY_AT:=$((TROTTER_STEPS - 1))}"
echo "target_bucket_lens='$TARGET_BUCKET_LENS' min_buckets=$MIN_BUCKETS topn=$TOPN trotter_steps=$TROTTER_STEPS occupancy_at=$OCCUPANCY_AT"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
# bench-campaign.sh hardcodes its own out_dir as benchmarks/results/<date>-<host>
# (job 7035746's real bug when a prior version of this script assumed otherwise).
bc_out_dir="benchmarks/results/${date_tag}-${node}"
mkdir -p "$bc_out_dir"

for target_len in $TARGET_BUCKET_LENS; do
    campaign_name="bucketsize-topn-tbl${target_len}"
    echo "== scripts/bench-campaign.sh $campaign_name (heavyhex_step, threads=1, target_bucket_len=$target_len)"

    scripts/bench-campaign.sh "$campaign_name" \
        "probe:--layers heavyhex_step --qubits 127 --threads 1 --reps $TROTTER_STEPS \
--truncation topn:$TOPN --initial z0 --min-buckets $MIN_BUCKETS \
--target-bucket-len $target_len --occupancy-at $OCCUPANCY_AT --format table" \
        || echo "  ($campaign_name campaign exited nonzero -- see $bc_out_dir/$campaign_name.txt for what did complete)"
done

out_dir="$repo/quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-bucketsize-topn"
mkdir -p "$out_dir"
cp -r "$bc_out_dir"/. "$out_dir"/
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/ (bucketsize-topn-tbl*.txt/.json/-probe.json/-report.html)"

#!/usr/bin/env bash
#SBATCH --job-name=quera-bucketsize
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11: bucket-size vs.
# throughput/occupancy slide data. Sweeps `--target-bucket-len` over
# {256, 512, 1024, 2048, 4096} on the canonical 127-qubit heavy-hex
# kicked-Ising Trotter step (phase_breakdown's `heavyhex_step` layer,
# --features phase-timing), single-thread, `min_buckets` fixed at 128 and
# truncation fixed at the loosest campaign cutoff (2^-12) -- the cheapest
# point that still exercises real truncation (evidence.md's E9 cost
# estimate: eps=2^-12 ~= 15 min single-thread at this scale; the tighter
# 2^-16 point has separately been measured at ~9h single-thread and is not
# used here). 20 Trotter steps, matching contract.md's primary depth.
#
# `PauliSum::rebucket` is grow-only (bucket/sum.rs:776-781), so the bucket
# count under one `--target-bucket-len` is a high-water mark a warm process
# could carry into the next value -- each of the 5 values below therefore
# gets its own fresh `cargo run` invocation via `scripts/bench-campaign.sh`'s
# `probe:` item (one campaign name per value), never a shared/reused process.
#
# Uses the new (this-session) `--occupancy-at` flag on phase_breakdown to
# sample the bucket-occupancy histogram once, at the same fixed late step
# across all 5 configs (default: the last Trotter step, index reps-1=19),
# so the resulting JSON sidecars carry `num_buckets`/`empty_buckets`/
# `occupancy_median`/`occupancy_p95`/`occupancy_max` alongside the usual
# phase-timing breakdown. See ARCHITECTURE.md and the phase_breakdown
# --help text for what each field means.
#
# KNOWN LIMITATION (same as campaign-genoa-memory.sbatch): phase_breakdown's
# `heavyhex_step` layer has THETA_ZZ = -pi/2 (matches contract.md) but a
# hard-coded THETA_H = 5*pi/16 baked into the example's source, not a CLI
# flag -- this job cannot be pointed at contract.md's primary 7*pi/32 kick
# angle. The lattice, x-then-zz layer order, per-term payload, and the
# truncation/bucket-size axes under test are otherwise the genuine canonical
# task; caption the resulting figure accordingly if this is ever reconciled
# against other campaign stages that do use 7*pi/32.
#
# Reuses the worktree-per-revision / shared-toolchain conventions of
# campaign-genoa.sbatch and campaign-genoa-memory.sbatch: this is a
# pure-Rust example, so there is no venv/maturin step, only the cargo
# worktree build. Does NOT call sbatch anywhere -- submission is the
# user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-bucketsize.sbatch
#
# Environment knobs (all optional):
#   PS_REV              commit to build (default: HEAD when the job starts), via a git worktree
#                        (never the live checkout, per ab-campaign.sbatch's note on job 7004970).
#   TARGET_BUCKET_LENS  space-separated --target-bucket-len values to sweep (default:
#                        "256 512 1024 2048 4096").
#   MIN_BUCKETS         fixed --min-buckets floor for every cell (default: 128).
#   MIN_ABS_COEFF       fixed truncation cutoff (default: 2.44140625e-4 = 2^-12, the loosest
#                        campaign rung -- do not tighten without re-checking the E9 cost
#                        estimate above; 2^-16 alone has been measured at ~9h single-thread
#                        at this scale).
#   TROTTER_STEPS       headline depth (default: 20, contract.md's primary point).
#   OCCUPANCY_AT        0-based rep index to sample bucket occupancy at (default:
#                        TROTTER_STEPS - 1, i.e. the last step). Must stay < TROTTER_STEPS.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

# --- toolchain (mirrors campaign-genoa.sbatch / campaign-genoa-memory.sbatch) ---------------
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
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# --- one-time verification: Slurm's ThreadsPerCore is NOT trusted ----------
# contract.md: "Slurm reports ThreadsPerCore=1 ... still must be verified
# with lscpu on an actual allocation before the first timed run." This job
# only ever runs --threads 1, so SMT exposure would not corrupt a
# multi-thread comparison, but it would still silently change which
# physical core the single thread lands on relative to other campaign
# stages -- checked and logged, not fatal here.
lscpu | grep -E 'Thread\(s\) per core'
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
echo "physical cores=$phys nproc=$(nproc)"

# --- worktree build (mirrors campaign-genoa.sbatch's / campaign-genoa-memory.sbatch's PS_REV pattern) --
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-bucketsize-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null' EXIT
cd "$src"

# scripts/bench-campaign.sh sources scripts/jcc-rustflags.sh itself and
# appends to RUSTFLAGS internally -- do not also export RUSTFLAGS here, see
# that file's "append, never assign" note (an exported RUSTFLAGS would
# replace, not add to, whatever the script builds internally).

# --- matrix ----------------------------------------------------------------
: "${TARGET_BUCKET_LENS:=256 512 1024 2048 4096}"
: "${MIN_BUCKETS:=128}"
: "${MIN_ABS_COEFF:=2.44140625e-4}"   # 2^-12, the loosest campaign cutoff
: "${TROTTER_STEPS:=20}"
: "${OCCUPANCY_AT:=$((TROTTER_STEPS - 1))}"
echo "target_bucket_lens='$TARGET_BUCKET_LENS' min_buckets=$MIN_BUCKETS min_abs_coeff=$MIN_ABS_COEFF trotter_steps=$TROTTER_STEPS occupancy_at=$OCCUPANCY_AT"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
# scripts/bench-campaign.sh hardcodes its own out_dir as
# benchmarks/results/<date>-<host> (its line 118) -- it does NOT honor a
# caller-supplied directory, so bc_out_dir here must match that exactly, not
# a campaign-specific suffix (job 7035746's real bug: a mismatched suffix
# left this script copying an empty directory, and the actual results were
# then deleted by the worktree-removal EXIT trap below before being noticed).
bc_out_dir="benchmarks/results/${date_tag}-${node}"
mkdir -p "$bc_out_dir"

for target_len in $TARGET_BUCKET_LENS; do
    campaign_name="bucketsize-tbl${target_len}"
    echo "== scripts/bench-campaign.sh $campaign_name (heavyhex_step, threads=1, target_bucket_len=$target_len)"

    scripts/bench-campaign.sh "$campaign_name" \
        "probe:--layers heavyhex_step --qubits 127 --threads 1 --reps $TROTTER_STEPS \
--truncation coeff:$MIN_ABS_COEFF --initial z0 --min-buckets $MIN_BUCKETS \
--target-bucket-len $target_len --occupancy-at $OCCUPANCY_AT --format table" \
        || echo "  ($campaign_name campaign exited nonzero -- see $bc_out_dir/$campaign_name.txt for what did complete)"
done

# --- deliver into this campaign's raw/ layout, per house convention --------
out_dir="$repo/quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-bucketsize"
mkdir -p "$out_dir"
cp -r "$bc_out_dir"/. "$out_dir"/
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/ (bucketsize-tbl*.txt/.json/-probe.json/-report.html)"

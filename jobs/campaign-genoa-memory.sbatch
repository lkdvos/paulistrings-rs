#!/usr/bin/env bash
#SBATCH --job-name=quera-c-memory
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Single-node genoa stage for campaign-2026-09-11's E3 (memory diagnosis,
# evidence.md rows 13-14, logged "not started" -- no task before this one
# drove crates/membench / scripts/bandwidth.sh for this campaign's host).
#
# Produces the raw ingredients for the "memory" slide (page 17): a sustained
# memory-bandwidth ceiling on this exact hardware class (scripts/bandwidth.sh)
# and a real per-phase/per-layer timing+traffic breakdown of the canonical
# 127-qubit heavy-hex kicked-Ising Trotter step (the phase_breakdown probe,
# --features phase-timing), both fed through the repo's existing
# scripts/bench-campaign.sh + scripts/perf-viz.py loop (benchmarks/PROFILING.md)
# rather than a bespoke script -- this IS the canonical change/measure/compare
# tool, we are only pointing it at this campaign's task and copying its
# output into this campaign's raw/ directory.
#
# Reuses the worktree-per-revision / shared-toolchain / JCC-rustflags
# conventions of campaign-genoa.sbatch and scripts/slurm/ab-campaign.sbatch.
# Does NOT call sbatch anywhere -- submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-memory.sbatch
#
# KNOWN LIMITATION (read before trusting the theta_h column of the resulting
# report against contract.md): phase_breakdown's `heavyhex_step` layer has
# THETA_ZZ = -pi/2 (matches contract.md exactly) but a hard-coded
# THETA_H = 5*pi/16 baked into the example's source -- it is not a CLI flag,
# so this job CANNOT be pointed at contract.md's primary 7*pi/32 kick angle.
# The lattice (127-qubit Eagle r3 heavy-hex), the x-then-zz layer order, the
# per-term payload (48 B/term at W=2), and the truncation/threading axes are
# otherwise the genuine canonical task. Term-count growth (and therefore
# `m`, bucket count, and wall time) will differ somewhat from a real
# 7*pi/32 run; the phase-share structure and the bytes/term traffic model are
# not expected to change in kind. Do not caption the resulting figure as
# "the campaign's theta_h=7pi/32 point" -- caption it as "a same-lattice,
# same-cutoff-family Trotter step, secondary kick angle (5pi/16)" and note
# the mismatch explicitly if this ever needs reconciling with E9's `theta_h`
# convergence numbers.
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a git worktree
#                    (never the live checkout, per ab-campaign.sbatch's note on job 7004970).
#   MIN_ABS_COEFF   space-separated dyadic grid (default: one point, 2^-16 = 1.5258789e-05,
#                    the loosest rung of campaign-genoa.sbatch's own C2 grid) -- this is a
#                    diagnosis job, not a sweep, so it defaults to the cheapest cell that still
#                    exercises real truncation. Widen only if the report's traffic model needs
#                    a second cutoff point to sanity-check trend.
#   TROTTER_STEPS   headline depth (default: 20, contract.md's primary point).
#   THREADS         space-separated total thread counts for phase_breakdown (default:
#                    "1 <physical cores>"). Physical cores only: this job asserts
#                    `nproc == physical core count` after module load and aborts if that
#                    disagrees, since the campaign's hard no-SMT constraint (contract.md,
#                    "Slurm reports ThreadsPerCore=1 ... still must be verified with lscpu on
#                    an actual allocation") is a correctness requirement here, not advisory --
#                    an inflated thread count would silently corrupt the bandwidth-ceiling
#                    comparison (research/HARDWARE.md's ceilings are all physical-core counts).
#   RUN_PERF_STAT   1 (default) or 0. Best-effort hardware-counter pass (scripts/perf-stat.sh)
#                    at the all-core cell, on top of the counter-free timing/traffic evidence
#                    the job always collects. `perf stat` needs kernel access this campaign's
#                    account may not have on a shared cluster node (perf_event_paranoid); a
#                    failure here is non-fatal (bench-campaign.sh already logs-and-continues)
#                    and the report is still valid without it -- it just loses the IPC/LLC-miss
#                    corroboration research/HARDWARE.md's ccqlin038 numbers had. Set to 0 to
#                    skip the attempt outright on a node known not to allow it.
#                    IMPORTANT if it does run: scripts/perf-stat.sh's Pass B ("% of per-socket
#                    ceiling") hard-codes ccqlin038's own read/write ceilings (39.0/18.6 GB/s,
#                    Cascade Lake) as RD_CEIL_1S/WR_CEIL_1S -- that percentage column is WRONG
#                    on genoa and must be ignored; do not quote it or substitute it into the
#                    memory-diagnosis figure. Pass A (IPC, LLC-miss rate, cycles/string) and
#                    Pass C (retired NUMA loads, if the host's perf lists those events) are
#                    frequency/host-agnostic ratios and remain valid corroboration on genoa.
#                    The figure's actual bandwidth-vs-ceiling comparison should come from this
#                    job's own scripts/bandwidth.sh + phase_breakdown wall-time pairing
#                    (perf-viz.py's roofline model), which is measured fresh on genoa, not from
#                    perf-stat.sh's Pass B percentage.
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

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# --- one-time verification: Slurm's ThreadsPerCore is NOT trusted ----------
# contract.md: "Slurm reports ThreadsPerCore=1 ... still must be verified
# with lscpu on an actual allocation before the first timed run." This job's
# whole point is a bandwidth-ceiling comparison, so an undetected SMT sibling
# would silently invalidate every "% of ceiling" number in the report.
lscpu | grep -E 'Thread\(s\) per core'
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
nproc_val=$(nproc)
if [[ "$phys" -ne "$nproc_val" ]]; then
    echo "abort: nproc ($nproc_val) != physical core count ($phys) -- SMT appears exposed on this node, which violates this campaign's hard no-SMT constraint. Re-check the allocation before running this job." >&2
    exit 4
fi
echo "physical cores == nproc == $phys (no SMT, confirmed)"

# --- worktree build (mirrors ab-campaign.sbatch's / campaign-genoa.sbatch's PS_REV pattern) --
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/quera-campaign-memory-target-${SLURM_JOB_ID:-local}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null' EXIT
cd "$src"

# scripts/bandwidth.sh and scripts/bench-campaign.sh both source
# scripts/jcc-rustflags.sh themselves -- do not also export RUSTFLAGS here,
# see that file's "append, never assign" note (an exported RUSTFLAGS would
# replace, not add to, whatever either script builds internally).

# --- matrix --------------------------------------------------------------
: "${MIN_ABS_COEFF:=1.5258789e-05}"   # 2^-16, campaign-genoa.sbatch's loosest C2 point
: "${TROTTER_STEPS:=20}"
: "${THREADS:=1 $phys}"
: "${RUN_PERF_STAT:=1}"
echo "min_abs_coeff='$MIN_ABS_COEFF' trotter_steps=$TROTTER_STEPS threads='$THREADS' run_perf_stat=$RUN_PERF_STAT"

date_tag=$(date +%Y-%m-%d)
node=$(hostname -s)
# scripts/bench-campaign.sh's own output convention (not this campaign's raw/
# layout) -- computed the same way it computes it internally, so we can drop
# scripts/bandwidth.sh's redirected bandwidth.txt into the same directory
# before bench-campaign.sh's own perf-viz.py auto-render picks it up.
bc_out_dir="benchmarks/results/${date_tag}-${node}"
mkdir -p "$bc_out_dir"

# scripts/bandwidth.sh prints to stdout by design (its header comment: "the
# first stdout line ... lands at the top of bandwidth.txt when this is
# redirected") -- bench-campaign.sh's own `bandwidth` item does NOT do this
# redirection (it only tees combined output into the campaign .txt log), so
# it does not by itself produce a bandwidth.txt perf-viz.py can read. Do the
# redirection explicitly, once, before running any probe cells.
echo "== scripts/bandwidth.sh -> $bc_out_dir/bandwidth.txt"
scripts/bandwidth.sh >"$bc_out_dir/bandwidth.txt" 2>"$bc_out_dir/bandwidth.stderr.log" \
    || echo "  (bandwidth.sh exited nonzero -- see $bc_out_dir/bandwidth.stderr.log; the timing/traffic evidence below still stands on its own, see the sbatch header's neutral-findings note)"

threads_csv=$(echo "$THREADS" | tr ' ' ',')

for eps in $MIN_ABS_COEFF; do
    campaign_name="memory-diagnosis-eps${eps}"
    echo "== scripts/bench-campaign.sh $campaign_name (heavyhex_step, threads=$threads_csv, eps=$eps)"

    items=(
        "probe:--layers heavyhex_step --qubits 127 --threads $threads_csv --reps $TROTTER_STEPS --truncation coeff:$eps --initial z0 --format table"
    )
    if [[ "$RUN_PERF_STAT" == "1" ]]; then
        items+=("perf-stat:--layers heavyhex_step --qubits 127 --threads $phys --reps $TROTTER_STEPS --truncation coeff:$eps --initial z0")
    fi

    scripts/bench-campaign.sh "$campaign_name" "${items[@]}" \
        || echo "  ($campaign_name campaign exited nonzero -- see $bc_out_dir/$campaign_name.txt for what did complete)"
done

# --- deliver into this campaign's raw/ layout, per house convention --------
out_dir="$repo/quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${node}-memory"
mkdir -p "$out_dir"
cp -r "$bc_out_dir"/. "$out_dir"/
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/ (bandwidth.txt, memory-diagnosis-eps*.txt/.json/-probe.json/-report.html)"

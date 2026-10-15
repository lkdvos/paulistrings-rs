#!/usr/bin/env bash
#SBATCH --job-name=ps-ab
#SBATCH --partition=ccq
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Paired A/B cells on an exclusive node (see README.md); each distinct rev builds once, before the first cell.
#
# Environment knobs (all optional):
#   A_REV   git rev for side A. Unset (default): runtime-knob A/B, both sides PS_REV, side B adding
#           KNOB_B (default --partitions <numa> --partition-cpus <node lists>).
#           Set (e.g. e7de227): code A/B of A_REV against PS_REV with the same probe args.
#           Both sides run whatever PROBE_EXTRA adds, e.g. "--partitions 2" for a partitioned code A/B
#           (THREADS is then the total over the partitions).
#   LAYERS  probe layers, space separated (default "su4 rotation_local rotation_remote gu2q";
#           for a code A/B use e.g. "rotation_zz cnot su4")
#   NS      --n values (default "1000000")
#   THREADS thread counts, space separated (default: the node's physical cores and half of them;
#           under MPI=1, the CPUs of one NUMA domain, per rank). Must be divisible by P.
#   REPS    --reps (default 8; rotation cells at m ~ 1e6 want >= 40, see PROFILING.md)
#   PAIRS   --pairs (default 5), ORDER --order (default abba)
#   PROBE_EXTRA  extra probe args for both sides (e.g. "--truncation coeff:1e-4")
#   KNOB_B  side B's extra args in a runtime-knob A/B (e.g. "--truncation atopn:100000")
#   MPI=1   one rank per NUMA domain over the allocation (scripts/slurm/ranks.sh), the probe under
#           `srun` with --mpi, each run's per-rank rows merged by ab-compare.sh --launch; needs A_REV or KNOB_B.
#           Outside Slurm the launcher is `mpirun -n 2`; LAUNCH overrides it either way.
#   MEMBENCH=1  also run crates/membench (1 core node-local, all cores) for this node's ceilings
#   PS_REV  commit to build (default: HEAD when the job starts); pass PS_REV=$(git rev-parse HEAD)
#           at submit time to pin it there.
#   SMOKE=1 the pre-submit check, run locally as `SMOKE=1 bash scripts/slurm/ab-campaign.sbatch`:
#           presets LAYERS=cnot NS=2000 THREADS=2 REPS=2 PAIRS=1.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh

if [ "${SMOKE:-0}" = 1 ]; then
    : "${LAYERS:=cnot}" "${NS:=2000}" "${THREADS:=2}" "${REPS:=2}" "${PAIRS:=1}"
fi
features=phase-timing
mpi_args=""
launch=""
if [ "${MPI:-0}" = 1 ]; then
    [ -n "${A_REV:-}${KNOB_B:-}" ] || { echo "MPI=1 needs A_REV (a code A/B) or KNOB_B (a knob A/B)" >&2; exit 2; }
    if command -v module >/dev/null 2>&1; then
        module purge
        module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
    fi
    export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"
    features=phase-timing,mpi
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        source scripts/slurm/ranks.sh
        launch="srun --ntasks=$ranks --ntasks-per-node=$per_node --cpus-per-task=$cpus_per_numa --cpu-bind=ldoms --mpi=pmix"
        : "${THREADS:=$cpus_per_numa}"
    else
        launch="mpirun -n 2"
    fi
    launch=${LAUNCH:-$launch}
    mpi_args="--mpi"
fi

use_shared_rust
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-local}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
command -v numactl >/dev/null && numactl -H | grep -E 'available|cpus|distances' -A0
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# NUMA domains -> P and the --partition-cpus list "a-b,c-d;e-f,..."
mapfile -t node_lists < <(for n in /sys/devices/system/node/node*/cpulist; do cat "$n"; done)
P=${#node_lists[@]}
PART_CPUS=$(IFS=';'; echo "${node_lists[*]}")
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
: "${THREADS:=$phys $((phys / 2))}"
: "${LAYERS:=su4 rotation_local rotation_remote gu2q}"
: "${NS:=1000000}"
: "${REPS:=8}"
: "${PAIRS:=5}"
: "${ORDER:=abba}"
echo "P=$P partition-cpus='$PART_CPUS' physical cores=$phys threads='$THREADS'${launch:+ launcher='$launch'}"

if [ "${MEMBENCH:-0}" = 1 ]; then
    echo "== membench ceilings (STREAM convention, best of 5)"
    cargo build --offline --release -p membench >/dev/null
    numactl --cpunodebind=0 --membind=0 "$CARGO_TARGET_DIR/release/membench" --threads 1 || true
    "$CARGO_TARGET_DIR/release/membench" --threads "$phys" || true
fi

# build_probe <rev>: builds <rev>'s probe in its own checkout and sets `built` to a copy of the binary.
build_probe() {
    local sha dir
    sha=$(git rev-parse --verify "$1^{commit}")
    echo "== building $sha ($(git log -1 --format=%s "$sha"))"
    dir="$CARGO_TARGET_DIR/src-$sha"
    checkout_rev "$sha" "$dir"
    (cd "$dir" && cargo build --offline --locked --release --features "$features" -p paulistrings --example phase_breakdown)
    built="$CARGO_TARGET_DIR/bin/phase_breakdown-$sha"
    mkdir -p "$CARGO_TARGET_DIR/bin"
    cp "$CARGO_TARGET_DIR/release/examples/phase_breakdown" "$built"
}

PS_REV=${PS_REV:-$(git rev-parse HEAD)}
build_probe "$PS_REV"
bin_b=$built
bin_a=$bin_b
if [ -n "${A_REV:-}" ]; then
    build_probe "$A_REV"
    bin_a=$built
fi

for layer in $LAYERS; do
  for n in $NS; do
    for t in $THREADS; do
      name="slurm-${SLURM_JOB_ID:-local}-${layer}-n${n}-t${t}"
      probe="--n $n --qubits 128 --layers $layer --threads $t --reps $REPS $mpi_args ${PROBE_EXTRA:-}"
      common=(--bin-a "$bin_a" --bin-b "$bin_b" --pairs "$PAIRS" --order "$ORDER" ${launch:+--launch "$launch"})
      if [ -n "${A_REV:-}" ]; then
        echo "== code A/B $A_REV vs $PS_REV: $probe"
        scripts/ab-compare.sh "$name" --a "$A_REV" --b "$PS_REV" --probe "$probe" "${common[@]}"
      else
        if [ -n "${KNOB_B:-}" ]; then
          knob=$KNOB_B
        else
          if (( t % P != 0 )); then echo "skip threads=$t: not divisible by P=$P"; continue; fi
          knob="--partitions $P --partition-cpus '$PART_CPUS'"
        fi
        echo "== knob A/B: $probe | B adds: $knob"
        scripts/ab-compare.sh "$name" --a "$PS_REV" --b "$PS_REV" --probe "$probe" \
            --probe-b "$probe $knob" "${common[@]}"
      fi
    done
  done
done
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: benchmarks/results/$(date +%Y-%m-%d)-$(hostname -s)/slurm-${SLURM_JOB_ID:-local}-*"

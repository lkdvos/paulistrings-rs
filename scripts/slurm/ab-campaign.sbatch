#!/usr/bin/env bash
#SBATCH --job-name=ps-ab
#SBATCH --partition=scc
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Paired A/B cells for the partitioned engine on an exclusive node (see README.md).
#
# Environment knobs (all optional):
#   A_REV   git rev for side A. Unset (default): runtime-knob A/B, both sides the
#           working tree, side B adds --partitions <numa> --partition-cpus <node lists>.
#           Set (e.g. e7de227): code A/B of the untouched path, both sides P=1.
#   LAYERS  probe layers, space separated (default "su4 rotation_local rotation_remote gu2q";
#           for a code A/B use e.g. "rotation_zz cnot su4")
#   NS      --n values (default "1000000")
#   THREADS total thread counts, space separated (default: physical cores on the node;
#           also half of them, i.e. one socket's worth). Must be divisible by P.
#   REPS    --reps (default 8; rotation cells at m ~ 1e6 want >= 40, see PROFILING.md)
#   PAIRS   --pairs (default 5), ORDER --order (default abba)
#   MEMBENCH=1  also run crates/membench (1 core node-local, all cores) for this node's ceilings
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
unset RUST_LOG
export RAYON_NUM_THREADS=   # the probe sizes its own pools; keep the env clean
unset RAYON_NUM_THREADS

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
command -v numactl >/dev/null && numactl -H | grep -E 'available|cpus|distances' -A0
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# NUMA domains -> P and the --partition-cpus list "a-b,c-d;e-f,..."
mapfile -t node_lists < <(for n in /sys/devices/system/node/node*/cpulist; do cat "$n"; done)
P=${#node_lists[@]}
PART_CPUS=$(IFS=';'; echo "${node_lists[*]}")
phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
half=$((phys / 2))
: "${THREADS:=$phys $half}"
: "${LAYERS:=su4 rotation_local rotation_remote gu2q}"
: "${NS:=1000000}"
: "${REPS:=8}"
: "${PAIRS:=5}"
: "${ORDER:=abba}"
echo "P=$P partition-cpus='$PART_CPUS' physical cores=$phys threads='$THREADS'"

if [ "${MEMBENCH:-0}" = 1 ]; then
    echo "== membench ceilings (STREAM convention, best of 5)"
    cargo build --offline --release -p membench >/dev/null
    first_node_cpu=${node_lists[0]%%[-,]*}
    numactl --cpunodebind=0 --membind=0 "$CARGO_TARGET_DIR/release/membench" --threads 1 || true
    "$CARGO_TARGET_DIR/release/membench" --threads "$phys" || true
fi

date_tag=$(date +%Y-%m-%d)
for layer in $LAYERS; do
  for n in $NS; do
    for t in $THREADS; do
      name="slurm-${layer}-n${n}-t${t}"
      probe="--n $n --qubits 128 --layers $layer --threads $t --reps $REPS"
      if [ -n "${A_REV:-}" ]; then
        echo "== code A/B $A_REV vs . : $probe"
        scripts/ab-compare.sh "$name" --a "$A_REV" --b . --probe "$probe" \
            --pairs "$PAIRS" --order "$ORDER"
      else
        if (( t % P != 0 )); then echo "skip threads=$t: not divisible by P=$P"; continue; fi
        echo "== knob A/B P=1 vs P=$P : $probe"
        scripts/ab-compare.sh "$name" --a . --b . --probe "$probe" \
            --probe-b "$probe --partitions $P --partition-cpus '$PART_CPUS'" \
            --pairs "$PAIRS" --order "$ORDER"
      fi
    done
  done
done
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: benchmarks/results/${date_tag}-$(hostname -s)/"

#!/usr/bin/env bash
# Measure the host's memory-bandwidth ceiling with crates/membench across the
# placement matrix, and cross-check against the uncore memory controllers.
#
# Run once per host (or after hardware changes); record the headline triad
# numbers in a research note (see benchmarks/PROFILING.md). The placement
# matrix (BANDWIDTH_RUNS) and the roofline ceiling labels (CEILING_MAP) come
# from scripts/host-topology.sh, keyed on this host's `hostname -s`.
set -euo pipefail

# The JCC-erratum padding is deliberately not in .cargo/config.toml — it is a
# ~1% tax on every part without the erratum, which is all of `ccq`. Measurement
# builds opt in, so the reference host cannot silently lose 9-13% mid-campaign.
# Append: an exported RUSTFLAGS replaces the config's rustflags wholesale.
# See scripts/jcc-rustflags.sh and research/HARDWARE.md.
. "$(dirname "$0")/jcc-rustflags.sh"
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$JCC_RUSTFLAGS"

cd "$(dirname "$0")/.."
source scripts/host-topology.sh

# --device N appends a `=== gpu<N> <name> ===` section measured on that CUDA device (membench's `cuda` feature).
DEVICE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE=$2; shift 2 ;;
        *) echo "bandwidth.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

MIB=${MIB:-512}
REPS=${REPS:-5}
BIN=${CARGO_TARGET_DIR:-target}/release/membench

if [[ -n $DEVICE ]]; then
    cargo build --release --offline -p membench --features cuda >/dev/null
else
    cargo build --release --offline -p membench >/dev/null
fi

run() { # run <label> <threads> [prefix...]
    local label=$1 threads=$2
    shift 2
    echo "=== $label ==="
    if [[ $# -gt 0 ]]; then
        "$@" "$BIN" --threads "$threads" --mib "$MIB" --reps "$REPS"
    else
        "$BIN" --threads "$threads" --mib "$MIB" --reps "$REPS"
    fi
}

# First stdout line: the ceiling-map perf-viz.py parses to label roofline
# ceilings, so it lands at the top of bandwidth.txt when this is redirected.
echo "# ceiling-map: ${CEILING_MAP}"

echo "memory-bandwidth ceiling — $(hostname -f)"
echo "date: $(date +%F)"
echo "commit: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "mib=$MIB reps=$REPS  (STREAM-convention nominal bytes; plain stores, RFO uncorrected)"
echo

for entry in "${BANDWIDTH_RUNS[@]}"; do
    IFS='|' read -r label threads prefix <<<"$entry"
    # shellcheck disable=SC2206  # intentional word-splitting of a
    # host-topology-defined, space-separated prefix command into argv.
    prefix_arr=($prefix)
    run "$label" "$threads" "${prefix_arr[@]}"
done

echo "=== uncore cross-check (node0, 8 physical, read+triad) ==="
# System-wide counters on a shared box: short window, and the alloc/first-touch
# traffic is inside the window too — read the per-socket split, not absolutes.
perf stat -a --per-socket -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ \
    numactl --cpunodebind=0 --membind=0 \
    "$BIN" --threads 8 --mib "$MIB" --reps "$REPS" --kernels read,triad 2>&1 \
    || echo "uncore cross-check unavailable on this host (perf exit $?)"

if [[ -n $DEVICE ]]; then
    echo
    echo "=== gpu$DEVICE $(nvidia-smi --query-gpu=name --format=csv,noheader -i "$DEVICE" 2>/dev/null || echo unknown) ==="
    echo "clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader -i "$DEVICE" 2>/dev/null || echo unknown)"
    "$BIN" --device "$DEVICE" --mib "$MIB" --reps "$REPS"
fi

echo
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"

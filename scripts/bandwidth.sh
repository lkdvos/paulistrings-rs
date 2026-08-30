#!/usr/bin/env bash
# Measure the host's memory-bandwidth ceiling with crates/membench across the
# placement matrix, and cross-check against the uncore memory controllers.
#
# Run once per host (or after hardware changes); record the headline triad
# numbers in a research note (see benchmarks/PROFILING.md). Placement masks
# below are for ccqlin038's topology: 2 sockets, node0 phys CPUs 0-7
# (HT 16-23), node1 phys 8-15 (HT 24-31). Adjust for other hosts.
set -euo pipefail
cd "$(dirname "$0")/.."

MIB=${MIB:-512}
REPS=${REPS:-5}
BIN=target/release/membench

cargo build --release --offline -p membench >/dev/null

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

echo "memory-bandwidth ceiling — $(hostname -f)"
echo "date: $(date +%F)"
echo "commit: $(git rev-parse HEAD)$(git diff --quiet || echo ' (dirty)')"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "mib=$MIB reps=$REPS  (STREAM-convention nominal bytes; plain stores, RFO uncorrected)"
echo

run "1 core, node0 local"    1  numactl --cpunodebind=0 --membind=0
run "1 core, node0 cpu / node1 mem (remote)" 1 numactl --cpunodebind=0 --membind=1
run "node0, 8 physical"      8  numactl --cpunodebind=0 --membind=0
run "node1, 8 physical"      8  numactl --cpunodebind=1 --membind=1
run "node0, 8 phys + 8 HT"   16 taskset -c 0-7,16-23 numactl --membind=0
run "both sockets, 16 physical" 16 taskset -c 0-15
run "both sockets, 16 phys, interleaved pages" 16 numactl --interleave=all --physcpubind=0-15
run "both sockets, 32 threads"  32

echo "=== uncore cross-check (node0, 8 physical, read+triad) ==="
# System-wide counters on a shared box: short window, and the alloc/first-touch
# traffic is inside the window too — read the per-socket split, not absolutes.
perf stat -a --per-socket -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ \
    numactl --cpunodebind=0 --membind=0 \
    "$BIN" --threads 8 --mib "$MIB" --reps "$REPS" --kernels read,triad 2>&1

echo
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"

#!/usr/bin/env bash
#SBATCH --job-name=ps-scaling
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=05:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Intra-node strong scaling on one exclusive node (see README.md): one process per (cell, thread count, placement), each pinned with numactl.
#   compact     the first T CPUs of NUMA node 0, memory bound to it; past one node, node 1 continues and memory is first-touch
#   spread      T CPUs of node 0 dealt round-robin over its L3 domains (one per CCD on Genoa), memory bound to node 0
#   interleave  compact CPUs past one node with pages interleaved over every node
#   numa        the partitioned engine at P = <numa nodes> over the whole node with `cut` partition rows (FINDINGS.md §Cut partition rows), the reference for the single-pool rows past one node
# membench measures the same placements' read and copy ceilings.
# Read the results with scripts/scaling-report.py on each scaling-<job>-<placement>.jsonl (done at the end of the job).
#
# Environment knobs (all optional):
#   CELLS           ';'-separated "layer|probe args" (default: rotation_zz on a 3e7-term steady sum, and heavyhex_step grown to 2.7e7 terms)
#   THREADS         compact thread counts (default 1 2 4 8 16 24 32 40 48 64 96, capped at the node's CPUs)
#   SPREAD_THREADS  spread thread counts (default 6 8 16 24, capped at node 0's CPUs)
#   PS_REV          commit to build (default: HEAD when the job starts)
#   SMOKE=1         the pre-submit check, run locally as `SMOKE=1 bash scripts/slurm/thread-scaling.sbatch`: tiny cells at 1 and 2 threads
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh

if [ "${SMOKE:-0}" = 1 ]; then
    : "${CELLS:=rotation_zz|--n 20000 --reps 2;heavyhex_step|--truncation coeff:1.220703125e-4 --reps 1}"
    : "${THREADS:=1 2}" "${SPREAD_THREADS:=2}" "${MEMBENCH_MIB:=64}"
fi
: "${CELLS:=rotation_zz|--n 20000000 --reps 20;heavyhex_step|--truncation coeff:3.0517578125e-05 --reps 4}"
: "${THREADS:=1 2 4 8 16 24 32 40 48 64 96}"
: "${SPREAD_THREADS:=6 8 16 24}"
: "${MEMBENCH_MIB:=2048}"

use_shared_rust
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

job=${SLURM_JOB_ID:-local}
out=$(results_dir)
echo "== node $(hostname -s), job $job, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)|L3'
numactl -H | grep -E 'available|cpus|distances' -A0
echo "L3 domains: $(cat /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list | sort -un | tr '\n' ' ')"
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# cpus_for <compact|spread> <T>: a cpulist of T CPUs, or nothing when the node has too few; `cpus_for node0 0` prints node 0's CPU count.
cpus_for() {
    python3 - "$1" "$2" <<'EOF'
import glob, re, sys

placement, threads = sys.argv[1], int(sys.argv[2])

def expand(text):
    cpus = []
    for part in text.strip().split(","):
        if "-" in part:
            low, high = map(int, part.split("-"))
            cpus += range(low, high + 1)
        elif part:
            cpus.append(int(part))
    return cpus

paths = sorted(glob.glob("/sys/devices/system/node/node[0-9]*/cpulist"), key=lambda p: int(re.search(r"node(\d+)/", p)[1]))
nodes = [expand(open(p).read()) for p in paths]
if placement == "node0":
    print(len(nodes[0]))
    sys.exit()
if placement == "compact":
    cpus = [c for node in nodes for c in node]
else:
    domains = {}
    for c in nodes[0]:
        shared = open(f"/sys/devices/system/cpu/cpu{c}/cache/index3/shared_cpu_list").read().strip()
        domains.setdefault(shared, []).append(c)
    groups = list(domains.values())
    cpus = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
if threads <= len(cpus):
    print(",".join(map(str, cpus[:threads])))
EOF
}

node0_cpus=$(cpus_for node0 0)
numa_nodes=$(ls -d /sys/devices/system/node/node[0-9]* | wc -l)
total_cpus=$(nproc --all)
echo "node 0: $node0_cpus CPUs; $numa_nodes NUMA nodes; $total_cpus CPUs"

# numa_prefix <placement> <T>: the numactl prefix for this placement, or nothing to skip the cell.
numa_prefix() {
    local placement=$1 t=$2 cpus
    cpus=$(cpus_for "$([ "$placement" = spread ] && echo spread || echo compact)" "$t")
    [ -n "$cpus" ] || return 0
    case "$placement" in
        compact | spread) (( t <= node0_cpus )) && echo "numactl --physcpubind=$cpus --membind=0" \
                              || { [ "$placement" = compact ] && echo "numactl --physcpubind=$cpus"; } ;;
        interleave) (( t > node0_cpus )) && echo "numactl --physcpubind=$cpus --interleave=all" ;;
    esac
    return 0
}

PS_REV=${PS_REV:-$(git rev-parse HEAD)}
dir="$CARGO_TARGET_DIR/src-$PS_REV"
echo "== building $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
checkout_rev "$PS_REV" "$dir"
(cd "$dir" && cargo build --offline --locked --release --features phase-timing -p paulistrings --example phase_breakdown \
    && cargo build --offline --locked --release -p membench)
probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"
membench="$CARGO_TARGET_DIR/release/membench"

echo "== membench ceilings (STREAM convention, best of 5, ${MEMBENCH_MIB} MiB arrays)"
for placement in compact spread interleave; do
    list=$THREADS
    [ "$placement" = spread ] && list=$SPREAD_THREADS
    for t in $list; do
        prefix=$(numa_prefix "$placement" "$t")
        [ -n "$prefix" ] || continue
        echo "placement=$placement $($prefix "$membench" --threads "$t" --mib "$MEMBENCH_MIB" --reps 5 --kernels read,copy | tr '\n' ' ')"
    done
done | tee "$out/membench-$job.txt"

IFS=';' read -ra cells <<<"$CELLS"
for cell in "${cells[@]}"; do
    layer=${cell%%|*}
    args=${cell#*|}
    for placement in compact spread interleave; do
        list=$THREADS
        [ "$placement" = spread ] && list=$SPREAD_THREADS
        for t in $list; do
            prefix=$(numa_prefix "$placement" "$t")
            [ -n "$prefix" ] || continue
            echo "== $placement T=$t: $layer $args | $prefix"
            $prefix "$probe" --layers "$layer" --threads "$t" $args \
                --json-out "$out/scaling-$job-$placement.jsonl" | { grep -E "^cell" || true; }
        done
    done
    if (( numa_nodes > 1 && total_cpus % numa_nodes == 0 )); then
        echo "== numa P=$numa_nodes T=$total_cpus: $layer $args"
        "$probe" --layers "$layer" --threads "$total_cpus" --partitions "$numa_nodes" --partition-rows cut $args \
            --json-out "$out/scaling-$job-numa.jsonl" | { grep -E "^cell" || true; }
    fi
done

for placement in compact spread interleave numa; do
    [ -s "$out/scaling-$job-$placement.jsonl" ] || continue
    echo "== scaling report: $placement"
    python3 scripts/scaling-report.py "$out/scaling-$job-$placement.jsonl"
done
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out/{membench,scaling}-$job*"

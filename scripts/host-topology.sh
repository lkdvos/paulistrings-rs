#!/usr/bin/env bash
# Sourced, never executed (after the caller's `set -euo pipefail` and `cd` to the repo root): host-specific CPU placement, keyed on `hostname -s`.
#   PLACEMENT_PREFIX  placement name -> command prefix (bench-campaign.sh's `scaling:` items)
#   BANDWIDTH_RUNS    "label|threads|prefix" entries, one per bandwidth.sh run, in order
#   CEILING_MAP       "key=label;..." roofline labels; perf-viz.py parses this exact format
# A partitioned cell (P > 1) runs under no placement prefix: benchmarks/PROFILING.md §Threading.
# Each host gets its own case arm; never generalize an existing one.

declare -gA PLACEMENT_PREFIX
declare -ga BANDWIDTH_RUNS
declare -g CEILING_MAP

_host_topology_host=$(hostname -s)

case "$_host_topology_host" in
  ccqlin038)
    # 2 sockets, node0 physical CPUs 0-7 (HT 16-23), node1 physical CPUs
    # 8-15 (HT 24-31). See benchmarks/PROFILING.md's Threading section.
    PLACEMENT_PREFIX=(
      [default]=""
      [node0]="numactl --cpunodebind=0 --membind=0"
      [phys16]="taskset -c 0-15"
      [smt16]="taskset -c 0-7,16-23 numactl --membind=0"
      [phys8]="taskset -c 0-7 numactl --membind=0"
    )

    BANDWIDTH_RUNS=(
      "1 core, node0 local|1|numactl --cpunodebind=0 --membind=0"
      "1 core, node0 cpu / node1 mem (remote)|1|numactl --cpunodebind=0 --membind=1"
      "node0, 8 physical|8|numactl --cpunodebind=0 --membind=0"
      "node1, 8 physical|8|numactl --cpunodebind=1 --membind=1"
      "node0, 8 phys + 8 HT|16|taskset -c 0-7,16-23 numactl --membind=0"
      "both sockets, 16 physical|16|taskset -c 0-15"
      "both sockets, 16 phys, interleaved pages|16|numactl --interleave=all --physcpubind=0-15"
      "both sockets, 32 threads|32|"
    )

    CEILING_MAP='1=1 core, node0 local;8=node0, 8 physical;16=both sockets, 16 physical;default=both sockets, 32 threads'
    ;;

  *)
    # Unknown host: no calibrated placement masks, so PLACEMENT_PREFIX has
    # only the no-op "default" entry -- any other placement name will abort
    # with the "unknown placement" error in the caller (e.g.
    # bench-campaign.sh's scaling: item). bandwidth.sh still gets something
    # useful: a 1-thread and an nproc-thread run, both unplaced.
    PLACEMENT_PREFIX=(
      [default]=""
    )

    BANDWIDTH_RUNS=(
      "1 thread|1|"
      "$(nproc) threads|$(nproc)|"
    )

    CEILING_MAP="default=$(nproc) threads, no calibrated placement"

    echo "warning: scripts/host-topology.sh has no entry for host '${_host_topology_host}'" \
      "-- using an uncalibrated default (1 thread / nproc threads, no NUMA placement)." \
      "Add a case arm for this host in scripts/host-topology.sh." >&2
    ;;
esac

unset _host_topology_host

#!/usr/bin/env bash
# Single source of truth for host-specific CPU placement.
#
# This file is meant to be *sourced*, not executed: `source
# scripts/host-topology.sh` from another script after that script has done
# its own `set -euo pipefail` and `cd`. It defines, keyed on `hostname -s`:
#
#   PLACEMENT_PREFIX  -- associative array, placement name -> command prefix
#                        string (used by bench-campaign.sh's `scaling:`
#                        items and by anything else that wants to run a
#                        command under one of the named placements).
#   BANDWIDTH_RUNS     -- array of "label|threads|prefix" entries, one per
#                        scripts/bandwidth.sh invocation, in run order.
#   CEILING_MAP        -- semicolon-separated "key=label" string consumed by
#                        perf-viz.py to label roofline ceilings. Pinned
#                        format -- do not reformat without updating the
#                        parser in perf-viz.py.
#
# Add a new host by adding a case arm below with its own topology; do not
# edit the ccqlin038 arm to "generalize" it -- each host gets its own arm.
#
# Never invokes any Slurm command.

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

#!/usr/bin/env bash
# A dated measurement campaign: one `=== <item> ===` section per item in benchmarks/results/<date>-<host>/<name>.txt, then the perf-viz.py report (see --help).
# A failing item is logged and the campaign continues; a malformed item aborts.
# `probe:` and `perf-stat:` arguments go through `eval` into an argv array, so quoting inside them survives; the items come from this script's own command line only.
# Never invokes any Slurm command.

set -euo pipefail

. "$(dirname "$0")/jcc-rustflags.sh"   # research/FINDINGS.md §JCC branch padding, opt-in

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: scripts/bench-campaign.sh <campaign-name> <item>...

Items (parsed on the first ':'):
  criterion:<filter>
      cargo bench --offline -p paulistrings --bench pauli_ops -- "<filter>",
      then merges a snapshot of $CARGO_TARGET_DIR/criterion (default target/)
      into <name>.json via scripts/criterion-report.py snapshot --merge.

  probe:<args>
      cargo run --offline --release --features phase-timing -p paulistrings \
        --example phase_breakdown -- <args> --json-out <name>.probe.jsonl
      Pass <args> as one shell word; quoting inside it is honoured, e.g.
      probe:'--layers cnot,su4 --n 1000000' or probe:'--partitions 1,2 --n 1000000'.

  perf-stat:<args>
      scripts/perf-stat.sh <args> (split like probe:<args>).

  scaling:<placement>
      cargo bench ... -- thread_scaling_bucketed under the command prefix
      PLACEMENT_PREFIX[<placement>] from scripts/host-topology.sh, snapshot
      into <name>-scaling-<placement>.json. An unknown placement aborts with
      the valid list; an uncalibrated host has only `default`.

  bandwidth
      scripts/bandwidth.sh.

Output: benchmarks/results/<date>-<host>/<name>.txt (a rerun appends behind a
divider), plus <name>-report.html from scripts/perf-viz.py.

Never invokes any Slurm command.
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

source scripts/host-topology.sh

campaign_name=$1
shift
items=("$@")

out_dir="benchmarks/results/$(date +%F)-$(hostname -s)"
mkdir -p "$out_dir"
out_file="$out_dir/${campaign_name}.txt"
snapshot_json="$out_dir/${campaign_name}.json"

if [[ -f "$out_file" ]]; then
  {
    echo
    echo "############################################################"
    echo "# RERUN: $(date -Iseconds)"
    echo "############################################################"
    echo
  } >>"$out_file"
fi

hostname_full=$(hostname -f 2>/dev/null || hostname)
commit=$(git rev-parse HEAD)
if ! git diff --quiet || ! git diff --cached --quiet; then
  commit="$commit (dirty)"
fi
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')

{
  echo "${campaign_name} measurement campaign — ${hostname_full}"
  echo "date: $(date +%F)"
  echo "commit: ${commit}"
  rustc -V
  echo "threads (nproc): $(nproc)"
  echo "governor: ${governor}"
  echo "cpu: ${cpu_model}"
  echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo
} >>"$out_file"

for item in "${items[@]}"; do
  if [[ "$item" == *:* ]]; then
    item_type="${item%%:*}"
    item_rest="${item#*:}"
  else
    item_type="$item"
    item_rest=""
  fi

  echo "=== ${item} ===" >>"$out_file"

  case "$item_type" in
    criterion)
      filter="$item_rest"
      if ! (cargo bench --offline -p paulistrings --bench pauli_ops -- "$filter") \
        2>&1 | tee -a "$out_file"; then
        echo "  (criterion bench exited nonzero for filter '${filter}' — continuing campaign)" \
          | tee -a "$out_file"
      fi
      if ! (python3 scripts/criterion-report.py snapshot "$snapshot_json" --merge --filter "$filter") \
        2>&1 | tee -a "$out_file"; then
        echo "  (criterion-report.py snapshot failed — continuing campaign)" \
          | tee -a "$out_file"
      fi
      ;;

    probe)
      args_arr=()
      eval "args_arr=( ${item_rest} )"
      if ! (cargo run --offline --release --features phase-timing -p paulistrings \
        --example phase_breakdown -- "${args_arr[@]}" \
        --json-out "$out_dir/${campaign_name}.probe.jsonl") 2>&1 | tee -a "$out_file"; then
        echo "  (phase_breakdown probe exited nonzero — continuing campaign)" \
          | tee -a "$out_file"
      fi
      ;;

    perf-stat)
      args_arr=()
      eval "args_arr=( ${item_rest} )"
      if ! (scripts/perf-stat.sh "${args_arr[@]}") 2>&1 | tee -a "$out_file"; then
        echo "  (perf-stat.sh exited nonzero — continuing campaign)" \
          | tee -a "$out_file"
      fi
      ;;

    scaling)
      placement="$item_rest"
      if [[ -z "${PLACEMENT_PREFIX[$placement]+set}" ]]; then
        echo "error: unknown scaling placement '${placement}' (expected one of: ${!PLACEMENT_PREFIX[*]})" >&2
        exit 1
      fi
      # shellcheck disable=SC2206  # intentional word-splitting of a
      # host-topology-defined, space-separated prefix command into argv.
      prefix=(${PLACEMENT_PREFIX[$placement]})
      if ! ("${prefix[@]}" cargo bench --offline -p paulistrings \
        --bench pauli_ops -- thread_scaling_bucketed) 2>&1 | tee -a "$out_file"; then
        echo "  (scaling run exited nonzero for placement '${placement}' — continuing campaign)" \
          | tee -a "$out_file"
      fi
      # Each placement overwrites target/criterion in place, so snapshot it
      # per placement — otherwise only the last placement survives as JSON.
      if ! (python3 scripts/criterion-report.py snapshot \
        "$out_dir/${campaign_name}-scaling-${placement}.json" \
        --filter thread_scaling_bucketed) 2>&1 | tee -a "$out_file"; then
        echo "  (criterion-report.py snapshot failed — continuing campaign)" \
          | tee -a "$out_file"
      fi
      ;;

    bandwidth)
      if ! (scripts/bandwidth.sh) 2>&1 | tee -a "$out_file"; then
        echo "  (bandwidth.sh exited nonzero — continuing campaign)" | tee -a "$out_file"
      fi
      ;;

    *)
      echo "error: unknown item type '${item_type}' in item '${item}'" >&2
      echo "       run with --help for the item menu" >&2
      exit 1
      ;;
  esac
done

{
  echo
  echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
} >>"$out_file"

if python3 scripts/perf-viz.py "$out_dir/$campaign_name"; then
  echo "report: $out_dir/${campaign_name}-report.html"
else
  echo "  (perf-viz.py failed — campaign data is intact, re-run it by hand)" >&2
fi

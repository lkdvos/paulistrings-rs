#!/usr/bin/env bash
# Reusable measurement-campaign harness for paulistrings-rs.
#
# Usage: scripts/bench-campaign.sh <campaign-name> <item>...
#
# Writes benchmarks/results/<date>-<hostname>/<campaign-name>.txt: a
# provenance header (commit, rustc version, thread count, governor, cpu,
# load) followed by one "=== <item> ===" section per item, each containing
# the invoked tool's combined stdout+stderr. Never touches Slurm.
#
# Design notes (see --help for the item menu):
#   - A failing item is logged and the campaign continues; the script only
#     aborts outright on a malformed item (unknown type, unknown scaling
#     placement) since that's a caller mistake, not a flaky measurement.
#   - Re-running the same campaign name on the same day/host appends to the
#     existing file behind a rerun divider, rather than overwriting it.
#   - `criterion:<filter>` re-snapshots target/criterion into
#     <campaign-name>.json via `criterion-report.py snapshot --merge` after
#     every such item, scoped with the same --filter. --merge means running
#     more than one criterion: item in one campaign accumulates into the
#     same JSON (each item's entries win for their own keys) instead of the
#     later snapshot overwriting the earlier one.
#   - `probe:<args>` and `perf-stat:<args>` are split into an argv array via
#     `eval "args_arr=( ${item_rest} )"` rather than plain unquoted word
#     splitting, so a quoted arg (e.g. probe:'--layers "a,b"') survives
#     intact. This is a deliberate `eval` on a string that only ever comes
#     from this script's own trusted argv (the campaign items an operator
#     types on the command line, not external/untrusted input) -- treat it
#     the same as any other shell-quoting trust boundary and don't route
#     unreviewed input through it.

set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: scripts/bench-campaign.sh <campaign-name> <item>...

Items (parsed on the first ':'):
  criterion:<filter>
      cargo bench --offline -p paulistrings --bench pauli_ops -- "<filter>"
      then snapshots target/criterion into
      benchmarks/results/<date>-<host>/<campaign-name>.json via
      scripts/criterion-report.py snapshot --merge --filter "<filter>"
      (--merge accumulates across multiple criterion: items in one
      campaign instead of the later one overwriting the earlier).

  probe:<args>
      cargo run --offline --release --features phase-timing \
        --example phase_breakdown -- <args>
      <args> is split into an argv array via `eval "args_arr=( ${args} )"`,
      so ordinary shell quoting inside <args> is honored -- pass it as a
      single shell word, e.g. probe:"--width 2 --n 1000000" or
      probe:'--layers "a,b" --n 1000000'.

  perf-stat:<args>
      scripts/perf-stat.sh <args>  (split into argv the same way as
      probe:<args>)
      Prints a skip message instead of failing if the script doesn't exist
      yet.

  scaling:<placement>
      Runs `cargo bench --offline -p paulistrings --bench pauli_ops -- \
      thread_scaling_bucketed` under a CPU-placement prefix looked up from
      PLACEMENT_PREFIX in scripts/host-topology.sh (keyed on this host's
      `hostname -s`). <placement> must be a key in that map -- an unknown
      placement aborts with the valid list. On the reference host
      (ccqlin038) the keys are:
        default  -- no prefix
        node0    -- numactl --cpunodebind=0 --membind=0
        phys16   -- taskset -c 0-15
        smt16    -- taskset -c 0-7,16-23 numactl --membind=0
        phys8    -- taskset -c 0-7 numactl --membind=0
      An unrecognized host only has `default` -- add an entry to
      scripts/host-topology.sh to calibrate placements for it.

  macro
      Times `cargo run --offline --release -p paulistrings --example
      ising_2d_quench`, then runs `git diff --exit-code` against the two
      committed CSV outputs it writes
      (crates/paulistrings/examples/output/ising_{4x4,6x6}.csv, per
      output_dir()/write_csv() in crates/paulistrings/examples/ising_2d_quench.rs)
      and logs "determinism gate: PASS" or "FAIL".

  bandwidth
      scripts/bandwidth.sh, if present; otherwise a skip message.

Output: benchmarks/results/$(date +%F)-$(hostname -s)/<campaign-name>.txt
(appended, with a rerun divider, if it already exists).

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
      # Split item_rest into an argv array via eval rather than unquoted
      # word-splitting, so quoted args survive. item_rest only ever comes
      # from this script's own argv (an operator-typed campaign item), so
      # this eval is over trusted input -- see the header comment.
      args_arr=()
      eval "args_arr=( ${item_rest} )"
      # Every probe cell also lands as a JSON line in the campaign's probe
      # sidecar, which perf-viz.py renders at the end of the run.
      if ! (cargo run --offline --release --features phase-timing \
        --example phase_breakdown -- "${args_arr[@]}" \
        --json-out "$out_dir/${campaign_name}-probe.json") 2>&1 | tee -a "$out_file"; then
        echo "  (phase_breakdown probe exited nonzero — continuing campaign)" \
          | tee -a "$out_file"
      fi
      ;;

    perf-stat)
      # See the probe: item above -- same eval-into-array rationale.
      args_arr=()
      eval "args_arr=( ${item_rest} )"
      if [[ ! -x scripts/perf-stat.sh ]]; then
        echo "  skip: scripts/perf-stat.sh not present yet" | tee -a "$out_file"
      else
        if ! (scripts/perf-stat.sh "${args_arr[@]}") 2>&1 | tee -a "$out_file"; then
          echo "  (perf-stat.sh exited nonzero — continuing campaign)" \
            | tee -a "$out_file"
        fi
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

    macro)
      if ! { time cargo run --offline --release -p paulistrings --example ising_2d_quench; } \
        2>&1 | tee -a "$out_file"; then
        echo "  (ising_2d_quench run exited nonzero — continuing campaign)" \
          | tee -a "$out_file"
      fi
      csv_4x4="crates/paulistrings/examples/output/ising_4x4.csv"
      csv_6x6="crates/paulistrings/examples/output/ising_6x6.csv"
      {
        if git diff --exit-code -- "$csv_4x4" "$csv_6x6"; then
          echo "determinism gate: PASS"
        else
          echo "determinism gate: FAIL"
        fi
      } 2>&1 | tee -a "$out_file"
      ;;

    bandwidth)
      if [[ ! -x scripts/bandwidth.sh ]]; then
        echo "  skip: scripts/bandwidth.sh not present yet" | tee -a "$out_file"
      else
        if ! (scripts/bandwidth.sh) 2>&1 | tee -a "$out_file"; then
          echo "  (bandwidth.sh exited nonzero — continuing campaign)" | tee -a "$out_file"
        fi
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

# Render the campaign's HTML report from whatever JSON/txt this run (and
# earlier runs of the same campaign name) left in $out_dir.
if [[ -x scripts/perf-viz.py || -f scripts/perf-viz.py ]]; then
  if python3 scripts/perf-viz.py "$out_dir/$campaign_name"; then
    echo "report: $out_dir/${campaign_name}-report.html"
  else
    echo "  (perf-viz.py failed — campaign data is intact, re-run it by hand)" >&2
  fi
fi

#!/usr/bin/env bash
# Hardware-counter capture for the phase_breakdown probe: cycles/string, IPC,
# LLC behavior (per-process), and DRAM bandwidth (system-wide uncore IMC).
#
# Usage: scripts/perf-stat.sh [probe args...]
#   e.g. scripts/perf-stat.sh --n 1000000 --threads 32 --layers rotation_zz
#
# PROBE=/path/to/binary skips the `cargo build` step and uses that prebuilt
# executable instead (must already exist and be executable) -- e.g. for an
# interleaved A/B comparison between two prebuilt binaries.
#
# Best used with a single (layer × threads) cell per invocation — cycles are
# whole-process, so with several cells the cycles/string figure is a blend
# (the script warns). Cycles-per-string is frequency-robust, which matters on
# this powersave-governed host.
#
# Caveats printed into the output: the uncore pass is unavoidably system-wide
# (-a) on a shared box; an idle baseline taken immediately before is
# subtracted, but treat the GB/s figure as approximate when the load average
# is non-trivial.
set -euo pipefail
cd "$(dirname "$0")/.."

# A caller-supplied prebuilt probe (PROBE=/path/to/binary, already built and
# executable) skips the cargo build step entirely -- useful for the
# interleaved A/B protocol in benchmarks/PROFILING.md, where the binaries
# under comparison are built once up front and must not be rebuilt (or
# overwritten in place) between runs.
_probe_prebuilt=0
if [[ -n "${PROBE:-}" && -x "${PROBE:-}" ]]; then
    _probe_prebuilt=1
fi
PROBE=${PROBE:-target/release/examples/phase_breakdown}
IDLE_SECS=${IDLE_SECS:-3}

if [[ "$_probe_prebuilt" == "1" ]]; then
    echo "perf-stat: using caller-supplied PROBE=$PROBE (skipping cargo build)"
else
    cargo build --release --offline --features phase-timing -p paulistrings \
        --example phase_breakdown >/dev/null
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "perf-stat: $PROBE $*"
echo "host: $(hostname -f)  date: $(date +%F)  commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
echo "load: $(cut -d' ' -f1-3 /proc/loadavg)"
echo

# ---- pass A: per-process core counters ------------------------------------
perf stat -x, -o "$tmp/core.csv" \
    -e duration_time,cycles,instructions,LLC-loads,LLC-load-misses,branches,branch-misses \
    -- "$PROBE" "$@" | tee "$tmp/probe.out"

echo
echo "--- pass A: per-process counters ---"
grep -v '^#' "$tmp/core.csv" | grep -v '^$' || true

# Work items from the probe's cell contract: `cell ... n=<N> layers=<L> ...`.
strings_processed=$(awk '/^cell /{for(i=1;i<=NF;i++){if($i~/^n=/)n=substr($i,3); if($i~/^layers=/)l=substr($i,8)}; s+=n*l} END{printf "%.0f", s}' "$tmp/probe.out")
cells=$(grep -c '^cell ' "$tmp/probe.out" || true)

read_counter() { # read_counter <event-name> <csv>
    awk -F, -v ev="$1" '$3==ev && $1+0==$1 {print $1; exit}' "$2"
}
cycles=$(read_counter cycles "$tmp/core.csv")
instr=$(read_counter instructions "$tmp/core.csv")
llc_loads=$(read_counter LLC-loads "$tmp/core.csv")
llc_miss=$(read_counter LLC-load-misses "$tmp/core.csv")

echo
echo "--- derived (pass A) ---"
if [[ -n "${cycles:-}" && -n "${instr:-}" ]]; then
    awk -v c="$cycles" -v i="$instr" 'BEGIN{printf "IPC: %.3f\n", i/c}'
fi
if [[ -n "${llc_loads:-}" && -n "${llc_miss:-}" && "$llc_loads" != "0" ]]; then
    awk -v l="$llc_loads" -v m="$llc_miss" 'BEGIN{printf "LLC load miss rate: %.2f%%\n", 100*m/l}'
fi
if [[ -n "${cycles:-}" && -n "$strings_processed" && "$strings_processed" != "0" ]]; then
    [[ "$cells" -gt 1 ]] && echo "note: $cells cells in one run — cycles/string is a blend across them"
    awk -v c="$cycles" -v s="$strings_processed" \
        'BEGIN{printf "cycles per input string: %.1f  (over %.0f string-layers)\n", c/s, s}'
    echo "note: cycles are whole-process (input generation + warm-up included);"
    echo "      the figure converges from above as --n and --reps grow"
else
    echo "cycles/string unavailable (no 'cell ... n= layers=' lines found in probe output)"
fi

# ---- pass B: system-wide DRAM traffic (uncore IMC) -------------------------
# perf pre-scales cas_count_* to MiB (the CSV unit column says so) — do NOT
# multiply by 64 again. Idle baseline first, subtracted as a rate.
echo
echo "--- pass B: DRAM bandwidth (system-wide; approximate on a shared box) ---"
perf stat -a -x, -o "$tmp/idle.csv" \
    -e duration_time,uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ \
    -- sleep "$IDLE_SECS" 2>/dev/null
perf stat -a -x, -o "$tmp/imc.csv" --per-socket \
    -e duration_time,uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ \
    -- "$PROBE" "$@" >/dev/null

grep -v '^#' "$tmp/imc.csv" | grep -v '^$' || true
echo

awk -F, '
    function num(x) { return x + 0 }
    # idle file: value,unit,event,... ; imc file (--per-socket): socket,ncpus,value,unit,event,...
    FNR == NR {
        if ($3 == "duration_time") idle_ns = num($1)
        if ($2 == "MiB" && $3 ~ /cas_count/) idle_mib += num($1)
        next
    }
    $5 == "duration_time" { run_ns = num($3) }
    $4 == "MiB" && $5 ~ /cas_count_read/  { rd[$1] += num($3); total += num($3) }
    $4 == "MiB" && $5 ~ /cas_count_write/ { wr[$1] += num($3); total += num($3) }
    END {
        if (run_ns == 0 || idle_ns == 0) { print "bandwidth: duration_time missing, skipping"; exit }
        run_s = run_ns / 1e9
        idle_rate = idle_mib / (idle_ns / 1e9)          # MiB/s, whole box, both dirs
        for (s in rd) {
            printf "%s: read %.2f GB/s  write %.2f GB/s\n",
                s, rd[s] * 1.048576e-3 / run_s, wr[s] * 1.048576e-3 / run_s
        }
        printf "total DRAM traffic: %.2f GB/s  (idle baseline %.2f GB/s, already excluded below)\n",
            total * 1.048576e-3 / run_s, idle_rate * 1.048576e-3
        printf "attributable to run (total - idle): %.2f GB/s over %.2f s\n",
            (total / run_s - idle_rate) * 1.048576e-3, run_s
    }
' "$tmp/idle.csv" "$tmp/imc.csv"

echo
echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"

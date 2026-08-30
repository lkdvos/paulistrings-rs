#!/usr/bin/env bash
# perf-record + flamegraph wrapper for paulistrings-rs.
#
# Records a `perf.data` for a chosen target, converts it offline with
# `perf script report flamegraph` (d3-flame-graph template — no network
# access needed), and drops a standalone interactive HTML plus a sidecar
# `.meta.txt` describing exactly what was profiled.
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE_MODE=${PROFILE_MODE:-dwarf}
FREQ=${FREQ:-397}

usage() {
  cat <<'EOF'
Usage:
  scripts/profile.sh probe [probe args...]
      Profile the `phase_breakdown` example. Builds it first with:
        cargo build --offline --profile profiling --features phase-timing \
          -p paulistrings --example phase_breakdown
      then perf-records ./target/profiling/examples/phase_breakdown with the
      given args.

  scripts/profile.sh bench <criterion-filter> [seconds]
      Profile a criterion bench via --profile-time. Builds the bench binary
      first with `cargo bench --offline -p paulistrings --bench pauli_ops
      --no-run`, locates the freshest target/release/deps/pauli_ops-* binary,
      and perf-records it directly (not through cargo, for clean attribution)
      as: <bin> --bench --profile-time <seconds> "<criterion-filter>".
      seconds defaults to 10.

  scripts/profile.sh bin <path> [args...]
      Profile an arbitrary prebuilt binary at <path> with the given args.
      No build step.

Env knobs:
  PROFILE_MODE=dwarf|fp   Stack-walking method (default: dwarf).
                          dwarf: perf record --call-graph dwarf,16384 — works
                          against the normal profiling-profile build (which
                          has line-tables-only debug info) and resolves
                          LTO-inlined frames.
                          fp: perf record --call-graph fp. For `probe`/`bench`
                          this exports RUSTFLAGS="-Cforce-frame-pointers=yes"
                          before the cargo build/bench step, which forces a
                          full rebuild (frame-pointer codegen differs from the
                          cached profiling/release artifacts). For `bin` it
                          only changes the perf record flag — you are
                          responsible for having built the binary with frame
                          pointers.
  FREQ=<hz>               Sampling frequency for `perf record -F` (default: 397).

Output:
  benchmarks/results/<date>-<host>/flamegraph-<name>-<shortcommit>[-dirty].html
  plus a sidecar <same>.meta.txt (host, date, full commit, rustc -V, mode,
  freq, exact command line profiled). The output path is printed at the end.

Caveats:
  - criterion --profile-time loops the routine without statistical sampling;
    ignore criterion:: and setup frames in the graph.
  - rayon idle spinning shows up as crossbeam_epoch::* frames.
EOF
}

if [[ $# -eq 0 || $1 == "-h" || $1 == "--help" ]]; then
  usage
  exit 0
fi

MODE=$1
shift

case $MODE in
  probe | bench | bin) ;;
  *)
    echo "error: unknown mode '$MODE'" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ $PROFILE_MODE != "dwarf" && $PROFILE_MODE != "fp" ]]; then
  echo "error: PROFILE_MODE must be 'dwarf' or 'fp', got '$PROFILE_MODE'" >&2
  exit 1
fi

case $PROFILE_MODE in
  dwarf) CALL_GRAPH="dwarf,16384" ;;
  fp) CALL_GRAPH="fp" ;;
esac

sanitize() {
  # Turn a criterion filter / arbitrary arg into a filesystem-safe name
  # fragment: '/' and ' ' become '-'.
  local s=$1
  s=${s//\//-}
  s=${s// /-}
  printf '%s' "$s"
}

# --- resolve mode-specific binary, args, and output name -------------------

BIN=""
ARGS=()
NAME=""

case $MODE in
  probe)
    if [[ $PROFILE_MODE == "fp" ]]; then
      export RUSTFLAGS="-Cforce-frame-pointers=yes"
    fi
    echo "==> building phase_breakdown (profiling profile, phase-timing feature)" >&2
    cargo build --offline --profile profiling --features phase-timing \
      -p paulistrings --example phase_breakdown
    BIN="./target/profiling/examples/phase_breakdown"
    ARGS=("$@")
    # Name by the distinguishing probe options (--layers/--threads), not the
    # first arg — several cells profiled back to back must not overwrite each
    # other's output.
    probe_tag="default"
    prev=""
    for a in "$@"; do
      case $prev in
        --layers) probe_tag=$a ;;
        --threads) probe_tag="${probe_tag}-t$a" ;;
      esac
      prev=$a
    done
    NAME="probe-$(sanitize "$probe_tag")"
    ;;

  bench)
    if [[ $# -lt 1 ]]; then
      echo "error: bench mode requires a criterion filter argument" >&2
      usage >&2
      exit 1
    fi
    FILTER=$1
    SECONDS_ARG=${2:-10}

    if [[ $PROFILE_MODE == "fp" ]]; then
      export RUSTFLAGS="-Cforce-frame-pointers=yes"
    fi
    echo "==> building pauli_ops bench (--no-run)" >&2
    cargo bench --offline -p paulistrings --bench pauli_ops --no-run

    BENCH_BIN=""
    for f in $(ls -t target/release/deps/pauli_ops-* 2>/dev/null); do
      [[ $f == *.d ]] && continue
      [[ -f $f && -x $f ]] || continue
      BENCH_BIN=$f
      break
    done
    if [[ -z $BENCH_BIN ]]; then
      echo "error: no pauli_ops bench executable found under target/release/deps/" >&2
      exit 1
    fi

    BIN=$BENCH_BIN
    ARGS=(--bench --profile-time "$SECONDS_ARG" "$FILTER")
    NAME="bench-$(sanitize "$FILTER")"
    ;;

  bin)
    if [[ $# -lt 1 ]]; then
      echo "error: bin mode requires a path to a binary" >&2
      usage >&2
      exit 1
    fi
    BIN=$1
    shift
    ARGS=("$@")
    if [[ ! -x $BIN ]]; then
      echo "error: '$BIN' does not exist or is not executable" >&2
      exit 1
    fi
    NAME="bin-$(sanitize "$(basename "$BIN")")"
    ;;
esac

# --- output paths ------------------------------------------------------

COMMIT=$(git rev-parse --short HEAD)
DIRTY_SUFFIX=""
if [[ -n $(git status --porcelain) ]]; then
  DIRTY_SUFFIX="-dirty"
fi

RESULTS_DIR="benchmarks/results/$(date +%F)-$(hostname -s)"
mkdir -p "$RESULTS_DIR"

OUT_BASE="flamegraph-${NAME}-${COMMIT}${DIRTY_SUFFIX}"
OUT_HTML="$RESULTS_DIR/${OUT_BASE}.html"
OUT_META="$RESULTS_DIR/${OUT_BASE}.meta.txt"

# Resolve to absolute paths before we cd into the scratch perf-data dir.
OUT_HTML_ABS=$(cd "$RESULTS_DIR" && pwd)/"${OUT_BASE}.html"

# --- record --------------------------------------------------------------

SCRATCH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/paulistrings-profile.XXXXXX")
trap 'rm -rf "$SCRATCH_DIR"' EXIT

PERF_DATA="$SCRATCH_DIR/perf.data"
PERF_STDERR="$SCRATCH_DIR/perf-record.stderr"

CMDLINE="$BIN${ARGS[*]:+ ${ARGS[*]}}"
echo "==> perf record (mode=$PROFILE_MODE, freq=$FREQ): $CMDLINE" >&2

set +e
perf record -F "$FREQ" --call-graph "$CALL_GRAPH" -o "$PERF_DATA" -- "$BIN" "${ARGS[@]}" \
  2> >(tee "$PERF_STDERR" >&2)
RECORD_STATUS=$?
set -e

if [[ $RECORD_STATUS -ne 0 ]]; then
  echo "error: perf record exited with status $RECORD_STATUS" >&2
  exit "$RECORD_STATUS"
fi

if grep -Eiq 'lost (chunks|samples)' "$PERF_STDERR"; then
  echo "warning: perf record reported lost chunks/samples — profile may be incomplete:" >&2
  grep -Ei 'lost (chunks|samples)' "$PERF_STDERR" >&2
fi

# --- convert to flamegraph -------------------------------------------------

# `perf script report flamegraph` only ever reads ./perf.data (its -i handling
# does not forward through the `report <script>` form), so convert from
# inside the scratch dir where we named the file exactly that.
echo "==> converting to flamegraph HTML" >&2
(cd "$SCRATCH_DIR" && perf script report flamegraph -- -o "$OUT_HTML_ABS")

# --- sidecar meta ----------------------------------------------------------

{
  echo "host: $(hostname -s)"
  echo "date: $(date -Iseconds)"
  echo "commit: $(git rev-parse HEAD)${DIRTY_SUFFIX}"
  echo "rustc: $(rustc -V)"
  echo "mode: $PROFILE_MODE"
  echo "freq: $FREQ"
  echo "command: $CMDLINE"
} > "$OUT_META"

echo "$OUT_HTML"

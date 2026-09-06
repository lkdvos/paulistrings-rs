#!/usr/bin/env bash
# Runs every figure script (F0-F7) against the real data under
# presentation/data/. Missing data files are reported and skipped rather than
# failing the whole run -- so an in-progress data collection still lets the
# figures that *are* ready get rebuilt -- but the script exits non-zero if
# anything was skipped, so CI/manual runs notice.
#
# Usage: presentation/plots/make_all.sh [--data-dir DIR]

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$SCRIPT_DIR/../data"
if [[ "${1:-}" == "--data-dir" ]]; then
    # Resolve relative to the caller's cwd (where this script was invoked
    # from), not the script's own directory, before the `cd` below.
    case "$2" in
        /*) DATA_DIR="$2" ;;
        *) DATA_DIR="$(pwd)/$2" ;;
    esac
fi

cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"
if [[ -x "../../.venv/bin/python" ]]; then
    PY="../../.venv/bin/python"
fi

skipped=()
failed=()

run() {
    local name="$1"
    shift
    echo "==> $name"
    if ! "$PY" "$@" --data-dir "$DATA_DIR"; then
        failed+=("$name")
    fi
}

require() {
    # Prints and records a skip if any of the given files (relative to
    # DATA_DIR) is missing; returns failure so the caller can skip the run.
    local name="$1"
    shift
    for f in "$@"; do
        if [[ ! -f "$DATA_DIR/$f" ]]; then
            echo "==> $name: SKIPPED (missing $DATA_DIR/$f)"
            skipped+=("$name")
            return 1
        fi
    done
    return 0
}

echo "==> fig0_term_growth"
if [[ -f "$DATA_DIR/term_growth.jsonl" ]]; then
    "$PY" fig0_term_growth.py || failed+=("fig0_term_growth")
else
    echo "==> fig0_term_growth: SKIPPED (missing $DATA_DIR/term_growth.jsonl)"
    skipped+=("fig0_term_growth")
fi

if require fig1_engine_ladder engine_ladder.jsonl; then
    run fig1_engine_ladder fig1_engine_ladder.py --all
fi

if require fig2_targetcpu targetcpu_default.jsonl targetcpu_native.jsonl; then
    run fig2_targetcpu fig2_targetcpu.py
fi

if require fig3_thread_scaling thread_scaling.jsonl; then
    run fig3_old_attempts fig3_thread_scaling.py --only threadmaps,mergesort
    run fig3_all fig3_thread_scaling.py
fi

if require fig4_superlinear thread_scaling.jsonl; then
    run fig4_superlinear fig4_superlinear.py --data thread_scaling.jsonl --out fig4_superlinear
fi
if require fig4_superlinear_large thread_scaling_large.jsonl; then
    run fig4_superlinear_large fig4_superlinear.py --data thread_scaling_large.jsonl --out fig4_superlinear_large
fi

if require fig5_bucket_sweep bucket_sweep.jsonl bucket_sweep_perf.jsonl; then
    run fig5_bucket_sweep fig5_bucket_sweep.py
fi

if require fig6_memory memory.jsonl; then
    run fig6_memory fig6_memory.py
fi

# F7 is optional: it looks for any file with a non-null layer_wall_ns and
# skips itself with a message if none exists. It never counts toward the
# skipped/failed totals below (and so never flips the exit code) since the
# task brief marks it optional.
echo "==> fig7_layer_profile"
if ! "$PY" fig7_layer_profile.py --data-dir "$DATA_DIR"; then
    echo "(fig7_layer_profile is optional; continuing)"
fi

echo
if [[ ${#skipped[@]} -gt 0 ]]; then
    echo "Skipped (missing data): ${skipped[*]}"
fi
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed: ${failed[*]}"
fi

if [[ ${#skipped[@]} -gt 0 || ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
echo "All figures rebuilt."

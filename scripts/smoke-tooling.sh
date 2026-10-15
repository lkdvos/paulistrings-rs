#!/usr/bin/env bash
# One pass over the measurement tooling at smoke size, to run after changing any of it (benchmarks/PROFILING.md §Tool one-liners).
# A step this host cannot run (no device, no MPI, no perf, no shared toolchain) is skipped with a message; any failure stops the script.
# A device step needs libnvrtc on LD_LIBRARY_PATH, an MPI step the MPI modules and LIBCLANG_PATH (CLAUDE.md §Commands).
# Never invokes any Slurm command: the Slurm template runs as plain `bash` under SMOKE=1.
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in
export IDLE_SECS=1 MIB=32 REPS=1
probe="${CARGO_TARGET_DIR:-target}/release/examples/phase_breakdown"
cell=(--n 2000 --threads 2 --layers cnot --reps 2)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
step() { echo; echo "=== smoke [${SECONDS}s]: $*"; }
skip() { echo; echo "=== smoke: skip $*"; }
build_probe() { cargo build --offline --release -p paulistrings --example phase_breakdown --features "$1"; }

step "criterion benches, one iteration each"
cargo bench --offline -p paulistrings --bench pauli_ops -- --test
step "probe: host and partitioned cells"
build_probe phase-timing
"$probe" "${cell[@]}" --json-out "$out/host.jsonl"
"$probe" "${cell[@]}" --partitions 2 --partition-cpus unpinned --json-out "$out/partitioned.jsonl"
step "ab-compare: a one-pair runtime-knob A/B"
scripts/ab-compare.sh smoke-tooling --a . --b . --probe "${cell[*]}" --probe-b "${cell[*]} --partitions 2" --pairs 1
step "bench-campaign: one item of each kind"
campaign="benchmarks/results/$(date +%F)-$(hostname -s)/smoke-tooling.txt"
rm -f "$campaign"   # a rerun appends, and an earlier run's failure must not fail this one
scripts/bench-campaign.sh smoke-tooling criterion:pauli_string_mul_assign/W=1 "probe:${cell[*]}" "perf-stat:${cell[*]}" bandwidth
if grep -E 'exited nonzero|snapshot failed|perf-viz.py failed' "$campaign"; then echo "bench-campaign logged a failure: $campaign" >&2; exit 1; fi
if command -v perf >/dev/null; then
    step "profile.sh bin"
    scripts/profile.sh bin "$probe" "${cell[@]}"
else skip "profile.sh: no perf"; fi
if [ -d "${SHARED_RUST:-$HOME/.local/rust-shared}/cargo" ]; then
    step "the Slurm A/B template under SMOKE=1"
    TMPDIR="$out" SMOKE=1 bash scripts/slurm/ab-campaign.sbatch
else skip "ab-campaign.sbatch: no shared toolchain (scripts/slurm/setup-shared-toolchain.sh)"; fi
if nvidia-smi -L >/dev/null 2>&1; then
    step "probe: device cells"
    build_probe phase-timing,cuda
    "$probe" "${cell[@]}" --device 0 --json-out "$out/device.jsonl"
    "$probe" "${cell[@]}" --device 0 --gpu-partitions 2 --json-out "$out/device.jsonl"
else skip "device cells: no CUDA device"; fi
if command -v mpirun >/dev/null && [ -n "${LIBCLANG_PATH:-}" ]; then
    step "probe under mpirun -n 2, a paired MPI A/B, and the mpi_ranks net"
    build_probe phase-timing,mpi
    mpirun -n 2 "$probe" "${cell[@]}" --mpi --json-out "$out/mpi.jsonl"
    scripts/ab-compare.sh smoke-tooling-mpi --a . --b . --features phase-timing,mpi --launch "mpirun -n 2" \
        --probe "${cell[*]} --mpi" --probe-b "${cell[*]} --mpi --truncation atopn:1000" --pairs 1
    scripts/mpi-test.sh --ranks 2
else skip "MPI steps: no mpirun or LIBCLANG_PATH"; fi
echo; echo "=== smoke: all steps passed in ${SECONDS}s"

#!/usr/bin/env bash
#SBATCH --job-name=ps-hash
#SBATCH --partition=ccq
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The prototype hash-table kernels (`hash-kernel` feature) against the sorted-column engine on an exclusive node: `examples/hash_kernel.rs`, one numactl-pinned process per (cell, threads, warm), memory bound to NUMA node 0, each cell REPEATS times.
#
# Environment knobs (all optional):
#   CELLS    ';'-separated "probe args" (default: rotation and Clifford layers at 1e6 terms, su4, and a growing kicked-Ising run)
#   THREADS  thread counts (default "1 16")
#   TARGETS  bucket targets for the hash engines (default 64,256,1024,4096,16384,inf); the sorted engine runs 256,1024,4096
#   REPEATS  repetitions per cell (default 3)
#   INDEXED  1 (default) also runs the indexed engine; 0 for a PS_REV that predates it
#   PS_REV   commit to build (default: HEAD when the job starts)
#   SMOKE=1  the pre-submit check, run locally as `SMOKE=1 bash scripts/slurm/hash-kernel.sbatch`
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh

if [ "${SMOKE:-0}" = 1 ]; then
    : "${CELLS:=--layers rotation_zz,cnot --n 20000 --reps 2;--layers tfim --qubits 32 --threshold 1e-2 --reps 3}"
    : "${THREADS:=1 2}" "${TARGETS:=256,inf}" "${REPEATS:=1}"
fi
: "${CELLS:=--layers rotation_zz,rotation_wide,cnot,clifford,depolarizing --n 1000000 --reps 10;--layers su4,gu2q --n 100000 --reps 4;--layers tfim --qubits 64 --threshold 1.220703125e-4 --reps 8;--layers heavyhex --qubits 127 --threshold 1.220703125e-4 --reps 3}"
: "${THREADS:=1 16}" "${TARGETS:=64,256,1024,4096,16384,inf}" "${REPEATS:=3}" "${INDEXED:=1}"

use_shared_rust
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in

rev=$(git rev-parse "${PS_REV:-HEAD}")
work="${TMPDIR:-/tmp}/ps-hash-${SLURM_JOB_ID:-local}"
checkout_rev "$rev" "$work"
(cd "$work" && cargo build --offline --release --features hash-kernel --example hash_kernel)
bin="$CARGO_TARGET_DIR/release/examples/hash_kernel"

out="$(results_dir)/hash-kernel-${SLURM_JOB_ID:-local}.txt"
echo "== node $(hostname -s), job ${SLURM_JOB_ID:-local}, rev $rev, $(date -Is)" | tee "$out"
lscpu | grep -E 'Model name|L2|L3|NUMA node0' | tee -a "$out"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$out"

# The first T CPUs of node 0, one per physical core.
cpus_for() {
    lscpu -p=CPU,CORE,NODE | grep -v '^#' | awk -F, '$3 == 0' | sort -t, -k2,2n -u | head -n "$1" | cut -d, -f1 | paste -sd,
}

IFS=';' read -ra cells <<< "$CELLS"
for repeat in $(seq "$REPEATS"); do
    for cell in "${cells[@]}"; do
        for threads in $THREADS; do
            cpus=$(cpus_for "$threads")
            for warm in "" "--warm"; do
                echo "## repeat $repeat threads $threads $warm :: $cell" | tee -a "$out"
                # shellcheck disable=SC2086
                numactl --physcpubind="$cpus" --membind=0 "$bin" $cell --threads "$threads" \
                    --targets "$TARGETS" --engines scatter,orbit,fused $warm | tee -a "$out"
            done
            if [ "$INDEXED" = 1 ]; then
                echo "## repeat $repeat threads $threads indexed :: $cell" | tee -a "$out"
                # shellcheck disable=SC2086
                numactl --physcpubind="$cpus" --membind=0 "$bin" $cell --threads "$threads" \
                    --targets "$TARGETS" --engines indexed | tee -a "$out"
            fi
            echo "## repeat $repeat threads $threads sorted :: $cell" | tee -a "$out"
            # shellcheck disable=SC2086
            numactl --physcpubind="$cpus" --membind=0 "$bin" $cell --threads "$threads" \
                --targets 256,1024,4096 --engines sorted | tee -a "$out"
        done
    done
done
echo "== done $(date -Is); results in $out"
